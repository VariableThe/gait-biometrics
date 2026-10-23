"""Evaluation entry points: activity classification and gait verification.

CLI:
    python -m engine.evaluate --checkpoint PATH --config PATH \
        --task {activity,verification} --subset INT [--out-dir PATH]

`--task activity` evaluates a registered classification checkpoint on the TEST
split and writes `<out_dir>/eval_activity.json`. `--task verification` loads
a siamese `EmbeddingNet` checkpoint, embeds the TEST windows, forms genuine
(same-subject) and impostor (different-subject) pairs, and writes
`<out_dir>/eval_verification.json`. `--out-dir` defaults to the checkpoint's
parent directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from data.adapter import load_windows
from engine.train import _git_commit, _stratified_subset
from models.common import load_checkpoint, seed_everything
from models.registry import build_model
from models.siamese import EmbeddingNet


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    """Accuracy, per-class precision/recall, and the confusion matrix.

    Rows of the confusion matrix are true classes, columns are predicted.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    accuracy = float((y_true == y_pred).sum() / n) if n else 0.0
    per_class: dict[str, dict[str, float]] = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[str(c)] = {"precision": float(precision), "recall": float(recall)}
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _validated_scores(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be equally sized 1D arrays")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be nonempty and binary (1 genuine, 0 impostor)")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not (np.any(labels == 1) and np.any(labels == 0)):
        raise ValueError("verification requires both genuine and impostor samples")
    return labels.astype(np.int64), scores


def _roc(labels, scores):
    labels, scores = _validated_scores(labels, scores)
    far, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # Arrays have equal lengths. Replace sklearn's +inf reject-all sentinel
    # with a finite threshold. Tied scores move together; do not interpolate it.
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    if not np.isfinite(thresholds[0]):
        raise ValueError("scores too large for a finite reject-all threshold")
    n_genuine = int((labels == 1).sum())
    # Compute count fractions to avoid 1 - .99 > .01 rejecting an attainable
    # FRR of exactly 1% because of subtraction roundoff.
    frr = np.rint((1.0 - tpr) * n_genuine) / n_genuine
    return far, frr, thresholds


def _eer_operating_point(labels, scores) -> tuple[float, float, float]:
    """Descriptive interpolated ROC crossing, not a realizable decision rule."""
    far, frr, _ = _roc(labels, scores)
    diff = far - frr
    hi = int(np.searchsorted(diff, 0.0))
    if diff[hi] == 0:
        eer = float(far[hi])
    else:
        lo = hi - 1
        weight = -diff[lo] / (diff[hi] - diff[lo])
        eer = float(far[lo] + weight * (far[hi] - far[lo]))
    return eer, eer, eer


def compute_eer(labels, scores) -> float:
    """EER with score ties grouped; linear interpolation is descriptive only."""
    return _eer_operating_point(labels, scores)[0]


def fixed_threshold_metrics(labels, scores, threshold: float) -> dict:
    """Count actual decisions using score >= threshold (ties are accepted)."""
    labels, scores = _validated_scores(labels, scores)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    accepted = scores >= threshold
    genuine, impostor = labels == 1, labels == 0
    fa = int(np.count_nonzero(accepted & impostor))
    fr = int(np.count_nonzero(~accepted & genuine))
    return {
        "threshold": threshold,
        "far": fa / int(impostor.sum()),
        "frr": fr / int(genuine.sum()),
        "false_accepts": fa,
        "false_rejects": fr,
        "n_genuine": int(genuine.sum()),
        "n_impostor": int(impostor.sum()),
        "decision_rule": "score >= threshold",
    }


def calibrate_threshold(labels, scores, target_far: float = 0.01) -> dict:
    """Least restrictive empirical threshold with FAR <= target.

    Caller must supply independent validation scores, never evaluation scores.
    """
    if not np.isfinite(target_far) or not 0 <= target_far <= 1:
        raise ValueError("target_far must be between 0 and 1")
    far, _, thresholds = _roc(labels, scores)
    index = int(np.flatnonzero(far <= target_far)[-1])
    return {
        **fixed_threshold_metrics(labels, scores, thresholds[index]),
        "target_far": float(target_far),
        "selection": "empirical_far_at_most_target",
    }


def _integer_ids(values, n: int, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    ids = np.asarray(values)
    if ids.ndim != 1 or len(ids) != n or ids.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a 1D integer array with one label per window")
    return ids


def _verification_pairs(subject_ids, n_pairs: int, seed: int, activity_ids=None):
    """Balanced subject-uniform sampling WITH replacement and no self-pairs.

    Activity strata have near-identical genuine/impostor budgets when supplied.
    Storage is O(N + n_pairs); repeated unordered pairs are counted explicitly.
    """
    if (
        isinstance(n_pairs, bool)
        or not isinstance(n_pairs, (int, np.integer))
        or n_pairs < 2
    ):
        raise ValueError("n_pairs must be an integer >= 2")
    subs = _integer_ids(subject_ids, len(subject_ids), "subject_ids")
    activities = (
        np.zeros(len(subs), dtype=np.int64)
        if activity_ids is None
        else _integer_ids(activity_ids, len(subs), "activity_ids")
    )
    groups = {}
    for i, (subject, activity) in enumerate(zip(subs, activities)):
        groups.setdefault(int(activity), {}).setdefault(int(subject), []).append(i)
    eligible = {
        a: g
        for a, g in sorted(groups.items())
        if len(g) >= 2 and any(len(rows) >= 2 for rows in g.values())
    }
    if not eligible:
        raise ValueError(
            "cannot form genuine/impostor pairs: need >=2 subjects and a subject "
            "with >=2 windows in a shared activity; enlarge subset or check activity_ids"
        )
    rng = random.Random(seed)
    strata = list(eligible)
    rng.shuffle(strata)
    subjects = {a: sorted(g) for a, g in eligible.items()}
    positive_subjects = {
        a: [s for s in subjects[a] if len(g[s]) >= 2] for a, g in eligible.items()
    }
    counts = [int(n_pairs) // 2, int(n_pairs) - int(n_pairs) // 2]
    pairs, labels = [], []
    for genuine, count in zip((True, False), counts):
        for k in range(count):
            activity = strata[k % len(strata)]
            group = eligible[activity]
            if genuine:
                subject = rng.choice(positive_subjects[activity])
                i, j = rng.sample(group[subject], 2)
            else:
                s1, s2 = rng.sample(subjects[activity], 2)
                i, j = rng.choice(group[s1]), rng.choice(group[s2])
            pairs.append((min(i, j), max(i, j)))
            labels.append(int(genuine))
    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    unique = [len(set(map(tuple, pairs[labels == lab]))) for lab in (1, 0)]
    per_subject = {}
    for s in np.unique(subs):
        touches = (subs[pairs[:, 0]] == s) | (subs[pairs[:, 1]] == s)
        per_subject[str(int(s))] = {
            "n_windows": int(np.count_nonzero(subs == s)),
            "n_genuine_pairs": int(np.count_nonzero(touches & (labels == 1))),
            "n_impostor_pairs": int(np.count_nonzero(touches & (labels == 0))),
        }
    context = {
        "sampling": "balanced_subject_uniform_with_replacement",
        "pairing_version": 2,
        "seed": int(seed),
        "requested_n_pairs": int(n_pairs),
        "n_genuine_pairs": counts[0],
        "n_impostor_pairs": counts[1],
        "n_unique_genuine_pairs": unique[0],
        "n_unique_impostor_pairs": unique[1],
        "n_unique_pairs": sum(unique),
        "n_repeated_pairs": int(n_pairs) - sum(unique),
        "same_activity_pairs": activity_ids is not None,
        "eligible_activity_ids": list(eligible) if activity_ids is not None else None,
        "excluded_activity_ids": sorted(set(groups) - set(eligible))
        if activity_ids is not None
        else [],
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
        "n_windows_used": int(len(np.unique(pairs))),
        "pair_independence": "not_independent_windows_and_pairs_can_recur",
        "per_activity": {
            str(a): {
                "n_genuine_pairs": int(
                    np.count_nonzero((activities[pairs[:, 0]] == a) & (labels == 1))
                ),
                "n_impostor_pairs": int(
                    np.count_nonzero((activities[pairs[:, 0]] == a) & (labels == 0))
                ),
            }
            for a in groups
        }
        if activity_ids is not None
        else None,
    }
    return pairs, labels, context


def _verification_scores(embeddings, subject_ids, n_pairs, seed, activity_ids=None):
    z = torch.as_tensor(embeddings).detach().to(device="cpu", dtype=torch.float64)
    if z.ndim != 2 or not z.shape[0] or not z.shape[1]:
        raise ValueError("embeddings must be a nonempty (N, D) matrix")
    subs = _integer_ids(subject_ids, len(z), "subject_ids")
    norms = torch.linalg.vector_norm(z, dim=1)
    if (
        not torch.isfinite(z).all()
        or not torch.isfinite(norms).all()
        or (norms == 0).any()
    ):
        raise ValueError("embeddings must be finite with nonzero row norms")
    z = z / norms[:, None]
    pairs, labels, context = _verification_pairs(subs, n_pairs, seed, activity_ids)
    # Preserve historical negative-cosine-distance scale: [-2, 0].
    scores = ((z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1).clamp(-1.0, 1.0) - 1.0).numpy()
    return labels, scores, context


def verification_metrics(
    embeddings,
    subject_ids,
    n_pairs: int = 10000,
    seed: int = 42,
    *,
    activity_ids=None,
    threshold: float | None = None,
) -> dict:
    """Descriptive pair metrics; supply an independently calibrated threshold.

    Original arguments and EER/AUC remain supported. ``far`` and ``frr`` keep
    legacy interpolated-EER aliases; ``fixed_threshold`` counts actual decisions.
    """
    labels, scores, context = _verification_scores(
        embeddings, subject_ids, n_pairs, seed, activity_ids
    )
    eer = compute_eer(labels, scores)
    far, frr, thresholds = _roc(labels, scores)
    nearest = int(np.argmin(np.abs(far - frr)))  # ties prefer stricter threshold
    at_far = int(np.flatnonzero(far <= 0.01)[-1])
    at_frr = int(np.flatnonzero(frr <= 0.01)[0])
    result = {
        "eer": eer,
        "far": eer,
        "frr": eer,
        "legacy_far_frr_semantics": "aliases_of_interpolated_eer_not_independent_rates",
        "eer_threshold": float(thresholds[nearest]),
        "eer_threshold_semantics": "nearest_empirical_crossing_on_evaluation_not_deployable",
        "far_at_eer_threshold": float(far[nearest]),
        "frr_at_eer_threshold": float(frr[nearest]),
        "far_at_frr_0.01": float(far[at_frr]),
        "frr_at_far_0.01": float(frr[at_far]),
        "operating_points_semantics": "empirical_constraint_at_most_0.01_on_evaluation",
        "roc_auc": float(roc_auc_score(labels, scores)),
        "n_pairs": len(labels),
        "score_semantics": "cosine_similarity_minus_one_higher_is_more_genuine",
        **context,
    }
    if threshold is not None:
        result["fixed_threshold"] = fixed_threshold_metrics(labels, scores, threshold)
    return result


def _load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by `save_checkpoint` or a bare state_dict.

    Delegates to `models.common.load_checkpoint`, which always unpickles with
    ``weights_only=True`` so a checkpoint cannot execute arbitrary code on load.
    """
    return load_checkpoint(path)


def _strip_embedding_prefix(state: dict) -> dict:
    """Strip the SiameseNet `embedding.` prefix so keys match `EmbeddingNet`."""
    if state and all(k.startswith("embedding.") for k in state):
        return {k[len("embedding.") :]: v for k, v in state.items()}
    return state


def _eval_activity(
    config: dict, checkpoint: str, subset: int | None, out_dir: str
) -> dict:
    source = config["source"]
    protocol = _checkpoint_protocol(checkpoint)
    data = _evaluation_data(config, "test", protocol)
    if subset is not None:
        data = _stratified_subset(data, int(subset), int(config.get("seed", 42)))

    n_channels = int(data.X.shape[1])
    n_classes = len(data.activity_names)
    model = build_model(
        config["model"], config.get("model_args", {}), n_channels, n_classes
    )
    model.load_state_dict(_load_checkpoint(checkpoint))
    model.eval()

    from data.adapter import build_loader

    loader = build_loader(
        data,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            out = model(xb)
            if isinstance(out, tuple):  # multitask: (activity_logits, subject_logits)
                out = out[0]
            y_true.append(yb.numpy())
            y_pred.append(out.argmax(1).numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = classification_metrics(y_true, y_pred, n_classes)
    result = {
        "task": "activity",
        "model": config["model"],
        "source": source,
        "subset": int(subset) if subset is not None else None,
        "n_samples": int(len(y_true)),
        "accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "activity_names": data.activity_names,
        "git_commit": _git_commit(),
    }
    path = os.path.join(out_dir, "eval_activity.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"activity evaluation: model={config['model']} source={source} n={len(y_true)}"
    )
    print(f"accuracy: {metrics['accuracy']:.4f}")
    for c, name in enumerate(data.activity_names):
        m = metrics["per_class"][str(c)]
        print(f"  {name:<20s} precision={m['precision']:.4f} recall={m['recall']:.4f}")
    print(f"wrote {path}")
    return result


def _checkpoint_protocol(checkpoint: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    protocol = (
        payload.get("metrics", {}).get("protocol", {})
        if isinstance(payload, dict)
        else {}
    )
    if not isinstance(protocol, dict):
        raise ValueError("checkpoint metrics.protocol must be a mapping")
    return protocol


def _take_windows(data, indices):
    from data.adapter import WindowData

    return WindowData(
        X=data.X[indices],
        y_activity=data.y_activity[indices],
        y_subject=data.y_subject[indices] if data.y_subject is not None else None,
        activity_names=data.activity_names,
        subject_ids=sorted(torch.unique(data.y_subject[indices]).tolist())
        if data.y_subject is not None
        else [],
        sampling_rate=data.sampling_rate,
    )


def _evaluation_data(config: dict, split: str, protocol: dict):
    source = config["source"]
    if protocol.get("source", source) != source:
        raise ValueError("config source differs from checkpoint protocol")
    if protocol.get("normalization", "none") != "none":
        raise ValueError(
            "checkpoint normalization requires an explicit matching inference transform"
        )
    for key in ("window_size", "stride"):
        if key in config and key in protocol and config[key] != protocol[key]:
            raise ValueError(f"config {key} differs from checkpoint protocol")
    activities = config.get("activity_ids", protocol.get("activity_ids"))
    if "activity_ids" in protocol and activities != protocol["activity_ids"]:
        raise ValueError("config activity_ids differ from checkpoint protocol")
    data = load_windows(
        source,
        split,
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
        data_root=config.get("data_root", "datasets"),
    )
    for key, actual in (
        ("n_channels", data.X.shape[1]),
        ("window_size", data.X.shape[2]),
        ("sampling_rate", data.sampling_rate),
    ):
        if key in protocol and protocol[key] != actual:
            raise ValueError(f"loaded {key} differs from checkpoint protocol")
    if activities is not None:
        ids = _integer_ids(activities, len(activities), "activity_ids")
        if not len(ids) or np.any(ids < 0) or np.any(ids >= len(data.activity_names)):
            raise ValueError(
                "activity_ids must name existing zero-indexed activity labels"
            )
        data = _take_windows(
            data, torch.from_numpy(np.isin(data.y_activity.numpy(), ids))
        )
    if not len(data.X):
        raise ValueError("no evaluation windows remain; check source and activity_ids")
    return data


def _embed_evaluation(model, data, config):
    from data.adapter import build_loader

    requested = str(config.get("device", "cpu"))
    device = torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    model.to(device).eval()
    loader = build_loader(
        data,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    with torch.inference_mode():
        return np.concatenate(
            [model(batch[0].to(device)).cpu().numpy() for batch in loader]
        )


def _eval_verification(
    config: dict,
    checkpoint: str,
    subset: int | None,
    out_dir: str,
    *,
    calibrate: bool = False,
    calibration_target_far: float = 0.01,
) -> dict:
    protocol = _checkpoint_protocol(checkpoint)
    data = _evaluation_data(config, "test", protocol)
    if data.y_subject is None:
        raise ValueError(
            "verification requires authentic subject labels; use har or wisdm_v1, not anonymous WISDM CSV"
        )
    test_subjects = set(data.y_subject.tolist())
    known_subjects = set(protocol.get("train_subject_ids", [])) | set(
        protocol.get("val_subject_ids", [])
    )
    if test_subjects & known_subjects:
        raise ValueError(
            "test subjects overlap checkpoint training/validation subjects"
        )
    if subset is not None:
        if subset < 1:
            raise ValueError("subset must be positive")
        data = _stratified_subset(data, int(subset), int(config.get("seed", 42)))
    model_args = dict(config.get("model_args", {}))
    embed_dim = int(model_args.get("embed_dim", 128))
    emb_net = EmbeddingNet(
        int(data.X.shape[1]),
        embed_dim=embed_dim,
        dropout=float(model_args.get("dropout", 0.3)),
    )
    emb_net.load_state_dict(_strip_embedding_prefix(_load_checkpoint(checkpoint)))
    n_pairs, seed = config.get("verification_pairs", 10000), int(config.get("seed", 42))
    pairing = config.get(
        "verification_pairing", protocol.get("verification_pairing", "same_activity")
    )
    if pairing not in ("same_activity", "unrestricted"):
        raise ValueError("verification_pairing must be same_activity or unrestricted")
    if (
        "verification_pairing" in protocol
        and pairing != protocol["verification_pairing"]
    ):
        raise ValueError("config verification_pairing differs from checkpoint protocol")
    calibration = None
    if calibrate:
        train_ids, val_ids = (
            protocol.get("train_subject_ids"),
            protocol.get("val_subject_ids"),
        )
        if not train_ids or not val_ids:
            raise ValueError(
                "--calibrate requires checkpoint protocol with train_subject_ids and held-out val_subject_ids; retrain with val_subjects >= 2"
            )
        if set(train_ids) & set(val_ids) or test_subjects & set(val_ids):
            raise ValueError(
                "calibration subjects must be disjoint from training and test subjects"
            )
        val_data = _evaluation_data(config, "train", protocol)
        if val_data.y_subject is None:
            raise ValueError("calibration source requires subject labels")
        val_data = _take_windows(
            val_data, torch.from_numpy(np.isin(val_data.y_subject.numpy(), val_ids))
        )
        if set(val_data.y_subject.tolist()) != set(val_ids):
            raise ValueError(
                "checkpoint validation subjects missing from loaded calibration data"
            )
        labels, scores, context = _verification_scores(
            _embed_evaluation(emb_net, val_data, config),
            val_data.y_subject,
            n_pairs,
            seed,
            val_data.y_activity if pairing == "same_activity" else None,
        )
        calibration = {
            **calibrate_threshold(labels, scores, calibration_target_far),
            "source_split": "train_held_out_subjects",
            "subject_ids": sorted(val_ids),
            "n_samples": len(val_data.X),
            "pair_context": context,
            "limitation": "validation also used for checkpoint selection; cross-session calibration unverified",
        }
    embeddings = _embed_evaluation(emb_net, data, config)
    metrics = verification_metrics(
        embeddings,
        data.y_subject,
        n_pairs=n_pairs,
        seed=seed,
        activity_ids=data.y_activity if pairing == "same_activity" else None,
        threshold=calibration["threshold"] if calibration is not None else None,
    )
    result = {
        "task": "verification",
        "source": config["source"],
        "subset": subset,
        "n_samples": len(embeddings),
        "embed_dim": embed_dim,
        **metrics,
        "protocol": "held_out_subject_within_recording_discrimination"
        if known_subjects
        else "within_recording_discrimination_training_subjects_unverified",
        "checkpoint_protocol": protocol,
        "calibration": calibration,
        "activity_ids": sorted(torch.unique(data.y_activity).tolist()),
        "activity_names": data.activity_names,
        "session_separation": "unverified_no_session_ids_in_WindowData",
        "overlap_exclusion": "not_possible_without_recording_offsets",
        "deployment_ready": False,
        "git_commit": _git_commit(),
    }
    path = os.path.join(out_dir, "eval_verification.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        f"verification source={config['source']} n={len(embeddings)} pairing={pairing}"
    )
    print(
        f"descriptive eer={metrics['eer']:.4f} roc_auc={metrics['roc_auc']:.4f} n_pairs={metrics['n_pairs']} unique={metrics['n_unique_pairs']}"
    )
    if calibration is not None:
        fixed = metrics["fixed_threshold"]
        print(
            f"validation-calibrated test FAR={fixed['far']:.4f} FRR={fixed['frr']:.4f} threshold={fixed['threshold']:.6f}"
        )
    print(f"wrote {path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint (activity or verification)"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to the .pt checkpoint"
    )
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument(
        "--task",
        required=True,
        choices=["activity", "verification"],
        help="evaluation task",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="evaluate on a stratified subset of the test split",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: checkpoint's parent)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate fixed threshold on checkpoint held-out validation subjects",
    )
    parser.add_argument("--calibration-target-far", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.calibrate and args.task != "verification":
        parser.error("--calibrate applies only to --task verification")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed_everything(int(config.get("seed", 42)))
    torch.set_num_threads(int(config.get("torch_threads", 2)))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    if args.task == "activity":
        _eval_activity(config, args.checkpoint, args.subset, out_dir)
    else:
        _eval_verification(
            config,
            args.checkpoint,
            args.subset,
            out_dir,
            calibrate=args.calibrate,
            calibration_target_far=args.calibration_target_far,
        )


if __name__ == "__main__":
    main()
