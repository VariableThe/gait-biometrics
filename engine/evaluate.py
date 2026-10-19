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
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

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


def _eer_operating_point(labels, scores) -> tuple[float, float, float]:
    """Return (eer, far, frr) at the threshold where FAR ~= FRR.

    `labels` are 1 for genuine and 0 for impostor; higher `scores` mean more
    similar. FAR and FRR are linearly interpolated at the crossing.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0:
        raise ValueError("cannot compute EER on empty input")
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        raise ValueError(
            "EER requires both genuine (label 1) and impostor (label 0) samples; "
            f"got {P} genuine / {N} impostor"
        )
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    fnr = 1.0 - tpr
    diff = fpr - fnr  # non-decreasing as the threshold sweeps down
    idx = int(np.searchsorted(diff, 0.0, side="left"))
    if idx <= 0:
        far, frr = float(fpr[0]), float(fnr[0])
    elif idx >= len(diff):
        far, frr = float(fpr[-1]), float(fnr[-1])
    else:
        d0, d1 = diff[idx - 1], diff[idx]
        w = 0.0 if d1 == d0 else (0.0 - d0) / (d1 - d0)
        far = float(fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1]))
        frr = float(fnr[idx - 1] + w * (fnr[idx] - fnr[idx - 1]))
    return 0.5 * (far + frr), far, frr


def compute_eer(labels, scores) -> float:
    """Equal Error Rate: the error rate at the threshold where FAR == FRR."""
    return _eer_operating_point(labels, scores)[0]


def verification_metrics(embeddings, subject_ids, n_pairs: int = 10000,
                         seed: int = 42) -> dict:
    """Form genuine/impostor pairs from `embeddings` and score them.

    Samples `n_pairs // 2` genuine (same-subject) and `n_pairs - n_pairs // 2`
    impostor (different-subject) pairs, computes cosine distances between the
    (L2-normalized) embeddings, and reports EER/FAR/FRR plus ROC AUC on the
    cosine similarities.
    """
    z = F.normalize(torch.as_tensor(embeddings, dtype=torch.float32), p=2, dim=1)
    subs = np.asarray(subject_ids)
    if len(z) != len(subs):
        raise ValueError(
            f"embeddings ({len(z)}) and subject_ids ({len(subs)}) length mismatch"
        )
    subject_to_indices: dict[int, list[int]] = {}
    for idx, s in enumerate(subs):
        subject_to_indices.setdefault(int(s), []).append(idx)
    all_subjects = sorted(subject_to_indices)
    valid = [s for s in all_subjects if len(subject_to_indices[s]) >= 2]
    if len(all_subjects) < 2:
        raise ValueError(
            "verification requires at least 2 subjects; "
            f"found {len(all_subjects)}"
        )
    if not valid:
        raise ValueError("no subject has at least 2 windows; cannot form genuine pairs")

    rng = random.Random(seed)
    n_genuine = int(n_pairs) // 2
    n_impostor = int(n_pairs) - n_genuine

    genuine: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n_genuine * 100 + 1000
    while len(genuine) < n_genuine and attempts < max_attempts:
        attempts += 1
        s = rng.choice(valid)
        i, j = rng.sample(subject_to_indices[s], 2)
        genuine.append((i, j))
    impostor: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n_impostor * 100 + 1000
    while len(impostor) < n_impostor and attempts < max_attempts:
        attempts += 1
        s1, s2 = rng.sample(all_subjects, 2)
        i = rng.choice(subject_to_indices[s1])
        j = rng.choice(subject_to_indices[s2])
        impostor.append((i, j))

    idx1 = [i for i, _ in genuine] + [i for i, _ in impostor]
    idx2 = [j for _, j in genuine] + [j for _, j in impostor]
    labels = np.array([1] * len(genuine) + [0] * len(impostor), dtype=np.int64)
    z1 = z[torch.tensor(idx1)]
    z2 = z[torch.tensor(idx2)]
    dist = (1.0 - (z1 * z2).sum(dim=1)).clamp(min=0.0).numpy()  # cosine distance
    scores = -dist  # similarity: higher = more similar

    eer, far, frr = _eer_operating_point(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    return {
        "eer": float(eer),
        "far": float(far),
        "frr": float(frr),
        "roc_auc": roc_auc,
        "n_pairs": int(len(labels)),
    }


def _load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by `save_checkpoint` or a bare state_dict.

    Delegates to `models.common.load_checkpoint`, which always unpickles with
    ``weights_only=True`` so a checkpoint cannot execute arbitrary code on load.
    """
    return load_checkpoint(path)


def _strip_embedding_prefix(state: dict) -> dict:
    """Strip the SiameseNet `embedding.` prefix so keys match `EmbeddingNet`."""
    if state and all(k.startswith("embedding.") for k in state):
        return {k[len("embedding."):]: v for k, v in state.items()}
    return state


def _eval_activity(config: dict, checkpoint: str, subset: int | None,
                   out_dir: str) -> dict:
    source = config["source"]
    data = load_windows(
        source,
        "test",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
    )
    if subset is not None:
        data = _stratified_subset(data, int(subset), int(config.get("seed", 42)))

    n_channels = int(data.X.shape[1])
    n_classes = len(data.activity_names)
    model = build_model(config["model"], config.get("model_args", {}), n_channels, n_classes)
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

    print(f"activity evaluation: model={config['model']} source={source} n={len(y_true)}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    for c, name in enumerate(data.activity_names):
        m = metrics["per_class"][str(c)]
        print(f"  {name:<20s} precision={m['precision']:.4f} recall={m['recall']:.4f}")
    print(f"wrote {path}")
    return result


def _eval_verification(config: dict, checkpoint: str, subset: int | None,
                       out_dir: str) -> dict:
    source = config["source"]
    data = load_windows(
        source,
        "test",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
    )
    if data.y_subject is None:
        raise ValueError(
            f"verification requires subject labels, but source {source!r} "
            "has y_subject=None; genuine/impostor pairs cannot be formed. "
            "Use source 'har'."
        )
    if subset is not None:
        data = _stratified_subset(data, int(subset), int(config.get("seed", 42)))

    from data.adapter import build_loader

    model_args = dict(config.get("model_args", {}))
    embed_dim = int(model_args.get("embed_dim", 128))
    dropout = float(model_args.get("dropout", 0.3))
    emb_net = EmbeddingNet(int(data.X.shape[1]), embed_dim=embed_dim, dropout=dropout)
    emb_net.load_state_dict(_strip_embedding_prefix(_load_checkpoint(checkpoint)))
    emb_net.eval()

    loader = build_loader(
        data,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    embs = []
    with torch.no_grad():
        for batch in loader:
            embs.append(emb_net(batch[0]).numpy())
    embeddings = np.concatenate(embs)

    n_pairs = int(config.get("verification_pairs", 10000))
    metrics = verification_metrics(
        embeddings, data.y_subject.numpy(), n_pairs=n_pairs,
        seed=int(config.get("seed", 42)),
    )
    result = {
        "task": "verification",
        "source": source,
        "subset": int(subset) if subset is not None else None,
        "n_samples": int(len(embeddings)),
        "embed_dim": embed_dim,
        **metrics,
        "git_commit": _git_commit(),
    }
    path = os.path.join(out_dir, "eval_verification.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"verification evaluation: source={source} n={len(embeddings)} "
        f"embed_dim={embed_dim}"
    )
    print(
        f"eer={metrics['eer']:.4f} far={metrics['far']:.4f} frr={metrics['frr']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} n_pairs={metrics['n_pairs']}"
    )
    print(f"wrote {path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint (activity or verification)"
    )
    parser.add_argument("--checkpoint", required=True, help="path to the .pt checkpoint")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument(
        "--task", required=True, choices=["activity", "verification"],
        help="evaluation task",
    )
    parser.add_argument("--subset", type=int, default=None,
                        help="evaluate on a stratified subset of the test split")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: checkpoint's parent)")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed_everything(int(config.get("seed", 42)))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    if args.task == "activity":
        _eval_activity(config, args.checkpoint, args.subset, out_dir)
    else:
        _eval_verification(config, args.checkpoint, args.subset, out_dir)


if __name__ == "__main__":
    main()
