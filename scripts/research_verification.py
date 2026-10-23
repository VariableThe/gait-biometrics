"""Validation-only research selection, explicit locking, then held-out scoring.

Run --stage train, then --stage lock, then --stage evaluate. Test data is loaded
only by the last stage after the saved validation registry is hash-locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.adapter import load_windows
from engine.evaluate import (
    _strip_embedding_prefix,
    _verification_scores,
    calibrate_threshold,
    verification_metrics,
)
from engine.train import _filter_activities, _split_validation
from models.common import load_checkpoint
from models.gait_features import extract_gait_features
from models.gait_metric import WithinClassMetric
from models.siamese import EmbeddingNet

METHODS = [
    ("legacy_hard", "legacy", "batch_hard", False),
    ("multiscale_hard", "multiscale", "batch_hard", False),
    ("multiscale_supcon", "multiscale", "supcon", False),
    ("multiscale_hybrid", "multiscale", "hybrid", False),
    ("temporal_hybrid", "temporal", "hybrid", False),
    ("multiscale_stats_hybrid", "multiscale", "hybrid", True),
]


def selection_key(row):
    """Predeclared tie rule: fewer processing stages before method order."""
    method_order = {name: index for index, (name, *_) in enumerate(METHODS)}
    return (
        row["validation"]["eer"],
        row["method"] != "features",
        row["postprocess"] not in ("standard", "none"),
        method_order.get(row["method"], -1),
        row["name"],
    )


def development_fingerprint(train, val):
    checksum = hashlib.sha256()
    for data in (train, val):
        for values in (data.X, data.y_subject, data.y_activity):
            checksum.update(values.contiguous().numpy().tobytes())
    return checksum.hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def data_for(source, split, root):
    return _filter_activities(load_windows(source, split, 128, 64, str(root)), [0])


def encode(model, data, device, batch_size=128):
    model.to(device).eval()
    with torch.inference_mode():
        return np.concatenate(
            [
                model(chunk.to(device)).cpu().numpy()
                for chunk in data.X.split(batch_size)
            ]
        )


def metrics(embeddings, data, threshold=None):
    return verification_metrics(
        embeddings,
        data.y_subject.numpy(),
        seed=42,
        activity_ids=data.y_activity.numpy(),
        threshold=threshold,
    )


def validate_training_run(expected, checkpoint_config, protocol, completed):
    """A sidecar alone cannot certify the seed or completed optimizer budget."""
    if any(checkpoint_config.get(key) != value for key, value in expected.items()):
        raise ValueError("actual checkpoint configuration differs from registered run")
    for key in ("epochs", "seed", "split_seed", "val_seed"):
        if completed.get(key) != expected[key]:
            raise ValueError("completed training metadata differs from registered run")
    if [row["epoch"] for row in completed.get("per_epoch", [])] != list(
        range(1, expected["epochs"] + 1)
    ):
        raise ValueError(
            "completed training history does not cover the registered epoch budget"
        )
    for key, expected_key in (
        ("train_seed", "seed"),
        ("split_seed", "split_seed"),
        ("val_seed", "val_seed"),
    ):
        if protocol.get(key) != expected[expected_key]:
            raise ValueError("actual checkpoint seed differs from registered run")
    if (
        protocol.get("config") != checkpoint_config
        or completed.get("protocol") != protocol
    ):
        raise ValueError("checkpoint and completed training protocols differ")


def train_stage(args, source):
    from engine.train_metric import load_metric_encoder, train_metric

    folder = args.out_root / source
    if (folder / "selection_lock.json").exists():
        raise ValueError(
            "selection already locked; use a new output root for a new study"
        )
    data = data_for(source, "train", args.data_root)
    train, val, _ = _split_validation(data, 4, 42)
    registry = {
        "source": source,
        "split_seed": 42,
        "pair_seed": 42,
        "train_subject_ids": train.subject_ids,
        "val_subject_ids": val.subject_ids,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "candidates": [],
        "test_loaded": False,
        "selection_rule": "minimum validation ensemble EER; ties prefer fewer stages then registered method order",
        "data_root": str(args.data_root.resolve()),
        "registered_methods": METHODS,
        "development_data_sha256": development_fingerprint(train, val),
    }
    registry_path = folder / "validation_registry.json"
    if (
        registry_path.exists()
        and json.loads(registry_path.read_text()).get("development_data_sha256")
        != registry["development_data_sha256"]
    ):
        raise ValueError(
            "development data changed since study training began; use a fresh output root"
        )
    write(registry_path, registry)

    def record(
        name, method, post, embeddings, artifacts, training_seconds, seed_metrics=None
    ):
        cache = folder / "validation" / f"{name}.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            embeddings=embeddings,
            subjects=val.y_subject.numpy(),
            activities=val.y_activity.numpy(),
        )
        result = {
            "name": name,
            "method": method,
            "postprocess": post,
            "validation": metrics(embeddings, val),
            "artifacts": artifacts,
            "validation_cache": str(cache.relative_to(folder)),
            "training_seconds": training_seconds,
            "per_seed_validation": seed_metrics,
            "n_stochastic_seeds": len(args.seeds) if method != "features" else 0,
        }
        registry["candidates"].append(result)
        write(folder / "validation_registry.json", registry)
        print(source, name, "validation EER", result["validation"]["eer"], flush=True)

    training_features = extract_gait_features(train.X, train.sampling_rate)
    validation_features = extract_gait_features(val.X, val.sampling_rate)
    for label, mode, shrink in [
        ("standard", "standard", "auto"),
        ("wccn_auto", "wccn", "auto"),
        ("wccn_0.2", "wccn", 0.2),
    ]:
        started = time.perf_counter()
        metric = WithinClassMetric(mode, shrink).fit(
            training_features, train.y_subject.numpy()
        )
        path = folder / "features" / f"{label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **metric.state_dict())
        record(
            "features_" + label,
            "features",
            label,
            metric.transform(validation_features),
            [str(path.relative_to(folder))],
            time.perf_counter() - started,
        )
    for name, architecture, objective, stats in METHODS:
        checkpoints, train_embeddings, val_embeddings, elapsed = [], [], [], 0.0
        for seed in args.seeds:
            path = folder / name / f"seed_{seed}"
            config = {
                "source": source,
                "data_root": str(args.data_root.resolve()),
                "activity_ids": [0],
                "epochs": args.epochs,
                "seed": seed,
                "split_seed": 42,
                "val_seed": 42,
                "val_subjects": 4,
                "validation_pairs": 10000,
                "objective": objective,
                "identities_per_batch": 8,
                "windows_per_identity": 4,
                "ce_weight": 0.5,
                "label_smoothing": 0.05,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "margin": 0.2,
                "temperature": 0.1,
                "device": args.device,
                "torch_threads": 2,
                "augment": {
                    "jitter_std": 0.01,
                    "scale_std": 0.05,
                    "roll_fraction": 0.05,
                },
                "model_args": {
                    "architecture": architecture,
                    "embed_dim": 128,
                    "width": 32,
                    "dropout": 0.1,
                    "stats_fusion": stats,
                    "normalization": True,
                },
            }
            config_path = path / "effective_config.json"
            if (path / "best.pt").exists():
                if (
                    not config_path.exists()
                    or json.loads(config_path.read_text()) != config
                ):
                    raise ValueError(
                        "existing checkpoint config differs from study plan"
                    )
                training_result = json.loads((path / "metrics.json").read_text())
            else:
                write(config_path, config)
                training_result = train_metric(config, str(path))
            elapsed += float(training_result.get("elapsed_seconds", 0))
            model, protocol, checkpoint_config = load_metric_encoder(
                str(path / "best.pt"), device=args.device
            )
            validate_training_run(config, checkpoint_config, protocol, training_result)
            if set(protocol["train_subject_ids"]) != set(train.subject_ids) or set(
                protocol["val_subject_ids"]
            ) != set(val.subject_ids):
                raise ValueError(
                    "checkpoint subject cohorts differ from fixed study split"
                )
            train_embeddings.append(encode(model, train, args.device))
            if (
                protocol.get("development_data_sha256")
                != registry["development_data_sha256"]
            ):
                raise ValueError("checkpoint was trained on different development data")
            val_embeddings.append(encode(model, val, args.device))
            checkpoints.append(str((path / "best.pt").relative_to(folder)))
            del model
        for post in ("none", "wccn_auto"):
            transformed = []
            artifacts = list(checkpoints)
            seeds_results = []
            for index, seed in enumerate(args.seeds):
                value = val_embeddings[index]
                if post == "wccn_auto":
                    metric = WithinClassMetric().fit(
                        train_embeddings[index], train.y_subject.numpy()
                    )
                    value = metric.transform(value)
                    path = folder / name / f"metric_{seed}.npz"
                    np.savez(path, **metric.state_dict())
                    artifacts.append(str(path.relative_to(folder)))
                transformed.append(value)
                seeds_results.append({"seed": seed, "eer": metrics(value, val)["eer"]})
            ensemble = np.concatenate(transformed, axis=1) / np.sqrt(len(transformed))
            record(
                name + "_" + post,
                name,
                post,
                ensemble,
                artifacts,
                elapsed,
                seeds_results,
            )
    return registry


def lock_stage(args, source):
    from engine.train_metric import load_metric_encoder

    folder = args.out_root / source
    registry_path = folder / "validation_registry.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("epochs") != 40 or registry.get("seeds") != [42, 43, 44]:
        raise ValueError(
            "only the complete preregistered 40-epoch, three-seed study can unlock test access"
        )
    expected_names = {
        "features_standard",
        "features_wccn_auto",
        "features_wccn_0.2",
        *(name + "_" + post for name, *_ in METHODS for post in ("none", "wccn_auto")),
    }
    candidates = registry["candidates"]
    for row in candidates:
        if row["method"] != "features" and (
            row.get("n_stochastic_seeds") != 3
            or [result["seed"] for result in row.get("per_seed_validation", [])]
            != [42, 43, 44]
        ):
            raise ValueError(
                "every neural candidate requires all three registered seeds"
            )
    if (
        len(candidates) != len(expected_names)
        or {row["name"] for row in candidates} != expected_names
    ):
        raise ValueError(
            f"study incomplete: expected {len(expected_names)} distinct validation candidates"
        )
    if (
        registry["source"] != source
        or registry["test_loaded"]
        or registry["registered_methods"] != [list(method) for method in METHODS]
    ):
        raise ValueError("registry source, development-only status or methods differ")
    if any(
        not np.isfinite(row["validation"]["eer"])
        or not 0 <= row["validation"]["eer"] <= 1
        for row in candidates
    ):
        raise ValueError("validation EER must be finite and in [0,1]")
    for name, *_ in METHODS:
        for seed in registry["seeds"]:
            run = folder / name / f"seed_{seed}"
            expected = json.loads((run / "effective_config.json").read_text())
            if (
                expected["epochs"] != 40
                or expected["seed"] != seed
                or expected["split_seed"] != 42
                or expected["val_seed"] != 42
            ):
                raise ValueError("run sidecar differs from registered budget or seeds")
            encoder, protocol, actual = load_metric_encoder(str(run / "best.pt"))
            validate_training_run(
                expected,
                actual,
                protocol,
                json.loads((run / "metrics.json").read_text()),
            )
            if (
                protocol["development_data_sha256"]
                != registry["development_data_sha256"]
            ):
                raise ValueError(
                    "checkpoint development data differs from pre-training registry"
                )
            del encoder
    train, val, _ = _split_validation(data_for(source, "train", args.data_root), 4, 42)
    if development_fingerprint(train, val) != registry.get("development_data_sha256"):
        raise ValueError("development data differs from the pre-training registry")
    if (
        train.subject_ids != registry["train_subject_ids"]
        or val.subject_ids != registry["val_subject_ids"]
    ):
        raise ValueError("registry development subjects differ from current data")
    winner = min(candidates, key=selection_key)
    artifacts = winner["artifacts"] + [winner["validation_cache"]]
    lock = {
        "source": source,
        "selection": winner,
        "registry_sha256": digest(registry_path),
        "artifact_sha256": {name: digest(folder / name) for name in artifacts},
        "test_loaded_before_lock": False,
        "seeds": registry["seeds"],
        "data_root": registry["data_root"],
        "train_subject_ids": registry["train_subject_ids"],
        "val_subject_ids": registry["val_subject_ids"],
        "selection_rule": "minimum validation ensemble EER; ties prefer fewer stages then registered method order",
        "development_data_sha256": registry["development_data_sha256"],
        "baseline_sha256": digest(
            args.baseline_root
            / ("verification_har" if source == "har" else "wisdm_verification")
            / "best.pt"
        ),
    }
    lock_path = folder / "selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("refusing to replace a different study lock")
    write(lock_path, lock)
    print("LOCKED", source, winner["name"], winner["validation"]["eer"], flush=True)
    return lock


def evaluate_stage(args, source):
    from sklearn.metrics import roc_auc_score

    from engine.enrollment import enrollment_scores, enrollment_split
    from engine.evaluate import compute_eer, fixed_threshold_metrics
    from engine.train_metric import load_metric_encoder

    folder = args.out_root / source
    lock = json.loads((folder / "selection_lock.json").read_text())
    if digest(folder / "validation_registry.json") != lock["registry_sha256"]:
        raise ValueError("validation registry changed after selection was locked")
    for name, expected in lock["artifact_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError("selected artifact changed after locking")
    train_all = data_for(source, "train", args.data_root)
    train, val, _ = _split_validation(train_all, 4, 42)
    if (
        source != lock["source"]
        or development_fingerprint(train, val) != lock["development_data_sha256"]
    ):
        raise ValueError("development data differs from the locked study")
    baseline_name = "verification_har" if source == "har" else "wisdm_verification"
    if (
        digest(args.baseline_root / baseline_name / "best.pt")
        != lock["baseline_sha256"]
    ):
        raise ValueError("baseline checkpoint changed after locking")
    selected = lock["selection"]
    with np.load(folder / selected["validation_cache"], allow_pickle=False) as cache:
        if not np.array_equal(
            cache["subjects"], val.y_subject.numpy()
        ) or not np.array_equal(cache["activities"], val.y_activity.numpy()):
            raise ValueError(
                "validation embedding cache rows differ from locked cohort"
            )
        val_embeddings = cache["embeddings"]
    # This is the only load of source='test' anywhere in this study runner.
    test = data_for(source, "test", args.data_root)
    if not set(test.subject_ids).isdisjoint(
        set(lock["train_subject_ids"]) | set(lock["val_subject_ids"])
    ):
        raise ValueError("test subjects overlap development cohorts")
    per_seed = []
    if selected["method"] == "features":
        state = dict(np.load(folder / selected["artifacts"][0], allow_pickle=False))
        test_embeddings = WithinClassMetric.from_state_dict(state).transform(
            extract_gait_features(test.X, test.sampling_rate)
        )
    else:
        blocks = []
        for index, seed in enumerate(lock["seeds"]):
            checkpoint = folder / selected["method"] / f"seed_{seed}" / "best.pt"
            model, _, _ = load_metric_encoder(str(checkpoint), device=args.device)
            block = encode(model, test, args.device)
            if selected["postprocess"] == "wccn_auto":
                state = dict(
                    np.load(
                        folder / selected["method"] / f"metric_{seed}.npz",
                        allow_pickle=False,
                    )
                )
                block = WithinClassMetric.from_state_dict(state).transform(block)
            blocks.append(block)
            per_seed.append({"seed": seed, "test_eer": metrics(block, test)["eer"]})
        test_embeddings = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
    payload = torch.load(
        args.baseline_root / baseline_name / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    protocol = payload["metrics"]["protocol"]
    if set(protocol["train_subject_ids"]) != set(train.subject_ids) or set(
        protocol["val_subject_ids"]
    ) != set(val.subject_ids):
        raise ValueError("frozen baseline development cohorts differ")
    baseline = EmbeddingNet(test.X.shape[1], embed_dim=128, dropout=0.3)
    baseline.load_state_dict(
        _strip_embedding_prefix(
            load_checkpoint(str(args.baseline_root / baseline_name / "best.pt"))
        )
    )
    baseline_val, baseline_test = (
        encode(baseline, val, args.device),
        encode(baseline, test, args.device),
    )
    result = {
        "source": source,
        "selection_lock_sha256": digest(folder / "selection_lock.json"),
        "winner": selected["name"],
        "per_seed_test": per_seed,
        "protocol": "held_out_subject_within_recording",
        "deployment_ready": False,
    }
    for label, ve, te in [
        ("baseline", baseline_val, baseline_test),
        ("selected", val_embeddings, test_embeddings),
    ]:
        labels, scores, _ = _verification_scores(
            ve, val.y_subject.numpy(), 10000, 42, val.y_activity.numpy()
        )
        calibration = calibrate_threshold(labels, scores, 0.01)
        pair = metrics(te, test, calibration["threshold"])
        enrollment = {}
        val_ei, val_pi, val_context = enrollment_split(val.y_subject.numpy(), 5, 2)
        test_ei, test_pi, test_context = enrollment_split(test.y_subject.numpy(), 5, 2)
        vl, vs, _ = enrollment_scores(ve, val.y_subject.numpy(), val_ei, val_pi)
        tl, ts, context = enrollment_scores(
            te, test.y_subject.numpy(), test_ei, test_pi
        )
        threshold = calibrate_threshold(vl, vs, 0.01)
        enrollment = {
            "eer": compute_eer(tl, ts),
            "roc_auc": float(roc_auc_score(tl, ts)),
            "fixed_threshold": fixed_threshold_metrics(tl, ts, threshold["threshold"]),
            "calibration": threshold,
            "validation_coverage": val_context,
            "test_coverage": test_context,
            "comparison_context": context,
            "session_separation": "not_established; conditional ordered-index purge only",
        }
        result[label] = {
            "pair": pair,
            "calibration": calibration,
            "enrollment": enrollment,
        }
    write(folder / "locked_test_results.json", result)
    print(
        json.dumps(
            {
                "source": source,
                "winner": result["winner"],
                "baseline_eer": result["baseline"]["pair"]["eer"],
                "selected_eer": result["selected"]["pair"]["eer"],
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["train", "lock", "evaluate"], default="train"
    )
    parser.add_argument(
        "--sources", nargs="+", choices=["har", "wisdm_v1"], default=["har", "wisdm_v1"]
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--out-root", type=Path, default=Path("runs/research_verification")
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("runs/confirmed_verification")
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(2)
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive epochs and distinct training seeds are required")
    for source in args.sources:
        {"train": train_stage, "lock": lock_stage, "evaluate": evaluate_stage}[
            args.stage
        ](args, source)


if __name__ == "__main__":
    main()
