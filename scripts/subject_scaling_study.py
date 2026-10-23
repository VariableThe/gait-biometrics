"""Registered size/objective ablation on TRAIN identities; never loads test data.

The original test cohorts have already been inspected. This development study
does not manufacture a fresh holdout by reporting them again. Calibration users
are excluded from every optimizer, fold score and checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from engine.evaluate import _verification_scores, calibrate_threshold
from engine.subject_protocol import fold_summary, make_subject_folds
from engine.train import _take
from models.gait_features import extract_gait_features
from models.gait_metric import WithinClassMetric
from scripts.research_verification import data_for, digest, encode, metrics, write

NEURAL = [
    (f"{head}_w{width}", head, width)
    for head in ("linear", "arcface")
    for width in (32, 96)
]
FEATURES = [
    ("features_standard", "standard", "auto"),
    ("features_wccn_auto", "wccn", "auto"),
    ("features_wccn_0.2", "wccn", 0.2),
]
SEEDS = [42, 43]
EPOCHS = 40


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "scripts/subject_scaling_study.py",
        "engine/train_metric.py",
        "engine/subject_protocol.py",
        "engine/evaluate.py",
        "engine/train.py",
        "data/adapter.py",
        "data/gait_transforms.py",
        "models/angular_margin.py",
        "models/gait_encoders.py",
        "models/metric_learning.py",
        "models/gait_features.py",
        "models/gait_metric.py",
        "scripts/research_verification.py",
        "models/common.py",
    ]
    return {name: digest(root / name) for name in paths}


def fingerprint(data):
    checksum = hashlib.sha256()
    for values in (data.X, data.y_subject, data.y_activity):
        array = values.contiguous().numpy()
        checksum.update(str((array.shape, str(array.dtype))).encode())
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def cohort(data, subjects):
    return _take(
        data, torch.where(torch.isin(data.y_subject, torch.tensor(subjects)))[0]
    )


def training_config(source, root, fold, head, width, seed, device):
    return {
        "source": source,
        "data_root": str(root),
        "device": device,
        "epochs": EPOCHS,
        "epoch_selection": "final",
        "seed": seed,
        "split_seed": 2026,
        "val_seed": 42,
        "val_subjects": 4,
        "train_subject_ids": fold["train_subject_ids"],
        "val_subject_ids": fold["val_subject_ids"],
        "activity_ids": [0],
        "window_size": 128,
        "stride": 64,
        "objective": "hybrid",
        "classifier_type": head,
        "angular_scale": 30.0,
        "angular_margin": 0.3,
        "identities_per_batch": 8,
        "windows_per_identity": 4,
        "ce_weight": 0.5,
        "label_smoothing": 0.05,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "margin": 0.2,
        "temperature": 0.1,
        "gradient_clip": 1.0,
        "torch_threads": 2,
        "validation_pairs": 10000,
        "eval_batch_size": 128,
        "precision": "float32",
        "model_args": {
            "architecture": "multiscale",
            "embed_dim": 128,
            "width": width,
            "dropout": 0.1,
            "stats_fusion": False,
            "normalization": True,
        },
        "augment": {"jitter_std": 0.01, "scale_std": 0.05, "roll_fraction": 0.05},
    }


def validate_run(expected, config, protocol, completed, epoch):
    """Validate actual checkpoint and complete history, not a requested sidecar."""
    from engine.train_metric import _resolved_config

    if config != _resolved_config(expected):
        raise ValueError("checkpoint config differs from registered run")
    if epoch != expected["epochs"] or completed.get("best_epoch") != epoch:
        raise ValueError("fixed final epoch checkpoint is incomplete")
    if [row["epoch"] for row in completed.get("per_epoch", [])] != list(
        range(1, epoch + 1)
    ):
        raise ValueError(
            "training history does not cover the complete registered budget"
        )
    if completed.get("protocol") != protocol or protocol.get("config") != config:
        raise ValueError("checkpoint/history protocols disagree")
    for key in ("train_subject_ids", "val_subject_ids"):
        if sorted(protocol[key]) != sorted(expected[key]):
            raise ValueError("checkpoint identities differ from registered fold")
    if protocol.get("test_split_loaded") is not False:
        raise ValueError("unexpected test access in training protocol")


def register(args, source, data):
    folder = args.out_root / source
    plan = {
        "study": "subject_scaling_v1",
        "source": source,
        "development_data_sha256": fingerprint(data),
        "split": make_subject_folds(data.y_subject.numpy()),
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "neural": NEURAL,
        "features": FEATURES,
        "epoch_selection": "fixed_final_epoch",
        "candidate_selection": "minimum_equal_fold_seed_mean_eer_then_registered_order",
        "calibration_target_far": 0.01,
        "calibration_used_for_selection": False,
        "source_test_loaded": False,
        "prior_test_inspection": True,
        "implementation_sha256": implementation_hashes(),
    }
    plan["training_configs"] = [
        training_config(source, args.data_root, fold, head, width, seed, args.device)
        for _, head, width in NEURAL
        for fold in plan["split"]["folds"]
        for seed in SEEDS
    ]
    # JSON canonicalization preserves tuple/list equality across resume.
    plan = json.loads(json.dumps(plan))
    path = folder / "pretraining_protocol.json"
    if not path.exists() and folder.exists() and any(folder.iterdir()):
        raise ValueError(
            "existing artifacts lack original pretraining registration; use a new study directory"
        )
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError(
            "registered data/cohort/budget changed; use a new study directory"
        )
    if not path.exists():
        write(path, plan)
    return plan


def neural_fold(
    args, source, data, name, head, width, fold, seed, *, require_existing=False
):
    from engine.train_metric import load_metric_encoder, train_metric

    run = args.out_root / source / name / f"fold_{fold['fold']}" / f"seed_{seed}"
    config = training_config(
        source, args.data_root, fold, head, width, seed, args.device
    )
    if not (run / "best.pt").exists():
        if require_existing:
            raise ValueError("lock requires every registered checkpoint before scoring")
        train_metric(config, str(run))
    model, protocol, loaded = load_metric_encoder(run / "best.pt", args.device)
    completed = json.loads((run / "metrics.json").read_text())
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    validate_run(config, loaded, protocol, completed, payload["epoch"])
    train, val = (
        cohort(data, fold["train_subject_ids"]),
        cohort(data, fold["val_subject_ids"]),
    )
    from engine.train_metric import _development_fingerprint

    if protocol["development_data_sha256"] != _development_fingerprint(train, val):
        raise ValueError("run was trained on different sensor data")
    result = metrics(encode(model, val, args.device), val)
    return {
        "fold": fold["fold"],
        "seed": seed,
        "eer": result["eer"],
        "roc_auc": result["roc_auc"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": completed["elapsed_seconds"],
        "checkpoint": str(run.relative_to(args.out_root / source) / "best.pt"),
        "checkpoint_sha256": digest(run / "best.pt"),
        "metrics_sha256": digest(run / "metrics.json"),
    }


def feature_candidates(data, plan):
    candidates = []
    development = cohort(data, plan["split"]["development_subject_ids"])
    features = extract_gait_features(development.X, development.sampling_rate)
    identities = development.y_subject.numpy()
    for name, mode, shrinkage in FEATURES:
        rows = []
        for fold in plan["split"]["folds"]:
            val_mask = np.isin(identities, fold["val_subject_ids"])
            metric = WithinClassMetric(mode, shrinkage).fit(
                features[~val_mask], identities[~val_mask]
            )
            val = cohort(data, fold["val_subject_ids"])
            result = metrics(metric.transform(features[val_mask]), val)
            rows.append(
                {
                    "fold": fold["fold"],
                    "seed": 42,
                    "eer": result["eer"],
                    "roc_auc": result["roc_auc"],
                }
            )
        candidates.append(
            {
                "name": name,
                "kind": "features",
                "summary": fold_summary(rows),
                "mode": mode,
                "shrinkage": shrinkage,
            }
        )
    return candidates


def neural_summary(rows):
    return fold_summary(
        [{k: row[k] for k in ("fold", "seed", "eer", "roc_auc")} for row in rows]
    )


def train_stage(args, source):
    folder = args.out_root / source
    if (folder / "selection_lock.json").exists():
        raise ValueError("study selection already locked")
    data = data_for(source, "train", args.data_root)
    plan = register(args, source, data)
    candidates = []
    for name, head, width in NEURAL:
        rows = []
        for fold in plan["split"]["folds"]:
            for seed in SEEDS:
                row = neural_fold(args, source, data, name, head, width, fold, seed)
                rows.append(row)
                print(
                    json.dumps({"source": source, "candidate": name, **row}), flush=True
                )
        candidates.append(
            {
                "name": name,
                "kind": "neural",
                "summary": neural_summary(rows),
                "runs": rows,
            }
        )
        write(folder / "development_results.partial.json", {"candidates": candidates})
    candidates.extend(feature_candidates(data, plan))
    write(
        folder / "development_results.json",
        {
            "protocol_sha256": digest(folder / "pretraining_protocol.json"),
            "candidates": candidates,
            "source_test_loaded": False,
        },
    )


def lock_stage(args, source):
    from engine.train_metric import load_metric_encoder

    folder = args.out_root / source
    if (folder / "selection_lock.json").exists():
        raise ValueError("selection is already locked")
    data = data_for(source, "train", args.data_root)
    plan = register(args, source, data)
    registry = json.loads((folder / "development_results.json").read_text())
    if registry["protocol_sha256"] != digest(folder / "pretraining_protocol.json"):
        raise ValueError("protocol changed after development scoring")
    expected_names = [row[0] for row in NEURAL + FEATURES]
    if [row["name"] for row in registry["candidates"]] != expected_names:
        raise ValueError("incomplete or reordered candidate study")
    # Revalidate every actual run before accepting any scalar selection result.
    for candidate, (name, head, width) in zip(registry["candidates"], NEURAL):
        expected_runs = [
            (fold, seed) for fold in plan["split"]["folds"] for seed in SEEDS
        ]
        if len(candidate["runs"]) != len(expected_runs):
            raise ValueError("incomplete fold/seed coverage")
        for saved, (fold, seed) in zip(candidate["runs"], expected_runs):
            if saved != neural_fold(
                args, source, data, name, head, width, fold, seed, require_existing=True
            ):
                raise ValueError(
                    "checkpoint or development scores changed since registration"
                )
        if candidate["summary"] != neural_summary(candidate["runs"]):
            raise ValueError("registered neural summary differs from actual scores")
    if registry["candidates"][len(NEURAL) :] != feature_candidates(data, plan):
        raise ValueError("registered descriptor scores differ from recomputed folds")
    winner = min(registry["candidates"], key=lambda row: row["summary"]["mean"]["eer"])
    # Freeze candidate selection BEFORE extracting any calibration representations.
    write(
        folder / "frozen_choice.json",
        {
            "winner": winner["name"],
            "registry_sha256": digest(folder / "development_results.json"),
            "source_test_loaded": False,
        },
    )
    calibration = cohort(data, plan["split"]["calibration_subject_ids"])
    artifacts = [
        "pretraining_protocol.json",
        "development_results.json",
        "frozen_choice.json",
    ]
    if winner["kind"] == "features":
        dev = cohort(data, plan["split"]["development_subject_ids"])
        metric = WithinClassMetric(winner["mode"], winner["shrinkage"]).fit(
            extract_gait_features(dev.X, dev.sampling_rate), dev.y_subject.numpy()
        )
        np.savez(folder / "selected_metric.npz", **metric.state_dict())
        artifacts.append("selected_metric.npz")
        embeddings = metric.transform(
            extract_gait_features(calibration.X, calibration.sampling_rate)
        )
    else:
        parts = []
        for row in winner["runs"]:
            model, _, _ = load_metric_encoder(folder / row["checkpoint"], args.device)
            parts.append(encode(model, calibration, args.device))
            artifacts.append(row["checkpoint"])
        embeddings = np.concatenate(parts, axis=1) / math.sqrt(len(parts))
    np.savez(
        folder / "calibration.npz",
        embeddings=embeddings,
        subjects=calibration.y_subject.numpy(),
        activities=calibration.y_activity.numpy(),
    )
    artifacts.append("calibration.npz")
    labels, scores, pair_metadata = _verification_scores(
        embeddings,
        calibration.y_subject.numpy(),
        10000,
        42,
        calibration.y_activity.numpy(),
    )
    lock = {
        "study": "subject_scaling_v1",
        "winner": winner["name"],
        "mean_cv_eer": winner["summary"]["mean"]["eer"],
        "calibration_subject_ids": calibration.subject_ids,
        "calibration": calibrate_threshold(labels, scores, 0.01),
        "calibration_pair_metadata": pair_metadata,
        "artifact_sha256": {name: digest(folder / name) for name in artifacts},
        "selection_comparator": "CV is per fold/seed; final neural calibration uses six-member ensemble (not CV prediction)",
        "source_test_loaded": False,
        "prior_test_inspection": True,
        "deployment_ready": False,
    }
    write(folder / "selection_lock.json", lock)
    print(
        json.dumps(
            {
                "source": source,
                "winner": winner["name"],
                "mean_cv_eer": lock["mean_cv_eer"],
                "source_test_loaded": False,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "lock"), required=True)
    parser.add_argument(
        "--sources", nargs="+", choices=("har", "wisdm_v1"), default=["har", "wisdm_v1"]
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/subject_scaling"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    for source in args.sources:
        {"train": train_stage, "lock": lock_stage}[args.stage](args, source)
    print(
        json.dumps({"stage": args.stage, "wall_seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
