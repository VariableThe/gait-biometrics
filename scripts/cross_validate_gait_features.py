"""Secondary HAR study: select descriptors by three subject folds, then freeze.

The HAR test cohort was inspected in the preceding neural study. Results from
this follow-up are exploratory, never a new blind-holdout performance claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engine.evaluate import (
    _verification_scores,
    calibrate_threshold,
    verification_metrics,
)
from engine.train import _split_validation
from models.gait_features import extract_gait_features
from models.gait_metric import WithinClassMetric
from scripts.research_verification import (
    data_for,
    development_fingerprint,
    digest,
    write,
)

METHODS = [
    ("standard", "standard", "auto"),
    ("wccn_auto", "wccn", "auto"),
    ("wccn_0.2", "wccn", 0.2),
]


def subject_folds(subject_ids):
    identities = np.asarray(sorted(subject_ids), dtype=np.int64)
    if len(identities) < 6:
        raise ValueError("three verification folds require at least six subjects")
    return np.array_split(np.random.default_rng(2026).permutation(identities), 3)


def development(root):
    return _split_validation(data_for("har", "train", root), 4, 42)[:2]


def train_stage(args):
    folder = args.out_root
    if (folder / "selection_lock.json").exists():
        raise ValueError("selection is already frozen; use a new output root")
    train, calibration = development(args.data_root)
    fingerprint = development_fingerprint(train, calibration)
    write(
        folder / "pretraining_protocol.json",
        {
            "development_data_sha256": fingerprint,
            "train_subject_ids": train.subject_ids,
            "calibration_subject_ids": calibration.subject_ids,
            "methods": METHODS,
            "fold_subject_ids": [
                fold.tolist() for fold in subject_folds(train.subject_ids)
            ],
            "selection": "minimum mean three-fold EER; ties use registered method order",
            "test_loaded": False,
            "previous_test_inspection": True,
        },
    )
    features = extract_gait_features(train.X, train.sampling_rate)
    calibration_features = extract_gait_features(
        calibration.X, calibration.sampling_rate
    )
    identities = train.y_subject.numpy()
    candidates = []
    for name, mode, shrinkage in METHODS:
        folds = []
        for subjects in subject_folds(train.subject_ids):
            held_out = np.isin(identities, subjects)
            metric = WithinClassMetric(mode, shrinkage).fit(
                features[~held_out], identities[~held_out]
            )
            scores = verification_metrics(
                metric.transform(features[held_out]), identities[held_out], seed=42
            )
            folds.append(
                {
                    "validation_subject_ids": subjects.tolist(),
                    "train_subject_ids": metric.training_subject_ids_,
                    "eer": scores["eer"],
                    "roc_auc": scores["roc_auc"],
                }
            )
        candidates.append(
            {
                "method": name,
                "mode": mode,
                "shrinkage": shrinkage,
                "folds": folds,
                "mean_eer": float(np.mean([fold["eer"] for fold in folds])),
            }
        )
    winner = min(candidates, key=lambda candidate: candidate["mean_eer"])
    metric = WithinClassMetric(winner["mode"], winner["shrinkage"]).fit(
        features, identities
    )
    np.savez(folder / "metric.npz", **metric.state_dict())
    embeddings = metric.transform(calibration_features)
    np.savez(
        folder / "calibration.npz",
        embeddings=embeddings,
        subjects=calibration.y_subject.numpy(),
        activities=calibration.y_activity.numpy(),
    )
    labels, scores, _ = _verification_scores(
        embeddings,
        calibration.y_subject.numpy(),
        10000,
        42,
        calibration.y_activity.numpy(),
    )
    lock = {
        "study": "secondary_har_feature_cross_validation",
        "exploratory_previous_test_inspection": True,
        "candidates": candidates,
        "winner": winner,
        "development_data_sha256": fingerprint,
        "calibration": calibrate_threshold(labels, scores, 0.01),
        "artifact_sha256": {
            name: digest(folder / name)
            for name in ("metric.npz", "calibration.npz", "pretraining_protocol.json")
        },
    }
    write(folder / "selection_lock.json", lock)
    print(json.dumps({"winner": winner["method"], "mean_cv_eer": winner["mean_eer"]}))


def evaluate_stage(args):
    folder = args.out_root
    lock = json.loads((folder / "selection_lock.json").read_text())
    for name, expected in lock["artifact_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError("frozen feature study artifact changed")
    train, calibration = development(args.data_root)
    if development_fingerprint(train, calibration) != lock["development_data_sha256"]:
        raise ValueError("development data changed since feature selection")
    with np.load(folder / "metric.npz", allow_pickle=False) as state:
        metric = WithinClassMetric.from_state_dict(state)
    test = data_for("har", "test", args.data_root)
    if not set(test.subject_ids).isdisjoint(
        set(train.subject_ids) | set(calibration.subject_ids)
    ):
        raise ValueError("test identities overlap development subjects")
    embeddings = metric.transform(extract_gait_features(test.X, test.sampling_rate))
    results = verification_metrics(
        embeddings,
        test.y_subject.numpy(),
        seed=42,
        activity_ids=test.y_activity.numpy(),
        threshold=lock["calibration"]["threshold"],
    )
    write(
        folder / "exploratory_test_results.json",
        {
            "source": "har",
            "winner": lock["winner"]["method"],
            "pair": results,
            "calibration": lock["calibration"],
            "selection_lock_sha256": digest(folder / "selection_lock.json"),
            "exploratory_previous_test_inspection": True,
            "deployment_ready": False,
        },
    )
    print(
        json.dumps(
            {"winner": lock["winner"]["method"], "exploratory_test_eer": results["eer"]}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/har_feature_cv"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"train": train_stage, "evaluate": evaluate_stage}[args.stage](args)


if __name__ == "__main__":
    main()
