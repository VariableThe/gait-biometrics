"""The holdout stage must reject incomplete or changed development evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import research_verification as study


@pytest.fixture
def registered_study(tmp_path, monkeypatch):
    train = SimpleNamespace(
        subject_ids=[1, 2],
        X=torch.ones(4, 3, 8),
        y_subject=torch.tensor([1, 1, 2, 2]),
        y_activity=torch.zeros(4, dtype=torch.long),
    )
    val = SimpleNamespace(
        subject_ids=[3, 4],
        X=torch.ones(4, 3, 8) * 2,
        y_subject=torch.tensor([3, 3, 4, 4]),
        y_activity=torch.zeros(4, dtype=torch.long),
    )
    args = SimpleNamespace(
        out_root=tmp_path / "study",
        data_root=tmp_path / "data",
        baseline_root=tmp_path / "baseline",
        device="cpu",
    )
    folder = args.out_root / "har"
    folder.mkdir(parents=True)
    baseline = args.baseline_root / "verification_har" / "best.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"frozen baseline fixture")
    (folder / "metric.npz").write_bytes(b"fitted development metric fixture")
    np.savez(
        folder / "validation.npz",
        embeddings=np.eye(4),
        subjects=val.y_subject.numpy(),
        activities=val.y_activity.numpy(),
    )
    names = [
        ("features_standard", "features", "standard"),
        ("features_wccn_auto", "features", "wccn_auto"),
        ("features_wccn_0.2", "features", "wccn_0.2"),
    ]
    names += [
        (name + "_" + post, name, post)
        for name, *_ in study.METHODS
        for post in ("none", "wccn_auto")
    ]
    registry = {
        "epochs": 40,
        "development_data_sha256": study.development_fingerprint(train, val),
        "source": "har",
        "test_loaded": False,
        "registered_methods": [list(method) for method in study.METHODS],
        "train_subject_ids": train.subject_ids,
        "val_subject_ids": val.subject_ids,
        "seeds": [42, 43, 44],
        "data_root": str(args.data_root),
        "candidates": [
            {
                "name": name,
                "method": method,
                "postprocess": post,
                "validation": {"eer": 0.1},
                "artifacts": ["metric.npz"],
                "validation_cache": "validation.npz",
                "n_stochastic_seeds": 3 if method != "features" else 0,
                "per_seed_validation": [
                    {"seed": seed, "eer": 0.1} for seed in [42, 43, 44]
                ],
            }
            for name, method, post in names
        ],
    }
    study.write(folder / "validation_registry.json", registry)
    calls = []

    def data(source, split, root):
        calls.append(split)
        if split != "train":
            pytest.fail("holdout was loaded before checking the study lock")
        return train

    monkeypatch.setattr(study, "data_for", data)
    monkeypatch.setattr(
        study, "_split_validation", lambda *a: (train, val, val.subject_ids)
    )
    return args, folder, registry, train, calls


def test_lock_prefers_simpler_tied_candidate_and_never_reads_test(registered_study):
    args, folder, _, _, calls = registered_study
    lock = study.lock_stage(args, "har")
    assert lock["selection"]["name"] == "features_standard"
    assert lock["baseline_sha256"] == study.digest(
        args.baseline_root / "verification_har/best.pt"
    )
    assert json.loads((folder / "selection_lock.json").read_text()) == lock
    assert calls == ["train"]


def test_lock_rejects_duplicate_candidate_displacing_missing_method(registered_study):
    args, folder, registry, _, calls = registered_study
    registry["candidates"][-1] = registry["candidates"][0]
    study.write(folder / "validation_registry.json", registry)
    with pytest.raises(ValueError, match="distinct validation candidates"):
        study.lock_stage(args, "har")
    assert calls == []


@pytest.mark.parametrize("change", ["epochs", "seeds", "coverage", "development"])
def test_lock_rejects_reduced_study_or_changed_training_values(
    registered_study, change
):
    args, folder, registry, train, calls = registered_study
    if change == "epochs":
        registry["epochs"] = 1
    elif change == "seeds":
        registry["seeds"] = [42]
    elif change == "coverage":
        registry["candidates"][-1]["per_seed_validation"] = [{"seed": 42, "eer": 0.1}]
    else:
        train.X += 1
    study.write(folder / "validation_registry.json", registry)
    with pytest.raises(
        ValueError, match="complete|three registered seeds|pre-training"
    ):
        study.lock_stage(args, "har")
    assert "test" not in calls


@pytest.mark.parametrize("change", ["registry", "artifact", "development", "baseline"])
def test_changed_evidence_rejected_before_test_load(registered_study, change):
    args, folder, registry, train, calls = registered_study
    study.lock_stage(args, "har")
    calls.clear()
    if change == "registry":
        registry["seeds"] = [99]
        study.write(folder / "validation_registry.json", registry)
    elif change == "artifact":
        (folder / "metric.npz").write_bytes(b"replacement")
    elif change == "development":
        train.X += 1
    else:
        (args.baseline_root / "verification_har/best.pt").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed|differs"):
        study.evaluate_stage(args, "har")
    assert "test" not in calls
