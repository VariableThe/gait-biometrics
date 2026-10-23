"""Synthetic controls only: no datasets, Torch imports, training jobs, or installs."""

import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from engine.subject_protocol import fold_summary, make_subject_folds
from scripts import invariant_transfer_study as study


def synthetic_data(source):
    ids = np.arange(2, 36, 3, dtype=np.int64)
    subjects = np.r_[np.repeat(ids, 4), 777]
    activities = np.r_[np.tile([0, 1, 0, 0], len(ids)), 1].astype(np.int64)
    channels = 9 if source == "har" else 3
    x = (
        np.broadcast_to(subjects[:, None, None], (len(subjects), channels, 128))
        .astype(np.float32)
        .copy()
    )
    x += np.arange(len(subjects), dtype=np.float32)[:, None, None] / 1000
    return SimpleNamespace(
        X=x,
        y_subject=subjects,
        y_activity=activities,
        subject_ids=list(range(100, 136)),
        sampling_rate=50.0 if source == "har" else 20.0,
    )


def mocked_runtime(monkeypatch, out_dir, sources):
    calls = {"load": [], "extract": [], "fit": [], "verify": []}

    def audit_exists():
        protocol = json.loads((out_dir / "pre_fit_protocol.json").read_text())
        assert protocol["test_loaded"] is False
        assert protocol["candidate_selection"] is False
        for source in protocol["sources"]:
            data = sources[source["source"]]
            assert source["source_train_fingerprint"] == study.input_fingerprint(
                data.X, data.y_subject, data.y_activity
            )
        return protocol

    def load(**kwargs):
        assert kwargs["split"] == "train"
        assert kwargs["window_size"] == 128 and kwargs["stride"] == 64
        calls["load"].append(kwargs)
        return sources[kwargs["source"]]

    def extractor(name):
        def extract(x, rate):
            audit_exists()
            calls["extract"].append((name, x.copy(), rate))
            width = 2 if name == "axis" else 3
            return x[:, :width, 0].copy()

        return extract

    class Metric:
        def __init__(self, mode, shrinkage):
            assert (mode, shrinkage) == ("wccn", "auto")

        def fit(self, values, subjects):
            audit_exists()
            self.training_subject_ids_ = sorted(map(int, np.unique(subjects)))
            self.shrinkage_ = 0.2
            # The source values encode subject identity: catch a wrong row mask.
            np.testing.assert_array_equal(np.floor(values[:, 0]).astype(int), subjects)
            calls["fit"].append((values.copy(), subjects.copy()))
            return self

        def transform(self, values):
            calls.setdefault("transform", []).append(values.copy())
            return values

    def verify(embeddings, subjects, **kwargs):
        assert kwargs["n_pairs"] == 10000 and kwargs["seed"] == 42
        np.testing.assert_array_equal(
            kwargs["activity_ids"], np.zeros(len(subjects), dtype=int)
        )
        np.testing.assert_array_equal(np.floor(embeddings[:, 0]).astype(int), subjects)
        calls["verify"].append((embeddings.copy(), subjects.copy()))
        index = (len(calls["verify"]) - 1) % 3
        return {
            "eer": [0.1, 0.2, 0.6][index],
            "roc_auc": [0.9, 0.8, 0.4][index],
            "n_pairs": 10000,
        }

    runtime = {
        "load": load,
        "folds": make_subject_folds,
        "summary": fold_summary,
        "axis": extractor("axis"),
        "invariant": extractor("invariant"),
        "metric": Metric,
        "verify": verify,
        "thread_limits": lambda **kwargs: nullcontext(),
    }
    monkeypatch.setattr(study, "_runtime", lambda: runtime)
    return calls


def test_import_and_help_never_import_torch_or_adapter():
    code = """
import builtins
import runpy
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch' or name in ('data.adapter', 'engine.evaluate'):
        raise AssertionError('eager runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['invariant_transfer_study', '--help']
runpy.run_module('scripts.invariant_transfer_study', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=study.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out-dir" in result.stdout


def test_canonical_fingerprints_cover_values_labels_shape_and_signed_zero():
    x = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    x[0, 0, 0] = -0.0
    canonical = x.astype("<f4")
    canonical[0, 0, 0] = 0.0
    assert study.array_fingerprint(x, "<f4") == study.array_fingerprint(
        np.asfortranarray(canonical.astype(">f4")), "<f4"
    )
    first = study.input_fingerprint(x, [2, 9], [0, 0])
    x[1, 0, 0] += 0.5
    assert study.input_fingerprint(x, [2, 9], [0, 0]) != first
    assert study.input_fingerprint(canonical, [2, 8], [0, 0]) != first
    assert study.input_fingerprint(canonical, [2, 9], [0, 1]) != first
    assert (
        study.array_fingerprint(canonical.reshape(2, 6), "<f4") != first["arrays"]["X"]
    )
    with pytest.raises(ValueError, match="finite"):
        study.array_fingerprint([np.nan], "<f4")


@pytest.mark.parametrize("source", ["har", "wisdm_v1", "all"])
def test_train_only_fixed_candidates_exact_masks_prefit_audit_and_equal_fold_summary(
    monkeypatch, tmp_path, source
):
    selected = study.SOURCES if source == "all" else (source,)
    data = {name: synthetic_data(name) for name in selected}
    out_dir = tmp_path / "exclusive"
    calls = mocked_runtime(monkeypatch, out_dir, data)
    result = study.run(tmp_path / "unused_data", out_dir, source)
    audit = json.loads((out_dir / "pre_fit_protocol.json").read_text())
    assert len(result["rows"]) == 3 * len(selected)
    assert [call["source"] for call in calls["load"]] == list(selected)
    assert len(calls["fit"]) == len(calls["verify"]) == 9 * len(selected)
    assert len(calls["extract"]) == 2 * len(selected)
    for source_index, name in enumerate(selected):
        original = data[name]
        walking = original.y_activity == 0
        protocol = make_subject_folds(original.y_subject[walking], **study.FOLD_PARAMS)
        metadata = audit["sources"][source_index]
        assert metadata["subject_protocol"] == protocol
        assert metadata["walking_eligibility"]["excluded"]["n_subjects"] == 0
        assert metadata["walking_eligibility"]["retained"]["n_windows"] == 36
        assert metadata["subject_protocol_comparison"]["membership_changed"] is False
        assert 777 not in protocol["all_subject_ids"]
        assert metadata["reserved_calibration"]["n_subjects"] == 4
        assert metadata["reserved_calibration"]["n_windows"] == 12
        assert metadata["development"]["n_windows"] == 24
        dev_mask = walking & np.isin(
            original.y_subject, protocol["development_subject_ids"]
        )
        assert (
            metadata["development_source_row_indices"]
            == np.flatnonzero(dev_mask).tolist()
        )
        assert metadata["development_fingerprint"] == study.input_fingerprint(
            original.X[dev_mask],
            original.y_subject[dev_mask],
            original.y_activity[dev_mask],
        )
        for _, x, _ in calls["extract"][source_index * 2 : source_index * 2 + 2]:
            np.testing.assert_array_equal(x, original.X[dev_mask])
        for candidate_index, candidate in enumerate(study.CANDIDATES):
            row = result["rows"][source_index * 3 + candidate_index]
            assert row["source"] == name and row["candidate"] == candidate
            assert (
                row["feature_dimension"]
                == {"axis": 2, "invariant": 3, "fusion": 5}[candidate]
            )
            assert row["summary"]["mean"]["eer"] == pytest.approx(0.3)
            assert row["summary"]["fold_sample_sd"]["eer"] == pytest.approx(
                np.std([0.1, 0.2, 0.6], ddof=1)
            )
            for fold_index, fold in enumerate(protocol["folds"]):
                call_index = source_index * 9 + candidate_index * 3 + fold_index
                fit_values, fit_ids = calls["fit"][call_index]
                val_values, val_ids = calls["verify"][call_index]
                assert sorted(set(fit_ids)) == fold["train_subject_ids"]
                assert sorted(set(val_ids)) == fold["val_subject_ids"]
                assert not set(fit_ids) & set(val_ids)
                assert not (set(fit_ids) | set(val_ids)) & set(
                    protocol["calibration_subject_ids"]
                )
                assert row["folds"][fold_index]["train"]["n_windows"] == len(fit_values)
                assert row["folds"][fold_index]["validation"]["n_windows"] == len(
                    val_values
                )
                if candidate == "fusion":
                    axis_fit = calls["fit"][source_index * 9 + fold_index][0]
                    inv_fit = calls["fit"][source_index * 9 + 3 + fold_index][0]
                    np.testing.assert_array_equal(
                        fit_values, np.concatenate((axis_fit, inv_fit), axis=1)
                    )
    assert json.loads((out_dir / "results.json").read_text()) == result
    assert (
        result["pre_fit_protocol_sha256"]
        == study.hashlib.sha256(
            (out_dir / "pre_fit_protocol.json").read_bytes()
        ).hexdigest()
    )
    assert "winner" not in result
    assert "fixed_threshold" not in (out_dir / "results.json").read_text()


@pytest.mark.parametrize("excluded_id", [6, 18])
def test_singleton_walking_id_is_filtered_before_folds_with_explicit_cohort_changes(
    monkeypatch, tmp_path, excluded_id
):
    # Fake official-style ID universe only: no adapter or source files loaded.
    global_ids = np.arange(1, 37, dtype=np.int64)
    shuffled = np.random.default_rng(42).permutation(global_ids)
    train_ids = np.sort(shuffled[: int(0.8 * len(global_ids))]) - 1
    reference = make_subject_folds(train_ids, **study.FOLD_PARAMS)
    assert reference["calibration_subject_ids"] == [18, 25, 28, 30]
    assert 6 in reference["folds"][2]["val_subject_ids"]
    # The excluded identity has three total windows, but only one is walking.
    subjects = np.r_[
        np.repeat(train_ids, np.where(train_ids == excluded_id, 1, 3)),
        excluded_id,
        excluded_id,
    ]
    activities = np.r_[np.zeros(len(subjects) - 2, dtype=np.int64), 1, 1]
    x = (
        np.broadcast_to(subjects[:, None, None], (len(subjects), 3, 128))
        .astype(np.float32)
        .copy()
    )
    x += np.arange(len(x), dtype=np.float32)[:, None, None] / 1000
    data = SimpleNamespace(
        X=x,
        y_subject=subjects,
        y_activity=activities,
        subject_ids=global_ids.tolist(),
        sampling_rate=20.0,
    )
    original_arrays = [value.copy() for value in (x, subjects, activities)]
    original_global_ids = data.subject_ids.copy()
    out_dir = tmp_path / "singletons"
    calls = mocked_runtime(monkeypatch, out_dir, {"wisdm_v1": data})
    runtime = study._runtime()
    fold_inputs = []

    def observed_folds(ids, **kwargs):
        fold_inputs.append(np.asarray(ids).copy())
        return make_subject_folds(ids, **kwargs)

    runtime["folds"] = observed_folds
    study.run(tmp_path / "unused", out_dir, "wisdm_v1")
    metadata = json.loads((out_dir / "pre_fit_protocol.json").read_text())["sources"][0]
    expected = make_subject_folds(
        train_ids[train_ids != excluded_id], **study.FOLD_PARAMS
    )
    assert excluded_id not in fold_inputs[0]
    assert metadata["subject_protocol"] == expected
    assert len(expected["calibration_subject_ids"]) == 4
    assert excluded_id not in expected["all_subject_ids"]
    eligibility = metadata["walking_eligibility"]
    assert eligibility["minimum_windows_per_subject"] == 2
    assert eligibility["retained"]["n_subjects"] == 27
    assert eligibility["retained"]["n_windows"] == 81
    assert eligibility["excluded"] == {
        "subject_ids": [excluded_id],
        "n_subjects": 1,
        "n_windows": 1,
        "windows_per_subject": {str(excluded_id): 1},
    }
    assert eligibility["exclusions"] == [
        {
            "subject_id": excluded_id,
            "original_subject_id": excluded_id + 1,
            "n_windows": 1,
            "reason": "fewer_than_two_walking_windows",
        }
    ]
    eligible_mask = (activities == 0) & (subjects != excluded_id)
    assert (
        eligibility["retained_source_row_indices"]
        == np.flatnonzero(eligible_mask).tolist()
    )
    assert (
        eligibility["excluded_source_row_indices"]
        == np.flatnonzero((activities == 0) & (subjects == excluded_id)).tolist()
    )
    assert metadata["eligible_walking_fingerprint"] == study.input_fingerprint(
        x[eligible_mask], subjects[eligible_mask], activities[eligible_mask]
    )
    comparison = metadata["subject_protocol_comparison"]
    assert comparison["membership_changed"] is True
    assert comparison["reference_unfiltered_subject_protocol"] == reference
    for key, protocol_key in (
        ("calibration", "calibration_subject_ids"),
        ("development", "development_subject_ids"),
    ):
        assert comparison[key] == {
            "removed_subject_ids": sorted(
                set(reference[protocol_key]) - set(expected[protocol_key])
            ),
            "added_subject_ids": sorted(
                set(expected[protocol_key]) - set(reference[protocol_key])
            ),
        }
    if excluded_id == 6:
        assert comparison["calibration"] == {
            "removed_subject_ids": [],
            "added_subject_ids": [],
        }
        assert comparison["development"]["removed_subject_ids"] == [6]
    else:
        assert comparison["calibration"]["removed_subject_ids"] == [18]
    for index, fold in enumerate(comparison["folds"]):
        for key, protocol_key in (
            ("train", "train_subject_ids"),
            ("validation", "val_subject_ids"),
        ):
            before = set(reference["folds"][index][protocol_key])
            after = set(expected["folds"][index][protocol_key])
            assert fold[key] == {
                "removed_subject_ids": sorted(before - after),
                "added_subject_ids": sorted(after - before),
            }
    forbidden = set(expected["calibration_subject_ids"]) | {excluded_id}
    for _, extracted, _ in calls["extract"]:
        assert not set(np.floor(extracted[:, 0, 0]).astype(int)) & forbidden
    for _, fitted_ids in calls["fit"]:
        assert not set(fitted_ids) & forbidden
    for _, verified_ids in calls["verify"]:
        assert not set(verified_ids) & forbidden
    for after, before in zip(
        (data.X, data.y_subject, data.y_activity), original_arrays
    ):
        np.testing.assert_array_equal(after, before)
    assert data.subject_ids == original_global_ids


def test_existing_output_is_rejected_before_runtime_import(monkeypatch, tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(
        study, "_runtime", lambda: pytest.fail("runtime should not load")
    )
    with pytest.raises(FileExistsError):
        study.run(tmp_path, tmp_path)
    assert sentinel.read_text() == "keep"


def test_invalid_source_is_rejected_without_creating_output(tmp_path):
    with pytest.raises(ValueError, match="source"):
        study.run(tmp_path, tmp_path / "never_created", "wisdm")
    assert not (tmp_path / "never_created").exists()


def test_code_hash_includes_runner_and_actual_dependencies():
    fingerprint = study.code_fingerprint()
    expected = (
        "scripts/invariant_transfer_study.py",
        "data/adapter.py",
        "engine/evaluate.py",
        "engine/subject_protocol.py",
        "engine/train.py",
        "models/gait_features.py",
        "models/gait_invariants.py",
        "models/gait_metric.py",
    )
    assert set(expected) <= fingerprint["files"].keys()
    for path in expected:
        assert (
            fingerprint["files"][path]
            == study.hashlib.sha256((study.ROOT / path).read_bytes()).hexdigest()
        )


def test_no_results_are_published_if_sources_change(monkeypatch, tmp_path):
    out_dir = tmp_path / "incomplete"
    calls = mocked_runtime(monkeypatch, out_dir, {"har": synthetic_data("har")})
    hashes = iter([{"sha256": "before"}, {"sha256": "after"}])
    monkeypatch.setattr(study, "code_fingerprint", lambda: next(hashes))
    with pytest.raises(ValueError, match="source code changed"):
        study.run(tmp_path, out_dir, "har")
    assert len(calls["fit"]) == 9
    assert (out_dir / "pre_fit_protocol.json").exists()
    assert not (out_dir / "results.json").exists()
