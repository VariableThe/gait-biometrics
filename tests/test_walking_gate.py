"""Synthetic-only contract, fit-scope, causality and registered replay checks."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from models import walking_gate as module
from models.walking_gate import (
    FEATURE_NAMES,
    SourceContract,
    WalkingGate,
    binary_labels,
    extract_features,
    select_threshold,
)
from scripts import walking_gate_study as study


def signals(n=36, seed=90):
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 20
    activity = np.arange(n, dtype=np.int64) % 3
    X = np.empty((n, 3, 128), dtype=np.float64)
    for i, label in enumerate(activity):
        phase = rng.uniform(0, 2 * np.pi)
        frequency, amplitude = ((1.8, 1.5), (3.4, 3.0), (0.2, 0.02))[label]
        X[i] = rng.normal(0, 0.02, (3, 128))
        X[i, 0] += 0.4 * amplitude * np.sin(2 * np.pi * frequency * t + phase)
        X[i, 1] += 0.3 * amplitude * np.cos(2 * np.pi * frequency * t + phase)
        X[i, 2] += 9.81 + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return X, activity


@pytest.fixture
def gate():
    X, activity = signals()
    model = WalkingGate.fit(
        X,
        activity,
        activity_names=["Walking", "Jogging", "Sitting"],
        subjects=np.ones(len(X), dtype=np.int64),
        fit_subject_ids=[1],
        forbidden_subject_ids=[2, 3],
        contract=study.CONTRACT,
    )
    return model, X, activity


def test_metadata_labels_are_authoritative_not_assumed_zero():
    assert binary_labels(
        np.array([0, 1, 2]), ["Sitting", "Walking", "Jogging"]
    ).tolist() == [0, 1, 0]
    for names in (["activity_0", "activity_1"], ["Walking", "Walking"]):
        with pytest.raises(ValueError, match="authoritative"):
            binary_labels(np.array([0]), names)
    with pytest.raises(ValueError, match="index authoritative"):
        binary_labels(np.array([7]), ["Walking", "Jogging"])


def test_training_statistics_only_and_no_inference_adaptation(gate):
    model, X, _ = gate
    features = extract_features(X, contract=study.CONTRACT)
    np.testing.assert_array_equal(model.mean, features.mean(axis=0))
    std = features.std(axis=0)
    np.testing.assert_array_equal(model.scale, np.where(std > 1e-12, std, 1))
    state = model.to_dict()
    shifted = X * 5 + 40
    assert not np.allclose(
        extract_features(np.r_[X, shifted], contract=study.CONTRACT).mean(axis=0),
        model.mean,
    )
    model.predict_proba(shifted, contract=study.CONTRACT)
    model.predict_proba(X, contract=study.CONTRACT)
    assert state == model.to_dict()
    probabilities = model.predict_proba(X, contract=study.CONTRACT)
    assert np.mean(probabilities[::3]) > np.mean(probabilities[1::3])
    assert np.mean(probabilities[::3]) > np.mean(probabilities[2::3])
    assert len(FEATURE_NAMES) == features.shape[1] == 18


def test_native_wisdm_units_cannot_be_silently_relabelled_si(gate):
    model, X, _ = gate
    assert study.CONTRACT.units == ("wisdm_native_10_per_g",) * 3
    si = SourceContract.from_dict({**study.CONTRACT.to_dict(), "units": ["m/s^2"] * 3})
    with pytest.raises(ValueError, match="contract"):
        model.predict_proba(X, contract=si)


def test_numeric_json_roundtrip_and_no_overwrite(gate, tmp_path):
    model, X, _ = gate
    model = model.with_threshold(0.7)
    path = tmp_path / "gate.json"
    model.save(path)
    restored = WalkingGate.load(path)
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(
        restored.predict_proba(X, contract=study.CONTRACT),
        model.predict_proba(X, contract=study.CONTRACT),
    )
    np.testing.assert_array_equal(
        restored.predict(X, contract=study.CONTRACT),
        restored.predict_proba(X, contract=study.CONTRACT) >= 0.7,
    )
    assert isinstance(study.read_json(path)["coefficients"], list)
    with pytest.raises(FileExistsError):
        model.save(path)
    state = model.to_dict()
    state["fit_provenance"]["n_fit_windows"] = 1000
    assert model.fit_provenance["n_fit_windows"] == len(X)


def resigned(state):
    state["integrity_sha256"] = module.hashlib.sha256(
        module.canonical({k: v for k, v in state.items() if k != "integrity_sha256"})
    ).hexdigest()
    return state


def test_finite_numeric_corruption_and_invalid_schema_refused(gate, tmp_path):
    model, _, _ = gate
    corrupt = model.to_dict()
    corrupt["coefficients"][0] += 0.01
    with pytest.raises(ValueError, match="integrity"):
        WalkingGate.from_dict(corrupt)
    for key, value in (
        ("scale", [0.0] * 18),
        ("mean", [0.0] * 17),
        ("coefficients", ["1"] * 18),
        ("threshold", 1.01),
        ("intercept", float("nan")),
        ("feature_version", "future_schema"),
    ):
        corrupt = model.to_dict()
        corrupt[key] = value
        with pytest.raises(ValueError):
            WalkingGate.from_dict(resigned(corrupt))
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        WalkingGate.load(path)


def test_contract_required_and_mismatches_refused(gate):
    model, X, _ = gate
    with pytest.raises(TypeError):
        model.predict_proba(X)
    for delta in (
        {"source": "santos_leg"},
        {"sample_rate": 25.0},
        {"channels": ("acc_z", "acc_x", "acc_y")},
        {"window_length": 129},
    ):
        contract = SourceContract.from_dict({**study.CONTRACT.to_dict(), **delta})
        with pytest.raises(ValueError, match="incompatible"):
            model.predict_proba(X, contract=contract)
    for delta in (
        {"units": ["g"] * 3},
        {"channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]},
        {"acceleration": "body_without_gravity"},
        {"sample_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            SourceContract.from_dict({**study.CONTRACT.to_dict(), **delta})
    with pytest.raises(ValueError, match="shape"):
        model.predict_proba(np.zeros((2, 6, 128)), contract=study.CONTRACT)
    for value in (float("nan"), float("inf"), 1e7):
        invalid = X.copy()
        invalid[0, 0, 0] = value
        with pytest.raises(ValueError, match="acceleration"):
            model.predict_proba(invalid, contract=study.CONTRACT)


def test_no_future_input_dependence_and_bounded_batch(gate):
    model, X, _ = gate
    early = model.predict_proba(X[:7], contract=study.CONTRACT, batch_size=2)
    changed = X.copy()
    changed[7:] = changed[7:] * 100 + 1000
    np.testing.assert_array_equal(
        early, model.predict_proba(changed, contract=study.CONTRACT)[:7]
    )
    np.testing.assert_array_equal(
        early,
        np.array(
            [
                model.predict_proba(window[None], contract=study.CONTRACT)[0]
                for window in X[:7]
            ]
        ),
    )
    observed = []
    original = module._features_batch

    def extract(batch, contract):
        observed.append(len(batch))
        return original(batch, contract)

    with patch.object(module, "_features_batch", side_effect=extract):
        probability = model.predict_proba(X, contract=study.CONTRACT, batch_size=3)
    assert max(observed) == 3 and sum(observed) == len(X)
    np.testing.assert_array_equal(
        probability,
        model.predict_proba(X, contract=study.CONTRACT, batch_size=64),
    )
    # A raw trailing window at t=128 cannot depend on later stream samples.
    stream = np.concatenate([X[0], X[1]], axis=1)
    before = model.predict_proba(stream[:, :128][None], contract=study.CONTRACT)
    stream[:, 128:] += 7000
    np.testing.assert_array_equal(
        before, model.predict_proba(stream[:, :128][None], contract=study.CONTRACT)
    )


def test_degenerate_windows_and_empty_prediction_are_safe(gate):
    model, _, _ = gate
    features = extract_features(np.zeros((3, 3, 128)), contract=study.CONTRACT)
    assert np.isfinite(features).all()
    assert np.all(features[:, FEATURE_NAMES.index("cadence_peak_hz")] == 0)
    assert np.all(features[:, FEATURE_NAMES.index("autocorrelation_peak_hz")] == 0)
    assert model.predict_proba(
        np.zeros((0, 3, 128)), contract=study.CONTRACT
    ).shape == (0,)
    for batch in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            model.predict_proba(
                np.zeros((0, 3, 128)), contract=study.CONTRACT, batch_size=batch
            )


def test_fit_leakage_guard_and_single_class_refusal(gate):
    _, X, activity = gate
    kwargs = {
        "activity_names": ["Walking", "Jogging", "Sitting"],
        "subjects": np.ones(len(X), dtype=np.int64),
        "fit_subject_ids": [1],
        "forbidden_subject_ids": [2],
        "contract": study.CONTRACT,
    }
    with pytest.raises(ValueError, match="leakage guard"):
        WalkingGate.fit(X, activity, **{**kwargs, "fit_subject_ids": [1, 2]})
    subjects = kwargs["subjects"].copy()
    subjects[-1] = 2
    with pytest.raises(ValueError, match="leakage guard"):
        WalkingGate.fit(X, activity, **{**kwargs, "subjects": subjects})
    with pytest.raises(ValueError, match="both classes"):
        WalkingGate.fit(X, np.zeros(len(X), dtype=np.int64), **kwargs)


def test_threshold_fixed_exact_tie_rule():
    assert select_threshold(np.array([0, 1]), [0.3, 0.3]) == 1.0
    assert select_threshold(np.array([0, 0, 1, 1]), [0.1, 0.2, 0.8, 0.9]) == 0.8
    rng = np.random.default_rng(1)
    for _ in range(10):
        labels = np.array([0, 1] * 10)
        probabilities = np.round(rng.uniform(size=20), 1)
        candidates = np.unique(np.r_[0, probabilities, 1])
        expected = max(
            candidates,
            key=lambda t: (
                int(np.sum((probabilities >= t) & (labels == 1))) * 10
                + int(np.sum((probabilities < t) & (labels == 0))) * 10,
                -int(np.sum((probabilities >= t) & (labels == 0))),
                t,
            ),
        )
        assert select_threshold(labels, probabilities) == expected


def synthetic_training():
    subjects = np.repeat(np.array(study.roles()["source_train"], dtype=np.int64), 6)
    X, activity = signals(len(subjects), seed=331)
    starts = np.arange(len(X), dtype=np.int64) * 131
    return {
        "X": X,
        "activity": activity,
        "subjects": subjects,
        "source": {
            "source": "wisdm_v1",
            "sha256": study.RAW_SHA256,
            "path": "SYNTHETIC_MOCK_NO_SOURCE_OPENED",
            "bytes": 0,
            "contract": study.CONTRACT.to_dict(),
            "adapter_split": "train",
        },
        "start_rows": starts,
        "stop_rows": starts + 128,
        "segment_ids": starts.copy(),
        "first_timestamps_ns": starts * 50_000_000 + 1,
        "last_timestamps_ns": (starts + 127) * 50_000_000 + 1,
    }


def test_subject_roles_fixed_before_fit_and_registration_complete():
    allocation = study.roles()
    assert [len(allocation[k]) for k in study.ROLES] == [16, 6, 6]
    assert set().union(*(set(allocation[k]) for k in study.ROLES)) == set(
        allocation["source_train"]
    )
    assert not set(allocation["source_train"]) & set(
        allocation["source_test_never_windowed"]
    )
    assert len(study.source_hashes()) == 6
    cache = study.cache_from_training(synthetic_training())
    for delta in ("role", "subjects", "support"):
        invalid = {k: v.copy() for k, v in cache.items()}
        if delta == "role":
            invalid["role"][0] = (invalid["role"][0] + 1) % 3
        elif delta == "subjects":
            invalid["subjects"][0] = allocation["source_test_never_windowed"][0]
        else:
            invalid["start_rows"][1] = invalid["start_rows"][0]
            invalid["stop_rows"][1] = invalid["stop_rows"][0]
            invalid["segment_ids"][1] = invalid["segment_ids"][0]
        with pytest.raises(ValueError, match="leakage guard|duplicate raw"):
            study.validate_cache(invalid)
    invalid = {k: v.copy() for k, v in cache.items()}
    fit_rows = np.flatnonzero(cache["role"] == 0)
    calibration_rows = np.flatnonzero(cache["role"] == 1)
    a, b = fit_rows[0], calibration_rows[0]
    invalid["start_rows"][b] = invalid["start_rows"][a] + 1
    invalid["stop_rows"][b] = invalid["start_rows"][b] + 128
    invalid["segment_ids"][b] = invalid["start_rows"][b]
    with pytest.raises(ValueError, match="overlapping support"):
        study.validate_cache(invalid)


def test_synthetic_registered_run_one_fit_and_cache_only_verify(tmp_path):
    args = SimpleNamespace(
        data_root="MUST_NOT_BE_OPENED", out_root=tmp_path / "study", threads=1
    )
    original = WalkingGate.fit_features
    fit_rows = []

    def guarded_fit(features, labels, **kwargs):
        registration = study.read_json(args.out_root / "registration.json")
        assert registration["plan"] == study.plan()
        assert registration["source_files"] == study.source_hashes()
        assert "versions" in registration and "arrays" in registration
        assert (args.out_root / "features.npz").is_file()
        assert set(kwargs["subjects"].tolist()) <= set(study.roles()["fit"])
        fit_rows.append(len(features))
        return original(features, labels, **kwargs)

    with patch.object(
        study, "load_training", return_value=synthetic_training()
    ) as load:
        with patch.object(WalkingGate, "fit_features", side_effect=guarded_fit) as fit:
            result = study.run(args)
            assert fit.call_count == 1 and fit_rows == [16 * 6]
        assert load.call_count == 1
    assert result["fit_calls"] == 1
    assert result["validation"]["pooled"]["windows"] == 36
    assert len(result["validation"]["per_person"]) == 6
    with (
        patch.object(
            study, "load_training", side_effect=AssertionError("no data reads")
        ),
        patch.object(
            WalkingGate, "fit_features", side_effect=AssertionError("no refit")
        ),
    ):
        verification = study.verify(SimpleNamespace(out_root=args.out_root))
    assert verification["verified"] and verification["new_fits"] == 0
    assert verification["validation"] == result["validation"]
    assert result["performance"]["batch_size"] == 64
    assert result["performance"]["lookahead_samples"] == 0
    with pytest.raises(ValueError, match="fresh out-root"):
        study.run(args)
    with pytest.raises(FileExistsError):
        study.verify(args)
    corrupt = study.read_json(args.out_root / "results.json")
    corrupt["threshold"] += 0.001
    (args.out_root / "results.json").write_bytes(module.canonical(corrupt))
    with pytest.raises(ValueError, match="frozen threshold"):
        study.verify(args)


def test_cache_numeric_corruption_rejected_and_validation_cannot_select(tmp_path):
    data = synthetic_training()
    first = SimpleNamespace(data_root="MOCK", out_root=tmp_path / "one")
    second = SimpleNamespace(data_root="MOCK", out_root=tmp_path / "two")
    changed = deepcopy(data)
    validation = np.isin(data["subjects"], study.roles()["validation"])
    changed["X"][validation] = changed["X"][validation] * 2 + 8
    changed["activity"][validation] = (changed["activity"][validation] + 1) % 3
    with patch.object(study, "load_training", side_effect=[data, changed]):
        study.run(first)
        study.run(second)
    assert (
        WalkingGate.load(first.out_root / "model.json").to_dict()
        == WalkingGate.load(second.out_root / "model.json").to_dict()
    )
    cache = study.read_cache(first.out_root / "features.npz")
    cache["features"][0, 0] += 0.01
    with (first.out_root / "features.npz").open("wb") as stream:
        np.savez_compressed(stream, **cache)
    with pytest.raises(ValueError, match="features.npz integrity"):
        study.verify(first)
