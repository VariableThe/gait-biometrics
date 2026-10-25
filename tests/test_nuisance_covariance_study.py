"""Synthetic signals and temporary caches only; exactly200 tiny fits in the run test."""

from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from engine import robustness_selection as selector
from scripts import nuisance_covariance_study as study


def synthetic_data():
    rng = np.random.default_rng(9034)
    phase = 2 * np.pi * np.arange(256) / 256
    subjects, days, trials, starts, windows, features, recordings = (
        [] for _ in range(7)
    )
    for subject in study.prior.DEV_IDS:
        identity = rng.normal(size=5)
        for day in (1, 2):
            for trial in (1, 2):
                recordings.append(
                    {
                        "member": f"synthetic/{subject}/{day}/{trial}",
                        "subject": subject,
                        "day": day,
                        "trial": trial,
                        "sha256": "synthetic-no-archive",
                        "windows": 2,
                    }
                )
                for start in (0, 256):
                    subjects.append(subject)
                    days.append(day)
                    trials.append(trial)
                    starts.append(start)
                    signal = np.array(
                        [
                            3 * np.sin(3 * phase),
                            0.7 * np.cos(5 * phase),
                            9.81 + 0.4 * np.sin(2 * phase),
                            10 + 2 * np.cos(3 * phase),
                            -30 + 0.7 * np.sin(6 * phase),
                            2 + 0.2 * np.cos(4 * phase),
                        ]
                    )
                    windows.append(signal + rng.normal(scale=0.03, size=(6, 256)))
                    features.append(identity + rng.normal(scale=0.05, size=5))
    return {
        "X": np.asarray(windows, np.float32),
        "subjects": np.asarray(subjects),
        "days": np.asarray(days),
        "trials": np.asarray(trials),
        "starts": np.asarray(starts),
        "feature_fixture": np.asarray(features),
        "metadata": {
            "members_sha256": "synthetic-members",
            "recordings": recordings,
            "sampling_rate_hz": 100,
            "exclusions": {"zero_window_recordings": ["synthetic-retained-exclusion"]},
        },
    }


def tiny_features(data):
    """A fixed five-dimensional sensor-dependent fixture; no covariance estimator."""
    x = data["X"].astype(np.float64)
    summary = np.column_stack(
        (
            x[:, 0].mean(axis=1),
            x[:, 2].mean(axis=1),
            x[:, 3].mean(axis=1),
            x[:, 4, ::17].mean(axis=1),
            x[:, 5].std(axis=1),
        )
    )
    return data["feature_fixture"] + 0.025 * summary + 0.003 * summary**2


def args_for(root):
    return SimpleNamespace(out_root=root, archive_path="synthetic-never-an-archive")


def test_fixed_contract_closure_and_no_historical_global_changes():
    plan = study.plan()
    assert study.ARMS == ("baseline", "nuisance")
    assert plan["dimensions"] == {"baseline": 362, "nuisance": 362}
    assert (
        plan["budget"]
        == study.budget()
        == {"fits": 200, "inner_scores": 160, "outer_scores": 280}
    )
    assert plan["folds"] == study.crossfit.folds()
    assert plan["geometry_contexts"] == {
        "clean_dev": 1,
        "outer_validation": 140,
        "training_views": 400,
        "total": 541,
    }
    assert study.CONDITIONS == selector.CONDITIONS == study.canonical.CONDITIONS
    assert set(study.canonical.SOURCES).issubset(study.SOURCES)
    for stem, directory in (
        ("nuisance_covariance_study", "scripts"),
        ("nuisance_metric", "models"),
        ("robustness_selection", "engine"),
    ):
        assert {
            f"{directory}/{stem}.py",
            f"tests/test_{stem}.py",
            f"docs/{stem}.md",
        }.issubset(study.SOURCES)
    assert set(study.hashes()) == set(study.SOURCES)
    assert len(study.SOURCES) == len(set(study.SOURCES))
    assert (
        plan["prior_dev_inspection"]
        and plan["prior_test_inspection"]
        and plan["prior_stress_inspection"]
    )
    assert plan["canonical_source"] == "108a559bc1a83355dc85335e9f640786846f894a"
    assert plan["inspected_evidence_head"] == "1c861bfc2104b204c7262d931d8cffa19cc6e676"
    assert plan["training_transform"]["noise_seed"] == 20261026
    assert not plan["training_transform"]["evaluation_only_metadata_reused"]
    assert "blocked_by_invalid_train_scale" in plan["invalid_train_scale"]
    assert study.canonical.ARMS == ("control", "canonical336", "canonical362")
    assert study.canonical.frozen.ARMS == (
        "control",
        "full_covariance",
        "log_covariance",
    )
    for name in (
        "_subset",
        "_store",
        "_replay",
        "score",
        "training_scale",
        "perturbed",
        "_array_hash",
        "_row_keys",
    ):
        assert getattr(study, name) is getattr(study.canonical, name)
    assert list(inspect.signature(study.load_development).parameters) == ["archive"]
    assert "canonical.load_development(archive)" in inspect.getsource(
        study.load_development
    )
    assert not set(study.prior.DEV_IDS).intersection(
        (*study.prior.CALIBRATION_IDS, *study.prior.TEST_IDS)
    )


def test_training_raw_views_against_independent_reference_and_float32_rounding():
    data = synthetic_data()
    ids = study.crossfit.folds()[0]["inner"][0]["train_ids"]
    train, scale, views = study._training_inputs(data, ids)
    original = train["X"].copy()
    assert tuple(views) == ("gain09", "gain11", "noise_plus", "noise_minus")
    assert set(train["subjects"]) == set(ids) and set(train["days"]) == {1, 2}
    x = original.astype(np.float64)
    noise = []
    for row in zip(
        train["subjects"], train["days"], train["trials"], train["starts"], strict=True
    ):
        key = "/".join(str(int(v)) for v in row)
        encoded = json.dumps(
            ["imu-evaluation-stress-v1", 20261026, ["str", key]],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal((6, 256))
        noise.append(z * (scale * 0.01)[:, None])
    noise = np.asarray(noise)
    expected = (x * 0.9, x * 1.1, x + noise, x - noise)
    for name, reference in zip(study.TRAIN_VIEWS, expected, strict=True):
        assert views[name]["X"].dtype == np.float32
        np.testing.assert_array_equal(views[name]["X"], reference.astype(np.float32))
        for key in ("subjects", "days", "trials", "starts"):
            np.testing.assert_array_equal(views[name][key], train[key])
        for day in (1, 2):
            assert not np.array_equal(
                views[name]["X"][train["days"] == day], original[train["days"] == day]
            )
    # A gravity-preserving dynamic-only gain would fail this DC assertion.
    assert np.allclose(
        views["gain09"]["X"][:, 2].mean(axis=1), 0.9 * original[:, 2].mean(axis=1)
    )
    rounded_plus_minus = (2 * x - views["noise_plus"]["X"].astype(np.float64)).astype(
        np.float32
    )
    assert np.any(rounded_plus_minus != views["noise_minus"]["X"])
    np.testing.assert_array_equal(train["X"], original)


def test_stable_original_keys_batching_reordering_and_train9_scale_isolation():
    data = synthetic_data()
    fold = study.crossfit.folds()[0]
    ids = fold["inner"][0]["train_ids"]
    train, scale, full = study._training_inputs(data, ids)
    x = data["X"][np.isin(data["subjects"], ids)].astype(np.float64)
    centered = x - x.mean(axis=2, keepdims=True)
    reference = np.repeat(
        [
            np.sqrt(
                np.sum(centered[:, start : start + 3] ** 2)
                / centered[:, start : start + 3].size
            )
            for start in (0, 3)
        ],
        3,
    )
    np.testing.assert_allclose(scale, reference, rtol=1e-15, atol=0)
    changed = {**data, "X": data["X"].copy()}
    changed["X"][~np.isin(data["subjects"], ids)] *= 10000
    _, isolated_scale, isolated = study._training_inputs(changed, ids)
    np.testing.assert_array_equal(scale, isolated_scale)
    for name in study.TRAIN_VIEWS:
        np.testing.assert_array_equal(full[name]["X"], isolated[name]["X"])
    assert not np.array_equal(scale, study.training_scale(changed, fold["refit_ids"]))
    order = np.random.default_rng(19).permutation(len(train["X"]))

    def sliced(indices):
        return {
            k: v[indices] if isinstance(v, np.ndarray) and len(v) == len(order) else v
            for k, v in train.items()
        }

    shuffled = study.training_transform(sliced(order), scale)
    pieces = [
        study.training_transform(sliced(indices), scale)
        for indices in np.array_split(np.arange(len(order)), 7)
    ]
    for name in study.TRAIN_VIEWS:
        np.testing.assert_array_equal(shuffled[name]["X"], full[name]["X"][order])
        np.testing.assert_array_equal(
            np.concatenate([p[name]["X"] for p in pieces]), full[name]["X"]
        )
    renamed = sliced(np.arange(len(order)))
    renamed["starts"] = renamed["starts"] + 1024
    new = study.training_transform(renamed, scale)
    assert not np.array_equal(new["noise_plus"]["X"], full["noise_plus"]["X"])


@pytest.mark.parametrize("fault", ["dtype", "scale", "keys", "ids"])
def test_training_transform_rejects_invalid_contract(fault):
    data = synthetic_data()
    ids = study.crossfit.folds()[0]["inner"][0]["train_ids"]
    train, scale, _views = study._training_inputs(data, ids)
    if fault == "ids":
        with pytest.raises(ValueError, match="nine/twelve"):
            study._training_inputs(data, ids[:-1])
        return
    if fault == "dtype":
        train["X"] = train["X"].astype(np.float64)
    elif fault == "scale":
        scale[1] *= 2
    else:
        train["starts"][0] = train["starts"][1]
    with pytest.raises(ValueError):
        study.training_transform(train, scale)


def test_real_canonical362_training_and_inherited_validation_without_fits(monkeypatch):
    monkeypatch.setattr(
        study, "fit_nuisance_metric", lambda *a: pytest.fail("unexpected metric fit")
    )
    data = synthetic_data()
    ids = study.crossfit.folds()[0]["inner"][0]["train_ids"]
    train, scale, views = study._training_inputs(data, ids)
    clean = study.representations(train)
    assert clean.shape == (72, 362)
    np.testing.assert_array_equal(
        clean, study.canonical.representations(train)["canonical362"]
    )
    for view in study.TRAIN_VIEWS:
        fresh = study.representations(views[view])
        assert fresh.shape == clean.shape and np.isfinite(fresh).all()
        np.testing.assert_array_equal(
            fresh[:, 336:], study.prior.features(views[view])["lagcov"][:, 336:]
        )
        assert not np.array_equal(fresh, clean)
    fold = study.crossfit.folds()[0]
    validation = study._subset(data, fold["validation_ids"])
    scale = study.training_scale(data, fold["refit_ids"])
    enrollment = study.representations(validation)[validation["days"] == 1]
    for condition in study.CONDITIONS:
        changed = study.perturbed(validation, condition, scale)
        np.testing.assert_array_equal(
            changed["X"][validation["days"] == 1],
            validation["X"][validation["days"] == 1],
        )
        np.testing.assert_array_equal(
            study.representations(changed)[validation["days"] == 1], enrollment
        )
        np.testing.assert_array_equal(
            changed["X"], study.canonical.perturbed(validation, condition, scale)["X"]
        )


def test_source_graph_cancellation_aborts_without_fitting():
    data = synthetic_data()
    ids = study.prior.DEV_IDS[:3]
    values = data["feature_fixture"].copy()
    mask = (data["subjects"] == ids[0]) & (data["days"] == 2) & (data["trials"] == 1)
    values[mask] = [[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]]

    class IdentityMetric:
        def transform(self, features):
            return features

    with pytest.raises(ValueError, match="registered"):
        study.score(
            IdentityMetric(),
            values,
            data,
            ids,
            study.crossfit.expected_graph(data, ids),
        )


def test_source_graph_preflight_refuses_missing_day_before_features_or_fit(
    tmp_path, monkeypatch
):
    data = synthetic_data()
    mask = ~((data["subjects"] == study.prior.DEV_IDS[0]) & (data["days"] == 2))
    data = {
        key: value[mask] if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    monkeypatch.setattr(study, "load_development", lambda _: data)
    monkeypatch.setattr(
        study, "representations", lambda _: pytest.fail("features before graph refusal")
    )
    monkeypatch.setattr(
        study, "_fit", lambda *a: pytest.fail("fit before graph refusal")
    )
    with pytest.raises(ValueError, match="both source days"):
        study.run(args_for(tmp_path))
    assert (tmp_path / "registration.json").exists()
    assert not list(tmp_path.glob("*_model.npz"))


@pytest.mark.parametrize("fault", ["source", "runtime", "plan"])
def test_source_and_runtime_refusal_before_archive(tmp_path, monkeypatch, fault):
    registration = study._registration()
    if fault == "source":
        registration["sources"]["engine/robustness_selection.py"] = "tampered"
    elif fault == "runtime":
        registration["python"] = "tampered"
    else:
        registration["plan"]["training_transform"]["noise_seed"] -= 1
    study.prior.write_new(tmp_path / "registration.json", registration)
    monkeypatch.setattr(
        study, "load_development", lambda _: pytest.fail("archive opened before guard")
    )
    with pytest.raises(ValueError, match="registration/runtime"):
        study.verify(args_for(tmp_path))


def test_budget_refusal_before_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "budget", lambda: {"fits": 199, "inner_scores": 160, "outer_scores": 280}
    )
    monkeypatch.setattr(
        study, "load_development", lambda _: pytest.fail("archive opened before budget")
    )
    with pytest.raises(ValueError, match="preflight fit/score budget"):
        study.run(args_for(tmp_path))


@pytest.mark.parametrize("fault", ["clean", "last_training_view"])
def test_all541_geometry_contexts_persist_before_any_feature_or_fit(
    tmp_path, monkeypatch, fault
):
    data = synthetic_data()
    calls = []
    if fault == "clean":
        data["X"][0] = 0
    else:
        transform = study.training_transform

        def invalid_last_training(d, scale):
            views = transform(d, scale)
            calls.append(1)
            if len(calls) == 100:
                views["noise_minus"]["X"][0] = 0
            return views

        monkeypatch.setattr(study, "training_transform", invalid_last_training)
    monkeypatch.setattr(study, "load_development", lambda _: data)
    monkeypatch.setattr(
        study, "representations", lambda _: pytest.fail("feature before geometry abort")
    )
    monkeypatch.setattr(
        study, "_fit", lambda *a: pytest.fail("fit before geometry abort")
    )
    with pytest.raises(ValueError, match="abort before features/fits"):
        study.run(args_for(tmp_path))
    manifest = study.prior.read(tmp_path / "development.json")
    geometry = manifest["geometry"]
    assert geometry["contexts"] == 541 and geometry["invalid_context_windows"] > 0
    assert manifest["source_coverage"] == data["metadata"]
    assert (
        sum(
            len(assignment["views"])
            for group in geometry["training"].values()
            for assignment in group.values()
        )
        == 400
    )
    record = (
        geometry["clean"]
        if fault == "clean"
        else geometry["training"]["19"]["refit"]["views"]["noise_minus"]
    )
    arrays = study.prior.load_npz(
        study.prior._verify_artifact(tmp_path, record["artifact"])
    )
    assert (~arrays["valid"]).sum() == 1
    assert arrays["reason"][~arrays["valid"]].tolist() == ["gravity_unidentifiable"]
    assert np.isnan(arrays["frames"][~arrays["valid"]]).all()
    assert (
        not list(tmp_path.glob("*_model.npz"))
        and not (tmp_path / "results.json").exists()
    )


@pytest.mark.parametrize(
    "fault", ["inner9_acceleration", "refit12_gyroscope", "all_dev"]
)
def test_invalid_train_rms_persists_all541_records_without_inventing_noise(
    tmp_path, monkeypatch, fault
):
    data = synthetic_data()
    first = study.crossfit.folds()[0]
    ids = first["inner"][0]["train_ids"]
    if fault == "inner9_acceleration":
        data["X"][np.isin(data["subjects"], ids), :3] = 0
    elif fault == "refit12_gyroscope":
        ids = first["refit_ids"]
        data["X"][np.isin(data["subjects"], ids), 3:] = 0
    else:
        data["X"][:] = 0
    original = data["X"].copy()
    coverage = study.normalization.coverage_plan(data)
    transform, perturb = study.training_transform, study.perturbed

    def positive_transform(selected, scale):
        assert np.isfinite(scale).all() and (np.asarray(scale) > 0).all()
        return transform(selected, scale)

    def positive_noise(selected, condition, scale):
        if condition.startswith("noise"):
            assert np.isfinite(scale).all() and (np.asarray(scale) > 0).all()
        return perturb(selected, condition, scale)

    monkeypatch.setattr(study, "training_transform", positive_transform)
    monkeypatch.setattr(study, "perturbed", positive_noise)
    monkeypatch.setattr(study, "load_development", lambda _: data)
    monkeypatch.setattr(
        study,
        "representations",
        lambda _: pytest.fail("features with invalid TRAIN RMS"),
    )
    monkeypatch.setattr(
        study, "_fit", lambda *a: pytest.fail("fit with invalid TRAIN RMS")
    )
    monkeypatch.setattr(
        study,
        "fit_nuisance_metric",
        lambda *a: pytest.fail("estimator with invalid RMS"),
    )
    with pytest.raises(ValueError, match="invalid TRAIN.*abort before features/fits"):
        study.run(args_for(tmp_path))
    manifest = study.prior.read(tmp_path / "development.json")
    geometry = manifest["geometry"]
    registration = study.prior.read(tmp_path / "registration.json")
    assert registration["sources"] == study.hashes() and len(study.SOURCES) == 50
    assert (
        registration["plan"]["invalid_train_scale"]
        == study.plan()["invalid_train_scale"]
    )
    assert manifest["registration_sha256"] == study.prior.digest(
        tmp_path / "registration.json"
    )
    assert manifest["source_coverage"] == data["metadata"]
    assert manifest["data"] == study.prior._manifest(data)
    assert manifest["coverage"] == coverage
    assert geometry["status"] == "failed_invalid_train_scale"
    assert geometry["contexts"] == len(list(tmp_path.glob("geometry_*.npz"))) == 541

    def reference(train_ids):
        # Independent RMS formula over the original assigned rows; no preflight helper.
        values = original[np.isin(data["subjects"], train_ids)].astype(np.float64)
        centered = values - values.mean(axis=-1, keepdims=True)
        return np.repeat(
            [
                np.sqrt(
                    np.sum(centered[:, start : start + 3] ** 2)
                    / centered[:, start : start + 3].size
                )
                for start in (0, 3)
            ],
            3,
        )

    contexts = [(geometry["clean"], data, False)]
    expected_errors = {}
    for fold in study.crossfit.folds():
        key = str(fold["fold"])
        refit_scale = reference(fold["refit_ids"])
        np.testing.assert_allclose(
            geometry["scales"][key], refit_scale, rtol=1e-15, atol=0
        )
        assert geometry["scale_train_ids"][key] == fold["refit_ids"]
        validation = study._subset(data, fold["validation_ids"])
        for condition in study.CONDITIONS:
            blocked = bool((refit_scale <= 0).any()) and condition.startswith("noise")
            contexts.append((geometry["folds"][key][condition], validation, blocked))
        for role, train_ids in (
            *(
                (i, assignment["train_ids"])
                for i, assignment in enumerate(fold["inner"])
            ),
            ("refit", fold["refit_ids"]),
        ):
            name = "refit" if role == "refit" else f"inner{role}"
            assignment = geometry["training"][key][name]
            selected = study._subset(data, train_ids)
            scale = reference(train_ids)
            invalid = bool((scale <= 0).any())
            assert assignment["fold"] == fold["fold"] and assignment["role"] == role
            assert assignment["train_ids"] == train_ids and assignment["arms"] == [
                "baseline",
                "nuisance",
            ]
            np.testing.assert_allclose(assignment["scale"], scale, rtol=1e-15, atol=0)
            assert assignment["clean_windows_sha256"] == study._array_hash(
                selected["X"]
            )
            if invalid:
                expected_errors[(fold["fold"], role)] = (train_ids, scale)
                assert assignment["scale_status"] == "invalid_train_scale"
            else:
                assert "scale_status" not in assignment
            assert tuple(assignment["views"]) == tuple(sorted(study.TRAIN_VIEWS))
            for view in study.TRAIN_VIEWS:
                contexts.append(
                    (
                        assignment["views"][view],
                        selected,
                        invalid and view.startswith("noise"),
                    )
                )

    actual_errors = {(e["fold"], e["role"]): e for e in geometry["scale_errors"]}
    assert len(actual_errors) == len(geometry["scale_errors"]) == len(expected_errors)
    assert actual_errors.keys() == expected_errors.keys()
    for key, (train_ids, scale) in expected_errors.items():
        error = actual_errors[key]
        assert (
            error["train_ids"] == train_ids and error["status"] == "invalid_train_scale"
        )
        assert "positive finite" in error["reason"]
        np.testing.assert_allclose(error["scale"], scale, rtol=1e-15, atol=0)
        assert error["invalid_triads"] == [
            {"triad": triad, "rms": 0.0, "reason": "not_positive"}
            for triad, value in zip(
                ("acceleration", "gyroscope"), scale[[0, 3]], strict=True
            )
            if value == 0
        ]
    target_role = "refit" if fault == "refit12_gyroscope" else 0
    assert (0, target_role) in actual_errors
    with pytest.raises(ValueError, match="positive finite"):
        study._training_inputs(data, ids)
    with pytest.raises(ValueError, match="positive finite"):
        transform(study._subset(data, ids), reference(ids))

    assert len(contexts) == 541
    blocked_count = 0
    for record, source, blocked in contexts:
        keys = np.column_stack(
            [source[k] for k in ("subjects", "days", "trials", "starts")]
        )
        assert record["rows"] == len(source["X"])
        assert record["row_keys_sha256"] == study._array_hash(keys)
        arrays = study.prior.load_npz(
            study.prior._verify_artifact(tmp_path, record["artifact"])
        )
        np.testing.assert_array_equal(arrays["row_keys"], keys)
        assert arrays["frames"].shape == (len(keys), 2, 3, 3)
        assert int(arrays["valid"].sum()) == record["valid"]
        assert record["valid"] + record["invalid"] == len(keys)
        if blocked:
            blocked_count += 1
            assert record["status"] == "blocked_by_invalid_train_scale"
            assert record["materialized"] is False and record["windows_sha256"] is None
            assert record["source_windows_sha256"] == study._array_hash(source["X"])
            assert record["valid"] == 0 and not arrays["valid"].any()
            assert arrays["reason"].tolist() == [
                "blocked_by_invalid_train_scale"
            ] * len(keys)
            for field in (
                "frames",
                "gravity_ratio",
                "horizontal_energy_ratio",
                "eigengap_ratio",
            ):
                assert np.isnan(arrays[field]).all()
        else:
            assert "status" not in record and isinstance(record["windows_sha256"], str)
            assert not np.isin(arrays["reason"], "blocked_by_invalid_train_scale").any()
            if fault == "refit12_gyroscope":
                assert arrays[
                    "valid"
                ].all()  # Valid geometry does not repair invalid gyro RMS.
    assert blocked_count == geometry["blocked_contexts"]
    assert geometry["materialized_contexts"] == 541 - blocked_count
    assert geometry["invalid_context_windows"] == sum(
        r["invalid"] for r, _, _ in contexts
    )
    if fault == "all_dev":
        assert len(actual_errors) == 100 and blocked_count == 240
    np.testing.assert_array_equal(data["X"], original)
    assert (
        not list(tmp_path.glob("*_model.npz"))
        and not (tmp_path / "results.json").exists()
    )
    inventory = {path.name: study.prior.digest(path) for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="fresh empty"):
        study.run(args_for(tmp_path))
    assert inventory == {
        path.name: study.prior.digest(path) for path in tmp_path.iterdir()
    }


def test_replay_requires_identical_decisions_even_inside_numeric_tolerance(tmp_path):
    cached = {
        "labels": np.array([1, 0]),
        "scores": np.array([0.1, 5e-8]),
        "metadata": {"synthetic": True},
    }
    record = study._store(tmp_path, "synthetic", cached)
    with pytest.raises(ValueError, match="decisions changed"):
        study._replay(
            tmp_path, {**cached, "scores": np.array([0.1, -5e-8])}, record, 0.0
        )


def test_rows_delegate_exact_ranking_decisions_and_selection(monkeypatch):
    data = synthetic_data()
    ids = study.prior.DEV_IDS[:3]

    class IdentityMetric:
        def transform(self, values):
            return values / np.linalg.norm(values, axis=1, keepdims=True)

    scored = study.score(
        IdentityMetric(),
        data["feature_fixture"],
        data,
        ids,
        study.crossfit.expected_graph(data, ids),
    )
    row = study._row(0, "nuisance", "clean", scored, 0.0)
    assert row["ranking"] == selector.ranking_metrics(
        scored["labels"], scored["scores"], scored["metadata"]["probe_subject_ids"]
    )
    decisions = np.ascontiguousarray(np.asarray(scored["scores"], np.float64) >= 0.0)
    payload = (
        json.dumps(
            [str(decisions.dtype), list(decisions.shape)], separators=(",", ":")
        ).encode()
        + decisions.tobytes()
    )
    assert row["decision_sha256"] == hashlib.sha256(payload).hexdigest()
    calls = []

    def selected(rows, *, dev_ids, folds):
        calls.append((rows, dev_ids, folds))
        return {"sentinel": True}

    monkeypatch.setattr(selector, "summarize_nuisance_study", selected)
    assert study.summarize([row]) == {"sentinel": True}
    assert calls == [([row], study.prior.DEV_IDS, study.crossfit.folds())]


def test_200_actual_tiny_fits_440_fit_free_replays_and_tampering(tmp_path, monkeypatch):
    data = synthetic_data()
    root = tmp_path / "synthetic"
    args = args_for(root)
    fits, loads, replays = [], [], []
    original_fit = study.fit_nuisance_metric
    original_calibrate = study.calibrate
    calibration_events = []

    def load(archive):
        assert study.prior.read(root / "registration.json") == study._registration()
        loads.append(archive)
        return data

    def representation(d):
        geometry = study.prior.read(root / "development.json")["geometry"]
        assert geometry["contexts"] == 541 and geometry["invalid_context_windows"] == 0
        return tiny_features(d)

    def fit(values, subjects, view_features=None):
        assert values.shape[1] == 5 and len(values) in (72, 96)
        ids = tuple(sorted(set(map(int, subjects))))
        if len(ids) == 12:
            assert len(calibration_events) == 2 * (len(fits) // 10 + 1)
        selected = study._subset(data, ids)
        np.testing.assert_array_equal(values, tiny_features(selected))
        if view_features is not None:
            assert view_features.shape == (len(values), 4, 5)
            assert np.any(view_features != values[:, None, :])
        fits.append((ids, view_features is not None))
        return original_fit(values, subjects, view_features)

    def calibrate_before_any_refit(labels, scores):
        # Eight completed inner fits (four per arm), then two cutoffs, then
        # either twelve-person refit. Check the actual clean pool's graph too.
        assert len(fits) % 10 == 8
        fold = study.crossfit.folds()[len(fits) // 10]
        graphs = [
            study.crossfit.expected_graph(data, inner["held_ids"])
            for inner in fold["inner"]
        ]
        assert labels.tolist() == [
            label for graph in graphs for label in graph["labels"]
        ]
        calibration_events.append(len(fits))
        return original_calibrate(labels, scores)

    monkeypatch.setattr(study, "load_development", load)
    monkeypatch.setattr(study, "representations", representation)
    monkeypatch.setattr(study, "fit_nuisance_metric", fit)
    monkeypatch.setattr(study, "calibrate", calibrate_before_any_refit)
    study.run(args)
    assert calibration_events == [n for fold in range(20) for n in (10 * fold + 8,) * 2]
    monkeypatch.setattr(study, "calibrate", original_calibrate)
    expected = [
        (tuple(sorted(ids)), arm == "nuisance")
        for fold in study.crossfit.folds()
        for role, ids in study._roles(fold)
        for arm in study.ARMS
    ]
    assert fits == expected and len(fits) == 200
    result_path = root / "results.json"
    result = study.prior.read(result_path)
    manifest = study.prior.read(root / "development.json")
    assert [len(result[k]) for k in ("models", "inner", "outer")] == [200, 160, 280]
    assert len(result["rows"]) == 280 and result["fit_calls"] == 200
    assert manifest["source_coverage"] == data["metadata"]
    for fold in study.crossfit.folds():
        for role, ids in study._roles(fold):
            assignment = manifest["geometry"]["training"][str(fold["fold"])][
                study._role_name(role)
            ]
            assert assignment["train_ids"] == ids and assignment["arms"] == list(
                study.ARMS
            )
            np.testing.assert_array_equal(
                assignment["scale"], study.training_scale(data, ids)
            )
            train, _scale, views = study._training_inputs(data, ids)
            for name in study.TRAIN_VIEWS:
                record = assignment["views"][name]
                assert record["row_keys_sha256"] == study._array_hash(
                    study._row_keys(train)
                )
                assert record["windows_sha256"] == study._array_hash(views[name]["X"])
    replay = study._replay

    def counted_replay(*args):
        replays.append(1)
        return replay(*args)

    def forbidden(*a, **kw):
        pytest.fail("verification attempted fitting")

    monkeypatch.setattr(study, "_replay", counted_replay)
    monkeypatch.setattr(study, "fit_nuisance_metric", forbidden)
    monkeypatch.setattr(study.WithinClassMetric, "fit", forbidden)
    from sklearn.covariance import LedoitWolf

    monkeypatch.setattr(LedoitWolf, "fit", forbidden)
    report = study.verify(args)
    assert report["verified"] and report["score_replays"] == len(replays) == 440
    assert report["geometry_contexts"] == 541 and report["models"] == 200
    assert len(fits) == 200 and loads == [args.archive_path, args.archive_path]
    original_result = result_path.read_bytes()

    # Direct restore exercises numerical attacks after valid artifact rehashing.
    # Swapping C and V preserves their sum and whitener, so checking only A fails.
    for arm in study.ARMS:
        record = next(
            r
            for r in result["models"]
            if r["fold"] == 0 and r["role"] == 0 and r["arm"] == arm
        )
        ids = record["train_ids"]
        views = study._training_features(data, ids)
        artifact = record["artifact"]
        path = root / artifact["path"]
        original_bytes = path.read_bytes()
        state = study.prior.load_npz(path)
        clean = tiny_features(study._subset(data, ids))
        if arm == "baseline":
            assert not np.any(state["diagnostic_nuisance_second_moment"])
        else:
            # Independent row/view outer products, then identity means.
            std = clean.std(axis=0).clip(min=1e-6)
            subjects = study._subset(data, ids)["subjects"]
            person_moments = []
            for subject in ids:
                moments = [
                    np.outer(displacement, displacement)
                    for index in np.flatnonzero(subjects == subject)
                    for view in views[index]
                    for displacement in [(view - clean[index]) / std]
                ]
                person_moments.append(np.mean(moments, axis=0))
            np.testing.assert_allclose(
                state["diagnostic_nuisance_second_moment"],
                np.mean(person_moments, axis=0),
                rtol=1e-12,
                atol=1e-12,
            )
        for fault in (
            "covariance_assignment",
            "nuisance",
            "eigenvalues",
            "floor",
            "projection",
            "counts",
            "shrinkage",
            "shape",
            "finite",
            "symmetry",
            "mean",
        ):
            changed = {k: v.copy() for k, v in state.items()}
            field = "diagnostic_nuisance_second_moment"
            if fault == "covariance_assignment":
                changed["diagnostic_clean_covariance"], changed[field] = (
                    changed[field],
                    changed["diagnostic_clean_covariance"],
                )
            elif fault == "nuisance":
                changed[field][0, 0] += 0.1
            elif fault == "eigenvalues":
                changed["diagnostic_eigenvalues"][0] += 0.1
            elif fault == "floor":
                changed["diagnostic_eigenvalue_floor"] *= 2
            elif fault == "projection":
                changed["projection"][0, 0] += 0.1
            elif fault == "counts":
                changed["diagnostic_original_rows"] += 1
            elif fault == "shrinkage":
                changed["shrinkage"] = np.asarray(float(state["shrinkage"]) / 2)
            elif fault == "shape":
                changed[field] = changed[field][:1]
            elif fault == "finite":
                changed[field][0, 0] = np.nan
            elif fault == "symmetry":
                changed[field][0, 1] += 0.1
            else:
                changed["mean"][0] += 0.1
            path.unlink()
            study.prior.save_npz(path, changed)
            tampered = deepcopy(record)
            tampered["artifact"]["sha256"] = study.prior.digest(path)
            if fault == "shrinkage":
                tampered["stats"]["shrinkage"] = float(changed["shrinkage"])
            with pytest.raises(ValueError):
                study._restore_model(
                    root, tampered, tiny_features(data), data, ids, views
                )
            path.write_bytes(original_bytes)

    # Subsequent verification attacks reuse the already checked geometry to keep
    # these tests bounded. The initial replay above checks all541 real contexts.
    monkeypatch.setattr(study, "geometry_coverage", lambda *a: manifest["geometry"])
    for fault, pattern in (
        ("assignment", "model assignment"),
        ("feature_hash", "assigned TRAIN statistics"),
        ("scale", "TRAIN dynamic RMS"),
        ("digest", "artifact"),
        ("threshold_order", "threshold assignments"),
        ("inventory", "inventory"),
        ("artifact_swap", "artifact assignment"),
        ("metadata", "score metadata"),
        ("decision", "cached ranking/counts/metrics/decisions"),
        ("ranking", "cached ranking/counts/metrics/decisions"),
        ("counts", "cached ranking/counts/metrics/decisions"),
        ("selection", "exact selection"),
        ("fits", "fit budget"),
    ):
        bad = deepcopy(result)
        if fault == "assignment":
            bad["models"][0]["train_ids"].pop()
        elif fault == "feature_hash":
            bad["models"][0]["stats"]["train_view_features_sha256"] = "wrong"
        elif fault == "scale":
            bad["scales"]["0"][0] += 1
        elif fault == "digest":
            bad["models"][0]["artifact"]["sha256"] = "wrong"
        elif fault == "threshold_order":
            bad["threshold_files"].reverse()
        elif fault == "inventory":
            bad["inner"][-1] = deepcopy(bad["inner"][0])
        elif fault == "artifact_swap":
            bad["models"][0]["artifact"], bad["models"][1]["artifact"] = (
                bad["models"][1]["artifact"],
                bad["models"][0]["artifact"],
            )
        elif fault == "metadata":
            bad["outer"][0]["metadata"]["probe_trial_ids"][0] = 999
        elif fault == "decision":
            bad["rows"][0]["decision_sha256"] = "0" * 64
        elif fault == "ranking":
            bad["rows"][0]["ranking"]["eer_exact"] = "1/7"
        elif fault == "counts":
            bad["rows"][0]["metrics"]["fixed_threshold"]["false_rejects"] += 1
        elif fault == "selection":
            bad["summary"]["winner"] = (
                "nuisance" if result["summary"]["winner"] == "baseline" else "baseline"
            )
        else:
            bad["fit_calls"] -= 1
        result_path.write_bytes(study.prior.canonical(bad))
        with pytest.raises(ValueError, match=pattern):
            study.verify(args)
        result_path.write_bytes(original_result)

    threshold_artifact = result["threshold_files"][0]
    path = root / threshold_artifact["path"]
    saved_bytes = path.read_bytes()
    thresholds = study.prior.read(path)
    thresholds["baseline"]["threshold"] = 10.0
    path.write_bytes(study.prior.canonical(thresholds))
    bad = deepcopy(result)
    bad["threshold_files"][0]["sha256"] = study.prior.digest(path)
    result_path.write_bytes(study.prior.canonical(bad))
    with pytest.raises(ValueError, match="frozen pooled threshold"):
        study.verify(args)
    path.write_bytes(saved_bytes)
    result_path.write_bytes(original_result)

    assignment = manifest["geometry"]["training"]["19"]["refit"]
    record = assignment["views"]["noise_minus"]
    path = root / record["artifact"]["path"]
    arrays = study.prior.load_npz(path)
    arrays["eigengap_ratio"][0] += 0.1
    path.unlink()
    study.prior.save_npz(path, arrays)
    bad_geometry = deepcopy(manifest["geometry"])
    bad_record = bad_geometry["training"]["19"]["refit"]["views"]["noise_minus"]
    bad_record["artifact"]["sha256"] = study.prior.digest(path)
    with pytest.raises(ValueError, match="geometry manifest"):
        study.canonical._geometry_record(
            root,
            path.stem,
            study._training_inputs(data, assignment["train_ids"])[2]["noise_minus"],
            bad_record,
        )
    with pytest.raises(ValueError, match="fresh empty"):
        study.run(args)
