"""Synthetic protocol checks; never load original Santos payloads."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import covariance_robustness_study as study


def synthetic_data():
    rng = np.random.default_rng(731)
    subjects, days, trials, starts, features = [], [], [], [], []
    for position, subject in enumerate(study.prior.DEV_IDS):
        base = rng.normal(size=5)
        for day in (1, 2):
            for trial in (1, 2):
                for window in (0, 1):
                    subjects.append(subject)
                    days.append(day)
                    trials.append(trial)
                    starts.append(256 * window)
                    features.append(base + rng.normal(scale=0.03, size=5))
    n = len(subjects)
    return {
        "X": rng.normal(size=(n, 6, 32)).astype(np.float32),
        "subjects": np.array(subjects),
        "days": np.array(days),
        "trials": np.array(trials),
        "starts": np.array(starts),
        "feature_fixture": np.array(features),
    }


def rows_fixture(control_errors=3, full_errors=3, log_errors=3):
    rows = []
    for fold in study.crossfit.folds():
        ids = fold["validation_ids"]
        for arm, rejected in zip(
            study.ARMS, (control_errors, full_errors, log_errors), strict=True
        ):
            for condition in study.CONDITIONS:
                per_person = {
                    str(s): {
                        "false_rejects": rejected,
                        "n_genuine": 10,
                        "false_accepts": 0,
                        "n_impostor": 20,
                    }
                    for s in ids
                }
                pairs = {
                    f"{c}<-{p}": {"false_accepts": 0, "claims": 10}
                    for c in ids
                    for p in ids
                    if c != p
                }
                rows.append(
                    {
                        "fold": fold["fold"],
                        "arm": arm,
                        "condition": condition,
                        "metrics": {
                            "claim_graph_sha256": f"fold{fold['fold']}",
                            "fixed_threshold": {
                                "false_accepts": 0,
                                "n_impostor": 60,
                                "false_rejects": 3 * rejected,
                                "n_genuine": 30,
                            },
                            "per_subject_errors": per_person,
                        },
                        "pair_errors": pairs,
                    }
                )
    return rows


def test_registered_roles_budget_and_no_original_calib_test():
    plan = study.plan()
    assert plan["budget"] == {"fits": 300, "inner_scores": 240, "outer_scores": 420}
    allowed = set(study.prior.DEV_IDS)
    assert not allowed.intersection(
        (*study.prior.CALIBRATION_IDS, *study.prior.TEST_IDS)
    )
    roles = {s: [0, 0] for s in allowed}
    for fold in plan["folds"]:
        validation, refit = set(fold["validation_ids"]), set(fold["refit_ids"])
        assert len(validation) == 3 and len(refit) == 12 and not validation & refit
        assert validation | refit == allowed
        held = []
        for inner in fold["inner"]:
            train, scoring = set(inner["train_ids"]), set(inner["held_ids"])
            assert len(train) == 9 and len(scoring) == 3
            assert not train & scoring and train | scoring == refit
            held.extend(scoring)
        assert len(held) == len(set(held)) == 12
        for s in validation:
            roles[s][0] += 1
        for s in refit:
            roles[s][1] += 1
    assert set(map(tuple, roles.values())) == {(4, 16)}


def test_loading_is_allowlisted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        study.prior, "load_data", lambda archive, ids: calls.append((archive, ids))
    )
    study.load_development("synthetic")
    assert calls == [("synthetic", study.prior.DEV_IDS)]


def test_exact_tie_retains_control_and_ties_use_fixed_order():
    summary = study.summarize(rows_fixture())
    assert summary["winner"] == "control" and summary["eligible"] == []
    summary = study.summarize(rows_fixture(full_errors=2, log_errors=2))
    assert summary["winner"] == "full_covariance"
    assert summary["eligible"] == ["full_covariance", "log_covariance"]
    assert summary["summaries"]["full_covariance"]["clean"]["frr_exact"] == "1/5"


def test_diagnostics_cannot_change_nomination():
    rows = rows_fixture()
    for row in rows:
        if row["arm"] != "control" and row["condition"] != "clean":
            row["metrics"]["fixed_threshold"]["false_rejects"] = 0
            for person in row["metrics"]["per_subject_errors"].values():
                person["false_rejects"] = 0
    assert study.summarize(rows)["winner"] == "control"


def test_far_and_bad_split_gates_override_rejection_gain():
    rows = rows_fixture(full_errors=1)
    # One fold is above2% FAR; mean FAR remains below1%, but baseline has0 bad folds.
    bad = next(
        r
        for r in rows
        if r["fold"] == 0
        and r["arm"] == "full_covariance"
        and r["condition"] == "clean"
    )
    bad["metrics"]["fixed_threshold"]["false_accepts"] = 2
    pair = next(iter(bad["pair_errors"]))
    bad["pair_errors"][pair]["false_accepts"] = 2
    bad["metrics"]["per_subject_errors"][pair.split("<-")[0]]["false_accepts"] = 2
    assert study.summarize(rows)["winner"] == "control"


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "graph", "person_count", "bad_count"]
)
def test_invalid_completeness_and_counts_fail(fault):
    rows = rows_fixture()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif fault == "graph":
        rows[-1]["metrics"]["claim_graph_sha256"] = "different"
    elif fault == "person_count":
        rows[-1]["metrics"]["per_subject_errors"].popitem()
    else:
        rows[0]["metrics"]["fixed_threshold"]["false_accepts"] = True
    with pytest.raises(ValueError):
        study.summarize(rows)


def test_scale_uses_training_people_only():
    data = synthetic_data()
    ids = study.prior.DEV_IDS[:9]
    scale = study.training_scale(data, ids)
    changed = {**data, "X": data["X"].copy()}
    changed["X"][~np.isin(data["subjects"], ids)] *= 10000
    np.testing.assert_array_equal(study.training_scale(changed, ids), scale)
    assert scale.shape == (6,) and (scale > 0).all()
    assert scale[0] == scale[1] == scale[2] and scale[3] == scale[4] == scale[5]


@pytest.mark.parametrize("condition", study.CONDITIONS)
def test_real_stress_integration_keeps_enrollment_and_keys(condition):
    data = synthetic_data()
    original = data["X"].copy()
    scale = study.training_scale(data, study.prior.DEV_IDS[:9])
    changed = study.perturbed(data, condition, scale)
    np.testing.assert_array_equal(data["X"], original)
    np.testing.assert_array_equal(
        changed["X"][data["days"] == 1], original[data["days"] == 1]
    )
    for key in ("subjects", "days", "trials", "starts"):
        np.testing.assert_array_equal(changed[key], data[key])
    again = study.perturbed(data, condition, scale)
    np.testing.assert_array_equal(changed["X"], again["X"])
    if condition != "clean":
        assert not np.array_equal(
            changed["X"][data["days"] == 2], original[data["days"] == 2]
        )


def test_real_representations_and_stresses_integrate():
    data = synthetic_data()
    values = study.representations(data)
    assert {key: value.shape for key, value in values.items()} == {
        "control": (120, 362),
        "full_covariance": (120, 494),
        "log_covariance": (120, 494),
    }
    for value in values.values():
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(
        values["control"][:, :336], values["full_covariance"][:, :336]
    )
    np.testing.assert_array_equal(
        values["control"][:, :336], values["log_covariance"][:, :336]
    )
    rotated = study.representations(study.perturbed(data, "rotation30", np.ones(6)))
    for arm in study.ARMS:
        np.testing.assert_array_equal(
            values[arm][data["days"] == 1], rotated[arm][data["days"] == 1]
        )
        assert not np.array_equal(
            values[arm][data["days"] == 2], rotated[arm][data["days"] == 2]
        )


def test_replay_checks_decisions_even_within_score_tolerance(tmp_path):
    frozen = {
        "labels": np.array([1, 0]),
        "scores": np.array([0.1, 5e-8]),
        "metadata": {"synthetic": True},
    }
    saved = study._store(tmp_path, "scores", frozen)
    fresh = {**frozen, "scores": np.array([0.1, -5e-8])}
    with pytest.raises(ValueError, match="decisions changed"):
        study._replay(tmp_path, fresh, saved, 0.0)


def test_trial_cancellation_cannot_change_registered_graph():
    data = synthetic_data()
    ids = study.prior.DEV_IDS[:3]
    values = data["feature_fixture"].copy()
    target = (data["subjects"] == ids[0]) & (data["days"] == 2) & (data["trials"] == 1)
    values[target] = np.array([[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]])

    class IdentityMetric:
        def transform(self, features):
            return features

    graph = study.crossfit.expected_graph(data, ids)
    with pytest.raises(ValueError, match="registered"):
        study.score(IdentityMetric(), values, data, ids, graph)


def test_preflight_runtime_guard_precedes_data_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "hashes", lambda: {"source": "synthetic"})
    registration = {
        "plan": study.plan(),
        "sources": study.hashes(),
        **study.normalization.runtime(),
    }
    registration["python"] = "mismatch"
    study.prior.write_new(tmp_path / "registration.json", registration)
    monkeypatch.setattr(
        study, "load_development", lambda _: pytest.fail("data loaded before guard")
    )
    with pytest.raises(ValueError, match="registration/runtime"):
        study.verify(SimpleNamespace(out_root=str(tmp_path), archive_path="never"))


def test_synthetic_end_to_end_counts_actual_fits_and_replays(tmp_path, monkeypatch):
    data = synthetic_data()
    root = tmp_path / "run"
    args = SimpleNamespace(out_root=str(root), archive_path="synthetic")
    loads = []
    monkeypatch.setattr(
        study, "load_development", lambda archive: (loads.append(archive), data)[1]
    )
    monkeypatch.setattr(study, "hashes", lambda: {"source": "synthetic"})
    monkeypatch.setattr(
        study.prior, "_manifest", lambda data: {"synthetic": len(data["X"])}
    )
    monkeypatch.setattr(
        study,
        "representations",
        lambda d: {a: d["feature_fixture"].copy() for a in study.ARMS},
    )
    monkeypatch.setattr(study, "perturbed", lambda d, condition, scale: d)
    calls = []
    fit = study._fit

    def counted(values, d, ids):
        calls.append(tuple(ids))
        return fit(values, d, ids)

    monkeypatch.setattr(study, "_fit", counted)
    study.run(args)
    expected = [
        tuple(ids)
        for fold in study.crossfit.folds()
        for arm in study.ARMS
        for ids in (*[inner["train_ids"] for inner in fold["inner"]], fold["refit_ids"])
    ]
    assert calls == expected and len(calls) == 300
    assert study.prior.read(root / "registration.json")["plan"] == study.plan()
    assert study.prior.read(root / "results.json")["summary"]["winner"] == "control"
    monkeypatch.setattr(
        study, "_fit", lambda *a: pytest.fail("verification must not fit")
    )
    study.verify(args)
    assert loads == ["synthetic", "synthetic"]
    report = study.prior.read(root / "verification.json")
    assert report["verified"] and report["outer_comparisons"] == 420
    result_path = root / "results.json"
    frozen_bytes = result_path.read_bytes()
    frozen = study.prior.read(result_path)
    for fault, pattern in (
        ("assignment", "model assignment"),
        ("scale", "TRAIN dynamic RMS"),
        ("hash", "artifact"),
        ("threshold", "artifact"),
    ):
        bad = deepcopy(frozen)
        if fault == "assignment":
            bad["models"][0]["train_ids"] = bad["models"][0]["train_ids"][1:]
        elif fault == "scale":
            bad["scales"]["0"][0] += 1
        elif fault == "hash":
            bad["models"][0]["artifact"]["sha256"] = "bad-digest"
        else:
            bad["threshold_files"][0]["sha256"] = "bad-digest"
        result_path.write_bytes(study.prior.canonical(bad))
        with pytest.raises(ValueError, match=pattern):
            study.verify(args)
    result_path.write_bytes(frozen_bytes)
    with pytest.raises(ValueError, match="fresh empty"):
        study.run(args)
