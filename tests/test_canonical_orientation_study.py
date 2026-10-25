"""Synthetic (N,6,256) only; tiny TRAIN metrics, no archive/artifact inputs."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from models.session_robust import extract_lagged_covariance_features
from scripts import canonical_orientation_study as study


def synthetic_data():
    rng = np.random.default_rng(731)
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
                        "sha256": "synthetic-not-an-archive",
                        "windows": 2,
                    }
                )
                for window in (0, 1):
                    subjects.append(subject)
                    days.append(day)
                    trials.append(trial)
                    starts.append(256 * window)
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
                    features.append(identity + rng.normal(scale=0.03, size=5))
    return {
        "X": np.asarray(windows, dtype=np.float32),
        "subjects": np.asarray(subjects),
        "days": np.asarray(days),
        "trials": np.asarray(trials),
        "starts": np.asarray(starts),
        "feature_fixture": np.asarray(features),
        "metadata": {
            "members_sha256": "synthetic-members",
            "recordings": recordings,
            "sampling_rate_hz": 100,
            "exclusions": {"zero_window_recordings": []},
        },
    }


def rows_fixture(errors=(3, 3, 3)):
    rows = []
    for fold in study.crossfit.folds():
        ids = fold["validation_ids"]
        for arm, rejected in zip(study.ARMS, errors, strict=True):
            for condition in study.CONDITIONS:
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
                            "per_subject_errors": {
                                str(s): {
                                    "false_rejects": rejected,
                                    "n_genuine": 10,
                                    "false_accepts": 0,
                                    "n_impostor": 20,
                                }
                                for s in ids
                            },
                        },
                        "pair_errors": {
                            f"{c}<-{p}": {"false_accepts": 0, "claims": 10}
                            for c in ids
                            for p in ids
                            if c != p
                        },
                    }
                )
    return rows


def _false_accepts(row, count):
    pair = next(iter(row["pair_errors"]))
    row["pair_errors"][pair]["false_accepts"] = count
    row["metrics"]["per_subject_errors"][pair.split("<-")[0]]["false_accepts"] = count
    row["metrics"]["fixed_threshold"]["false_accepts"] = count


def test_registration_dimensions_origins_roles_and_budget():
    plan = study.plan()
    assert study.ARMS == ("control", "canonical336", "canonical362")
    assert plan["dimensions"] == {
        "control": 362,
        "canonical336": 336,
        "canonical362": 362,
    }
    assert (
        plan["budget"]
        == study.budget()
        == {"fits": 300, "inner_scores": 240, "outer_scores": 420}
    )
    assert study.CONDITIONS == (
        "clean",
        "rotation30",
        "rotation90",
        "gain09",
        "gain11",
        "noise01",
        "noise05",
    )
    assert study.CONDITIONS == study.frozen.CONDITIONS
    assert set(study.frozen.SOURCES).issubset(study.SOURCES)
    assert {
        "models/gait_canonical.py",
        "tests/test_gait_canonical.py",
        "docs/gait_canonical.md",
        "scripts/canonical_orientation_study.py",
        "tests/test_canonical_orientation_study.py",
        "docs/canonical_orientation_study.md",
    }.issubset(study.SOURCES)
    assert set(study.hashes()) == set(study.SOURCES)
    assert plan["origins"]["copied_protocol_run_verify_summarize"] in study.SOURCES
    assert plan["geometry_guards"] == {
        "gravity_ratio_strictly_above": 1e-6,
        "horizontal_energy_ratio_strictly_above": 1e-12,
        "eigengap_ratio_strictly_above": 1e-6,
        "frozen_before_dev": True,
    }
    allowed = set(study.prior.DEV_IDS)
    assert not allowed.intersection(
        (*study.prior.CALIBRATION_IDS, *study.prior.TEST_IDS)
    )
    roles = {s: [0, 0] for s in allowed}
    assert plan["folds"] == study.crossfit.folds()
    for fold in plan["folds"]:
        validation, refit = set(fold["validation_ids"]), set(fold["refit_ids"])
        assert len(validation) == 3 and len(refit) == 12
        assert not validation & refit and validation | refit == allowed
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
    for name in (
        "_store",
        "_replay",
        "_row",
        "_fit",
        "_subset",
        "score",
        "training_scale",
        "perturbed",
    ):
        assert getattr(study, name) is getattr(study.frozen, name)


def test_loading_has_no_subject_override():
    # Check the actual call without opening an archive or changing old helpers.
    import inspect

    assert list(inspect.signature(study.load_development).parameters) == ["archive"]
    assert "return frozen.load_development(archive)" in inspect.getsource(
        study.load_development
    )
    assert "prior.load_data(archive, prior.DEV_IDS)" in inspect.getsource(
        study.frozen.load_development
    )


def test_exact_ties_and_fixed_order():
    assert study.summarize(rows_fixture())["winner"] == "control"
    selected = study.summarize(rows_fixture((3, 2, 2)))
    assert selected["winner"] == "canonical336"
    assert selected["eligible"] == ["canonical336", "canonical362"]
    assert selected["summaries"]["canonical336"]["clean"]["frr_exact"] == "1/5"
    assert study.summarize(rows_fixture((3, 2, 1)))["winner"] == "canonical362"


def test_unequal_fold_person_denominators_and_rational_ties():
    rows = rows_fixture()
    for row in rows:
        people = row["metrics"]["per_subject_errors"]
        for subject, person in people.items():
            n = 10 * (1 + (int(subject) + row["fold"]) % 3)
            person.update(n_genuine=n, false_rejects=n // 10)
        for pair, counts in row["pair_errors"].items():
            counts["claims"] = people[pair.split("<-")[1]]["n_genuine"]
        for subject, person in people.items():
            person["n_impostor"] = sum(
                c["claims"]
                for p, c in row["pair_errors"].items()
                if p.split("<-")[0] == subject
            )
        fixed = row["metrics"]["fixed_threshold"]
        fixed.update(
            n_genuine=sum(p["n_genuine"] for p in people.values()),
            false_rejects=sum(p["false_rejects"] for p in people.values()),
            n_impostor=sum(p["claims"] for p in row["pair_errors"].values()),
        )
    tied = study.summarize(rows)
    assert tied["winner"] == "control"
    assert {tied["summaries"][a]["clean"]["frr_exact"] for a in study.ARMS} == {"1/10"}
    changed = next(
        r
        for r in rows
        if r["fold"] == 0 and r["arm"] == "canonical336" and r["condition"] == "clean"
    )
    person = next(iter(changed["metrics"]["per_subject_errors"].values()))
    person["false_rejects"] -= 1
    changed["metrics"]["fixed_threshold"]["false_rejects"] -= 1
    selected = study.summarize(rows)
    assert selected["winner"] == "canonical336"
    expected = (
        sum(
            (
                Fraction(
                    r["metrics"]["fixed_threshold"]["false_rejects"],
                    r["metrics"]["fixed_threshold"]["n_genuine"],
                )
                for r in rows
                if r["arm"] == "canonical336" and r["condition"] == "clean"
            ),
            Fraction(),
        )
        / 20
    )
    assert selected["summaries"]["canonical336"]["clean"]["frr_exact"] == str(expected)


def test_stresses_and_display_floats_cannot_promote():
    rows = rows_fixture()
    for row in rows:
        row["metrics"]["fixed_threshold"].update(far=-100.0, frr=-100.0)
        if row["arm"] != "control" and row["condition"] != "clean":
            row["metrics"]["fixed_threshold"]["false_rejects"] = 0
            for person in row["metrics"]["per_subject_errors"].values():
                person["false_rejects"] = 0
    selected = study.summarize(rows)
    assert selected["winner"] == "control" and not selected["stress_used_for_selection"]


@pytest.mark.parametrize("bad_folds,expected", [(12, "canonical336"), (13, "control")])
def test_exact_one_percent_far_boundary(bad_folds, expected):
    rows = rows_fixture((3, 2, 3))
    for row in rows:
        if (
            row["arm"] == "canonical336"
            and row["condition"] == "clean"
            and row["fold"] < bad_folds
        ):
            _false_accepts(row, 1)
    assert study.summarize(rows)["winner"] == expected


def test_far_tie_break_and_bad_fold_gate():
    rows = rows_fixture((3, 2, 2))
    candidate = next(
        r for r in rows if r["arm"] == "canonical336" and r["condition"] == "clean"
    )
    _false_accepts(candidate, 1)
    assert study.summarize(rows)["winner"] == "canonical362"
    rows = rows_fixture((3, 1, 3))
    candidate = next(
        r for r in rows if r["arm"] == "canonical336" and r["condition"] == "clean"
    )
    _false_accepts(candidate, 2)  # Mean <1%, but one additional fold exceeds2%.
    assert study.summarize(rows)["winner"] == "control"


@pytest.mark.parametrize("no_worse,expected", [(9, "control"), (10, "canonical336")])
def test_ten_of_fifteen_people_gate(no_worse, expected):
    rows = rows_fixture()
    people = set(map(str, study.prior.DEV_IDS[:no_worse]))
    for row in rows:
        if row["arm"] == "canonical336" and row["condition"] == "clean":
            for subject, value in row["metrics"]["per_subject_errors"].items():
                value["false_rejects"] = 2 if subject in people else 4
            row["metrics"]["fixed_threshold"]["false_rejects"] = sum(
                p["false_rejects"]
                for p in row["metrics"]["per_subject_errors"].values()
            )
    assert study.summarize(rows)["winner"] == expected


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "graph", "person", "count", "pair", "fold_type"]
)
def test_complete_exact_counts_required(fault):
    rows = rows_fixture()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif fault == "graph":
        rows[-1]["metrics"]["claim_graph_sha256"] = "different"
    elif fault == "person":
        rows[-1]["metrics"]["per_subject_errors"].popitem()
    elif fault == "pair":
        rows[-1]["pair_errors"].popitem()
    elif fault == "fold_type":
        rows[0]["fold"] = False
    else:
        rows[-1]["metrics"]["fixed_threshold"]["false_accepts"] = True
    with pytest.raises(ValueError):
        study.summarize(rows)


def test_training_noise_scale_excludes_validation():
    data = synthetic_data()
    fold = study.crossfit.folds()[0]
    scale = study.training_scale(data, fold["refit_ids"])
    changed = {**data, "X": data["X"].copy()}
    changed["X"][np.isin(data["subjects"], fold["validation_ids"])] *= 10000
    np.testing.assert_array_equal(
        study.training_scale(changed, fold["refit_ids"]), scale
    )
    values = data["X"][np.isin(data["subjects"], fold["refit_ids"])].astype(np.float64)
    values -= values.mean(axis=2, keepdims=True)
    expected = np.repeat(
        [np.sqrt(np.mean(values[:, :3] ** 2)), np.sqrt(np.mean(values[:, 3:] ** 2))], 3
    )
    np.testing.assert_array_equal(scale, expected)


@pytest.mark.parametrize("condition", study.CONDITIONS)
def test_real_canonical_and_stress_integration(condition):
    data = synthetic_data()
    scale = study.training_scale(data, study.crossfit.folds()[0]["refit_ids"])
    original = data["X"].copy()
    clean = study.representations(data)
    changed = study.perturbed(data, condition, scale)
    values = study.representations(changed)
    np.testing.assert_array_equal(data["X"], original)
    np.testing.assert_array_equal(
        study.perturbed(data, condition, scale)["X"], changed["X"]
    )
    for key in ("subjects", "days", "trials", "starts"):
        np.testing.assert_array_equal(changed[key], data[key])
    np.testing.assert_array_equal(
        changed["X"][data["days"] == 1], original[data["days"] == 1]
    )
    for arm in study.ARMS:
        assert values[arm].shape == (120, study.DIMENSIONS[arm])
        assert np.isfinite(values[arm]).all()
        np.testing.assert_array_equal(
            values[arm][data["days"] == 1], clean[arm][data["days"] == 1]
        )
    np.testing.assert_array_equal(
        values["control"], study.prior.features(changed)["lagcov"]
    )
    np.testing.assert_array_equal(
        values["canonical336"],
        study.canonical.extract_canonical_features(
            changed["X"], 100
        ).require_all_valid(),
    )
    np.testing.assert_array_equal(
        values["canonical362"][:, :336], values["canonical336"]
    )
    np.testing.assert_array_equal(
        values["canonical362"][:, 336:],
        extract_lagged_covariance_features(changed["X"]),
    )
    if condition != "clean":
        assert not np.array_equal(
            changed["X"][data["days"] == 2], original[data["days"] == 2]
        )


def test_source_graph_cancellation_fails_closed():
    data = synthetic_data()
    ids = study.prior.DEV_IDS[:3]
    values = data["feature_fixture"].copy()
    selected = (
        (data["subjects"] == ids[0]) & (data["days"] == 2) & (data["trials"] == 1)
    )
    values[selected] = [[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]]

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


def test_replay_checks_decisions_inside_score_tolerance(tmp_path):
    cached = {
        "labels": np.array([1, 0]),
        "scores": np.array([0.1, 5e-8]),
        "metadata": {"synthetic": True},
    }
    record = study._store(tmp_path, "scores", cached)
    fresh = {**cached, "scores": np.array([0.1, -5e-8])}
    with pytest.raises(ValueError, match="decisions changed"):
        study._replay(tmp_path, fresh, record, 0.0)


@pytest.mark.parametrize("fault", ["source", "runtime", "plan"])
def test_verification_guard_before_dev_load(tmp_path, monkeypatch, fault):
    registration = study._registration()
    if fault == "source":
        registration["sources"]["models/gait_canonical.py"] = "changed"
    elif fault == "runtime":
        registration["python"] = "changed"
    else:
        registration["plan"]["geometry_guards"]["eigengap_ratio_strictly_above"] = 0.0
    study.prior.write_new(tmp_path / "registration.json", registration)
    monkeypatch.setattr(
        study, "load_development", lambda _: pytest.fail("DEV loaded before guard")
    )
    with pytest.raises(ValueError, match="registration/runtime"):
        study.verify(SimpleNamespace(out_root=tmp_path, archive_path="never-open"))


def test_bad_preflight_budget_before_dev_load(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "budget", lambda: {"fits": 301, "inner_scores": 240, "outer_scores": 420}
    )
    monkeypatch.setattr(
        study, "load_development", lambda _: pytest.fail("DEV loaded before budget")
    )
    with pytest.raises(ValueError, match="preflight fit/score budget"):
        study.run(SimpleNamespace(out_root=tmp_path, archive_path="never-open"))


@pytest.mark.parametrize("fault", ["clean", "late_stress"])
def test_any_degenerate_geometry_aborts_before_features_and_first_fit(
    tmp_path, monkeypatch, fault
):
    data = synthetic_data()
    if fault == "clean":
        data["X"][0] = 0
    else:
        perturb = study.perturbed

        def late_failure(d, condition, scale):
            changed = perturb(d, condition, scale)
            if condition == "noise05" and set(d["subjects"]) == set(
                study.crossfit.folds()[-1]["validation_ids"]
            ):
                changed["X"][np.flatnonzero(d["days"] == 2)[0]] = 0
            return changed

        monkeypatch.setattr(study, "perturbed", late_failure)
    monkeypatch.setattr(study, "load_development", lambda _: data)
    monkeypatch.setattr(
        study,
        "representations",
        lambda _: pytest.fail("features extracted before geometry abort"),
    )
    monkeypatch.setattr(
        study, "_fit", lambda *a: pytest.fail("fit before geometry abort")
    )
    with pytest.raises(ValueError, match="abort before features/fits"):
        study.run(SimpleNamespace(out_root=tmp_path, archive_path="synthetic"))
    manifest = study.prior.read(tmp_path / "development.json")
    geometry = manifest["geometry"]
    assert geometry["contexts"] == 141 and geometry["invalid_context_windows"] > 0
    record = (
        geometry["clean"] if fault == "clean" else geometry["folds"]["19"]["noise05"]
    )
    saved = study.prior.load_npz(
        study.prior._verify_artifact(tmp_path, record["artifact"])
    )
    assert (~saved["valid"]).sum() == 1
    assert saved["reason"][~saved["valid"]].tolist() == ["gravity_unidentifiable"]
    assert np.isnan(saved["frames"][~saved["valid"]]).all()
    assert np.isnan(saved["horizontal_energy_ratio"][~saved["valid"]]).all()
    assert record["rows"] == len(saved["row_keys"])
    assert (
        not list(tmp_path.glob("*_model.npz"))
        and not (tmp_path / "results.json").exists()
    )


def test_synthetic_run_counts_300_actual_fits_and_replays_all_artifacts(
    tmp_path, monkeypatch
):
    data = synthetic_data()
    root = tmp_path / "run"
    args = SimpleNamespace(out_root=root, archive_path="synthetic")
    loads, calls, features = [], [], []
    historical_arms = study.frozen.ARMS

    def load(archive):
        assert study.prior.read(root / "registration.json") == study._registration()
        loads.append(archive)
        return data

    def distinct_features(d):
        base = d["feature_fixture"].copy()
        base[:, :3] += 0.01 * d["X"][:, 3:].mean(axis=2)
        return {
            "control": base,
            "canonical336": base + 0.07 * base**2,
            "canonical362": base * np.array([1.0, 1.2, 0.8, 1.5, 0.9]) + 0.04 * base**3,
        }

    expected_clean = distinct_features(data)

    def tiny_representations(d):
        manifest = study.prior.read(root / "development.json")
        assert manifest["geometry"]["contexts"] == 141
        assert manifest["geometry"]["invalid_context_windows"] == 0
        features.append(len(d["X"]))
        return distinct_features(d)

    original_fit = study.WithinClassMetric.fit

    def counted_fit(self, values, subjects):
        assert values.shape[1] == 5 and len(values) <= 96
        assert self.mode == "wccn" and self.shrinkage == "auto"
        assert (root / "development.json").exists()
        arm = study.ARMS[(len(calls) // 5) % 3]
        np.testing.assert_array_equal(
            values, expected_clean[arm][np.isin(data["subjects"], np.unique(subjects))]
        )
        calls.append(tuple(sorted(set(map(int, subjects)))))
        return original_fit(self, values, subjects)

    monkeypatch.setattr(study, "load_development", load)
    monkeypatch.setattr(study, "representations", tiny_representations)
    monkeypatch.setattr(study.WithinClassMetric, "fit", counted_fit)
    study.run(args)
    expected = [
        tuple(sorted(ids))
        for f in study.crossfit.folds()
        for a in study.ARMS
        for ids in (*[i["train_ids"] for i in f["inner"]], f["refit_ids"])
    ]
    assert calls == expected and len(calls) == 300
    assert features == [120] + [24] * 120
    result_path = root / "results.json"
    frozen = study.prior.read(result_path)
    assert [len(frozen[k]) for k in ("models", "inner", "outer")] == [300, 240, 420]
    assert frozen["fit_calls"] == 300 and frozen["summary"]["winner"] in study.ARMS
    manifest = study.prior.read(root / "development.json")
    assert manifest["source_coverage"] == data["metadata"]
    for fold in study.crossfit.folds():
        key = str(fold["fold"])
        np.testing.assert_array_equal(
            frozen["scales"][key], study.training_scale(data, fold["refit_ids"])
        )
        selected = np.isin(data["subjects"], fold["validation_ids"])
        for condition in study.CONDITIONS:
            record = manifest["geometry"]["folds"][key][condition]
            saved = study.prior.load_npz(
                study.prior._verify_artifact(root, record["artifact"])
            )
            np.testing.assert_array_equal(
                saved["row_keys"], study._row_keys(data)[selected]
            )
            assert saved["valid"].all() and set(saved["reason"]) == {"valid"}
    monkeypatch.setattr(
        study.WithinClassMetric,
        "fit",
        lambda *a: pytest.fail("verification attempted fitting"),
    )
    report = study.verify(args)
    assert report["verified"] and report["geometry_contexts"] == 141
    assert loads == ["synthetic", "synthetic"] and len(calls) == 300
    assert study.frozen.ARMS == historical_arms
    original_result = result_path.read_bytes()

    for fault, pattern in (
        ("assignment", "model assignment"),
        ("stats", "assigned TRAIN statistics"),
        ("scale", "TRAIN dynamic RMS"),
        ("hash", "artifact"),
        ("threshold_hash", "artifact"),
        ("threshold_order", "threshold assignments"),
        ("inventory", "inventory"),
        ("duplicate", "inventory"),
        ("role_type", "artifact assignments"),
        ("artifact_swap", "artifact assignment"),
        ("metadata", "score metadata"),
        ("counts", "cached counts/metrics"),
        ("selection", "exact selection"),
        ("fits", "fit budget"),
    ):
        bad = deepcopy(frozen)
        if fault == "assignment":
            bad["models"][0]["train_ids"] = bad["models"][0]["train_ids"][1:]
        elif fault == "stats":
            bad["models"][0]["stats"]["train_windows"] += 1
        elif fault == "scale":
            bad["scales"]["0"][0] += 1
        elif fault == "hash":
            bad["models"][0]["artifact"]["sha256"] = "bad"
        elif fault == "threshold_hash":
            bad["threshold_files"][0]["sha256"] = "bad"
        elif fault == "threshold_order":
            bad["threshold_files"].reverse()
        elif fault == "inventory":
            bad["outer"][0]["condition"] = "unregistered"
        elif fault == "duplicate":
            bad["inner"][-1] = deepcopy(bad["inner"][0])
        elif fault == "role_type":
            bad["models"][0]["role"] = False
        elif fault == "artifact_swap":
            bad["models"][0]["artifact"], bad["models"][1]["artifact"] = (
                bad["models"][1]["artifact"],
                bad["models"][0]["artifact"],
            )
        elif fault == "metadata":
            bad["outer"][0]["metadata"]["probe_trial_ids"][0] = 999
        elif fault == "counts":
            bad["rows"][0]["metrics"]["fixed_threshold"]["false_accepts"] += 1
        elif fault == "selection":
            bad["summary"]["winner"] = (
                "control"
                if frozen["summary"]["winner"] == "canonical336"
                else "canonical336"
            )
        else:
            bad["fit_calls"] = 299
        result_path.write_bytes(study.prior.canonical(bad))
        with pytest.raises(ValueError, match=pattern):
            study.verify(args)
    result_path.write_bytes(original_result)

    # Rehash tampered numeric files too, so replay checks more than a digest.
    for fault, pattern in (
        ("geometry", "geometry manifest"),
        ("mean", "centering/scaling"),
        ("threshold", "frozen pooled threshold"),
    ):
        bad = deepcopy(frozen)
        changed_manifest = deepcopy(manifest)
        if fault == "geometry":
            artifact = changed_manifest["geometry"]["folds"]["19"]["noise05"][
                "artifact"
            ]
        elif fault == "mean":
            artifact = bad["models"][0]["artifact"]
        else:
            artifact = bad["threshold_files"][0]
        path = root / artifact["path"]
        original_bytes = path.read_bytes()
        original_manifest = (root / "development.json").read_bytes()
        if fault == "threshold":
            threshold = study.prior.read(path)
            threshold["control"]["threshold"] = (
                10.0  # All reject; cache decisions agree.
            )
            path.write_bytes(study.prior.canonical(threshold))
        else:
            arrays = study.prior.load_npz(path)
            arrays["eigengap_ratio" if fault == "geometry" else "mean"][0] += 0.1
            path.unlink()
            study.prior.save_npz(path, arrays)
        artifact["sha256"] = study.prior.digest(path)
        if fault == "geometry":
            (root / "development.json").write_bytes(
                study.prior.canonical(changed_manifest)
            )
            bad["development_sha256"] = study.prior.digest(root / "development.json")
        result_path.write_bytes(study.prior.canonical(bad))
        with pytest.raises(ValueError, match=pattern):
            study.verify(args)
        path.write_bytes(original_bytes)
        (root / "development.json").write_bytes(original_manifest)
        result_path.write_bytes(original_result)
    with pytest.raises(ValueError, match="fresh empty"):
        study.run(args)
