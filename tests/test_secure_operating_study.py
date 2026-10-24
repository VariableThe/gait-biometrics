"""Protocol tests: independent calibration, constrained selection, artifact replay."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scripts import secure_operating_study as study


def test_roles_are_balanced_and_disjoint():
    counts = {
        role: Counter()
        for role in ("train_subject_ids", "calibration_subject_ids", "val_subject_ids")
    }
    for fold in study.folds():
        roles = [set(fold[name]) for name in counts]
        assert [len(role) for role in roles] == [9, 3, 3]
        assert (
            not roles[0] & roles[1]
            and not roles[0] & roles[2]
            and not roles[1] & roles[2]
        )
        assert set.union(*roles) == set(study.prior.DEV_IDS)
        for name, counter in counts.items():
            counter.update(fold[name])
    assert set(counts["train_subject_ids"].values()) == {12}
    assert set(counts["calibration_subject_ids"].values()) == {4}
    assert set(counts["val_subject_ids"].values()) == {4}


def rows_for_selection():
    return [
        {
            "fold": fold,
            "candidate": candidate,
            "inner_calibration": {"metrics": {"claim_graph_sha256": "cal"}},
            "validation": {
                "metrics": {
                    "claim_graph_sha256": "val",
                    "eer": 0.05,
                    "fixed_threshold": {"far": 0.02, "frr": 0.1},
                    "per_subject_errors": {"1": {"frr": 0.1}},
                }
            },
        }
        for fold in range(20)
        for candidate in study.CANDIDATES
    ]


def test_selection_prioritizes_feasible_false_accept_constraint():
    rows = rows_for_selection()
    chosen = study.CANDIDATES[2]
    for row in rows:
        if row["candidate"] == chosen:
            row["validation"]["metrics"]["fixed_threshold"] = {"far": 0.005, "frr": 0.3}
    result = study.selection(rows)
    assert result["winner"] == chosen
    assert result["constraint_met"]
    assert result["eligible_candidates"] == [chosen]


def test_no_feasible_candidate_is_explicit_and_ties_are_stable():
    result = study.selection(rows_for_selection())
    assert not result["constraint_met"]
    assert result["winner"] == study.CANDIDATES[0]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "graph", "nonfinite"])
def test_selection_rejects_incomplete_or_inconsistent_evidence(corruption):
    rows = rows_for_selection()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    elif corruption == "graph":
        rows[0]["validation"]["metrics"]["claim_graph_sha256"] = "other"
    else:
        rows[0]["validation"]["metrics"]["fixed_threshold"]["far"] = float("nan")
    with pytest.raises(ValueError):
        study.selection(rows)


def test_fitting_rejects_original_calibration_or_test_people():
    for ids in (study.prior.CALIBRATION_IDS, study.prior.TEST_IDS, [7, 7]):
        with pytest.raises(ValueError):
            study._fit(None, None, ids, "auto")


def synthetic_population():
    rng = np.random.default_rng(71)
    subjects, days, trials, signals = [], [], [], []
    ids = (*study.prior.DEV_IDS, *study.prior.CALIBRATION_IDS, *study.prior.TEST_IDS)
    for subject in ids:
        center = rng.normal(size=6)
        for day in (1, 2):
            for trial in (1, 2, 3):
                subjects.append(subject)
                days.append(day)
                trials.append(trial)
                signals.append(center + rng.normal(scale=0.15, size=6))
    return {
        "subjects": np.array(subjects),
        "days": np.array(days),
        "trials": np.array(trials),
        "X": np.array(signals),
    }


@pytest.mark.parametrize("fail_test", [False, True])
def test_end_to_end_reserved_roles_and_consumed_attempt(tmp_path, fail_test):
    population = synthetic_population()
    reads = []

    def load(_archive, ids):
        reads.append(tuple(ids))
        if tuple(ids) == tuple(study.prior.TEST_IDS):
            assert (args.out_root / "test_attempt.json").exists()
            if fail_test:
                raise ValueError("injected TEST loader failure")
        mask = np.isin(population["subjects"], ids)
        return {key: value[mask] for key, value in population.items()}

    def manifest(data):
        return {"data_sha256": study.prior.object_hash(data)}

    args = SimpleNamespace(out_root=tmp_path / "study", archive_path="synthetic.zip")
    with (
        patch.object(
            study.prior._previous(), "audit_archive", return_value={"fixture": True}
        ),
        patch.object(study.prior, "load_data", side_effect=load),
        patch.object(study.prior, "_manifest", side_effect=manifest),
        patch.object(study, "_features", side_effect=lambda data: data["X"]),
        patch.object(study, "hashes", return_value={"fixture": "frozen"}),
    ):
        study.develop(args)
        assert set(reads) == {tuple(study.prior.DEV_IDS)}
        reg = study.prior.read(args.out_root / "registration.json")
        evidence = study.prior.read(args.out_root / "development.json")
        first = evidence["rows"][0]
        first["threshold"]["threshold"] += 1
        (args.out_root / "development.json").write_bytes(
            study.prior.canonical(evidence)
        )
        with pytest.raises(ValueError):
            study.verify(args)
        assert tuple(study.prior.CALIBRATION_IDS) not in reads
        first["threshold"]["threshold"] -= 1
        # Exact floating point restoration uses the original frozen CALIB scores.
        saved = study.prior.load_npz(
            study.prior._verify_artifact(
                args.out_root, first["inner_calibration"]["scores"]
            )
        )
        first["threshold"] = study.prior.calibrate(saved["labels"], saved["scores"])
        (args.out_root / "development.json").write_bytes(
            study.prior.canonical(evidence)
        )
        study.lock(args)
        assert tuple(study.prior.CALIBRATION_IDS) in reads
        assert tuple(study.prior.TEST_IDS) not in reads
        if fail_test:
            with pytest.raises(ValueError, match="injected TEST loader failure"):
                study.evaluate(args)
            assert not (args.out_root / "test_result.json").exists()
        else:
            result = study.evaluate(args)
            assert result["winner"] in study.CANDIDATES
        assert reads.count(tuple(study.prior.TEST_IDS)) == 1
        assert study.prior.read(args.out_root / "registration.json") == reg
        with pytest.raises(ValueError, match="already attempted"):
            study.evaluate(args)
        assert reads.count(tuple(study.prior.TEST_IDS)) == 1
        with pytest.raises(ValueError, match="lock already attempted"):
            study.lock(args)
