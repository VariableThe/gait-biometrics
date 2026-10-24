"""Exact promotion gates, fitting budgets and DEV-only replay on tiny signals."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts import claimant_normalization_study as study


def rows(rejects=None, false_accepts=None):
    result = []
    for fold in study.crossfit.folds():
        ids = fold["validation_ids"]
        for arm in study.ARMS:
            people = {
                str(s): {"n_genuine": 10, "false_rejects": (rejects or {}).get(arm, 2)}
                for s in ids
            }
            pairs = {
                f"{claim}<-{probe}": {"claims": 10, "false_accepts": 0}
                for claim in ids
                for probe in ids
                if claim != probe
            }
            next(iter(pairs.values()))["false_accepts"] = (false_accepts or {}).get(
                arm, 0
            )
            result.append(
                {
                    "fold": fold["fold"],
                    "arm": arm,
                    "pair_errors": pairs,
                    "metrics": {
                        "claim_graph_sha256": f"synthetic-{fold['fold']}",
                        "per_subject_errors": people,
                        "fixed_threshold": {
                            "n_genuine": 30,
                            "n_impostor": 60,
                            "false_rejects": sum(
                                p["false_rejects"] for p in people.values()
                            ),
                            "false_accepts": sum(
                                p["false_accepts"] for p in pairs.values()
                            ),
                        },
                    },
                }
            )
    return result


def test_exact_ties_retain_raw_despite_display_roundoff():
    evidence = rows()
    for row in evidence:
        row["metrics"]["fixed_threshold"]["frr"] = (
            0.20000000000000004 if row["arm"] == "raw" else 0.2
        )
    original = deepcopy(evidence)
    assert study.summarize(evidence)["winner"] == "raw"
    assert evidence == original


def test_fixed_arm_order_breaks_real_improvement_tie():
    selection = study.summarize(rows(rejects={"raw": 2, "z": 1, "s": 1}))
    assert selection["winner"] == "z"
    assert selection["eligible"] == ["z", "s"]
    assert selection["summaries"]["z"]["people_no_worse_than_raw"] == 15


def test_lower_frr_does_not_override_far_gate():
    selection = study.summarize(
        rows(rejects={"raw": 2, "z": 1, "s": 2}, false_accepts={"z": 1})
    )
    assert selection["winner"] == "raw"
    assert selection["summaries"]["z"]["far_exact"] == "1/60"


def test_one_bad_split_blocks_promotion_despite_eligible_mean_far():
    evidence = rows(rejects={"raw": 2, "z": 1, "s": 2})
    row = next(r for r in evidence if r["fold"] == 0 and r["arm"] == "z")
    next(iter(row["pair_errors"].values()))["false_accepts"] = 2
    row["metrics"]["fixed_threshold"]["false_accepts"] = 2
    selection = study.summarize(evidence)
    assert selection["summaries"]["z"]["far"] < 0.01
    assert selection["summaries"]["z"]["splits_far_above_2pct"] == 1
    assert selection["winner"] == "raw"


def test_mean_improvement_cannot_override_per_person_gate():
    evidence = rows(rejects={"raw": 2, "z": 0, "s": 2})
    harmed = set(map(str, study.prior.DEV_IDS[:6]))
    for row in evidence:
        if row["arm"] != "z":
            continue
        people = row["metrics"]["per_subject_errors"]
        for person, counts in people.items():
            counts["false_rejects"] = 3 if person in harmed else 0
        row["metrics"]["fixed_threshold"]["false_rejects"] = sum(
            person["false_rejects"] for person in people.values()
        )
    selection = study.summarize(evidence)
    assert selection["summaries"]["z"]["frr"] < selection["summaries"]["raw"]["frr"]
    assert selection["summaries"]["z"]["people_no_worse_than_raw"] == 9
    assert selection["winner"] == "raw"


def test_incomplete_duplicate_and_changed_graph_evidence_fail():
    evidence = rows()
    with pytest.raises(ValueError, match="complete unique"):
        study.summarize(evidence[:-1])
    changed = deepcopy(evidence)
    changed[-1] = changed[0]
    with pytest.raises(ValueError, match="complete unique"):
        study.summarize(changed)
    evidence[1]["metrics"]["claim_graph_sha256"] = "different"
    with pytest.raises(ValueError, match="share each claim graph"):
        study.summarize(evidence)


def test_registered_sources_exist():
    assert len(study.hashes()) == len(study.SOURCES)
    assert set(study.plan()["payload_allowlist"]) == set(study.prior.DEV_IDS)


def test_empirical_c_coverage_does_not_inherit_nominal_d_requirement():
    subjects, days, trials = np.asarray(
        [
            (subject, day, trial)
            for subject in study.prior.DEV_IDS
            for day in (1, 2)
            for trial in range(1, 10)
        ]
    ).T
    data = {"subjects": subjects, "days": days, "trials": trials}
    with pytest.raises(ValueError, match="nominal rank"):
        study.crossfit.coverage_plan(data)
    coverage = study.coverage_plan(data)
    assert coverage["0"]["pool"]["n_impostor"] == 216
    assert coverage["0"]["pool"]["n_genuine"] == 108
    assert len(coverage["0"]["cohort_inner"]) == 4


def test_replay_tolerance_cannot_hide_threshold_decision_changes():
    fresh, cached = {"scores": np.array([-5e-8])}, {"scores": np.array([5e-8])}
    assert np.allclose(fresh["scores"], cached["scores"], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="changed decisions"):
        study._same_decisions(fresh, cached, 0)
    study._same_decisions(fresh, cached, 1)


def test_synthetic_run_replay_fit_budget_and_tamper_refusal(tmp_path):
    rng = np.random.default_rng(73)
    values, subjects, days, trials = [], [], [], []
    for subject in study.prior.DEV_IDS:
        center = rng.normal(size=4)
        for day in (1, 2):
            for trial in range(1, 11):
                values.append(center + rng.normal(scale=0.1, size=4))
                subjects.append(subject)
                days.append(day)
                trials.append(trial)
    data = {
        key: np.asarray(value)
        for key, value in zip(
            ("X", "subjects", "days", "trials"),
            (values, subjects, days, trials),
            strict=True,
        )
    }
    loads = []

    def load(_path, ids):
        assert tuple(ids) == study.prior.DEV_IDS
        loads.append(tuple(ids))
        return data

    args = SimpleNamespace(archive_path="synthetic.zip", out_root=tmp_path / "study")
    with (
        patch.object(
            study.prior._previous(), "audit_archive", return_value={"fixture": True}
        ),
        patch.object(study.prior, "load_data", side_effect=load),
        patch.object(
            study.prior,
            "_manifest",
            side_effect=lambda d: {"sha": study.prior.object_hash(d)},
        ),
        patch.object(study.secure, "_features", side_effect=lambda d: d["X"]),
        patch.object(study.secure, "_fit", wraps=study.secure._fit) as fit,
    ):
        study.run(args)
        assert fit.call_count == 100
        validation = study.verify(args)
        assert validation["verified"] and validation["outer_comparisons"] == 60
        assert fit.call_count == 100 and len(loads) == 2
        registration = study.prior.read(args.out_root / "registration.json")
        registration["python"] = "different"
        (args.out_root / "registration.json").write_bytes(
            study.prior.canonical(registration)
        )
        with pytest.raises(ValueError, match="registered runtime python"):
            study.verify(args)
        assert len(loads) == 2
        registration["python"] = study.runtime()["python"]
        (args.out_root / "registration.json").write_bytes(
            study.prior.canonical(registration) + b"\n"
        )
        evidence = study.prior.read(args.out_root / "results.json")
        evidence["rows"][0]["threshold"]["threshold"] += 0.1
        (args.out_root / "results.json").write_bytes(study.prior.canonical(evidence))
        with pytest.raises(ValueError, match="frozen-cache threshold"):
            study.verify(args)
        with pytest.raises(ValueError, match="fresh empty"):
            study.run(args)
