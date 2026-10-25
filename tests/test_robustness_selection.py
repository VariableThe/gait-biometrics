"""Count-only safety gates and independently checked tied-score ranks."""

import hashlib
from copy import deepcopy
from fractions import Fraction
from typing import TypedDict

import numpy as np
import pytest

from engine import robustness_selection as selection


class Fold(TypedDict):
    fold: int
    validation_ids: list[int]


def fixture():
    ids = tuple(range(15))
    folds: list[Fold] = []
    for repeat in range(4):
        ordered = ids[repeat:] + ids[:repeat]
        for offset in range(0, 15, 3):
            folds.append(
                {
                    "fold": len(folds),
                    "validation_ids": list(ordered[offset : offset + 3]),
                }
            )
    rows = []
    for fold in folds:
        people = list(map(str, fold["validation_ids"]))
        for arm in selection.ARMS:
            for condition in selection.CONDITIONS:
                rejected = (
                    2
                    if condition in ("clean", "rotation30", "rotation90")
                    else (
                        20
                        if condition == "noise05"
                        else 12
                        if arm == "baseline"
                        else 10
                    )
                )
                graph = hashlib.sha256(str(fold["fold"]).encode()).hexdigest()
                rows.append(
                    {
                        "fold": fold["fold"],
                        "arm": arm,
                        "condition": condition,
                        "decision_sha256": graph,
                        "metrics": {
                            "claim_graph_sha256": graph,
                            "fixed_threshold": {
                                "false_accepts": 0,
                                "n_impostor": 120,
                                "false_rejects": 3 * rejected,
                                "n_genuine": 60,
                            },
                            "per_subject_errors": {
                                s: {
                                    "false_rejects": rejected,
                                    "n_genuine": 20,
                                    "false_accepts": 0,
                                    "n_impostor": 40,
                                }
                                for s in people
                            },
                        },
                        "pair_errors": {
                            f"{c}<-{p}": {"false_accepts": 0, "claims": 20}
                            for c in people
                            for p in people
                            if c != p
                        },
                        "ranking": {
                            "eer_exact": "1/4",
                            "auc_exact": "3/4",
                            "per_subject": {
                                s: {"eer_exact": "1/4", "auc_exact": "3/4"}
                                for s in people
                            },
                        },
                    }
                )
    return ids, folds, rows


def summarize(rows, ids=None, folds=None):
    default_ids, default_folds, _ = fixture()
    return selection.summarize_nuisance_study(
        rows, dev_ids=ids or default_ids, folds=folds or default_folds
    )


def set_person_rejects(row, person, count):
    entry = row["metrics"]["per_subject_errors"][str(person)]
    row["metrics"]["fixed_threshold"]["false_rejects"] += count - entry["false_rejects"]
    entry["false_rejects"] = count


@pytest.mark.parametrize(
    ("labels", "scores", "eer", "auc"),
    [
        ([1, 0], [1.0, 0.0], "0", "1"),
        ([1, 0], [0.0, 1.0], "1", "0"),
        ([1, 0], [1.0, 1.0], "1/2", "1/2"),
        ([1, 0, 0, 1, 0], [3.0, 2.0, 2.0, 2.0, 1.0], "2/7", "5/6"),
    ],
)
def test_rational_roc_references(labels, scores, eer, auc):
    result = selection.ranking_metrics(labels, scores, [7] * len(labels))
    assert result == {
        "eer_exact": eer,
        "auc_exact": auc,
        "per_subject": {"7": {"eer_exact": eer, "auc_exact": auc}},
    }
    order = np.arange(len(labels))[::-1]
    assert (
        selection.ranking_metrics(
            np.asarray(labels)[order], np.asarray(scores)[order], [7] * len(labels)
        )
        == result
    )


def test_auc_matches_pairwise_tie_credit_without_roc_interpolation():
    rng = np.random.default_rng(289)
    for _ in range(40):
        labels = np.r_[np.ones(7, int), np.zeros(11, int)]
        scores = rng.integers(-2, 4, len(labels)).astype(float)
        credit = (
            sum(
                Fraction(1) if x > y else Fraction(1, 2) if x == y else Fraction()
                for x in scores[:7]
                for y in scores[7:]
            )
            / 77
        )
        assert (
            Fraction(
                selection.ranking_metrics(labels, scores, [1] * len(labels))[
                    "auc_exact"
                ]
            )
            == credit
        )


@pytest.mark.parametrize(
    ("labels", "scores", "probes"),
    [
        ([0, 0], [0.0, 1.0], [1, 1]),
        ([0, 1], [0.0, np.nan], [1, 1]),
        ([0, 1], [0.0, np.inf], [1, 1]),
        ([0, 1], [0j, 1j], [1, 1]),
        ([0, 1], [0.0, 1.0], [-1, -1]),
        ([0, 1], [0.0, 1.0], [1.0, 1.0]),
        ([0, 1], [0.0, 1.0], [1, 2]),
        ([0.5, 1], [0.0, 1.0], [1, 1]),
        ([0, 1], [0.0, 1.0], [1]),
    ],
)
def test_ranking_rejects_invalid_inputs(labels, scores, probes):
    with pytest.raises(ValueError):
        selection.ranking_metrics(labels, scores, probes)


def test_all_gates_pass_for_fixed_safe_10pp_stress_improvement():
    ids, folds, rows = fixture()
    result = selection.summarize_nuisance_study(rows, dev_ids=ids, folds=folds)
    assert result["winner"] == "nuisance"
    assert result["failed_gates"] == []
    assert result["robust_improvement_exact"] == "1/10"
    assert result["summaries"]["nuisance"]["clean"]["frr_exact"] == "1/10"
    assert result["stress_used_for_selection"] is True


def test_no_robust_gain_retains_baseline_despite_floating_display_changes():
    _, _, rows = fixture()
    for row in rows:
        if row["arm"] == "nuisance" and row["condition"] in selection.ROBUST_CONDITIONS:
            for s in row["metrics"]["per_subject_errors"]:
                set_person_rejects(row, s, 12)
        row["metrics"]["eer"] = -1000
        row["metrics"]["fixed_threshold"]["frr"] = 0.10000000000000002
    result = summarize(rows)
    assert result["winner"] == "baseline"
    assert "robust_mean_frr_improves_at_least_5pp" in result["failed_gates"]


def test_exact_5pp_boundary_qualifies_and_one_more_reject_does_not():
    _, _, rows = fixture()
    for row in rows:
        if row["arm"] == "nuisance" and row["condition"] in selection.ROBUST_CONDITIONS:
            for s in row["metrics"]["per_subject_errors"]:
                set_person_rejects(row, s, 11)
    assert summarize(rows)["winner"] == "nuisance"
    target = next(
        r for r in rows if r["arm"] == "nuisance" and r["condition"] == "noise01"
    )
    set_person_rejects(target, next(iter(target["metrics"]["per_subject_errors"])), 12)
    assert summarize(rows)["winner"] == "baseline"


def test_eer_gate_uses_exact_interpolation_evidence_not_stored_float():
    _, _, rows = fixture()
    for row in rows:
        if row["arm"] == "nuisance" and row["condition"] == "clean":
            row["ranking"]["eer_exact"] = "25000000000000001/100000000000000000"
            row["metrics"]["eer"] = 0.25
    assert "clean_eer_no_worse" in summarize(rows)["failed_gates"]


def test_person_safety_and_rotation_decision_gates():
    _, _, original = fixture()
    rows = deepcopy(original)
    target = next(
        r for r in rows if r["arm"] == "nuisance" and r["condition"] == "clean"
    )
    set_person_rejects(target, next(iter(target["metrics"]["per_subject_errors"])), 4)
    assert (
        "clean_no_person_more_than_one_extra_reject" in summarize(rows)["failed_gates"]
    )
    rows = deepcopy(original)
    target = next(
        r for r in rows if r["arm"] == "nuisance" and r["condition"] == "rotation90"
    )
    target["decision_sha256"] = "f" * 64
    assert "rotations_retain_clean_decisions" in summarize(rows)["failed_gates"]


def test_claimant_gate_detects_one_false_accept_even_below_target_far():
    _, _, rows = fixture()
    target = next(
        r for r in rows if r["arm"] == "nuisance" and r["condition"] == "noise05"
    )
    pair = next(iter(target["pair_errors"]))
    claimant = pair.split("<-")[0]
    target["pair_errors"][pair]["false_accepts"] = 1
    target["metrics"]["fixed_threshold"]["false_accepts"] = 1
    target["metrics"]["per_subject_errors"][claimant]["false_accepts"] = 1
    result = summarize(rows)
    assert result["gates"]["noise05_far_at_most_1pct"]
    assert not result["gates"]["noise05_each_claimant_false_accepts_no_worse"]


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "graph", "people", "fraction", "negative", "decision"],
)
def test_rejects_incomplete_or_inconsistent_evidence(corruption):
    _, _, rows = fixture()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif corruption == "graph":
        rows[0]["metrics"]["claim_graph_sha256"] = "wrong"
    elif corruption == "people":
        rows[0]["ranking"]["per_subject"].pop(
            next(iter(rows[0]["ranking"]["per_subject"]))
        )
    elif corruption == "fraction":
        rows[0]["ranking"]["eer_exact"] = "2/8"
    elif corruption == "negative":
        rows[0]["metrics"]["fixed_threshold"]["false_rejects"] = -1
    else:
        rows[0]["decision_sha256"] = "short"
    with pytest.raises(ValueError):
        summarize(rows)


def test_equal_fold_mean_with_unequal_denominators_is_not_pooled_rate():
    _, _, rows = fixture()
    for row in rows:
        if row["fold"] == 0:
            person = next(iter(row["metrics"]["per_subject_errors"]))
            row["metrics"]["per_subject_errors"][person]["n_genuine"] += 7
            row["metrics"]["fixed_threshold"]["n_genuine"] += 7
    result = summarize(rows)
    summary = result["summaries"]["baseline"]["clean"]
    expected = (Fraction(6, 67) + 19 * Fraction(6, 60)) / 20
    assert summary["frr_exact"] == str(expected)
    assert expected != Fraction(
        summary["repeated_false_rejects"], summary["repeated_genuine_claims"]
    )


def test_global_stress_gain_cannot_hide_one_person_regression():
    _, _, rows = fixture()
    for row in rows:
        if (
            row["arm"] == "nuisance"
            and row["condition"] in selection.ROBUST_CONDITIONS
            and "0" in row["metrics"]["per_subject_errors"]
        ):
            set_person_rejects(row, "0", 14)
    result = summarize(rows)
    assert result["gates"]["robust_mean_frr_improves_at_least_5pp"]
    assert result["gates"]["robust_at_least_10_people_no_worse"]
    assert not result["gates"]["robust_no_person_worsens_more_than_5pp"]


def test_one_extra_reject_cannot_turn_a_person_completely_unusable():
    _, _, rows = fixture()
    first = True
    for row in rows:
        if (
            row["condition"] != "clean"
            or "0" not in row["metrics"]["per_subject_errors"]
        ):
            continue
        count = 19 if row["arm"] == "baseline" and first else 20
        set_person_rejects(row, "0", count)
        if row["arm"] == "baseline":
            first = False
    result = summarize(rows)
    assert result["gates"]["clean_no_person_more_than_one_extra_reject"]
    assert not result["gates"]["clean_no_new_completely_rejected_person"]


@pytest.mark.parametrize(
    ("count", "all_folds", "gate"),
    [
        (2, True, "noise01_far_at_most_1pct"),
        (3, False, "noise01_fold_tail_no_worse"),
    ],
)
def test_far_mean_and_fold_tail_are_separate_gates(count, all_folds, gate):
    _, _, rows = fixture()
    for row in rows:
        if row["arm"] != "nuisance" or row["condition"] != "noise01":
            continue
        if not all_folds and row["fold"] != 0:
            continue
        pair = next(iter(row["pair_errors"]))
        claimant = pair.split("<-")[0]
        row["pair_errors"][pair]["false_accepts"] = count
        row["metrics"]["fixed_threshold"]["false_accepts"] = count
        row["metrics"]["per_subject_errors"][claimant]["false_accepts"] = count
    result = summarize(rows)
    assert not result["gates"][gate]
    if not all_folds:
        assert result["gates"]["noise01_far_at_most_1pct"]
