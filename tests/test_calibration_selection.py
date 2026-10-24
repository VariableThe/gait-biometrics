"""Synthetic count-only selection tests; no study artifacts or data are read."""

import json
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from engine.calibration_selection import summarize_calibration

DEV_IDS = tuple(range(1, 16))
SCORERS = ("mean_multi", "multi")
POLICIES = ("A", "B", "C", "D")


def selection_rows(*, rejects=None, accepts=None, genuine=None, claims=None):
    """Build actual 30-genuine/60-impostor fold counts unless overridden.

    Each of fifteen people validates in four synthetic triples. Every error is
    allocated to a person or a directed pair before fixed counts are totaled.
    Denominator callbacks never depend on scorer/policy, preserving the graph.
    """
    rows = []
    for fold in range(20):
        subjects = DEV_IDS[3 * (fold % 5) : 3 * (fold % 5 + 1)]
        for scorer in SCORERS:
            for policy in POLICIES:
                people = {}
                for subject in subjects:
                    n = genuine(fold, subject) if genuine else 10
                    fr = (
                        rejects(fold, scorer, policy, subject)
                        if rejects
                        else {"A": 2, "B": 1, "C": 2, "D": 3}[policy]
                    )
                    people[str(subject)] = {
                        "false_rejects": fr,
                        "n_genuine": n,
                        "false_accepts": 0,
                        "n_impostor": 0,
                        "frr": fr / n,
                    }
                pairs = {}
                for claim in subjects:
                    for probe in subjects:
                        if claim == probe:
                            continue
                        n = claims(fold, claim, probe) if claims else 10
                        fa = (
                            accepts(fold, scorer, policy, claim, probe)
                            if accepts
                            else 0
                        )
                        pairs[f"{claim}<-{probe}"] = {"claims": n, "false_accepts": fa}
                        people[str(claim)]["false_accepts"] += fa
                        people[str(claim)]["n_impostor"] += n
                fixed = {
                    name: sum(person[name] for person in people.values())
                    for name in (
                        "false_rejects",
                        "n_genuine",
                        "false_accepts",
                        "n_impostor",
                    )
                }
                fixed["far"] = fixed["false_accepts"] / fixed["n_impostor"]
                fixed["frr"] = fixed["false_rejects"] / fixed["n_genuine"]
                for person in people.values():
                    person["far"] = person["false_accepts"] / person["n_impostor"]
                rows.append(
                    {
                        "fold": fold,
                        "scorer": scorer,
                        "policy": policy,
                        "metrics": {
                            "claim_graph_sha256": f"synthetic-graph-{fold}",
                            "fixed_threshold": fixed,
                            "per_subject_errors": people,
                        },
                        "pair_errors": pairs,
                    }
                )
    return rows


def summarize(rows, **kwargs):
    return summarize_calibration(rows, dev_ids=DEV_IDS, **kwargs)


def test_promotes_from_counts_and_preserves_json_contract_without_mutation():
    rows = selection_rows()
    original = deepcopy(rows)
    result = summarize(rows)
    assert rows == original
    assert set(result) == {
        "winner",
        "promotion_gate_passed",
        "eligible_policies",
        "summary",
        "normalization_gate",
        "interpretation",
    }
    assert result["winner"] == "B"
    assert result["promotion_gate_passed"]
    assert result["eligible_policies"] == ["B"]
    baseline = result["summary"]["mean_multi/A"]
    assert baseline["frr_exact"] == "1/5"
    assert baseline["far_exact"] == "0/1"
    assert baseline["people_no_worse_than_A"] == 15
    assert baseline["pair_errors"]["1<-2"] == {"claims": 40, "false_accepts": 0}
    assert result["summary"]["mean_multi/B"][
        "per_person_mean_frr_exact"
    ] == dict.fromkeys(map(str, DEV_IDS), "1/10")
    assert "people_no_worse_than_A" not in result["summary"]["multi/B"]
    assert result["normalization_gate"] == {
        "false_accepts": 0,
        "top_two_pairs_false_accepts": 0,
        "triggered": False,
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_exact_tie_60_of_600_and_four_of_40_cannot_promote_float_noise():
    def rejects(fold, _scorer, policy, _subject):
        if policy == "A":
            return 1
        if policy == "B":
            return (1, 1, 2, 0)[fold // 5]
        return 2

    rows = selection_rows(rejects=rejects)
    for policy in ("A", "B"):
        group = [
            r for r in rows if r["scorer"] == "mean_multi" and r["policy"] == policy
        ]
        assert (
            sum(r["metrics"]["fixed_threshold"]["false_rejects"] for r in group) == 60
        )
        assert sum(r["metrics"]["fixed_threshold"]["n_genuine"] for r in group) == 600
        for subject in DEV_IDS:
            person = [
                r["metrics"]["per_subject_errors"][str(subject)]
                for r in group
                if str(subject) in r["metrics"]["per_subject_errors"]
            ]
            assert sum(p["false_rejects"] for p in person) == 4
            assert sum(p["n_genuine"] for p in person) == 40
        floating_mean = float(
            np.mean([r["metrics"]["fixed_threshold"]["frr"] for r in group])
        )
        assert floating_mean == (0.10000000000000002 if policy == "A" else 0.1)
    result = summarize(rows)
    assert result["winner"] == "A"
    assert not result["promotion_gate_passed"]
    assert result["eligible_policies"] == []
    assert result["summary"]["mean_multi/A"]["frr_exact"] == "1/10"
    assert result["summary"]["mean_multi/B"]["frr_exact"] == "1/10"
    assert result["summary"]["mean_multi/B"]["people_no_worse_than_A"] == 15


def test_stored_display_rates_are_ignored_including_missing_or_nonfinite_values():
    rows = selection_rows()
    expected = summarize(rows)
    for row in rows:
        fixed = row["metrics"]["fixed_threshold"]
        fixed["far"] = float("nan")
        del fixed["frr"]
        for person in row["metrics"]["per_subject_errors"].values():
            person["frr"] = "not a rate"
            person["far"] = float("inf")
    assert summarize(rows) == expected


def test_genuine_person_counts_suffice_when_optional_person_far_counts_are_absent():
    rows = selection_rows()
    expected = summarize(rows)
    for row in rows:
        for person in row["metrics"]["per_subject_errors"].values():
            del person["false_accepts"], person["n_impostor"]
    assert summarize(rows) == expected


@pytest.mark.parametrize("order", [POLICIES, ("D", "C", "B", "A")])
def test_exact_ties_among_eligible_use_supplied_policy_order_not_row_order(order):
    rows = selection_rows(rejects=lambda _f, _s, p, _who: 3 if p == "A" else 1)
    result = summarize(reversed(rows), policies=order, scorers=tuple(reversed(SCORERS)))
    assert result["eligible_policies"] == [p for p in order if p != "A"]
    assert result["winner"] == result["eligible_policies"][0]


def test_exact_frr_gain_smaller_than_float_resolution_still_promotes():
    n = 10**20
    rows = selection_rows(
        genuine=lambda _f, _who: n,
        rejects=lambda _f, _s, p, _who: n // 2 - (p == "B"),
    )
    result = summarize(rows)
    assert result["summary"]["mean_multi/A"]["frr"] == 0.5
    assert result["summary"]["mean_multi/B"]["frr"] == 0.5
    assert result["winner"] == "B"


@pytest.mark.parametrize("metric", ["frr", "far"])
def test_tie_break_compares_exact_rates_even_when_display_floats_are_equal(metric):
    n = 10**20

    def rejects(_fold, _scorer, policy, _subject):
        if policy in ("A", "D"):
            return n // 2
        return n // 4 + int(metric == "frr" and policy == "B")

    def accepts(_fold, _scorer, policy, claim, probe):
        if metric == "far" and policy in ("B", "C") and probe == claim + 1:
            return 3 * n // 200 + (policy == "B")
        return 0

    result = summarize(
        selection_rows(
            genuine=lambda _f, _who: n,
            claims=lambda _f, _c, _p: n,
            rejects=rejects,
            accepts=accepts,
        )
    )
    b, c = (result["summary"][f"mean_multi/{p}"] for p in ("B", "C"))
    assert b[metric] == c[metric]
    assert Fraction(b[f"{metric}_exact"]) > Fraction(c[f"{metric}_exact"])
    assert result["eligible_policies"] == ["B", "C"]
    assert result["winner"] == "C"


@pytest.mark.parametrize("extra_error, winner", [(0, "B"), (1, "A")])
def test_mean_far_boundary_is_inclusive_and_exact(extra_error, winner):
    n = 10**20
    rows = selection_rows(
        claims=lambda _f, _c, _p: n,
        accepts=lambda _f, _s, p, c, probe: (
            3 * n // 100 + extra_error if p == "B" and probe == c + 1 else 0
        ),
    )
    result = summarize(rows)
    b = result["summary"]["mean_multi/B"]
    assert b["far"] == 0.01
    assert (Fraction(b["far_exact"]) > Fraction(1, 100)) == bool(extra_error)
    assert b["splits_far_above_2pct"] == 0
    assert result["winner"] == winner


@pytest.mark.parametrize(
    "baseline_fa, winner, bad", [(0, "A", 0), (12, "A", 0), (18, "B", 1)]
)
def test_bad_split_gate_separately_blocks_low_mean_far_and_counts_strictly_above_2pct(
    baseline_fa, winner, bad
):
    rows = selection_rows(
        claims=lambda _f, _c, _p: 100,
        accepts=lambda f, _s, p, c, probe: (
            {"A": baseline_fa, "B": 18}.get(p, 0)
            if f == 0 and (c, probe) == (1, 2)
            else 0
        ),
    )
    result = summarize(rows)
    a, b = (result["summary"][f"mean_multi/{p}"] for p in ("A", "B"))
    assert a["splits_far_above_2pct"] == bad
    assert b["splits_far_above_2pct"] == 1
    assert Fraction(b["far_exact"]) <= Fraction(1, 100)
    assert Fraction(b["frr_exact"]) < Fraction(a["frr_exact"])
    assert b["people_no_worse_than_A"] == 15
    assert result["winner"] == winner


def test_bad_split_boundary_uses_exact_counts_below_float_resolution():
    n = 10**20
    rows = selection_rows(
        claims=lambda _f, _c, _p: n,
        accepts=lambda f, _s, p, c, probe: (
            3 * n // 25 + (p == "B")
            if f == 0 and p in ("A", "B") and (c, probe) == (1, 2)
            else 0
        ),
    )
    assert rows[0]["metrics"]["fixed_threshold"]["far"] == 0.02
    assert rows[1]["metrics"]["fixed_threshold"]["far"] == 0.02
    result = summarize(rows)
    assert result["summary"]["mean_multi/A"]["splits_far_above_2pct"] == 0
    assert result["summary"]["mean_multi/B"]["splits_far_above_2pct"] == 1
    assert result["winner"] == "A"


@pytest.mark.parametrize("worse, winner", [(5, "B"), (6, "A")])
def test_person_gate_requires_at_least_ten_despite_strict_overall_gain(worse, winner):
    rows = selection_rows(
        rejects=lambda _f, _s, p, who: (6 if who <= worse else 0) if p == "B" else 5
    )
    result = summarize(rows)
    a, b = (result["summary"][f"mean_multi/{p}"] for p in ("A", "B"))
    assert Fraction(b["frr_exact"]) < Fraction(a["frr_exact"])
    assert b["people_no_worse_than_A"] == 15 - worse
    assert result["winner"] == winner


def test_person_no_worse_has_no_epsilon_even_when_float_rates_match():
    n = 10**20
    result = summarize(
        selection_rows(
            genuine=lambda _f, _who: n,
            rejects=lambda _f, _s, p, who: (
                (n // 2 + 1 if who <= 6 else 0) if p == "B" else n // 2
            ),
        )
    )
    a, b = (result["summary"][f"mean_multi/{p}"] for p in ("A", "B"))
    assert a["per_person_mean_frr"]["1"] == b["per_person_mean_frr"]["1"]
    assert b["people_no_worse_than_A"] == 9
    assert result["winner"] == "A"


def test_fold_and_person_rates_are_equal_weight_means_instead_of_pooled_counts():
    rows = selection_rows(
        genuine=lambda f, _who: 10 if f < 10 else 20,
        claims=lambda f, _c, _p: 10 if f < 10 else 20,
        rejects=lambda f, _s, p, _who: (1 if f < 10 else 9) if p == "A" else 1,
        accepts=lambda _f, _s, p, c, probe: int(p == "A" and probe == c + 1),
    )
    baseline = summarize(rows)["summary"]["mean_multi/A"]
    assert baseline["frr_exact"] == "11/40"
    assert baseline["far_exact"] == "1/40"
    assert set(baseline["per_person_mean_frr_exact"].values()) == {"11/40"}


def test_multi_scorer_cannot_promote_an_ineligible_blend():
    rows = selection_rows(
        rejects=lambda _f, scorer, policy, _who: (
            0 if scorer == "multi" and policy == "B" else 2
        )
    )
    assert summarize(rows)["winner"] == "A"


@pytest.mark.parametrize("pair_count, triggered", [(4, True), (5, False)])
def test_normalization_diagnostic_uses_baseline_pair_totals_and_inclusive_half(
    pair_count, triggered
):
    selected = {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1)}
    if pair_count == 4:
        selected.remove((3, 1))
    result = summarize(
        selection_rows(
            accepts=lambda _f, _s, p, c, probe: int(p == "A" and (c, probe) in selected)
        )
    )
    assert result["normalization_gate"] == {
        "false_accepts": 4 * pair_count,
        "top_two_pairs_false_accepts": 8,
        "triggered": triggered,
    }


def test_normalization_diagnostic_compares_exactly_below_float_resolution():
    n = 10**20
    errors = {(1, 2): n, (1, 3): n, (2, 1): n, (2, 3): n, (3, 1): 1}
    result = summarize(
        selection_rows(
            claims=lambda _f, _c, _p: n,
            accepts=lambda f, _s, p, c, probe: (
                errors.get((c, probe), 0) if f == 0 and p == "A" else 0
            ),
        )
    )
    assert result["normalization_gate"] == {
        "false_accepts": 4 * n + 1,
        "top_two_pairs_false_accepts": 2 * n,
        "triggered": False,
    }


@pytest.mark.parametrize(
    "corruption",
    ["missing", "extra", "duplicate", "fold", "bool_fold", "scorer", "policy"],
)
def test_requires_complete_unique_20_by_two_by_four(corruption):
    rows = selection_rows()
    if corruption == "missing":
        rows.pop()
    elif corruption == "extra":
        rows.append(deepcopy(rows[0]))
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif corruption == "fold":
        rows[0]["fold"] = 20
    elif corruption == "bool_fold":
        rows[0]["fold"] = False
    else:
        rows[0][corruption] = "unknown"
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize(
    "corruption", ["hash", "missing_hash", "empty_hash", "people", "pair_claims"]
)
def test_common_graph_includes_subject_and_pair_denominators(corruption):
    rows = selection_rows()
    metrics = rows[0]["metrics"]
    if corruption == "hash":
        metrics["claim_graph_sha256"] = "different"
    elif corruption == "missing_hash":
        del metrics["claim_graph_sha256"]
    elif corruption == "empty_hash":
        metrics["claim_graph_sha256"] = ""
    elif corruption == "people":
        people = metrics["per_subject_errors"]
        people["1"]["n_genuine"] += 1
        people["2"]["n_genuine"] -= 1
    else:
        rows[0]["pair_errors"]["1<-2"]["claims"] += 1
        rows[0]["pair_errors"]["1<-3"]["claims"] -= 1
    with pytest.raises(ValueError, match="graph|claim_graph_sha256"):
        summarize(rows)


def test_each_expected_person_must_validate_four_times_even_with_common_graphs():
    rows = selection_rows()
    for i, row in enumerate(rows):
        if row["fold"] == 0:
            replacement = deepcopy(rows[8 + i])
            replacement["fold"] = 0
            rows[i] = replacement
    with pytest.raises(ValueError, match="four times"):
        summarize(rows)


@pytest.mark.parametrize("corruption", ["missing", "unknown", "duplicate_alias"])
def test_requires_expected_unique_person_ids(corruption):
    rows = selection_rows()
    people = rows[0]["metrics"]["per_subject_errors"]
    if corruption == "missing":
        people.pop("1")
    elif corruption == "unknown":
        people["99"] = people.pop("1")
    else:
        people[1] = deepcopy(people["1"])
    with pytest.raises(ValueError):
        summarize(rows)


COUNT_PATHS = (
    [
        ("metrics", "fixed_threshold", field)
        for field in ("false_rejects", "n_genuine", "false_accepts", "n_impostor")
    ]
    + [
        ("metrics", "per_subject_errors", "1", field)
        for field in ("false_rejects", "n_genuine", "false_accepts", "n_impostor")
    ]
    + [("pair_errors", "1<-2", field) for field in ("false_accepts", "claims")]
)


def parent_for(row, path):
    for name in path[:-1]:
        row = row[name]
    return row


@pytest.mark.parametrize("path", COUNT_PATHS)
@pytest.mark.parametrize(
    "invalid",
    [True, False, -1, 1.0, "1", None, Fraction(1, 1), float("nan"), float("inf")],
)
def test_all_counts_require_actual_nonnegative_integers_excluding_bool(path, invalid):
    rows = selection_rows()
    parent_for(rows[0], path)[path[-1]] = invalid
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize("path", COUNT_PATHS)
def test_missing_raw_counts_are_rejected(path):
    rows = selection_rows()
    del parent_for(rows[0], path)[path[-1]]
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize(
    "path", [p for p in COUNT_PATHS if p[-1] in ("n_genuine", "n_impostor", "claims")]
)
def test_all_denominators_must_be_positive(path):
    rows = selection_rows()
    parent_for(rows[0], path)[path[-1]] = 0
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize(
    "path", [p for p in COUNT_PATHS if p[-1] in ("false_rejects", "false_accepts")]
)
def test_error_counts_cannot_exceed_denominators(path):
    rows = selection_rows()
    parent_for(rows[0], path)[path[-1]] = 1_000
    with pytest.raises(ValueError, match="must not exceed"):
        summarize(rows)


@pytest.mark.parametrize(
    "field", ["false_rejects", "n_genuine", "false_accepts", "n_impostor"]
)
def test_person_and_pair_counts_must_sum_to_fixed_counts(field):
    rows = selection_rows()
    rows[0]["metrics"]["fixed_threshold"][field] += 1
    with pytest.raises(ValueError, match="sum to fixed_threshold"):
        summarize(rows)


def test_optional_person_impostor_counts_must_match_pair_counts():
    rows = selection_rows()
    rows[0]["metrics"]["per_subject_errors"]["1"]["false_accepts"] = 1
    with pytest.raises(ValueError, match="impostor counts must match"):
        summarize(rows)


@pytest.mark.parametrize("pair", ["1->2", "1<-1", "1<-99", "01<-2"])
def test_directed_pairs_must_name_distinct_expected_validation_people(pair):
    rows = selection_rows()
    pairs = rows[0]["pair_errors"]
    pairs[pair] = pairs.pop("1<-2")
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize(
    "ids", [DEV_IDS[:-1], (*DEV_IDS[:-1], 1), (*DEV_IDS[:-1], True)]
)
def test_dev_ids_require_fifteen_unique_integer_ids(ids):
    with pytest.raises(ValueError):
        summarize_calibration(selection_rows(), dev_ids=ids)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scorers": ("multi",)},
        {"scorers": ("multi", "other")},
        {"scorers": ("mean_multi", "mean_multi")},
        {"policies": ("A", "B", "C")},
        {"policies": ("B", "C", "D", "E")},
        {"policies": ("A", "B", "B", "D")},
    ],
)
def test_configured_comparisons_preserve_required_shape_and_baseline(kwargs):
    with pytest.raises(ValueError):
        summarize(selection_rows(), **kwargs)
