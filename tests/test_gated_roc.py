"""Synthetic, independently enumerated checks; no datasets or fitted models."""

import json
from copy import deepcopy
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from engine.gated_roc import (
    apply_gated_policy,
    calibrate_gated_threshold,
    gated_roc,
)
from engine.robustness_selection import ranking_metrics

REJECT = {"kind": "reject_all", "threshold": None}


def enumerated_states(labels, scores):
    """Direct acceptances, without sorting claims or cumulative counts."""
    thresholds = [None, *sorted({s for s in scores if s is not None}, reverse=True)]
    ng, ni = labels.count(1), labels.count(0)
    states = []
    for threshold in thresholds:
        accepts = [
            threshold is not None and score is not None and score >= threshold
            for score in scores
        ]
        tp = sum(a for a, label in zip(accepts, labels, strict=True) if label == 1)
        fp = sum(a for a, label in zip(accepts, labels, strict=True) if label == 0)
        states.append(
            {
                "policy": REJECT.copy()
                if threshold is None
                else {"kind": "score_at_least", "threshold": threshold},
                "accepts": accepts,
                "counts": {"tp": tp, "fp": fp, "fn": ng - tp, "tn": ni - fp},
                "far": Fraction(fp, ni),
                "frr": Fraction(ng - tp, ng),
            }
        )
    return states


def reference_eer(points):
    """Intersect the equal-error line with directly enumerated line segments."""
    for index, (far, frr) in enumerate(points):
        if far < frr:
            continue
        if far == frr:
            return "exact_threshold_state", far, [index, index], Fraction()
        x, y = points[index - 1]
        weight = (y - x) / (far - x + y - frr)
        crossing = (x * frr - far * y) / (x - y - far + frr)
        return "interpolated", crossing, [index - 1, index], weight
    return "undefined_no_crossing", None, None, None


def assert_eer(actual, points, policies):
    status, value, indexes, weight = reference_eer(points)
    assert actual["status"] == status
    assert actual["eer_exact"] == (None if value is None else str(value))
    assert actual["eer"] == (None if value is None else float(value))
    if indexes is None:
        assert actual["bracket"] is None
    else:
        assert actual["bracket"] == {
            "state_indexes": indexes,
            "policies": [policies[index] for index in indexes],
            "upper_weight": float(weight),
            "upper_weight_exact": str(weight),
        }


@pytest.mark.parametrize("length", [2, 3, 4])
def test_exhaustive_binary_labels_ties_failures_and_calibration(length):
    for binary in product((0, 1), repeat=length):
        if len(set(binary)) != 2:
            continue
        labels = list(binary)
        for choices in product((None, -1.0, 1.0), repeat=length):
            scores = list(choices)
            valid = [score is not None for score in scores]
            reference = enumerated_states(labels, scores)
            result = gated_roc(labels, scores, valid)
            ng, ni = labels.count(1), labels.count(0)
            vg = sum(v for v, label in zip(valid, labels, strict=True) if label == 1)
            vi = sum(v for v, label in zip(valid, labels, strict=True) if label == 0)
            summary = {
                "n_genuine": ng,
                "n_impostor": ni,
                "n_valid_genuine": vg,
                "n_valid_impostor": vi,
                "n_failed_genuine": ng - vg,
                "n_failed_impostor": ni - vi,
            }
            assert result["counts"] == summary
            assert len(result["states"]) == len(reference)
            for actual, expected in zip(result["states"], reference, strict=True):
                assert actual["policy"] == expected["policy"]
                assert actual["counts"] == {**summary, **expected["counts"]}
                for rate in ("far", "frr"):
                    assert actual[rate] == float(expected[rate])
                    assert actual[rate + "_exact"] == str(expected[rate])
                decisions = apply_gated_policy(scores, valid, actual["policy"])
                assert decisions.dtype == np.bool_
                assert decisions.tolist() == expected["accepts"]
                assert not decisions[~np.asarray(valid, bool)].any()
            policies = [state["policy"] for state in reference]
            assert_eer(
                result["eer"], [(s["far"], s["frr"]) for s in reference], policies
            )
            assert result["terminal"] == {
                "state_index": len(reference) - 1,
                **result["states"][-1],
            }
            assert result["terminal"]["far_exact"] == str(Fraction(vi, ni))
            assert result["terminal"]["frr_exact"] == str(Fraction(ng - vg, ng))
            assert (result["eer"]["eer"] is not None) == (
                Fraction(vg, ng) + Fraction(vi, ni) >= 1
            )
            for name, successful, total in (("genuine", vg, ng), ("impostor", vi, ni)):
                fraction = Fraction(successful, total)
                assert result["coverage"][name] == {
                    "valid_fraction": float(fraction),
                    "valid_fraction_exact": str(fraction),
                    "failure_fraction": float(1 - fraction),
                    "failure_fraction_exact": str(1 - fraction),
                }
            conditional = result["conditional"]["eer"]
            if vg and vi:
                conditional_points = [
                    (
                        Fraction(s["counts"]["fp"], vi),
                        Fraction(vg - s["counts"]["tp"], vg),
                    )
                    for s in reference
                ]
                assert_eer(conditional, conditional_points, policies)
                fg, fi = Fraction(ng - vg, ng), Fraction(ni - vi, ni)
                for state, (far_valid, frr_valid) in zip(
                    reference, conditional_points, strict=True
                ):
                    assert state["frr"] == fg + (1 - fg) * frr_valid
                    assert state["far"] == (1 - fi) * far_valid
            else:
                assert conditional == {
                    "status": "undefined_empty_valid_class",
                    "eer": None,
                    "eer_exact": None,
                    "bracket": None,
                }
            for target in (0, Fraction(1, 3), 1):
                expected_index = max(
                    index
                    for index, state in enumerate(reference)
                    if state["far"] <= target
                )
                selected = calibrate_gated_threshold(
                    labels, scores, valid, target_far=target
                )
                assert selected["state_index"] == expected_index
                assert selected["policy"] == reference[expected_index]["policy"]
                assert selected["counts"] == result["states"][expected_index]["counts"]
                assert selected["far_exact"] == str(reference[expected_index]["far"])
                assert selected["frr_exact"] == str(reference[expected_index]["frr"])
            if all(valid):
                legacy = ranking_metrics(labels, scores, [7] * length)
                assert result["eer"]["eer_exact"] == legacy["eer_exact"]
                assert conditional == result["eer"]


def test_gated_crossing_and_exact_one_third_conditional_regression():
    labels = [1] * 4 + [0] * 4
    scores = [0.9, 0.6, 0.3, None, 0.8, 0.5, 0.2, None]
    result = gated_roc(labels, scores, [s is not None for s in scores])
    assert result["eer"]["status"] == "exact_threshold_state"
    assert result["eer"]["eer_exact"] == "1/2"
    conditional = result["conditional"]["eer"]
    assert conditional["status"] == "exact_threshold_state"
    assert conditional["eer_exact"] == "1/3"
    assert conditional["bracket"]["state_indexes"] == [3, 3]
    assert (
        conditional["bracket"]["policies"]
        == [{"kind": "score_at_least", "threshold": 0.6}] * 2
    )
    assert result["terminal"]["far_exact"] == "3/4"
    assert result["terminal"]["frr_exact"] == "1/4"


def test_twenty_percent_coverage_has_only_a_conditional_eer():
    labels = [1] * 5 + [0] * 5
    scores = [0.9, *([None] * 4), 0.1, *([None] * 4)]
    result = gated_roc(labels, scores, [s is not None for s in scores])
    assert result["eer"] == {
        "status": "undefined_no_crossing",
        "eer": None,
        "eer_exact": None,
        "bracket": None,
    }
    assert result["conditional"]["eer"]["eer_exact"] == "0"
    assert result["terminal"]["far_exact"] == "1/5"
    assert result["terminal"]["frr_exact"] == "4/5"


def test_terminal_one_third_equality_is_an_attainable_full_eer():
    # The frozen floating prototype incorrectly calls this undefined: subtracting
    # 2/3 from 1 rounds above the independently rounded impostor rate of 1/3.
    scores = [0.9, 0.6, None, 0.8, None, None]
    result = gated_roc([1, 1, 1, 0, 0, 0], scores, [s is not None for s in scores])
    assert result["terminal"]["far_exact"] == "1/3"
    assert result["terminal"]["frr_exact"] == "1/3"
    assert result["eer"]["status"] == "exact_threshold_state"
    assert result["eer"]["eer_exact"] == "1/3"
    assert result["eer"]["bracket"]["state_indexes"] == [3, 3]
    assert (
        result["eer"]["bracket"]["policies"]
        == [{"kind": "score_at_least", "threshold": 0.6}] * 2
    )


@pytest.mark.parametrize(("scores", "eer"), [([1.0, None], "0"), ([None, 1.0], "1")])
def test_empty_valid_class_can_have_full_metric_boundary_crossing(scores, eer):
    result = gated_roc([1, 0], scores, [s is not None for s in scores])
    assert result["eer"]["status"] == "exact_threshold_state"
    assert result["eer"]["eer_exact"] == eer
    assert result["conditional"]["eer"]["status"] == "undefined_empty_valid_class"


def test_all_invalid_has_only_explicit_reject_all():
    result = gated_roc([1, 0], [None, None], [False, False])
    assert len(result["states"]) == 1
    assert result["states"][0]["policy"] == REJECT
    assert result["states"][0]["counts"]["fn"] == 1
    assert result["eer"]["status"] == "undefined_no_crossing"
    assert result["conditional"]["eer"]["status"] == "undefined_empty_valid_class"
    selected = calibrate_gated_threshold(
        [1, 0], [None, None], [False, False], target_far=1
    )
    assert selected["policy"] == REJECT
    assert selected["state_index"] == 0
    assert selected["far_exact"] == "0"
    assert selected["frr_exact"] == "1"


def test_ties_interpolate_and_cannot_be_partially_selected():
    result = gated_roc([1, 0], [0.5, 0.5], [True, True])
    assert len(result["states"]) == 2
    assert result["eer"]["status"] == "interpolated"
    assert result["eer"]["eer_exact"] == "1/2"
    assert result["eer"]["bracket"]["state_indexes"] == [0, 1]
    assert result["eer"]["bracket"]["upper_weight_exact"] == "1/2"
    for target in (0, 0.5, np.nextafter(1.0, 0.0)):
        assert (
            calibrate_gated_threshold(
                [1, 0], [0.5, 0.5], [True, True], target_far=target
            )["policy"]
            == REJECT
        )


def test_float_max_and_min_are_real_states_without_overflow_sentinel():
    maximum = float(np.finfo(np.float64).max)
    labels, scores, valid = [1, 0], [maximum, -maximum], [True, True]
    with np.errstate(all="raise"):
        result = gated_roc(labels, scores, valid)
        selected = calibrate_gated_threshold(labels, scores, valid, target_far=0)
    assert [s["policy"]["threshold"] for s in result["states"]] == [
        None,
        maximum,
        -maximum,
    ]
    assert selected["policy"] == {"kind": "score_at_least", "threshold": maximum}
    assert apply_gated_policy(scores, valid, selected["policy"]).tolist() == [
        True,
        False,
    ]
    json.dumps(result, allow_nan=False)


def test_float64_representation_defines_rank_ties_for_large_integers():
    result = gated_roc([1, 0], [2**53, 2**53 + 1], [True, True])
    assert len(result["states"]) == 2
    assert result["states"][1]["policy"]["threshold"] == float(2**53)
    assert result["eer"]["eer_exact"] == "1/2"
    assert apply_gated_policy(
        [2**53, 2**53 + 1], [True, True], result["states"][1]["policy"]
    ).tolist() == [True, True]


def test_exact_far_cutoffs_do_not_use_binary_display_fuzz():
    labels = [1] + [0] * 100
    scores, valid = [2.0] + [1.0] + [0.0] * 99, [True] * 101
    for target in (0.01, Fraction(1, 100)):
        result = calibrate_gated_threshold(labels, scores, valid, target_far=target)
        assert result["policy"]["threshold"] == 1.0
        assert result["far_exact"] == result["target_far_exact"] == "1/100"
    lower = calibrate_gated_threshold(
        labels, scores, valid, target_far=np.nextafter(0.01, 0.0)
    )
    assert lower["policy"]["threshold"] == 2.0
    labels, scores = [1, 0, 0, 0], [2.0, 1.0, 0.0, 0.0]
    assert (
        calibrate_gated_threshold(labels, scores, [True] * 4, target_far=1 / 3)[
            "far_exact"
        ]
        == "0"
    )
    assert (
        calibrate_gated_threshold(
            labels, scores, [True] * 4, target_far=Fraction(1, 3)
        )["far_exact"]
        == "1/3"
    )


def test_calibration_keeps_original_impostor_denominator_and_last_zero_far_state():
    result = calibrate_gated_threshold(
        [1, 1, 0, 0, 0, 0],
        [0.9, 0.8, 0.7, None, None, None],
        [True, True, True, False, False, False],
        target_far=0.25,
    )
    assert result["policy"]["threshold"] == 0.7
    assert result["far_exact"] == "1/4"
    assert result["counts"]["n_impostor"] == 4
    result = calibrate_gated_threshold(
        [1, 1, 0], [0.9, 0.8, 0.7], [True] * 3, target_far=0
    )
    assert result["policy"]["threshold"] == 0.8


@pytest.mark.parametrize(
    ("scores", "valid"),
    [
        ([1.0, 0.0], [True, False]),
        ([1.0, float("nan")], [True, False]),
        ([1.0, float("inf")], [True, False]),
        ([1.0, float("-inf")], [True, True]),
        ([1.0, None], [True, True]),
        ([1.0, False], [True, True]),
        ([1.0, 0j], [True, True]),
        ([1.0, "0.5"], [True, True]),
        ([1.0, 10**400], [True, True]),
        ([1.0, 0.0], [True, 1]),
        ([1.0, 0.0], [1, 0]),
        ([1.0, 0.0], np.array([1, 0], dtype=np.int8)),
        ([1.0, 0.0], [True, None]),
        ([1.0], [True, True]),
        ([[1.0, 0.0]], [True, True]),
        ([1.0, 0.0], [[True, True]]),
        (1.0, True),
        (np.ma.array([1.0, 0.0], mask=[False, True]), [True, True]),
        ([1.0, 0.0], np.ma.array([True, True], mask=[False, True])),
    ],
)
def test_nullable_and_mask_validation_applies_even_to_reject_all(scores, valid):
    with pytest.raises(ValueError):
        gated_roc([1, 0], scores, valid)
    with pytest.raises(ValueError):
        calibrate_gated_threshold([1, 0], scores, valid)
    with pytest.raises(ValueError):
        apply_gated_policy(scores, valid, REJECT)


@pytest.mark.parametrize(
    "labels",
    [[], [1], [1, 1], [0, 0], [0, 2], [0.0, 1.0], [0, "1"], [0, None], [[0, 1]]],
)
def test_original_binary_classes_are_required(labels):
    for function in (gated_roc, calibrate_gated_threshold):
        with pytest.raises(ValueError):
            function(labels, [None, None], [False, False])


@pytest.mark.parametrize(
    "target",
    [True, np.bool_(False), -0.1, 1.1, np.nan, np.inf, None, "0.1", 0j, [0.1]],
)
def test_invalid_target_far(target):
    with pytest.raises(ValueError):
        calibrate_gated_threshold(
            [1, 0], [None, None], [False, False], target_far=target
        )


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {"kind": "reject_all"},
        {"kind": "reject_all", "threshold": None, "extra": True},
        {"kind": "reject_all", "threshold": 0.0},
        {"kind": "accept_all", "threshold": None},
        {"kind": None, "threshold": None},
        {"kind": np.array(["reject_all"]), "threshold": None},
        *[
            {"kind": "score_at_least", "threshold": value}
            for value in (None, True, np.nan, np.inf, -np.inf, 0j, "0.5", [0.5])
        ],
    ],
)
def test_policy_schema_is_strict(policy):
    with pytest.raises(ValueError):
        apply_gated_policy([1.0, None], [True, False], policy)


def test_source_inputs_are_immutable_and_returned_decisions_are_fresh():
    labels = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, None, None], dtype=object)
    valid = np.array([True, True, False, False])
    policy = {"kind": "score_at_least", "threshold": 0.8}
    saved = deepcopy((labels, scores, valid, policy))
    for array in (labels, scores, valid):
        array.setflags(write=False)
    result = gated_roc(labels, scores, valid)
    calibration = calibrate_gated_threshold(labels, scores, valid)
    accepts = apply_gated_policy(scores, valid, policy)
    assert accepts.tolist() == [True, True, False, False]
    accepts[:] = True
    assert apply_gated_policy(scores, valid, policy).tolist() == [
        True,
        True,
        False,
        False,
    ]
    for current, original in zip((labels, scores, valid), saved[:3], strict=True):
        np.testing.assert_array_equal(current, original)
    assert policy == saved[3]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert json.loads(json.dumps(calibration, allow_nan=False)) == calibration
    assert set(result) == {
        "schema_version",
        "denominator",
        "counts",
        "coverage",
        "states",
        "terminal",
        "eer",
        "conditional",
    }
    assert set(calibration) == {
        "schema_version",
        "denominator",
        "target_far",
        "target_far_exact",
        "selection_contract",
        "state_index",
        "policy",
        "counts",
        "far",
        "far_exact",
        "frr",
        "frr_exact",
    }
    assert calibration["selection_contract"] == (
        "most_permissive_attainable_state_with_empirical_far_at_most_target"
    )
    assert apply_gated_policy([], [], REJECT).dtype == np.bool_
    assert apply_gated_policy([], [], policy).shape == (0,)
