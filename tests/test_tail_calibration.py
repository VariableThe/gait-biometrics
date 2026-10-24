"""Synthetic CPU-only tail tests; no datasets, fitted models or held-out reads."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np
from scipy.stats import beta, binom

from engine.tail_calibration import (
    empirical_threshold,
    nominal_order_statistic_threshold,
    nominal_violation_bound,
)
from scripts import mentor_backend_study as frozen


class NominalTailTests(unittest.TestCase):
    def test_small_samples_cannot_supply_requested_nominal_confidence(self):
        for n, delta in ((60, 0.1), (196, 0.1), (229, 0.1), (298, 0.05)):
            with self.subTest(n=n, delta=delta):
                result = nominal_order_statistic_threshold(np.arange(n), delta=delta)
                self.assertEqual(
                    result["status"], "insufficient_independent_calibration"
                )
                for field in ("threshold", "count", "rank", "allowed_count"):
                    self.assertIsNone(result[field])
                self.assertIsNone(result["nominal_violation_bound"])
                self.assertEqual(result["n"], n)
                self.assertAlmostEqual(result["best_nominal_violation_bound"], 0.99**n)
                self.assertGreater(result["best_nominal_violation_bound"], delta)
                json.dumps(result, allow_nan=False)

    def test_230_and_240_use_maximum_at_90_percent(self):
        for n in (230, 240):
            with self.subTest(n=n):
                scores = np.arange(n, dtype=np.float64)
                result = nominal_order_statistic_threshold(scores)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["rank"], n)
                self.assertEqual(result["allowed_count"], 0)
                self.assertEqual(result["count"], 0)
                self.assertEqual(result["threshold"], np.nextafter(n - 1.0, np.inf))
                self.assertAlmostEqual(result["nominal_violation_bound"], 0.99**n)
                self.assertLessEqual(result["nominal_violation_bound"], 0.1)
                self.assertGreater(binom.cdf(1, n, 0.01), 0.1)

    def test_299_supports_95_percent(self):
        result = nominal_order_statistic_threshold(np.arange(299), delta=0.05)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rank"], 299)
        self.assertEqual(result["count"], 0)
        self.assertAlmostEqual(result["nominal_violation_bound"], 0.0495362566376625)

    def test_496_permits_one_at_95_percent(self):
        result = nominal_order_statistic_threshold(np.arange(496), delta=0.05)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rank"], 495)
        self.assertEqual(result["allowed_count"], 1)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["threshold"], np.nextafter(494.0, np.inf))
        self.assertAlmostEqual(result["nominal_violation_bound"], 0.0411091305387242)
        self.assertGreater(binom.cdf(2, 496, 0.01), 0.05)

    def test_bound_matches_direct_binomial_sum_and_uniform_order_distribution(self):
        # For iid Uniform scores, the ascending kth order statistic is
        # Beta(k, n+1-k); FAR > alpha iff it is below 1-alpha.
        for n in (1, 2, 10, 20):
            for count in range(n):
                bound = nominal_violation_bound(n, allowed_count=count, alpha=0.2)
                direct = math.fsum(
                    math.comb(n, i) * 0.2**i * 0.8 ** (n - i) for i in range(count + 1)
                )
                self.assertAlmostEqual(bound, direct)
                self.assertAlmostEqual(bound, beta.cdf(0.8, n - count, count + 1))

    def test_rank_is_maximal_for_multiple_targets(self):
        scores = np.linspace(-1, 1, 500)
        for alpha, delta in ((0.01, 0.1), (0.02, 0.05), (0.2, 0.3), (0.9, 0.95)):
            with self.subTest(alpha=alpha, delta=delta):
                result = nominal_order_statistic_threshold(
                    scores, alpha=alpha, delta=delta
                )
                j = result["allowed_count"]
                self.assertEqual(result["status"], "ok")
                self.assertLessEqual(result["nominal_violation_bound"], delta)
                self.assertGreater(binom.cdf(j + 1, len(scores), alpha), delta)
                self.assertEqual(result["rank"], len(scores) - j)
                self.assertEqual(result["count"], j)

    def test_equal_cdf_is_feasible(self):
        delta = float(binom.cdf(1, 496, 0.01))
        result = nominal_order_statistic_threshold(np.arange(496), delta=delta)
        self.assertEqual(result["allowed_count"], 1)
        below = nominal_order_statistic_threshold(
            np.arange(496), delta=np.nextafter(delta, 0.0)
        )
        self.assertEqual(below["allowed_count"], 0)

    def test_boundary_ties_are_rejected_together_without_recomputing_bound(self):
        for top_count in (0, 1, 2, 5):
            with self.subTest(top_count=top_count):
                scores = np.ones(496)
                scores[:top_count] = 2.0
                result = nominal_order_statistic_threshold(scores, delta=0.05)
                self.assertEqual(result["allowed_count"], 1)
                expected = int(top_count == 1)
                self.assertEqual(result["count"], expected)
                boundary = result["order_statistic"]
                self.assertFalse(
                    (scores[scores == boundary] >= result["threshold"]).any()
                )
                self.assertEqual(result["count"], np.count_nonzero(scores > boundary))
                self.assertEqual(
                    result["nominal_violation_bound"],
                    nominal_violation_bound(496, allowed_count=1),
                )

    def test_confidence_increase_cannot_lower_threshold(self):
        scores = np.linspace(-2, 3, 1000)
        previous = -np.inf
        previous_count = len(scores)
        for delta in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
            result = nominal_order_statistic_threshold(scores, delta=delta)
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["threshold"], previous)
            self.assertLessEqual(result["allowed_count"], previous_count)
            previous = result["threshold"]
            previous_count = result["allowed_count"]
        self.assertEqual(
            nominal_order_statistic_threshold(scores, delta=1e-6)["status"],
            "insufficient_independent_calibration",
        )

    def test_nextafter_handles_roundoff_subnormals_and_finite_extremes(self):
        maximum = np.finfo(np.float64).max
        boundaries = (
            -maximum,
            -1e300,
            -0.0,
            0.0,
            np.nextafter(0.0, 1.0),
            0.1,
            1e300,
            np.nextafter(maximum, -np.inf),
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary), np.errstate(all="raise"):
                result = nominal_order_statistic_threshold(np.full(240, boundary))
                self.assertEqual(result["status"], "ok")
                self.assertTrue(math.isfinite(result["threshold"]))
                self.assertGreater(result["threshold"], boundary)
                self.assertEqual(result["count"], 0)
                json.dumps(result, allow_nan=False)

    def test_no_infinity_substitute_for_unrepresentable_threshold(self):
        with np.errstate(all="raise"):
            result = nominal_order_statistic_threshold(
                np.full(240, np.finfo(np.float64).max)
            )
        self.assertEqual(result["status"], "threshold_not_representable")
        self.assertIsNone(result["threshold"])
        self.assertIsNone(result["count"])
        self.assertEqual(result["rank"], 240)
        self.assertLess(result["nominal_violation_bound"], 0.1)
        json.dumps(result, allow_nan=False)

        # max itself is still a valid accepting threshold when the boundary
        # is its predecessor and one accepted score is permitted.
        maximum = np.finfo(np.float64).max
        scores = np.full(496, np.nextafter(maximum, -np.inf))
        scores[0] = maximum
        result = nominal_order_statistic_threshold(scores, delta=0.05)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["threshold"], maximum)
        self.assertEqual(result["count"], 1)

    def test_inputs_are_not_mutated_and_order_does_not_matter(self):
        scores = np.random.default_rng(2026).normal(size=496)
        original = scores.copy()
        scores.flags.writeable = False
        result = nominal_order_statistic_threshold(scores, delta=0.05)
        self.assertEqual(
            result, nominal_order_statistic_threshold(scores[::-1], delta=0.05)
        )
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(result["interpretation"], "nominal_iid_reference_only")
        self.assertIn("no guarantee", " ".join(result["assumptions"]))
        json.dumps(result, allow_nan=False)

    def test_nominal_count_and_application_use_float64_scores(self):
        scores = np.ones(240, dtype=np.float32)
        result = nominal_order_statistic_threshold(scores)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 0)
        accepted = np.asarray(scores, dtype=np.float64) >= result["threshold"]
        self.assertEqual(int(accepted.sum()), result["count"])

    def test_invalid_score_vectors(self):
        invalid = (
            [],
            0.5,
            [[0.1, 0.2]],
            [np.nan],
            [np.inf],
            [-np.inf],
            ["0.1"],
            [1 + 0j],
            [True, False],
            np.array([0.1], dtype=object),
            [None],
        )
        for scores in invalid:
            with self.subTest(scores=repr(scores)), self.assertRaises(ValueError):
                nominal_order_statistic_threshold(scores)

    def test_probabilities_are_strict_real_scalars_excluding_bool(self):
        invalid = (
            True,
            False,
            np.bool_(True),
            "0.1",
            0.1 + 0j,
            [0.1],
            np.array(0.1),
            np.array([0.1]),
            None,
            np.nan,
            np.inf,
            -np.inf,
            0,
            1,
            -0.1,
            1.1,
        )
        for name in ("alpha", "delta"):
            for value in invalid:
                with (
                    self.subTest(name=name, value=repr(value)),
                    self.assertRaises(ValueError),
                ):
                    nominal_order_statistic_threshold(np.arange(240), **{name: value})
        result = nominal_order_statistic_threshold(
            np.arange(240), alpha=np.float64(0.01), delta=np.float32(0.1)
        )
        self.assertEqual(result["status"], "ok")

    def test_fixed_count_summary_validates_integer_counts(self):
        self.assertAlmostEqual(nominal_violation_bound(60), 0.5471566423907614)
        self.assertAlmostEqual(nominal_violation_bound(196), 0.1394755687752252)
        self.assertAlmostEqual(
            nominal_violation_bound(np.int64(496), allowed_count=np.int64(1)),
            0.0411091305387242,
        )
        for n in (True, np.bool_(True), 0, -1, 240.0, "240", np.array(240)):
            with self.subTest(n=repr(n)), self.assertRaises(ValueError):
                nominal_violation_bound(n)
        for count in (True, np.bool_(True), -1, 240, 1.0, "1", np.array(1)):
            with self.subTest(count=repr(count)), self.assertRaises(ValueError):
                nominal_violation_bound(240, allowed_count=count)
        with self.assertRaises(ValueError):
            nominal_violation_bound(240, alpha=True)


class EmpiricalTailTests(unittest.TestCase):
    def test_default_result_has_exact_frozen_original_parity(self):
        rng = np.random.default_rng(101)
        for n_impostor in (1, 60, 100, 196, 230, 240, 299, 496):
            labels = np.r_[np.zeros(n_impostor, dtype=int), np.ones(37, dtype=int)]
            for tied in (False, True):
                scores = rng.normal(size=len(labels))
                if tied:
                    scores = np.round(scores, 1)
                order = rng.permutation(len(labels))
                with self.subTest(n_impostor=n_impostor, tied=tied):
                    result = empirical_threshold(labels[order], scores[order])
                    self.assertEqual(
                        result, frozen.calibrate(labels[order], scores[order])
                    )
                    json.dumps(result, allow_nan=False)

    def test_original_dtype_comparison_is_preserved_for_exact_result_parity(self):
        for dtype in (np.float16, np.float32, np.float64, np.longdouble, np.int64):
            with self.subTest(dtype=dtype):
                # The original compares its Python-float threshold to the
                # original dtype in fixed_metrics, even for reject-all.
                scores = np.array([1, 1], dtype=dtype)
                self.assertEqual(
                    empirical_threshold([0, 1], scores),
                    frozen.calibrate([0, 1], scores),
                )

    def test_original_uses_genuine_roc_point_instead_of_max_impostor_nextafter(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        result = empirical_threshold(labels, scores)
        self.assertEqual(result, frozen.calibrate(labels, scores))
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["false_accepts"], 0)
        self.assertEqual(result["false_rejects"], 1)
        self.assertEqual(result["decision_rule"], "score >= threshold")
        self.assertNotEqual(result["threshold"], np.nextafter(0.8, np.inf))

    def test_tied_group_that_exceeds_budget_is_not_split(self):
        labels = np.r_[np.zeros(100, dtype=int), [1, 1]]
        scores = np.r_[[0.8, 0.8], np.full(98, 0.1), [0.9, 0.8]]
        result = empirical_threshold(labels, scores)
        self.assertEqual(result, frozen.calibrate(labels, scores))
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["false_accepts"], 0)
        self.assertEqual(result["false_rejects"], 1)

    def test_exact_target_far_and_equal_score_are_accepted(self):
        labels = np.r_[np.zeros(100, dtype=int), [1]]
        scores = np.r_[[0.9], np.full(99, 0.1), [0.9]]
        result = empirical_threshold(labels, scores)
        self.assertEqual(result, frozen.calibrate(labels, scores))
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["false_accepts"], 1)
        self.assertEqual(result["false_rejects"], 0)
        self.assertEqual(result["far"], 0.01)

    def test_custom_target_matches_frozen_grouped_roc_and_endpoint_rules(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([0.9, 0.9, 0.8, 0.85, 0.7, 0.75, 0.6, 0.65])
        far, _, thresholds = frozen.roc(labels, scores)
        for target in (0, 0.01, 0.25, 0.5, 0.75, 1):
            result = empirical_threshold(labels, scores, target)
            threshold = float(thresholds[np.flatnonzero(far <= target)[-1]])
            expected = {
                **frozen.fixed_metrics(labels, scores, threshold),
                "target_far": target,
                "selection": "empirical_far_at_most_target",
            }
            self.assertEqual(result, expected)

    def test_all_tied_scores_select_finite_reject_all_point(self):
        result = empirical_threshold([0, 1], [0.0, 0.0])
        self.assertEqual(result, frozen.calibrate([0, 1], [0.0, 0.0]))
        self.assertEqual(result["threshold"], np.nextafter(0.0, np.inf))
        self.assertEqual(result["false_accepts"], 0)
        self.assertEqual(result["false_rejects"], 1)

    def test_finite_extremes_do_not_overflow_group_detection(self):
        maximum = np.nextafter(np.finfo(np.float64).max, -np.inf)
        with np.errstate(all="raise"):
            result = empirical_threshold([1, 0], [maximum, -maximum])
        self.assertEqual(result["threshold"], maximum)
        self.assertEqual(result["false_accepts"], 0)
        self.assertEqual(result["false_rejects"], 0)
        json.dumps(result, allow_nan=False)
        with (
            np.errstate(all="raise"),
            self.assertRaisesRegex(ValueError, "finite reject-all"),
        ):
            empirical_threshold([0, 1], [0.0, np.finfo(np.float64).max])

    def test_invalid_empirical_inputs(self):
        for labels, scores in (
            ([0, 1], []),
            ([0, 1], [0.1]),
            ([[0, 1]], [[0.1, 0.2]]),
            ([0, 0], [0.1, 0.2]),
            ([1, 1], [0.1, 0.2]),
            ([0, 2], [0.1, 0.2]),
            ([0, np.nan], [0.1, 0.2]),
            ([0, 1], [0.1, np.inf]),
            ([0, 1], [0.1, np.nan]),
            ([0, 1], ["0.1", "0.2"]),
        ):
            with (
                self.subTest(labels=labels, scores=scores),
                self.assertRaises(ValueError),
            ):
                empirical_threshold(labels, scores)
        for target in (
            True,
            np.bool_(False),
            "0.1",
            np.array(0.1),
            [0.1],
            np.nan,
            np.inf,
            -0.1,
            1.1,
        ):
            with self.subTest(target=repr(target)), self.assertRaises(ValueError):
                empirical_threshold([0, 1], [0.1, 0.2], target)

    def test_empirical_inputs_are_not_mutated(self):
        labels, scores = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.8, 0.1])
        original_labels, original_scores = labels.copy(), scores.copy()
        labels.flags.writeable = scores.flags.writeable = False
        empirical_threshold(labels, scores)
        np.testing.assert_array_equal(labels, original_labels)
        np.testing.assert_array_equal(scores, original_scores)


if __name__ == "__main__":
    unittest.main()
