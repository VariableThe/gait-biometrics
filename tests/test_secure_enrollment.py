"""Synthetic CPU/NumPy-only enrollment checks; no dataset or saved scores."""

from __future__ import annotations

import builtins
import json
import math
import unittest
from unittest import mock

import numpy as np

_IMPORT = builtins.__import__
_METHODS = ("mean", "multi", "mean_multi", "top2", "lse_0.1")
_GRAPH_KEYS = (
    "claim_subject_ids",
    "probe_subject_ids",
    "probe_trial_ids",
    "score_order",
    "coverage",
)


def _without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("secure enrollment must remain Torch-free")
    return _IMPORT(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_torch):
    from engine.secure_enrollment import _logmeanexp, score_secure_trials
    from engine.template_scoring import CoverageError, score_trials


def _data(rows, embeddings):
    subjects, days, trials = np.asarray(rows, dtype=np.int64).T
    return np.asarray(embeddings, dtype=np.float64), subjects, days, trials


def _known():
    return _data(
        [(7, 1, 1), (7, 1, 2), (7, 1, 3), (7, 2, 1), (16, 1, 1), (16, 2, 1)],
        [[1, 0], [1, 0], [0, 1], [0, 1], [0, -1], [0, -1]],
    )


def _varied(*, windows=False):
    rows = [
        (s, d, t)
        for s in (29, 7, 16)
        for d in (2, 1)
        for t in ((8, 1, 4) if d == 1 else ((9,) if s == 16 else (6, 2)))
        for _ in range(t % 3 + 1 if windows else 1)
    ]
    return _data(rows, np.random.default_rng(71).normal(size=(len(rows), 7)))


def _claim_map(result):
    _, scores, metadata = result
    keys = zip(
        metadata["claim_subject_ids"],
        metadata["probe_subject_ids"],
        metadata["probe_trial_ids"],
        strict=True,
    )
    return dict(zip(keys, scores, strict=True))


class SecureEnrollmentTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("builtins.__import__", side_effect=_without_torch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_handcomputed_scores_dtypes_metadata_and_json(self):
        a = 1 / math.sqrt(5)
        expected = {
            "mean": [a - 1, -2, -a - 1, 0],
            "multi": [0, -2, -1, 0],
            "mean_multi": [0.5 * (a - 1), -2, -1 - 0.5 * a, 0],
            "top2": [-0.5, -2, -1, 0],
            "lse_0.1": [
                0.1 * math.log((2 + math.exp(10)) / 3) - 1,
                -2,
                0.1 * math.log((2 + math.exp(-10)) / 3) - 1,
                0,
            ],
        }
        for method in _METHODS:
            with self.subTest(method=method):
                labels, scores, metadata = score_secure_trials(*_known(), method=method)
                np.testing.assert_array_equal(labels, [1, 0, 0, 1])
                np.testing.assert_allclose(scores, expected[method], rtol=0, atol=2e-16)
                self.assertEqual(labels.dtype, np.dtype("int64"))
                self.assertEqual(scores.dtype, np.dtype("float64"))
                self.assertEqual(metadata["method"], method)
                self.assertEqual(metadata["claim_subject_ids"], [7, 16, 7, 16])
                self.assertEqual(metadata["probe_subject_ids"], [7, 7, 16, 16])
                self.assertEqual(metadata["probe_trial_ids"], [1, 1, 1, 1])
                self.assertEqual(metadata["score_range"], [-2.0, 0.0])
                self.assertFalse(metadata["calibration"]["threshold_calibrated"])
                self.assertEqual(metadata["calibration"]["candidate"]["method"], method)
                json.dumps(metadata, allow_nan=False)

    def test_mean_and_multi_exactly_match_existing_scorer(self):
        data = _varied(windows=True)
        for method in ("mean", "multi"):
            with self.subTest(method=method):
                expected = score_trials(*data, template_method=method)
                actual = score_secure_trials(*data, method=method)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                for key in (*_GRAPH_KEYS, "templates", "multi_fusion"):
                    self.assertEqual(actual[2][key], expected[2][key])

    def test_fusion_averages_already_transformed_scores(self):
        data = _varied(windows=True)
        mean = score_secure_trials(*data, method="mean")
        multi = score_secure_trials(*data, method="multi")
        fused = score_secure_trials(*data, method="mean_multi")
        np.testing.assert_array_equal(fused[1], 0.5 * mean[1] + 0.5 * multi[1])
        self.assertEqual(fused[2]["mean_templates"], mean[2]["templates"])
        self.assertEqual(fused[2]["templates"], multi[2]["templates"])
        self.assertEqual(
            fused[2]["calibration"]["candidate"]["weights"],
            {"mean": 0.5, "multi": 0.5},
        )

    def test_common_exhaustive_directed_graph_with_unequal_probe_counts(self):
        data = _varied(windows=True)
        baseline = score_trials(*data, template_method="mean")
        for method in _METHODS:
            with self.subTest(method=method):
                actual = score_secure_trials(*data, method=method)
                np.testing.assert_array_equal(actual[0], baseline[0])
                for key in _GRAPH_KEYS:
                    self.assertEqual(actual[2][key], baseline[2][key])
                metadata = actual[2]
                keys = list(_claim_map(actual))
                self.assertEqual(
                    keys, sorted(keys, key=lambda key: (key[1], key[2], key[0]))
                )
                self.assertEqual(len(keys), 15)  # Five probes, three claims each.
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(metadata["coverage"]["n_genuine"], 5)
                self.assertEqual(metadata["coverage"]["n_impostor"], 10)
                self.assertEqual(metadata["coverage"]["far_resolution"], 0.1)
                for subject, trial in {(p, t) for _, p, t in keys}:
                    self.assertEqual(
                        [c for c, p, t in keys if (p, t) == (subject, trial)],
                        [7, 16, 29],
                    )

    def test_cancellations_share_baseline_coverage_and_multi_joins_true_keys(self):
        rows = [
            (7, 1, 1),
            (7, 2, 1),
            (16, 1, 1),
            (16, 2, 4),
            (7, 2, 9),
            (7, 1, 3),
            (7, 1, 3),
            (7, 2, 2),
            (7, 2, 2),
            (10, 1, 1),
            (10, 1, 2),
            (10, 2, 1),
            (29, 1, 1),
            (30, 2, 1),
        ]
        vectors = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
        data = _data(rows, vectors)
        baseline = score_trials(*data, template_method="mean")
        original_multi = score_trials(*data, template_method="multi")
        self.assertEqual(
            original_multi[2]["coverage"]["eligible_subject_ids"], [7, 10, 16]
        )
        original_scores = _claim_map(original_multi)
        for method in _METHODS:
            with self.subTest(method=method):
                actual = score_secure_trials(*data, method=method)
                for key in _GRAPH_KEYS:
                    self.assertEqual(actual[2][key], baseline[2][key])
                coverage = actual[2]["coverage"]
                self.assertEqual(coverage["requested_subject_ids"], [7, 10, 16, 29, 30])
                self.assertEqual(coverage["eligible_subject_ids"], [7, 16])
                self.assertEqual(
                    {row["subject"] for row in coverage["exclusions"]}, {7, 10, 29, 30}
                )
                self.assertEqual(len(actual[1]), 6)
                self.assertIn("all exclusions", actual[2]["coverage_policy"])
                if method == "multi":
                    for key, score in _claim_map(actual).items():
                        self.assertEqual(score, original_scores[key])

    def test_insufficient_common_coverage_raises_with_same_report_for_every_method(
        self,
    ):
        data = _data(
            [(7, 1, 1), (7, 2, 1), (16, 1, 1), (16, 1, 2), (16, 2, 1)],
            [[1, 0], [1, 0], [1, 0], [-1, 0], [0, 1]],
        )
        reports = []
        for method in _METHODS:
            with (
                self.subTest(method=method),
                self.assertRaises(CoverageError) as caught,
            ):
                score_secure_trials(*data, method=method)
            reports.append(caught.exception.coverage)
        self.assertTrue(all(report == reports[0] for report in reports))
        self.assertEqual(reports[0]["requested_subject_ids"], [7, 16])
        self.assertEqual(reports[0]["eligible_subject_ids"], [7])
        json.dumps(reports[0], allow_nan=False)

    def test_day2_changes_never_fit_templates_or_change_other_probe_scores(self):
        data = _varied(windows=True)
        embeddings, subjects, days, trials = data
        changed = embeddings.copy()
        changed[(subjects == 7) & (days == 2) & (trials == 2)] *= -1
        for method in _METHODS:
            with self.subTest(method=method):
                first = score_secure_trials(*data, method=method)
                second = score_secure_trials(
                    changed, subjects, days, trials, method=method
                )
                self.assertEqual(first[2]["templates"], second[2]["templates"])
                self.assertEqual(
                    first[2].get("mean_templates"), second[2].get("mean_templates")
                )
                for key in _GRAPH_KEYS:
                    self.assertEqual(first[2][key], second[2][key])
                other = ~(
                    (np.asarray(first[2]["probe_subject_ids"]) == 7)
                    & (np.asarray(first[2]["probe_trial_ids"]) == 2)
                )
                np.testing.assert_array_equal(first[1][other], second[1][other])
                self.assertTrue(np.any(first[1][~other] != second[1][~other]))

    def test_exact_row_permutation_invariance_for_distinct_trials(self):
        data = _varied()
        permutation = np.random.default_rng(12).permutation(len(data[0]))
        permuted = tuple(values[permutation] for values in data)
        for method in _METHODS:
            with self.subTest(method=method):
                first = score_secure_trials(*data, method=method)
                second = score_secure_trials(*permuted, method=method)
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[2], second[2])

    def test_window_permutation_agrees_within_float_reduction_precision_and_no_mutation(
        self,
    ):
        data = _varied(windows=True)
        originals = tuple(values.copy() for values in data)
        permutation = np.random.default_rng(23).permutation(len(data[0]))
        for values in data:
            values.flags.writeable = False
        permuted = tuple(values[permutation] for values in data)
        for method in _METHODS:
            with self.subTest(method=method):
                first = score_secure_trials(*data, method=method)
                second = score_secure_trials(*permuted, method=method)
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_allclose(first[1], second[1], rtol=0, atol=1e-14)
                for key in _GRAPH_KEYS:
                    self.assertEqual(first[2][key], second[2][key])
        for values, original in zip(data, originals, strict=True):
            np.testing.assert_array_equal(values, original)

    def test_top2_and_lse_use_all_original_trials_including_duplicate_directions(self):
        embeddings, subjects, days, trials = _known()
        embeddings[3] = [1, 0]
        for method in ("top2", "lse_0.1"):
            with self.subTest(method=method):
                _, scores, metadata = score_secure_trials(
                    embeddings, subjects, days, trials, method=method
                )
                expected = (
                    0
                    if method == "top2"
                    else 0.1 * math.log((2 * math.exp(10) + 1) / 3) - 1
                )
                self.assertAlmostEqual(scores[0], expected, places=15)
                self.assertEqual(
                    metadata["templates"][0]["prototype_trial_ids"], [1, 2, 3]
                )
                self.assertEqual(metadata["templates"][0]["n_prototypes"], 3)

    def test_window_normalization_and_equal_trial_weighting(self):
        data = _data(
            [(7, 1, 1)] * 3 + [(7, 1, 2), (7, 2, 1), (16, 1, 1), (16, 2, 1)],
            [[100, 0], [2, 0], [1, 0], [0, 1], [1, 0], [-1, 0], [-1, 0]],
        )
        expected = {
            "mean": 1 / math.sqrt(2) - 1,
            "multi": 0,
            "mean_multi": 0.5 * (1 / math.sqrt(2) - 1),
            "top2": -0.5,
            "lse_0.1": 0.1 * math.log((math.exp(10) + 1) / 2) - 1,
        }
        for method in _METHODS:
            with self.subTest(method=method):
                _, scores, _ = score_secure_trials(*data, method=method)
                self.assertAlmostEqual(scores[0], expected[method], places=15)

    def test_single_trial_fallback_and_lse_has_no_template_count_bonus(self):
        for count in (1, 2, 10):
            data = _data(
                [(7, 1, trial) for trial in range(1, count + 1)]
                + [(7, 2, 1), (16, 1, 1), (16, 2, 1)],
                [[1, 0]] * count + [[0, 1], [0, -1], [0, -1]],
            )
            for method in _METHODS:
                with self.subTest(count=count, method=method):
                    _, scores, _ = score_secure_trials(*data, method=method)
                    np.testing.assert_array_equal(scores, [-1, -2, -1, 0])

    def test_logmeanexp_stays_finite_for_large_shifts_and_extreme_trial_gaps(self):
        correction = 0.1 * (math.log1p(math.exp(-20)) - math.log(2))
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            for maximum in (-1e6, 0, 1e6):
                with self.subTest(maximum=maximum):
                    value = _logmeanexp(np.array([maximum, maximum - 2]))
                    self.assertTrue(math.isfinite(value))
                    self.assertAlmostEqual(value, maximum + correction, places=10)
            self.assertEqual(_logmeanexp(np.full(10, -2.0)), -2.0)
        # An enrollment trial antipodal to the probe is retained, with a gap of 2.
        data = _data(
            [(7, 1, 1), (7, 1, 2), (7, 1, 3), (7, 2, 1), (16, 1, 1), (16, 2, 1)],
            [[1, 0], [1, 0], [-1, 0], [1, 0], [0, 1], [0, 1]],
        )
        _, scores, _ = score_secure_trials(*data, method="lse_0.1")
        self.assertAlmostEqual(scores[0], 0.1 * math.log((2 + math.exp(-20)) / 3))

    def test_authentic_large_and_zero_identity_ids_are_preserved(self):
        large = 2**53 + 1
        data = _data(
            [(large, 2, 10), (0, 1, 10), (large, 1, 10), (0, 2, 10)],
            [[0, 1], [1, 0], [0, 1], [1, 0]],
        )
        for method in _METHODS:
            with self.subTest(method=method):
                labels, scores, metadata = score_secure_trials(*data, method=method)
                np.testing.assert_array_equal(labels, [1, 0, 0, 1])
                np.testing.assert_array_equal(scores, [0, -1, -1, 0])
                self.assertEqual(metadata["claim_subject_ids"], [0, large, 0, large])
                self.assertEqual(metadata["probe_subject_ids"], [0, 0, large, large])
                self.assertEqual(metadata["probe_trial_ids"], [10] * 4)

    def test_strict_inputs_fail_before_coverage_can_hide_bad_rows(self):
        data = _known()
        invalid = [
            (0, np.zeros_like(data[0])),
            (0, np.full_like(data[0], np.nan)),
            (0, np.full_like(data[0], np.inf)),
            (0, np.full_like(data[0], 1e-13)),
            (0, data[0].astype(complex)),
            (0, data[0].astype(bool)),
            (0, data[0].astype(object)),
            (0, data[0][:, :0]),
            (0, data[0][0]),
            (0, data[0][:-1]),
            (1, data[1].astype(float)),
            (1, data[1].astype(bool)),
            (1, data[1][:, None]),
            (1, np.array([-1, 7, 7, 7, 16, 16])),
            (1, np.full(6, 2**63, dtype=np.uint64)),
            (2, data[2][:-1]),
            (2, np.array([1, 1, 1, 3, 1, 2])),
            (2, data[2].astype(float)),
            (3, np.array([0, 2, 3, 1, 1, 1])),
            (3, np.array([11, 2, 3, 1, 1, 1])),
            (3, np.asarray(1)),
        ]
        for method in _METHODS:
            for index, value in invalid:
                args = list(data)
                args[index] = value
                with (
                    self.subTest(method=method, index=index, value=value),
                    self.assertRaises(ValueError),
                ):
                    score_secure_trials(*args, method=method)
            # The malformed vector belongs to a subject that would otherwise be
            # excluded for missing day1; it must still fail global validation.
            extra = tuple(
                np.concatenate((values, tail))
                for values, tail in zip(
                    data, (np.array([[np.nan, 0]]), [29], [2], [1]), strict=True
                )
            )
            with (
                self.subTest(method=method, excluded_bad_row=True),
                self.assertRaises(ValueError),
            ):
                score_secure_trials(*extra, method=method)
        empty = (np.empty((0, 2)),) + (np.empty(0, dtype=np.int64),) * 3
        for method in _METHODS:
            with self.subTest(method=method, empty=True), self.assertRaises(ValueError):
                score_secure_trials(*empty, method=method)
        for method in ("robust", "lse_1.0", "", None, True, [], np.array(["mean"])):
            with self.subTest(method=method), self.assertRaises(ValueError):
                score_secure_trials(*data, method=method)

    def test_finite_extreme_embeddings_keep_direction_and_score_range(self):
        data = _data(
            [(7, 1, 1), (7, 2, 1), (16, 1, 1), (16, 2, 1)],
            [[1e308, 0], [1e308, 0], [0, 1e308], [0, 1e308]],
        )
        for method in _METHODS:
            with self.subTest(method=method):
                _, scores, _ = score_secure_trials(*data, method=method)
                np.testing.assert_array_equal(scores, [0, -1, -1, 0])
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= -2) & (scores <= 0)).all())


if __name__ == "__main__":
    unittest.main()
