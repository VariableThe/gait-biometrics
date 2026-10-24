"""Synthetic numerical and validation regressions; no datasets or saved studies."""

from __future__ import annotations

import io
import unittest
import warnings

import numpy as np
import torch

from engine.enrollment import enrollment_scores
from engine.evaluate import (
    _roc,
    _verification_pairs,
    _verification_scores,
    calibrate_threshold,
    classification_metrics,
    compute_eer,
    fixed_threshold_metrics,
)
from models.gait_metric import WithinClassMetric


class ScoreValidationTests(unittest.TestCase):
    def test_float32_scores_keep_stricter_float64_cutoff(self):
        scores = np.full(4, -0.5, dtype=np.float32)
        threshold = np.nextafter(-0.5, np.inf)
        result = fixed_threshold_metrics([1, 0, 1, 0], scores, threshold)
        self.assertEqual(result["threshold"], threshold)
        self.assertEqual((result["false_accepts"], result["false_rejects"]), (0, 2))
        calibrated = calibrate_threshold([1, 0, 1, 0], scores, 0)
        self.assertGreater(calibrated["threshold"], float(scores.max()))
        self.assertEqual((calibrated["far"], calibrated["frr"]), (0, 1))

    def test_threshold_and_target_require_finite_real_scalars(self):
        invalid = (
            True,
            np.bool_(False),
            [0.5],
            np.array([0.5]),
            np.ones((1, 1)),
            "0.5",
            0.5 + 0j,
            None,
            np.nan,
            np.inf,
            -np.inf,
        )
        for value in invalid:
            for function in (fixed_threshold_metrics, calibrate_threshold):
                with (
                    self.subTest(value=value, function=function.__name__),
                    self.assertRaises(ValueError),
                ):
                    function([1, 0], [1, 0], value)
        for value in (0, 1, np.float32(0.5), np.array(0.5)):
            result = calibrate_threshold([1, 0], [1, 0], value)
            self.assertIsInstance(result["target_far"], float)
            self.assertEqual(result["target_far"], float(value))
        for value in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                calibrate_threshold([1, 0], [1, 0], value)

    def test_scores_and_labels_cannot_be_silently_coerced(self):
        for scores in (
            np.array([1 + 9j, 0 + 2j]),
            [True, False],
            ["1", "0"],
            np.array([1, 0], dtype=object),
            [1, np.nan],
            [1, np.inf],
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                fixed_threshold_metrics([1, 0], scores, 0.5)
        for labels in ([1 + 0j, 0 + 0j], ["1", "0"], [0.2, 1], [1, 2]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                compute_eer(labels, [1, 0])
        self.assertEqual(compute_eer([True, False], [1, 0]), 0)

    def test_finite_reject_all_boundary_is_explicit_and_warning_free(self):
        maximum = np.finfo(np.float64).max
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "finite reject-all"):
                calibrate_threshold([1, 0], [maximum, maximum], 0)
            below = np.nextafter(maximum, 0)
            result = calibrate_threshold([1, 0], [below, below], 0)
        self.assertEqual(caught, [])
        self.assertEqual(result["threshold"], maximum)
        self.assertEqual((result["far"], result["frr"]), (0, 1))
        direct = fixed_threshold_metrics([1, 0], [maximum, maximum], maximum)
        self.assertEqual((direct["far"], direct["frr"]), (1, 0))

    def test_every_roc_point_matches_actual_float32_decisions(self):
        labels = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([-0.5, -0.5, -0.8, -0.9, -0.9, -1], dtype=np.float32)
        fars, frrs, thresholds = _roc(labels, scores)
        for far, frr, threshold in zip(fars, frrs, thresholds):
            actual = fixed_threshold_metrics(labels, scores, threshold)
            self.assertEqual(actual["far"], far)
            self.assertEqual(actual["frr"], frr)

    @unittest.skipUnless(
        np.finfo(np.longdouble).max > np.finfo(np.float64).max,
        "platform has no extended-precision floats",
    )
    def test_extended_precision_overflow_cannot_enter_decisions(self):
        huge = np.longdouble("1e400")
        for function in (fixed_threshold_metrics, calibrate_threshold):
            with (
                self.subTest(function=function.__name__),
                self.assertRaises(ValueError),
            ):
                function([1, 0], [1, 0], huge)
        with self.assertRaises(ValueError):
            compute_eer([1, 0], np.array([huge, 0], dtype=np.longdouble))


class ClassificationValidationTests(unittest.TestCase):
    def test_known_confusion_and_empty_input(self):
        result = classification_metrics([0, 1, 1], [0, 0, 1], np.int64(2))
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(classification_metrics([], [], 2)["accuracy"], 0)

    def test_invalid_labels_and_lengths_do_not_produce_metrics(self):
        for truth, predicted in (
            ([-1, 0], [0, 0]),
            ([0.8, 1.2], [0, 1]),
            ([0, 1], [0]),
            ([0], [0, 1]),
            ([[0, 1]], [[0, 1]]),
            ([0, 2], [0, 1]),
            ([0, 1], [-1, 0]),
            ([True, False], [0, 1]),
            (["0", "1"], [0, 1]),
            ([np.nan, 1], [0, 1]),
        ):
            with (
                self.subTest(truth=truth, predicted=predicted),
                self.assertRaises(ValueError),
            ):
                classification_metrics(truth, predicted, 2)
        for count in (True, np.bool_(True), 0, -1, 2.5, np.array(2)):
            with self.subTest(n_classes=count), self.assertRaises(ValueError):
                classification_metrics([0], [0], count)


class EmbeddingNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_pair_and_enrollment_scores_preserve_extreme_finite_directions(self):
        base = np.eye(2)[[0, 0, 1, 1]]
        expected = _verification_scores(base, [0, 0, 1, 1], 20, 7)
        for magnitude in (np.nextafter(0.0, 1.0), 1e-300, 1e300, np.finfo(float).max):
            with self.subTest(magnitude=magnitude):
                values = base * magnitude
                labels, scores, context = _verification_scores(
                    values, [0, 0, 1, 1], 20, 7
                )
                np.testing.assert_array_equal(labels, expected[0])
                np.testing.assert_array_equal(scores, expected[1])
                self.assertEqual(context, expected[2])
                labels, scores, _ = enrollment_scores(
                    values, [0, 0, 1, 1], [0, 2], [1, 3]
                )
                np.testing.assert_array_equal(labels, [1, 0, 0, 1])
                np.testing.assert_array_equal(scores, [0, -1, -1, 0])

    def test_complex_embeddings_and_invalid_subject_ids_are_rejected(self):
        values = np.eye(2)[[0, 0, 1, 1]]
        with self.assertRaises(ValueError):
            _verification_scores(values.astype(complex) + 1j, [0, 0, 1, 1], 20, 7)
        with self.assertRaises(ValueError):
            enrollment_scores(values.astype(complex) + 1j, [0, 0, 1, 1], [0, 2], [1, 3])
        for ids in ([-1, -1, 1, 1], [0.0, 0.0, 1.0, 1.0], [True] * 4, 0):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                _verification_pairs(ids, 20, 7)
        with self.assertRaises(ValueError):
            _verification_pairs([0, 0, 1, 1], 20, 7, [-1] * 4)

    def test_non_axis_directions_are_stable_without_mutating_tensors(self):
        base = np.array([[1, 2], [2, 3], [-1, 2], [-2, 3]], dtype=float)
        expected = _verification_scores(base, [0, 0, 1, 1], 20, 7)[1]
        for magnitude in (1e-160, 1e-300, 1e300):
            with self.subTest(magnitude=magnitude):
                values = torch.tensor(base * magnitude, requires_grad=True)
                original = values.detach().clone()
                actual = _verification_scores(values, [0, 0, 1, 1], 20, 7)[1]
                np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
                torch.testing.assert_close(values, original, atol=0, rtol=0)
                self.assertIsNone(values.grad)


def _metric_state():
    return {
        "mean": np.zeros(2),
        "scale": np.ones(2),
        "projection": np.eye(2),
        "training_subject_ids": np.array([3, 7], dtype=np.int64),
    }


class MetricStateValidationTests(unittest.TestCase):
    def test_valid_numeric_archive_roundtrip_owns_its_arrays(self):
        state = _metric_state()
        with io.BytesIO() as buffer:
            np.savez(buffer, **state)
            buffer.seek(0)
            with np.load(buffer, allow_pickle=False) as archive:
                metric = WithinClassMetric.from_state_dict(archive)
        for key, value in metric.state_dict().items():
            np.testing.assert_array_equal(value, state[key])
        metric.mean_[0] = 1
        self.assertEqual(state["mean"][0], 0)

    def test_invalid_numeric_state_shapes_values_and_training_ids_fail(self):
        bad_states = [
            {**_metric_state(), "mean": 0},
            {**_metric_state(), "mean": np.zeros((2, 1)), "scale": np.ones((2, 1))},
            {
                **_metric_state(),
                "mean": [],
                "scale": [],
                "projection": np.empty((0, 0)),
            },
            {**_metric_state(), "scale": [1, 0]},
            {**_metric_state(), "projection": np.ones((2, 1))},
            {**_metric_state(), "projection": np.eye(2).astype(complex)},
            {**_metric_state(), "mean": ["0", "0"]},
            {},
            None,
        ]
        for key in ("mean", "scale", "projection"):
            for value in (np.nan, np.inf):
                state = _metric_state()
                state[key].flat[0] = value
                bad_states.append(state)
        for ids in (
            [],
            [3],
            [3, 3],
            [3.1, 7.9],
            [True, False],
            [-1, 7],
            [[3, 7]],
            np.array([3, 2**63], dtype=np.uint64),
        ):
            bad_states.append({**_metric_state(), "training_subject_ids": ids})
        for state in bad_states:
            with self.subTest(state=state), self.assertRaises(ValueError):
                WithinClassMetric.from_state_dict(state)

    def test_transform_preserves_tiny_and_huge_nonzero_directions(self):
        metric = WithinClassMetric.from_state_dict(_metric_state())
        for magnitude in (np.nextafter(0.0, 1.0), 1e-300, 1e300, np.finfo(float).max):
            with self.subTest(magnitude=magnitude):
                actual = metric.transform([[magnitude, magnitude]])
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, [[2**-0.5, 2**-0.5]], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "no cosine direction"):
            metric.transform([[0, 0]])
        with self.assertRaises(ValueError):
            metric.transform([[1 + 1j, 0]])

    def test_unrepresentable_transform_fails_instead_of_returning_nan(self):
        state = _metric_state()
        state["projection"] *= np.finfo(float).max
        metric = WithinClassMetric.from_state_dict(state)
        with self.assertRaisesRegex(ValueError, "finite"):
            metric.transform([[2, 2]])

    def test_fit_rejects_invalid_ids_complex_features_and_nonfinite_statistics(self):
        values = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=float)
        for ids in ([-1, -1, 7, 7], np.array([3, 3, 2**63, 2**63], dtype=np.uint64)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                WithinClassMetric().fit(values, ids)
        with self.assertRaises(ValueError):
            WithinClassMetric().fit(values.astype(complex), [3, 3, 7, 7])
        with self.assertRaisesRegex(ValueError, "finite"):
            WithinClassMetric("standard").fit(values * 1e300, [3, 3, 7, 7])

    def test_ordinary_transform_arithmetic_is_unchanged(self):
        rng = np.random.default_rng(41)
        values = rng.normal(size=(20, 4))
        metric = WithinClassMetric(shrinkage=0.2).fit(values, np.repeat([3, 7], 10))
        transformed = ((values - metric.mean_) / metric.scale_) @ metric.projection_
        expected = (
            transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
        ).astype(np.float32)
        np.testing.assert_array_equal(metric.transform(values), expected)


if __name__ == "__main__":
    unittest.main()
