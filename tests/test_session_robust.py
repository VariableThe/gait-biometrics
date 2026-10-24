"""Tiny synthetic CPU tests; no real signals, Torch, or evaluation fitting."""

import io
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from models.session_robust import (
    NuisanceProjectedMetric,
    extract_lagged_covariance_features,
)


def _rotation(rng):
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    return rotation


class LaggedCovarianceTests(unittest.TestCase):
    def test_dimensions_dtype_empty_and_custom_lags(self):
        for channels in (3, 6, 9):
            for count in (0, 2):
                output = extract_lagged_covariance_features(
                    np.zeros((count, channels, 24))
                )
                self.assertEqual(output.shape, (count, 13 * (channels // 3)))
                self.assertEqual(output.dtype, np.float32)
        self.assertEqual(
            extract_lagged_covariance_features(np.ones((2, 6, 8)), lags=(0, 2)).shape,
            (2, 14),
        )

    def test_common_and_independent_constant_proper_rotations(self):
        rng = np.random.default_rng(72)
        values = rng.normal(size=(2, 9, 48))
        expected = extract_lagged_covariance_features(values)
        for common in (True, False):
            rotated = values.copy()
            shared = _rotation(rng)
            for group in range(3):
                rotation = shared if common else _rotation(rng)
                rotated[:, group * 3 : group * 3 + 3] = (
                    rotation @ values[:, group * 3 : group * 3 + 3]
                )
            np.testing.assert_allclose(
                extract_lagged_covariance_features(rotated),
                expected,
                rtol=2e-6,
                atol=2e-6,
            )

    def test_manual_aligned_covariance_and_raw_units(self):
        values = np.random.default_rng(2).normal(size=(1, 3, 28))
        lags = (0, 2, 5)
        stacked = np.concatenate([values[0, :, lag : lag + 23] for lag in lags])
        covariance = np.cov(stacked)
        trace = np.trace(covariance)
        expected = np.r_[
            np.log(np.maximum(np.linalg.eigvalsh(covariance / trace), 1e-8)),
            np.log(trace),
        ]
        np.testing.assert_allclose(
            extract_lagged_covariance_features(values, lags=lags)[0],
            expected,
            atol=1e-6,
        )
        scaled = extract_lagged_covariance_features(values * 7, lags=lags)[0]
        np.testing.assert_allclose(scaled[:-1], expected[:-1], atol=1e-6)
        self.assertAlmostEqual(
            float(scaled[-1] - expected[-1]), 2 * np.log(7), places=5
        )

    def test_zero_constant_tiny_and_extreme_are_finite_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for value in (0.0, 7.0, 1e308):
                output = extract_lagged_covariance_features(np.full((1, 3, 24), value))
                np.testing.assert_allclose(output, np.log(1e-8), atol=1e-6)
            for scale in (1e-300, 1e300, np.finfo(np.float64).max):
                values = np.tile([1.0, -1.0], 12)[None, None, :] * scale
                output = extract_lagged_covariance_features(
                    np.repeat(values, 3, axis=1)
                )
                self.assertTrue(np.isfinite(output).all())
            tiny = np.random.default_rng(7).normal(size=(1, 3, 24)) * 1e-5
            covariance = np.cov(
                np.concatenate([tiny[0, :, lag : lag + 4] for lag in (0, 5, 10, 20)])
            )
            expected = np.r_[
                np.log(
                    np.maximum(
                        np.linalg.eigvalsh(
                            covariance / max(np.trace(covariance), 1e-8)
                        ),
                        1e-8,
                    )
                ),
                np.log(max(np.trace(covariance), 1e-8)),
            ]
            np.testing.assert_allclose(
                extract_lagged_covariance_features(tiny)[0], expected, atol=2e-6
            )

    def test_strict_validation(self):
        for values in (
            np.zeros((1, 4, 24)),
            np.zeros((1, 3, 21)),
            np.zeros((3, 24)),
            np.ones((1, 3, 24), dtype=bool),
            np.ones((1, 3, 24), dtype=complex),
            np.full((1, 3, 24), np.nan),
            np.full((1, 3, 24), np.inf),
            np.full((1, 3, 24), "1"),
        ):
            with (
                self.subTest(shape=values.shape, dtype=values.dtype),
                self.assertRaises(ValueError),
            ):
                extract_lagged_covariance_features(values)
        for lags in ((), (0, 0), (-1, 0), (0, True), (0, 2.0), 2, (0, 100)):
            with self.subTest(lags=lags), self.assertRaises(ValueError):
                extract_lagged_covariance_features(np.zeros((1, 3, 24)), lags=lags)


class NuisanceProjectedMetricTests(unittest.TestCase):
    def setUp(self):
        self.subjects = np.repeat([3, 7, 11], 8)
        self.days = np.tile(np.repeat([1, 2], 4), 3)
        rng = np.random.default_rng(10)
        self.values = rng.normal(size=(24, 6))
        self.values[:, 0] += 3 * (self.days == 2)
        self.values[:, 1] += np.repeat([0.0, 2.0, -2.0], 8) * (self.days == 2)

    def test_manual_nuisance_removal_and_existing_wccn(self):
        from models.gait_metric import WithinClassMetric

        metric = NuisanceProjectedMetric(shrinkage=0.2).fit(
            self.values, self.subjects, self.days
        )
        mean = self.values.mean(axis=0)
        scale = self.values.std(axis=0).clip(min=1e-6)
        standardized = (self.values - mean) / scale
        differences = np.stack(
            [
                standardized[(self.subjects == identity) & (self.days == 2)].mean(
                    axis=0
                )
                - standardized[(self.subjects == identity) & (self.days == 1)].mean(
                    axis=0
                )
                for identity in (3, 7, 11)
            ]
        )
        _, _, right = np.linalg.svd(differences, full_matrices=False)
        projected = standardized - (standardized @ right[:2].T) @ right[:2]
        np.testing.assert_allclose(
            metric.basis_.T @ metric.basis_, right[:2].T @ right[:2], atol=1e-12
        )
        np.testing.assert_allclose(projected @ metric.basis_.T, 0, atol=1e-12)
        expected = (
            WithinClassMetric("wccn", 0.2)
            .fit(projected, self.subjects)
            .transform(projected)
        )
        np.testing.assert_allclose(metric.transform(self.values), expected, atol=1e-6)
        self.assertEqual(metric.training_subject_ids_, [3, 7, 11])
        np.testing.assert_array_equal(metric.training_day_counts_, [[4, 4]] * 3)

    def test_safe_npz_roundtrip_no_refit_or_state_mutation(self):
        from models.gait_metric import WithinClassMetric

        metric = NuisanceProjectedMetric().fit(self.values, self.subjects, self.days)
        state = metric.state_dict()
        self.assertTrue(all(value.dtype.kind in "fiu" for value in state.values()))
        archive = io.BytesIO()
        np.savez(archive, **state)
        archive.seek(0)
        shifted = self.values[:2] + 2
        expected = metric.transform(shifted)
        with (
            patch.object(
                WithinClassMetric,
                "fit",
                side_effect=AssertionError("no inference fitting"),
            ),
            patch.object(
                NuisanceProjectedMetric,
                "fit",
                side_effect=AssertionError("no inference fitting"),
            ),
        ):
            with np.load(archive, allow_pickle=False) as saved:
                restored = NuisanceProjectedMetric.from_state_dict(saved)
            np.testing.assert_array_equal(restored.transform(shifted), expected)
        for key, value in state.items():
            np.testing.assert_array_equal(metric.state_dict()[key], value)
            np.testing.assert_array_equal(restored.state_dict()[key], value)
        restored.mean_[0] += 1
        np.testing.assert_array_equal(metric.mean_, state["mean"])

    def test_missing_days_subjects_and_validation(self):
        with self.assertRaises(ValueError):
            NuisanceProjectedMetric().transform(self.values)
        for subjects, days in (
            (np.ones(24, dtype=int), self.days),
            (self.subjects, np.ones(24, dtype=int)),
            (self.subjects, np.where(self.days == 2, 3, 1)),
            (self.subjects.astype(float), self.days),
            (self.subjects, self.days.astype(float)),
            (self.subjects[:-1], self.days),
        ):
            with self.assertRaises(ValueError):
                NuisanceProjectedMetric().fit(self.values, subjects, days)
        missing = self.days.copy()
        missing[self.subjects == 3] = 1
        with self.assertRaises(ValueError):
            NuisanceProjectedMetric().fit(self.values, self.subjects, missing)
        for directions in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                NuisanceProjectedMetric(directions)
        metric = NuisanceProjectedMetric(0, 0.5).fit(
            self.values, self.subjects, self.days
        )
        self.assertEqual(metric.basis_.shape, (0, 6))
        for invalid in (
            np.zeros((1, 7)),
            np.full((1, 6), np.nan),
            np.ones((1, 6), dtype=bool),
        ):
            with self.assertRaises(ValueError):
                metric.transform(invalid)

    def test_zero_difference_rank_and_corrupt_states(self):
        values = np.tile(
            [[1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, -2.0], [0.0, 1.0, -2.0]],
            (1, 1),
        )
        metric = NuisanceProjectedMetric().fit(values, [3, 3, 7, 7], [1, 2, 1, 2])
        self.assertEqual(metric.basis_.shape, (0, 3))
        self.assertTrue(np.isfinite(metric.transform(values)).all())
        for key, value in (
            ("scale", np.zeros(3)),
            ("basis", np.ones((1, 3))),
            ("training_day_counts", np.zeros((2, 2), dtype=int)),
            ("mean", np.full(3, np.nan)),
            ("training_subject_ids", np.array([3.0, 7.0])),
            ("version", np.array(2)),
        ):
            state = metric.state_dict()
            state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                NuisanceProjectedMetric.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
