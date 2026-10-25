"""Independent synthetic checks; no datasets, fitted estimators, or artifacts."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from models.lag_covariance import extract_lag_covariance_features
from models.session_robust import extract_lagged_covariance_features


def _manual_covariance(triad, lags, shrinkage=0.2):
    """Small scalar-loop reference independent of the extractor's arithmetic."""
    length = triad.shape[1] - max(lags)
    channels = np.array(
        [
            [triad[axis, time + lag] for time in range(length)]
            for lag in lags
            for axis in range(3)
        ]
    )
    means = [sum(channel) / length for channel in channels]
    covariance = np.array(
        [
            [
                sum(
                    (channels[i, time] - means[i]) * (channels[j, time] - means[j])
                    for time in range(length)
                )
                / (length - 1)
                for j in range(len(channels))
            ]
            for i in range(len(channels))
        ]
    )
    tau = max(sum(covariance[i, i] for i in range(len(channels))), 1e-8)
    matrix = (1 - shrinkage) * covariance / tau
    for i in range(len(channels)):
        matrix[i, i] += shrinkage / len(channels)
    return matrix, np.log(tau)


def _pack(matrix):
    return np.array(
        [
            matrix[i, j] * (1 if i == j else np.sqrt(2))
            for i in range(len(matrix))
            for j in range(i, len(matrix))
        ]
    )


def _unpack(vector, width):
    matrix = np.zeros((width, width))
    offset = 0
    for i in range(width):
        for j in range(i, width):
            matrix[i, j] = matrix[j, i] = vector[offset] / (1 if i == j else np.sqrt(2))
            offset += 1
    return matrix


class LagCovarianceTests(unittest.TestCase):
    def test_dimensions_dtype_and_empty_batches(self):
        for channels in (3, 6, 9):
            for count in (0, 2):
                for logarithm in (False, True):
                    with self.subTest(
                        channels=channels, count=count, logarithm=logarithm
                    ):
                        result = extract_lag_covariance_features(
                            np.zeros((count, channels, 24)), logarithm=logarithm
                        )
                        self.assertEqual(result.shape, (count, 79 * (channels // 3)))
                        self.assertEqual(result.dtype, np.float64)
                        self.assertTrue(np.isfinite(result).all())
        # Integration is just concatenation; the parent study owns it.
        features = extract_lag_covariance_features(
            np.zeros((2, 6, 24)), logarithm=False
        )
        self.assertEqual(
            np.concatenate((np.zeros((2, 336)), features), axis=1).shape, (2, 494)
        )

    def test_manual_aligned_sample_covariance_and_feature_order(self):
        values = np.random.default_rng(821).normal(size=(2, 6, 28))
        for lags, shrinkage in (
            ((0, 5, 10, 20), 0.2),
            ((3, 0, 1), 0.45),
            ((2, 1), 0.2),
            ((0,), 1.0),
        ):
            actual = extract_lag_covariance_features(
                values, logarithm=False, lags=lags, shrinkage=shrinkage
            )
            for window in range(len(values)):
                expected = []
                for group in range(2):
                    matrix, log_trace = _manual_covariance(
                        values[window, 3 * group : 3 * group + 3], lags, shrinkage
                    )
                    expected.extend([*_pack(matrix), log_trace])
                np.testing.assert_allclose(
                    actual[window], expected, rtol=2e-13, atol=2e-14
                )

    def test_diagonal_matrix_log_without_an_eigendecomposition_reference(self):
        # Orthogonal zero-mean rows have variances 4/3, 16/3, and 0.
        values = np.array([[[-1, 1, -1, 1], [-2, -2, 2, 2], [0, 0, 0, 0]]])
        diagonal = 0.8 * np.array([0.2, 0.8, 0.0]) + 0.2 / 3
        result = extract_lag_covariance_features(values, logarithm=True, lags=(0,))[0]
        expected = np.r_[_pack(np.diag(np.log(diagonal))), np.log(20 / 3)]
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_frobenius_inner_products_and_distances(self):
        values = np.random.default_rng(12).normal(size=(2, 3, 36))
        for logarithm in (False, True):
            features = extract_lag_covariance_features(values, logarithm=logarithm)
            matrices = []
            for triad in values:
                matrix, _ = _manual_covariance(triad, (0, 5, 10, 20))
                if logarithm:
                    # Independently diagonalize the already-shrunk S, rather
                    # than reconstructing matrices from extractor coordinates.
                    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                    matrix = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
                matrices.append(matrix)
            first, second = matrices
            np.testing.assert_allclose(
                features[:, :-1], [_pack(matrix) for matrix in matrices], atol=2e-13
            )
            self.assertGreater(np.abs(np.triu(first, 1)).sum(), 0.01)
            self.assertAlmostEqual(
                float(features[0, :-1] @ features[1, :-1]),
                float(np.sum(first * second)),
                places=12,
            )
            self.assertAlmostEqual(
                float(np.sum((features[0, :-1] - features[1, :-1]) ** 2)),
                float(np.sum((first - second) ** 2)),
                places=12,
            )

    def test_logarithm_exponentiates_to_full_symmetric_positive_matrix(self):
        values = np.random.default_rng(63).normal(size=(2, 6, 30))
        full = extract_lag_covariance_features(values, logarithm=False)
        logged = extract_lag_covariance_features(values, logarithm=True)
        for full_row, log_row in zip(full, logged, strict=True):
            for offset in (0, 79):
                matrix = _unpack(full_row[offset : offset + 78], 12)
                log_matrix = _unpack(log_row[offset : offset + 78], 12)
                np.testing.assert_array_equal(log_matrix, log_matrix.T)
                self.assertTrue(np.isfinite(log_matrix).all())
                self.assertGreaterEqual(
                    np.linalg.eigvalsh(matrix).min(), 0.2 / 12 - 1e-14
                )
                eigenvalues, eigenvectors = np.linalg.eigh(log_matrix)
                recovered = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
                np.testing.assert_allclose(recovered, matrix, rtol=2e-13, atol=2e-14)
                self.assertEqual(log_row[offset + 78], full_row[offset + 78])

    def test_constants_rank_deficiency_floor_and_extremes_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for logarithm in (False, True):
                diagonal = np.log(0.2 / 12) if logarithm else 0.2 / 12
                expected = np.r_[_pack(np.eye(12) * diagonal), np.log(1e-8)]
                for constant in (0.0, 0.1, 7.0, 1e308, -np.finfo(np.float64).max):
                    result = extract_lag_covariance_features(
                        np.full((1, 3, 24), constant), logarithm=logarithm
                    )
                    np.testing.assert_allclose(
                        result[0], expected, rtol=1e-14, atol=1e-14
                    )
                # A rank-one alternating signal has an analytically known trace.
                for amplitude in (
                    np.nextafter(0.0, 1.0),
                    1e-300,
                    1e-5,
                    1e150,
                    1e300,
                    np.finfo(np.float64).max,
                ):
                    values = np.tile([amplitude, -amplitude], 12)[None, None, :]
                    values = np.repeat(values, 3, axis=1)
                    result = extract_lag_covariance_features(
                        values, logarithm=logarithm
                    )
                    expected_log_trace = max(
                        np.log(16.0) + 2 * np.log(amplitude), np.log(1e-8)
                    )
                    self.assertTrue(np.isfinite(result).all())
                    self.assertAlmostEqual(
                        float(result[0, -1]), expected_log_trace, places=11
                    )
                    if amplitude == 1e-5:
                        matrix, log_trace = _manual_covariance(
                            values[0], (0, 5, 10, 20)
                        )
                        self.assertAlmostEqual(
                            float(result[0, -1]), log_trace, places=12
                        )
                        if not logarithm:
                            np.testing.assert_allclose(
                                result[0, :-1], _pack(matrix), atol=1e-15
                            )

    def test_integer_inputs_and_extended_float_when_available(self):
        values = np.tile([np.iinfo(np.int64).min, np.iinfo(np.int64).max], 12)
        for logarithm in (False, True):
            result = extract_lag_covariance_features(
                np.tile(values, (1, 3, 1)), logarithm=logarithm
            )
            self.assertTrue(np.isfinite(result).all())
            if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
                amplitude = np.finfo(np.longdouble).max
                extended = np.tile(np.array([amplitude, -amplitude]), 12)
                result = extract_lag_covariance_features(
                    np.tile(extended, (1, 3, 1)), logarithm=logarithm
                )
                self.assertTrue(np.isfinite(result).all())
                expected = float(np.log(np.longdouble(16)) + 2 * np.log(amplitude))
                self.assertAlmostEqual(float(result[0, -1]), expected, places=9)

    def test_tiny_positive_and_unit_shrinkage(self):
        values = np.zeros((1, 3, 24))
        alpha = np.nextafter(0.0, 1.0)
        result = extract_lag_covariance_features(
            values, logarithm=True, shrinkage=alpha
        )
        expected = np.r_[_pack(np.eye(12) * (np.log(alpha) - np.log(12))), np.log(1e-8)]
        np.testing.assert_allclose(result[0], expected, atol=1e-12)
        values = np.random.default_rng(44).normal(size=(1, 3, 24))
        result = extract_lag_covariance_features(values, logarithm=True, shrinkage=1)
        np.testing.assert_allclose(
            result[0, :-1], _pack(np.eye(12) * -np.log(12)), atol=1e-13
        )

    def test_translation_and_separate_physical_units(self):
        values = np.random.default_rng(9).normal(size=(3, 6, 48))
        offsets = np.array([5, -7, 20, 100, -10, -30])[None, :, None]
        for logarithm in (False, True):
            reference = extract_lag_covariance_features(values, logarithm=logarithm)
            translated = extract_lag_covariance_features(
                values + offsets, logarithm=logarithm
            )
            np.testing.assert_allclose(translated, reference, rtol=2e-12, atol=2e-13)
            scaled = values.copy()
            scaled[:, 3:] *= 17
            result = extract_lag_covariance_features(scaled, logarithm=logarithm)
            np.testing.assert_array_equal(result[:, :79], reference[:, :79])
            np.testing.assert_allclose(
                result[:, 79:-1], reference[:, 79:-1], atol=1e-13
            )
            np.testing.assert_allclose(
                result[:, -1] - reference[:, -1], 2 * np.log(17), atol=1e-13
            )
            for group in range(2):
                independent = extract_lag_covariance_features(
                    values[:, 3 * group : 3 * group + 3], logarithm=logarithm
                )
                np.testing.assert_array_equal(
                    reference[:, 79 * group : 79 * (group + 1)], independent
                )

    def test_full_matrix_retains_orientation_while_spectrum_control_does_not(self):
        values = np.random.default_rng(87).normal(size=(2, 3, 48))
        values[:, 0] *= 4
        angle = 0.63
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        rotated = rotation @ values
        for logarithm in (False, True):
            first = extract_lag_covariance_features(values, logarithm=logarithm)
            second = extract_lag_covariance_features(rotated, logarithm=logarithm)
            self.assertGreater(np.linalg.norm(first[:, :-1] - second[:, :-1]), 0.1)
            np.testing.assert_allclose(first[:, -1], second[:, -1], atol=1e-13)
        np.testing.assert_allclose(
            extract_lagged_covariance_features(values),
            extract_lagged_covariance_features(rotated),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_batches_are_bounded_and_match_individual_windows(self):
        values = np.random.default_rng(39).normal(size=(259, 3, 24))
        original = np.concatenate
        for logarithm in (False, True):

            def bounded_stack(arrays, *args, **kwargs):
                self.assertLessEqual(arrays[0].shape[0], 128)
                return original(arrays, *args, **kwargs)

            with patch(
                "models.lag_covariance.np.concatenate", side_effect=bounded_stack
            ):
                result = extract_lag_covariance_features(values, logarithm=logarithm)
            for index in (0, 127, 128, 255, 256, 258):
                independent = extract_lag_covariance_features(
                    values[index : index + 1], logarithm=logarithm
                )
                np.testing.assert_allclose(
                    result[index], independent[0], rtol=2e-13, atol=2e-14
                )
        values[-1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_lag_covariance_features(values, logarithm=False)

    def test_readonly_strided_inputs_are_not_modified(self):
        for dtype in (np.float32, np.float64):
            original = np.random.default_rng(1).normal(size=(3, 6, 64)).astype(dtype)
            before = original.copy()
            values = original[:, :, ::2]
            values.flags.writeable = False
            for logarithm in (False, True):
                extract_lag_covariance_features(values, logarithm=logarithm)
                np.testing.assert_array_equal(original, before)

    def test_numpy_settings_and_integer_lags(self):
        values = np.arange(90, dtype=np.uint64).reshape(1, 3, 30)
        actual = extract_lag_covariance_features(
            values,
            logarithm=np.bool_(True),
            lags=(np.int64(0), np.uint64(20)),
            shrinkage=np.float32(0.2),
        )
        expected = extract_lag_covariance_features(
            values, logarithm=True, lags=(0, 20), shrinkage=float(np.float32(0.2))
        )
        np.testing.assert_array_equal(actual, expected)

    def test_bad_inputs(self):
        invalid = (
            np.zeros((1, 4, 24)),
            np.zeros((1, 3, 21)),
            np.zeros((3, 24)),
            np.zeros((1, 3, 0)),
            np.zeros((1, 3, 24, 1)),
            np.ones((1, 3, 24), dtype=bool),
            np.ones((1, 3, 24), dtype=complex),
            np.ones((1, 3, 24), dtype=object),
            np.full((1, 3, 24), "1"),
            np.full((1, 3, 24), np.nan),
            np.full((1, 3, 24), np.inf),
            np.full((1, 3, 24), -np.inf),
            None,
            [[[1], [2, 3]]],
        )
        for values in invalid:
            with self.subTest(values=type(values)), self.assertRaises(ValueError):
                extract_lag_covariance_features(values, logarithm=False)
        values = np.zeros((1, 3, 24))
        for lags in (
            (),
            (0, 0),
            (-1, 0),
            (0, True),
            (np.bool_(False),),
            (0, 2.0),
            (0, np.nan),
            (0, 1j),
            ("0",),
            (0, []),
            None,
            2,
            (0, 23),
            (0, 10**200),
            (0, np.uint64(2**64 - 1)),
        ):
            with self.subTest(lags=lags), self.assertRaises(ValueError):
                extract_lag_covariance_features(values, logarithm=False, lags=lags)
        for shrinkage in (
            False,
            True,
            np.bool_(True),
            0,
            -0.1,
            1.1,
            np.nan,
            np.inf,
            -np.inf,
            10**200,
            "0.2",
            0.2 + 0j,
            np.array(0.2),
            None,
        ):
            with self.subTest(shrinkage=shrinkage), self.assertRaises(ValueError):
                extract_lag_covariance_features(
                    values, logarithm=True, shrinkage=shrinkage
                )
        for logarithm in (0, 1, "false", None, np.array(True), np.nan):
            with self.subTest(logarithm=logarithm), self.assertRaises(ValueError):
                # Deliberately violate the static contract to test validation.
                extract_lag_covariance_features(values, logarithm=logarithm)  # ty: ignore[invalid-argument-type]
        with self.assertRaises(TypeError):
            extract_lag_covariance_features(values)  # ty: ignore[missing-argument]


if __name__ == "__main__":
    unittest.main()
