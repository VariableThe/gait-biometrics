"""Independent synthetic CPU checks; no datasets, artifacts, fitting, or Torch."""

import itertools
import unittest
import warnings
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from models.gait_canonical import (
    CANONICAL_FEATURE_DIM,
    GRAVITY_REL_TOL,
    HORIZONTAL_ENERGY_REL_TOL,
    HORIZONTAL_GAP_REL_TOL,
    estimate_canonical_frames,
    extract_canonical_features,
)
from models.gait_features import extract_gait_features
from models.gait_invariants import extract_invariant_features


def _window():
    phase = 2 * np.pi * np.arange(256) / 256
    return np.array(
        [
            3 * np.sin(3 * phase) + 0.4 * np.cos(6 * phase),
            0.7 * np.cos(5 * phase) + 0.1 * np.sin(10 * phase),
            9.81 + 0.4 * np.sin(2 * phase),
            10 + 2 * np.cos(3 * phase) + 0.3 * np.sin(8 * phase),
            -30 + 0.7 * np.sin(6 * phase),
            2 + 0.2 * np.cos(4 * phase),
        ]
    )


def _rotation(rng):
    """Rodrigues reference, independent of the frame estimator's PCA."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _rotate(window, rotation):
    return np.vstack((rotation @ window[:3], rotation @ window[3:]))


def _fusion(window, rate=100):
    return np.concatenate(
        (
            extract_gait_features(window[None], rate)[0],
            extract_invariant_features(window[None], rate)[0],
        )
    )


class CanonicalGeometryTests(unittest.TestCase):
    def test_float32_geometry_does_not_overflow_validation_scalar(self):
        values = _window()[None].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = extract_canonical_features(values, 100)
        self.assertTrue(result.valid[0])
        self.assertTrue(np.isfinite(result.features).all())

    def test_known_axes_and_independent_conditioning_reference(self):
        window = _window()
        result = estimate_canonical_frames(window[None])
        result.require_all_valid()
        self.assertEqual(result.reason.tolist(), ["valid"])
        basis = result.frames[0, 0]
        np.testing.assert_allclose(np.abs(basis), np.eye(3), atol=3e-15)
        # Scalar-loop sample covariance reference in the known horizontal plane.
        means = [sum(channel) / 256 for channel in window[:3]]
        rms = np.sqrt(sum(sum(channel * channel) for channel in window[:3]) / 256)
        self.assertAlmostEqual(result.gravity_ratio[0], np.linalg.norm(means) / rms)
        variances = [sum((window[i] - means[i]) ** 2) / 255 for i in (0, 1)]
        self.assertAlmostEqual(
            result.horizontal_energy_ratio[0], sum(variances) / rms**2
        )
        self.assertAlmostEqual(
            result.eigengap_ratio[0], 1 - variances[1] / variances[0]
        )

    def test_random_so3_proper_frames_shared_triads_and_exact_reconstruction(self):
        rng = np.random.default_rng(632)
        source = _window()
        for _ in range(12):
            rotation = _rotation(rng)
            window = _rotate(source, rotation)
            result = estimate_canonical_frames(window[None])
            self.assertTrue(result.valid[0])
            for basis in result.frames[0]:
                np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=7e-15)
                self.assertAlmostEqual(np.linalg.det(basis), 1.0, places=14)
                # Known forward LINE, gravity DIRECTION, and cross product.
                self.assertAlmostEqual(abs(basis[:, 0] @ rotation[:, 0]), 1.0)
                np.testing.assert_allclose(basis[:, 2], rotation[:, 2], atol=3e-15)
                np.testing.assert_allclose(
                    basis[:, 1], np.cross(basis[:, 2], basis[:, 0]), atol=3e-15
                )
                for left in (0, 3):
                    projected = basis.T @ window[left : left + 3]
                    np.testing.assert_allclose(
                        basis @ projected, window[left : left + 3], atol=5e-14
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(projected, axis=0),
                        np.linalg.norm(window[left : left + 3], axis=0),
                        atol=4e-14,
                    )
                # Cross-triad inner products are preserved only if BOTH use
                # this same frame. Gyro's own dominant axis is different.
                np.testing.assert_allclose(
                    np.sum((basis.T @ window[:3]) * (basis.T @ window[3:]), axis=0),
                    np.sum(window[:3] * window[3:], axis=0),
                    atol=2e-13,
                )
            np.testing.assert_array_equal(
                result.frames[0, 1], result.frames[0, 0] * [-1, -1, 1]
            )

    def test_frame_pair_equivariance_as_unordered_set(self):
        rng = np.random.default_rng(109)
        window = _window()
        original = estimate_canonical_frames(window[None])
        for _ in range(16):
            rotation = _rotation(rng)
            rotated = estimate_canonical_frames(_rotate(window, rotation)[None])
            expected = np.einsum("ij,sjk->sik", rotation, original.frames[0])
            errors = [
                np.max(np.abs(rotated.frames[0] - expected[order]))
                for order in ([0, 1], [1, 0])
            ]
            self.assertLess(min(errors), 5e-15)
            np.testing.assert_allclose(
                rotated.gravity_ratio, original.gravity_ratio, atol=1e-15
            )
            np.testing.assert_allclose(
                rotated.horizontal_energy_ratio,
                original.horizontal_energy_ratio,
                atol=2e-15,
            )
            np.testing.assert_allclose(
                rotated.eigengap_ratio, original.eigengap_ratio, atol=2e-15
            )

    def test_fixed_relative_guards_and_scale_independence(self):
        self.assertEqual(GRAVITY_REL_TOL, 1e-6)
        self.assertEqual(HORIZONTAL_ENERGY_REL_TOL, 1e-12)
        self.assertEqual(HORIZONTAL_GAP_REL_TOL, 1e-6)
        phase = 2 * np.pi * np.arange(256) / 256
        windows, expected, reasons = [], [], []
        # Gravity near zero relative to dynamic acceleration.
        for magnitude, valid in ((1e-8, False), (1e-4, True)):
            window = _window()
            window[2] = magnitude
            windows.append(window)
            expected.append(valid)
            reasons.append("valid" if valid else "gravity_unidentifiable")
        # Weak horizontal variation relative to the gravity/DC energy.
        for amplitude, valid in ((1e-8, False), (1e-4, True)):
            window = _window()
            window[0], window[1], window[2] = amplitude * np.sin(phase), 0, 1
            windows.append(window)
            expected.append(valid)
            reasons.append("valid" if valid else "horizontal_unidentifiable")
        # Eigenvalue gap, rather than an arbitrary component sign, decides
        # whether the horizontal line is identifiable.
        for gap, valid in ((1e-8, False), (1e-4, True)):
            window = _window()
            window[0], window[1], window[2] = (
                np.sin(phase),
                np.sqrt(1 - gap) * np.cos(phase),
                1,
            )
            windows.append(window)
            expected.append(valid)
            reasons.append("valid" if valid else "horizontal_degenerate")
        baseline = estimate_canonical_frames(np.array(windows))
        np.testing.assert_array_equal(baseline.valid, expected)
        self.assertEqual(baseline.reason.tolist(), reasons)
        for scale in (1e-200, 1e-30, 1e30, 1e200):
            result = estimate_canonical_frames(np.array(windows) * scale)
            np.testing.assert_array_equal(result.valid, expected)
            np.testing.assert_allclose(
                result.gravity_ratio, baseline.gravity_ratio, atol=1e-15
            )
            np.testing.assert_allclose(
                result.eigengap_ratio, baseline.eigengap_ratio, atol=2e-14
            )


class CanonicalDescriptorTests(unittest.TestCase):
    def test_balanced_spectral_peaks_keep_lowest_frequency_under_rotation(self):
        phase = 2 * np.pi * np.arange(256) / 256
        window = _window()
        window[0] = np.cos(5 * phase) + 1.0000387261797732 * np.cos(29 * phase)
        window[1] = 0.4 * np.sin(9 * phase)
        window[2] = 9.81 + 0.05 * np.sin(3 * phase)
        original = extract_canonical_features(window[None], 100)
        self.assertTrue(original.valid[0])
        self.assertEqual(original.features[0, 24], np.float32(5 * 100 / 256))
        rng = np.random.default_rng(7461)
        for _ in range(64):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= np.linalg.det(rotation)
            rotated = extract_canonical_features(_rotate(window, rotation)[None], 100)
            self.assertTrue(rotated.valid[0])
            self.assertEqual(rotated.features[0, 24], original.features[0, 24])
            np.testing.assert_allclose(
                rotated.features, original.features, atol=3e-6, rtol=3e-6
            )

    def test_manual_native_unit_shared_frame_fusion_pooling(self):
        window = _window()
        result = extract_canonical_features(window[None], 100)
        self.assertEqual(result.features.shape, (1, CANONICAL_FEATURE_DIM))
        self.assertEqual(result.features.dtype, np.float32)
        # The reference uses the known physical axes, with neither estimated
        # eigenvectors nor the implementation's projection/reshape path.
        alternative = window.copy()
        alternative[[0, 1, 3, 4]] *= -1
        expected = (
            (_fusion(window).astype(np.float64) + _fusion(alternative)) / 2
        ).astype(np.float32)
        np.testing.assert_allclose(
            result.require_all_valid()[0], expected, atol=2e-6, rtol=2e-6
        )
        # Means cancel on unoriented horizontal axes; gravity and gyro retain
        # native vertical amplitudes. No g conversion or gyro radians conversion.
        self.assertAlmostEqual(float(result.features[0, 26 * 2]), 9.81, places=5)
        self.assertAlmostEqual(float(result.features[0, 26 * 5]), 2.0, places=6)
        for index in (0, 1, 3, 4):
            self.assertEqual(float(result.features[0, 26 * index]), 0.0)

    def test_random_so3_and_all_proper_signed_axis_permutations(self):
        rng = np.random.default_rng(178)
        window = _window()
        expected = extract_canonical_features(window[None], 100).features
        rotations = [_rotation(rng) for _ in range(12)]
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                rotation = np.eye(3)[list(permutation)] * np.array(signs)[:, None]
                if np.linalg.det(rotation) > 0:
                    rotations.append(rotation)
        self.assertEqual(len(rotations), 36)
        for rotation in rotations:
            result = extract_canonical_features(_rotate(window, rotation)[None], 100)
            np.testing.assert_allclose(result.features, expected, atol=3e-6, rtol=3e-6)

    def test_eigensolver_sign_and_frame_order_do_not_change_pooled_features(self):
        values = np.stack(
            (_window(), _rotate(_window(), _rotation(np.random.default_rng(7))))
        )
        expected = extract_canonical_features(values, 100)
        original_eigh = np.linalg.eigh

        def sign_reversed(matrix):
            eigenvalues, eigenvectors = original_eigh(matrix)
            eigenvectors[..., -1] *= -1
            return eigenvalues, eigenvectors

        with patch("models.gait_canonical.np.linalg.eigh", side_effect=sign_reversed):
            reversed_sign = extract_canonical_features(values, 100)
        np.testing.assert_array_equal(reversed_sign.features, expected.features)
        np.testing.assert_allclose(
            reversed_sign.geometry.frames, expected.geometry.frames[:, ::-1], atol=1e-15
        )
        swapped = replace(
            expected.geometry, frames=expected.geometry.frames[:, ::-1].copy()
        )
        with patch(
            "models.gait_canonical.estimate_canonical_frames", return_value=swapped
        ):
            reversed_order = extract_canonical_features(values, 100)
        np.testing.assert_array_equal(reversed_order.features, expected.features)
        # Pool BEFORE a nonlinear metric: averaging metric outputs is generally
        # different and would define another method.
        first = _fusion(values[0]).astype(np.float64)
        opposite = values[0].copy()
        opposite[[0, 1, 3, 4]] *= -1
        second = _fusion(opposite).astype(np.float64)
        self.assertGreater(
            np.linalg.norm((first**2 + second**2) / 2 - ((first + second) / 2) ** 2),
            1.0,
        )

    def test_no_future_or_neighbor_dependence(self):
        rng = np.random.default_rng(84)
        current = np.stack((_window(), _rotate(_window(), _rotation(rng))))
        original = extract_canonical_features(current, 100)
        later = rng.normal(size=(67, 6, 256))
        later[:, 2] += 9.81
        extended = extract_canonical_features(np.concatenate((current, later)), 100)
        np.testing.assert_array_equal(extended.features[:2], original.features)
        np.testing.assert_array_equal(
            extended.geometry.frames[:2], original.geometry.frames
        )
        changed = current.copy()
        changed[1] = 0
        changed_result = extract_canonical_features(changed, 100)
        np.testing.assert_array_equal(
            changed_result.features[:1], original.features[:1]
        )
        np.testing.assert_array_equal(
            changed_result.geometry.frames[:1], original.geometry.frames[:1]
        )
        np.testing.assert_array_equal(
            extract_canonical_features(current[:1], 100).features, original.features[:1]
        )

    def test_readonly_source_repeatability_batch_boundary_and_bounded_temporaries(self):
        values = np.repeat(_window()[None], 65, axis=0)
        before = values.copy()
        values.setflags(write=False)
        sizes = []

        def record_axis(batch, rate):
            sizes.append(len(batch))
            return extract_gait_features(batch, rate)

        def record_invariant(batch, rate):
            sizes.append(len(batch))
            return extract_invariant_features(batch, rate)

        with (
            patch(
                "models.gait_canonical.extract_gait_features", side_effect=record_axis
            ),
            patch(
                "models.gait_canonical.extract_invariant_features",
                side_effect=record_invariant,
            ),
        ):
            together = extract_canonical_features(values, 100)
        self.assertEqual(sizes, [128, 128, 2, 2])
        separate = np.concatenate(
            (
                extract_canonical_features(values[:31], 100).features,
                extract_canonical_features(values[31:], 100).features,
            )
        )
        np.testing.assert_array_equal(together.features, separate)
        np.testing.assert_array_equal(
            together.features, extract_canonical_features(values, 100).features
        )
        np.testing.assert_array_equal(values, before)

    def test_shared_acceleration_frame_is_not_reestimated_from_gyro(self):
        window = _rotate(_window(), _rotation(np.random.default_rng(99)))
        result = extract_canonical_features(window[None], 100)
        reference = []
        for basis in result.geometry.frames[0]:
            reference.append(
                _fusion(np.vstack((basis.T @ window[:3], basis.T @ window[3:])))
            )
        expected = np.mean(np.array(reference, dtype=np.float64), axis=0).astype(
            np.float32
        )
        np.testing.assert_allclose(result.features[0], expected, atol=2e-6, rtol=2e-6)
        changed_gyro = window.copy()
        changed_gyro[3:] *= 25
        np.testing.assert_array_equal(
            estimate_canonical_frames(changed_gyro[None]).frames, result.geometry.frames
        )
        amplified = extract_canonical_features(changed_gyro[None], 100)
        self.assertAlmostEqual(float(amplified.features[0, 26 * 5]), 50.0, places=4)


class CanonicalValidityTests(unittest.TestCase):
    def test_degenerate_frames_fail_closed_and_preserve_source_trial_coverage(self):
        phase = 2 * np.pi * np.arange(256) / 256
        zero = np.zeros((6, 256))
        constant = zero.copy()
        constant[2] = 9.81
        vertical_only = constant.copy()
        vertical_only[2] += np.sin(phase)
        circle = constant.copy()
        circle[0], circle[1] = np.sin(phase), np.cos(phase)
        no_gravity = _window()
        no_gravity[2] -= 9.81
        windows = np.stack(
            (_window(), zero, constant, circle, no_gravity, vertical_only, _window())
        )
        result = extract_canonical_features(windows, 100)
        np.testing.assert_array_equal(
            result.valid, [True, False, False, False, False, False, True]
        )
        self.assertEqual(len(result.features), len(windows))
        self.assertEqual(
            result.geometry.reason.tolist(),
            [
                "valid",
                "gravity_unidentifiable",
                "horizontal_unidentifiable",
                "horizontal_degenerate",
                "gravity_unidentifiable",
                "horizontal_unidentifiable",
                "valid",
            ],
        )
        self.assertTrue(np.isnan(result.features[~result.valid]).all())
        self.assertTrue(np.isnan(result.geometry.frames[~result.valid]).all())
        self.assertTrue(np.isfinite(result.features[result.valid]).all())
        self.assertEqual(int(result.valid.sum()), 2)
        with self.assertRaisesRegex(ValueError, "5/7 windows"):
            result.require_all_valid()
        with self.assertRaisesRegex(ValueError, "5/7 windows"):
            result.geometry.require_all_valid()
        np.testing.assert_array_equal(result.features[0], result.features[-1])
        # No descriptor is evaluated on an invalid frame, even when the other
        # triad contains strong signals that might otherwise look informative.
        circle[3:] = _window()[3:]
        with (
            patch(
                "models.gait_canonical.extract_gait_features",
                side_effect=AssertionError("invalid frame scored"),
            ),
            patch(
                "models.gait_canonical.extract_invariant_features",
                side_effect=AssertionError("invalid frame scored"),
            ),
        ):
            invalid = extract_canonical_features(circle[None], 100)
        self.assertFalse(invalid.valid[0])
        self.assertTrue(np.isnan(invalid.features).all())

    def test_degeneracy_decisions_are_rotation_equivariant(self):
        rng = np.random.default_rng(23)
        phase = 2 * np.pi * np.arange(256) / 256
        ambiguous = _window()
        ambiguous[0], ambiguous[1] = np.sin(phase), np.cos(phase)
        constant = np.zeros((6, 256))
        constant[2] = 9.81
        for window in (np.zeros((6, 256)), constant, ambiguous):
            baseline = estimate_canonical_frames(window[None])
            self.assertFalse(baseline.valid[0])
            for _ in range(8):
                rotated = estimate_canonical_frames(
                    _rotate(window, _rotation(rng))[None]
                )
                np.testing.assert_array_equal(rotated.valid, baseline.valid)
                np.testing.assert_array_equal(rotated.reason, baseline.reason)
                self.assertTrue(np.isnan(rotated.frames).all())

    def test_empty_shapes_and_integer_inputs(self):
        result = extract_canonical_features(np.empty((0, 6, 256)), 100)
        self.assertEqual(result.require_all_valid().shape, (0, 336))
        self.assertEqual(result.geometry.frames.shape, (0, 2, 3, 3))
        for field in (
            result.valid,
            result.geometry.reason,
            result.geometry.gravity_ratio,
            result.geometry.eigengap_ratio,
        ):
            self.assertEqual(field.shape, (0,))
        values = np.rint(10 * _window())[None].astype(np.int64)
        np.testing.assert_array_equal(
            extract_canonical_features(values, 100).features,
            extract_canonical_features(values.astype(np.float64), 100).features,
        )

    def test_nonfinite_nonnumeric_and_incompatible_shapes_are_rejected(self):
        invalid = [
            np.zeros((6, 256)),
            np.zeros((1, 3, 256)),
            np.zeros((1, 9, 256)),
            np.zeros((1, 6, 255)),
            np.zeros((1, 6, 257)),
            np.zeros((1, 6, 256), dtype=bool),
            np.zeros((1, 6, 256), dtype=complex),
            np.zeros((1, 6, 256), dtype=object),
            np.full((1, 6, 256), "sensor"),
            [[1], [2, 3]],
        ]
        for value in (np.nan, np.inf, -np.inf):
            window = _window()[None]
            window[0, 5, -1] = value
            invalid.append(window)
            # Validation must still inspect gyro when acceleration is invalid.
            window = np.zeros((1, 6, 256))
            window[0, 4, 0] = value
            invalid.append(window)
        late_invalid = np.repeat(_window()[None], 65, axis=0)
        late_invalid[-1, 4, -1] = np.nan
        invalid.append(late_invalid)
        for values in invalid:
            for function in (
                estimate_canonical_frames,
                lambda x: extract_canonical_features(x, 100),
            ):
                with (
                    self.subTest(
                        shape=np.shape(values)
                        if isinstance(values, np.ndarray)
                        else "ragged"
                    ),
                    self.assertRaises(ValueError),
                ):
                    function(values)

    def test_invalid_rates_and_numeric_range_fail_without_runtime_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for rate in (
                True,
                "100",
                None,
                0,
                -1,
                np.nan,
                np.inf,
                10**400,
                1e100,
                np.nextafter(0.0, 1.0),
            ):
                with (
                    self.subTest(rate=rate),
                    self.assertRaisesRegex(ValueError, "sampling_rate"),
                ):
                    extract_canonical_features(_window()[None], rate)
            for scale in (1e40, 1e200, 1e306):
                with (
                    self.subTest(scale=scale),
                    self.assertRaisesRegex(ValueError, "finite"),
                ):
                    extract_canonical_features(_window()[None] * scale, 100)
            tiny = extract_canonical_features(_window()[None] * 1e-200, 100)
            self.assertTrue(tiny.valid[0])
            self.assertTrue(np.isfinite(tiny.features).all())


if __name__ == "__main__":
    unittest.main()
