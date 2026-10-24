"""Synthetic CPU/NumPy tests; no official signals, Torch, fitting, or GPU."""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.santos import (
    SANTOS_ARCHIVE_PREFIX,
    SANTOS_CSV_HEADER,
    fingerprint_santos_members,
    load_santos_windows,
)
from models.gait_features import extract_gait_features
from models.gait_invariants import (
    FEATURES_PER_VECTOR_GROUP,
    extract_invariant_features,
    invariant_feature_names,
)


def proper_rotation(rng):
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    return rotation


class InvariantFeatureTests(unittest.TestCase):
    def test_dimension_names_dtype_empty_and_short_windows(self):
        for channels in (3, 6, 9):
            names = invariant_feature_names(channels)
            self.assertEqual(len(names), FEATURES_PER_VECTOR_GROUP * channels // 3)
            self.assertEqual(len(set(names)), len(names))
            self.assertEqual(names[0], "group0[0,1,2].norm.mean")
            self.assertEqual(names[63], "group0[0,1,2].norm_spectrum.centroid_hz")
            for count in (0, 2):
                features = extract_invariant_features(
                    np.zeros((count, channels, 4)), 100
                )
                self.assertEqual(features.shape, (count, len(names)))
                self.assertEqual(features.dtype, np.float32)
                self.assertTrue(np.isfinite(features).all())

    def test_zero_and_constant_vector_conventions(self):
        X = np.zeros((2, 3, 128))
        X[1] = np.array([3, 4, 0])[:, None]
        features = extract_invariant_features(X, 100)
        np.testing.assert_array_equal(features[0], np.zeros(64))
        names = invariant_feature_names(3)
        values = dict(
            zip((name.split("].", 1)[1] for name in names), features[1], strict=True)
        )
        for name in ("mean", "rms", "min", "max", "q10", "q25", "q50", "q75", "q90"):
            self.assertAlmostEqual(values[f"norm.{name}"], 5.0)
        self.assertEqual(values["norm.std"], 0.0)
        self.assertEqual(values["mean_vector_norm"], 5.0)
        for name, value in values.items():
            if name.startswith("lag_cosine."):
                self.assertAlmostEqual(value, 1.0)
            elif not name.startswith("norm.") and name != "mean_vector_norm":
                self.assertEqual(value, 0.0, name)

    def test_common_and_independent_proper_rotations(self):
        rng = np.random.default_rng(197)
        for channels in (3, 6, 9):
            X = rng.normal(size=(4, channels, 128))
            X[:, :3] += np.array([0.0, 0.0, 9.81])[None, :, None]
            original = extract_invariant_features(X, 50)
            common = proper_rotation(rng)
            for independent in (False, True):
                rotated = X.copy()
                for start in range(0, channels, 3):
                    rotation = proper_rotation(rng) if independent else common
                    rotated[:, start : start + 3] = np.einsum(
                        "ab,nbt->nat", rotation, X[:, start : start + 3]
                    )
                np.testing.assert_allclose(
                    extract_invariant_features(rotated, 50),
                    original,
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_explicit_nonadjacent_group_order_and_validation(self):
        rng = np.random.default_rng(82)
        X = rng.normal(size=(3, 6, 32))
        groups = ((0, 2, 4), (1, 3, 5))
        interleaved = extract_invariant_features(X, 20, vector_groups=groups)
        contiguous = extract_invariant_features(X[:, [0, 2, 4, 1, 3, 5]], 20)
        np.testing.assert_array_equal(interleaved, contiguous)
        self.assertTrue(
            invariant_feature_names(6, vector_groups=groups)[0].startswith(
                "group0[0,2,4]"
            )
        )
        for groups in (
            (),
            ((0, 1, 2),),
            ((0, 1), (2, 3, 4, 5)),
            ((0, 1, 2), (2, 4, 5)),
            ((0, 1, 2), (3, 4, 6)),
            ((0, 1, 2), (3, 4, True)),
            ((0, 1, 2), (3, 4, 5.0)),
        ):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                extract_invariant_features(X, 20, vector_groups=groups)
        with self.assertRaises(ValueError):
            invariant_feature_names(6, vector_groups=7)

    def test_directional_dynamics_are_not_redundant_with_norm(self):
        length = 128
        phase = 2 * np.pi * np.arange(length) / 32
        X = np.zeros((2, 3, length))
        X[0, 0] = 1
        X[1, 0], X[1, 1] = np.cos(phase), np.sin(phase)
        features = extract_invariant_features(X, 32)
        np.testing.assert_allclose(features[0, :10], features[1, :10], atol=1e-7)
        self.assertGreater(np.linalg.norm(features[0, 10:] - features[1, 10:]), 0.5)
        # These coordinates are related by a time-varying rotation. Invariance
        # to that operation would erase genuine directional motion information.
        dominant = invariant_feature_names(3).index(
            "group0[0,1,2].vector_spectrum.dominant_hz"
        )
        self.assertGreater(features[1, dominant], 0)

    def test_known_vector_frequency_and_covariance_shape(self):
        rate, length, frequency = 64, 256, 4
        phase = 2 * np.pi * frequency * np.arange(length) / rate
        X = np.zeros((1, 3, length))
        X[0, 0], X[0, 1] = np.cos(phase), np.sin(phase)
        features = extract_invariant_features(X, rate)[0]
        values = dict(
            zip(
                (name.split("].", 1)[1] for name in invariant_feature_names(3)),
                features,
                strict=True,
            )
        )
        np.testing.assert_allclose(features[31:34], [0, 0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(values["vector_spectrum.dominant_hz"], frequency)
        self.assertAlmostEqual(
            values["vector_spectrum.centroid_hz"], frequency, places=3
        )
        self.assertEqual(values["norm_spectrum.dominant_hz"], 0.0)
        self.assertGreater(values["vector_spectrum.power_3_5hz"], 0.99)

    def test_batch_independence_repeatability_and_large_finite_values(self):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(257, 3, 16)).astype(np.float32)
        together = extract_invariant_features(X, 20)
        separately = np.concatenate(
            [
                extract_invariant_features(X[:1], 20),
                extract_invariant_features(X[1:], 20),
            ]
        )
        np.testing.assert_array_equal(together, separately)
        np.testing.assert_array_equal(together, extract_invariant_features(X, 20))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            large = extract_invariant_features(X[:2].astype(np.float64) * 1e30, 20)
            tiny = extract_invariant_features(X[:2].astype(np.float64) * 1e-200, 20)
        self.assertTrue(np.isfinite(large).all())
        self.assertTrue(np.isfinite(tiny).all())
        np.testing.assert_allclose(large[:, 31:], together[:2, 31:], atol=1e-6)
        np.testing.assert_allclose(tiny[:, 31:], together[:2, 31:], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "float32"):
            extract_invariant_features(np.full((1, 3, 8), 1e100), 20)

    def test_invalid_input_and_rates_are_rejected(self):
        for rate in (0, -1, True, "100", None, np.nan, np.inf):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                extract_invariant_features(np.zeros((1, 3, 8)), rate)
        for X in (
            np.zeros((3, 8)),
            np.zeros((1, 4, 8)),
            np.zeros((1, 3, 3)),
            np.full((1, 3, 8), np.nan),
            np.full((1, 3, 8), np.inf),
            np.zeros((1, 3, 8), dtype=complex),
            np.zeros((1, 3, 8), dtype=bool),
            np.full((1, 3, 8), "value"),
        ):
            with (
                self.subTest(shape=X.shape, dtype=X.dtype),
                self.assertRaises(ValueError),
            ):
                extract_invariant_features(X, 20)
        for channels in (True, 2, 12, 3.0):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                invariant_feature_names(channels)

    def test_large_finite_rejection_and_rotated_constants_are_warning_free(self):
        rng = np.random.default_rng(71)
        X = np.broadcast_to(
            np.array([1.25, -9.81, 0.5])[None, :, None], (2, 3, 127)
        ).copy()
        rotation = proper_rotation(rng)
        rotated = np.einsum("ab,nbt->nat", rotation, X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            np.testing.assert_allclose(
                extract_invariant_features(X, 100),
                extract_invariant_features(rotated, 100),
                atol=1e-6,
            )
            for value in (1e100, np.finfo(np.float64).max):
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, "float32"),
                ):
                    extract_invariant_features(np.full((1, 3, 8), value), 100)
            with self.assertRaisesRegex(ValueError, "sampling_rate"):
                extract_invariant_features(np.ones((1, 3, 8)), 10**400)

    def test_axis_magnitude_baseline_api_is_unchanged(self):
        X = np.random.default_rng(9).normal(size=(2, 6, 32))
        self.assertEqual(extract_gait_features(X, 20).shape, (2, 26 * 8))


class SantosMultisensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.zip"
        self.member = (
            f"{SANTOS_ARCHIVE_PREFIX}processed_data/user01/"
            "capture_user01_day1_0001_imu_walk.csv"
        )

    def archive(self, rows, extra=()):
        payload = (",".join(SANTOS_CSV_HEADER) + "\n" + "\n".join(rows) + "\n").encode()
        with ZipFile(self.path, "w") as archive:
            archive.writestr(self.member, payload)
            for name, data in extra:
                archive.writestr(name, data)

    def test_opt_in_exact_raw_gyro_default_parity_and_member_provenance(self):
        self.archive([f"{i},1,2,3,{90 + i},{-180 - i},{360 + i}" for i in range(7)])
        default = load_santos_windows(self.path, [1], window_size=3, stride=3)
        accel = load_santos_windows(
            self.path, [1], window_size=3, stride=3, channels="accel"
        )
        six = load_santos_windows(
            self.path, [1], window_size=3, stride=3, channels="accel_gyro"
        )
        self.assertEqual(six["X"].shape, (2, 6, 3))
        self.assertEqual(six["X"].dtype, np.float32)
        np.testing.assert_array_equal(default["X"], accel["X"])
        np.testing.assert_array_equal(default["X"], six["X"][:, :3])
        np.testing.assert_array_equal(six["X"][0, 3:, 0], [90, -180, 360])
        for name in (
            "subjects",
            "days",
            "trials",
            "starts",
            "frames",
            "recording_indices",
        ):
            np.testing.assert_array_equal(default[name], six[name])
        self.assertEqual(default["metadata"], accel["metadata"])
        old_metadata = {
            key: value
            for key, value in default["metadata"].items()
            if key not in ("channel_names", "channel_units")
        }
        self.assertEqual(old_metadata["channels"], ["acc x", "acc y", "acc z"])
        self.assertEqual(old_metadata["units"], "m/s^2")
        for key in ("recordings", "members_sha256", "exclusions", "subject_coverage"):
            self.assertEqual(default["metadata"][key], six["metadata"][key])
        meta = six["metadata"]
        self.assertEqual(meta["channel_names"], list(SANTOS_CSV_HEADER[1:]))
        self.assertEqual(meta["channel_units"], ["m/s^2"] * 3 + ["degrees/sec"] * 3)
        self.assertEqual(meta["omitted_channels"], [])
        self.assertEqual(default["metadata"]["channel_units"], ["m/s^2"] * 3)
        self.assertEqual(
            meta["members_sha256"],
            fingerprint_santos_members(self.path, [1])["members_sha256"],
        )
        json.dumps(meta, allow_nan=False)

    def test_both_modes_retain_invalid_rows_and_continuity_breaks(self):
        self.archive(
            [
                "10,1,2,3,4,5,6",
                "11,1,2,3,4,5,6",
                "12,1,2,3,nan,5,6",
                "15,1,2,3,4,5,6",
                "16,1,2,3,4,5,6",
                "16,1,2,3,4,5,6",
                "17,1,2,3,4,5,6",
            ]
        )
        outputs = [
            load_santos_windows(self.path, [1], window_size=2, stride=1, channels=mode)
            for mode in ("accel", "accel_gyro")
        ]
        for output in outputs:
            np.testing.assert_array_equal(output["starts"], [0, 3, 5])
            self.assertEqual(
                output["metadata"]["recordings"][0]["invalid_row_offsets"], [2]
            )
            self.assertEqual(
                output["metadata"]["recordings"][0]["continuity_breaks"][0]["reason"],
                "duplicate",
            )
        self.assertEqual(
            outputs[0]["metadata"]["recordings"], outputs[1]["metadata"]["recordings"]
        )

    def test_opt_in_empty_shape_and_unselected_payload_guard(self):
        other = self.member.replace("user01", "user03")
        self.archive(["0,1,2,3,4,5,6"], extra=[(other, b"not opened")])
        original_open = ZipFile.open

        def guarded(archive, name, *args, **kwargs):
            self.assertIsInstance(name, ZipInfo)
            self.assertEqual(name.filename, self.member)
            return original_open(archive, name, *args, **kwargs)

        with patch.object(ZipFile, "open", guarded):
            output = load_santos_windows(
                self.path, [1], window_size=4, channels="accel_gyro"
            )
        self.assertEqual(output["X"].shape, (0, 6, 4))
        self.assertEqual(output["frames"].shape, (0, 4))
        for mode in (None, "gyro", "ACCEL", True, ["accel"]):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "channels"),
            ):
                load_santos_windows(self.path, [1], channels=mode)


if __name__ == "__main__":
    unittest.main()
