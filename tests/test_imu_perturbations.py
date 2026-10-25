"""Synthetic CPU tests only: no datasets, artifacts, Torch, training, or fitting."""

import hashlib
import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.imu_perturbations import (
    MODEST_STRESS_NAMES,
    IMUPerturbation,
    apply_imu_perturbation,
    iter_imu_perturbation_batches,
    modest_stress_grid,
)

_DEFAULT_KEYS = object()


class IMUPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(38).normal(size=(7, 6, 31))
        self.keys = tuple(f"subject/recording/window-{i}" for i in range(len(self.x)))

    def apply(self, spec, x=None, keys: Any = _DEFAULT_KEYS, **kwargs):
        return apply_imu_perturbation(
            self.x if x is None else x,
            spec,
            window_keys=self.keys if keys is _DEFAULT_KEYS else keys,
            **kwargs,
        )

    def noisy_spec(self, **kwargs):
        return IMUPerturbation(
            noise_fraction=(0.01, 0.05),
            reference_std=(1, 2, 3, 4, 5, 6),
            reference_id="frozen TRAIN split/channel-std-v1",
            **kwargs,
        )

    def test_identity_is_bit_exact_detached_including_signed_zero(self):
        for dtype, uint in ((np.float32, np.uint32), (np.float64, np.uint64)):
            x = self.x.astype(dtype)[:, :, ::2]
            x[0, 0, :2] = (0.0, -0.0)
            original = x.copy()
            x.flags.writeable = False
            result = self.apply(IMUPerturbation(), x=x, batch_size=2)
            np.testing.assert_array_equal(
                result.windows.view(uint), original.view(uint)
            )
            self.assertEqual(result.windows.dtype, dtype)
            self.assertFalse(np.shares_memory(result.windows, x))
            self.assertTrue(result.metadata["identity"])
            self.assertIsNone(result.timestamps)
            np.testing.assert_array_equal(x, original)

    def test_independent_rotation_reference_norms_dots_cross_products(self):
        for axis, degrees in (
            ((1, 1, 1), 15),
            ((2, -3, 4), -45),
            ((0, 0, 1), 90),
            ((1, 0, 0), 180),
        ):
            with self.subTest(axis=axis, degrees=degrees):
                unit_axis = np.array(axis) / np.linalg.norm(axis)
                reference = Rotation.from_rotvec(unit_axis * np.deg2rad(degrees))
                out = self.apply(
                    IMUPerturbation(rotation_axis=axis, rotation_degrees=degrees)
                ).windows
                for group in (slice(0, 3), slice(3, 6)):
                    expected = reference.apply(
                        self.x[:, group, :].transpose(0, 2, 1).reshape(-1, 3)
                    )
                    expected = expected.reshape(
                        len(self.x), self.x.shape[2], 3
                    ).transpose(0, 2, 1)
                    np.testing.assert_allclose(
                        out[:, group], expected, rtol=2e-14, atol=2e-14
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(out[:, group], axis=1),
                        np.linalg.norm(self.x[:, group], axis=1),
                        rtol=2e-14,
                        atol=2e-14,
                    )
                np.testing.assert_allclose(
                    (out[:, :3] * out[:, 3:]).sum(axis=1),
                    (self.x[:, :3] * self.x[:, 3:]).sum(axis=1),
                    atol=2e-14,
                )
                cross = np.cross(self.x[:, :3], self.x[:, 3:], axis=1)
                reference_cross = np.einsum("ij,njt->nit", reference.as_matrix(), cross)
                np.testing.assert_allclose(
                    np.cross(out[:, :3], out[:, 3:], axis=1),
                    reference_cross,
                    atol=2e-14,
                )
                # Recover the actual matrix from basis vectors, independently of implementation.
                basis = np.concatenate((np.eye(3), np.eye(3)), axis=0)[None]
                matrix = self.apply(
                    IMUPerturbation(rotation_axis=axis, rotation_degrees=degrees),
                    x=basis,
                    keys=["basis"],
                ).windows[0, :3]
                np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=14)

    def test_rotation_equivariance_under_independent_coordinate_change(self):
        change = Rotation.from_quat([0.2, -0.4, 0.1, 0.7]).as_matrix()
        axis = np.array([1.0, 2.0, -3.0])
        rotated = self.apply(
            IMUPerturbation(rotation_axis=tuple(axis), rotation_degrees=45)
        ).windows
        changed_input = self.x.copy()
        changed_output = rotated.copy()
        for group in (slice(0, 3), slice(3, 6)):
            changed_input[:, group] = np.einsum("ij,njt->nit", change, self.x[:, group])
            changed_output[:, group] = np.einsum(
                "ij,njt->nit", change, rotated[:, group]
            )
        equivariant = self.apply(
            IMUPerturbation(rotation_axis=tuple(change @ axis), rotation_degrees=45),
            x=changed_input,
        ).windows
        np.testing.assert_allclose(equivariant, changed_output, atol=3e-14)
        restored = self.apply(
            IMUPerturbation(rotation_axis=tuple(axis), rotation_degrees=-45), x=rotated
        ).windows
        np.testing.assert_allclose(restored, self.x, atol=3e-14)

    def test_axis_extremes_normalize_without_overflow_or_underflow(self):
        expected = self.apply(IMUPerturbation(rotation_degrees=45)).windows
        for magnitude in (1e-320, 1e300):
            out = self.apply(
                IMUPerturbation(rotation_axis=(magnitude,) * 3, rotation_degrees=45)
            ).windows
            np.testing.assert_array_equal(out, expected)

    def test_native_gain_bias_and_declared_transform_order(self):
        bias = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
        spec = IMUPerturbation(
            rotation_axis=(0, 0, 1),
            rotation_degrees=90,
            gain=(0.9, 1.1),
            bias=bias,
            units=("m/s^2", "rad/s"),
        )
        # The analytic right-handed z rotation sends (x,y,z) to (-y,x,z).
        expected = self.x[:, [1, 0, 2, 4, 3, 5], :].copy()
        expected[:, [0, 3]] *= -1
        expected *= np.array([0.9] * 3 + [1.1] * 3)[None, :, None]
        expected += np.array(bias)[None, :, None]
        np.testing.assert_allclose(self.apply(spec).windows, expected, atol=2e-15)
        self.assertEqual(spec.to_dict()["units"], ["m/s^2", "rad/s"])

    def test_noise_uses_fixed_train_channel_scales_without_probe_estimation(self):
        zeros = np.zeros_like(self.x)
        raw_spec = IMUPerturbation(
            noise_fraction=(1, 1), reference_std=(1,) * 6, reference_id="TRAIN"
        )
        raw = self.apply(raw_spec, x=zeros).windows
        scaled = self.apply(self.noisy_spec(), x=zeros).windows
        sigma = np.array([0.01, 0.02, 0.03, 0.2, 0.25, 0.3])
        np.testing.assert_allclose(scaled, raw * sigma[None, :, None], rtol=1e-15)
        # Adding arbitrary evaluation offsets does not change the noise distribution/scale.
        shifted = self.apply(self.noisy_spec(), x=self.x * 17 + 1000).windows
        np.testing.assert_allclose(shifted - (self.x * 17 + 1000), scaled, atol=1.2e-13)
        zero_reference = IMUPerturbation(
            noise_fraction=(1, 1),
            reference_std=(0,) * 6,
            reference_id="TRAIN constant channels",
        )
        np.testing.assert_array_equal(self.apply(zero_reference).windows, self.x)
        long_noise = self.apply(
            raw_spec, x=np.zeros((1, 6, 20000)), keys=["distribution"]
        ).windows[0]
        np.testing.assert_allclose(long_noise.mean(axis=1), 0, atol=0.025)
        np.testing.assert_allclose(long_noise.std(axis=1), 1, atol=0.025)

    def test_streaming_exact_equivalence_for_all_transforms_and_external_batches(self):
        spec = self.noisy_spec(
            rotation_degrees=45,
            gain=(0.9, 1.1),
            bias=(0.1,) * 6,
            hold_every=10,
            time_scale=1.1,
        )
        times = (
            np.arange(self.x.shape[2], dtype=np.float64)[None, :]
            + np.arange(len(self.x))[:, None] * 100
        )
        before = self.x.copy()
        expected = self.apply(spec, timestamps=times, batch_size=7)
        for size in (1, 2, 3, 128):
            actual = self.apply(spec, timestamps=times, batch_size=size)
            np.testing.assert_array_equal(actual.windows, expected.windows)
            self.assertEqual(actual.metadata, expected.metadata)
            pieces = list(
                iter_imu_perturbation_batches(
                    self.x,
                    spec,
                    window_keys=self.keys,
                    timestamps=times,
                    batch_size=size,
                )
            )
            self.assertTrue(all(len(p.windows) <= size for p in pieces))
            np.testing.assert_array_equal(
                np.concatenate([p.windows for p in pieces]), expected.windows
            )
            np.testing.assert_array_equal(
                np.concatenate([p.timestamps for p in pieces]), times
            )
        external = [
            self.apply(
                spec, x=self.x[a:b], keys=self.keys[a:b], timestamps=times[a:b]
            ).windows
            for a, b in ((0, 2), (2, 3), (3, 7))
        ]
        np.testing.assert_array_equal(np.concatenate(external), expected.windows)
        np.testing.assert_array_equal(self.x, before)
        self.assertFalse(np.shares_memory(expected.timestamps, times))
        self.assertFalse(np.shares_memory(expected.windows, self.x))

    def test_seed_and_typed_keys_control_noise_without_global_rng_state(self):
        spec = self.noisy_spec()
        with (
            patch("numpy.random.seed", side_effect=AssertionError("global seed")),
            patch("numpy.random.normal", side_effect=AssertionError("global rng")),
        ):
            baseline = self.apply(spec, seed=89).windows
            np.testing.assert_array_equal(self.apply(spec, seed=89).windows, baseline)
        self.assertFalse(np.array_equal(self.apply(spec, seed=90).windows, baseline))
        changed = self.apply(
            spec, keys=[f"other-{i}" for i in range(len(self.x))], seed=89
        ).windows
        self.assertFalse(np.array_equal(changed, baseline))
        int_key = self.apply(spec, x=self.x[:1], keys=[1]).windows
        str_key = self.apply(spec, x=self.x[:1], keys=["1"]).windows
        self.assertFalse(np.array_equal(int_key, str_key))
        # Selecting a row externally does not change its keyed result.
        np.testing.assert_array_equal(
            self.apply(spec, x=self.x[4:5], keys=self.keys[4:5], seed=89).windows,
            baseline[4:5],
        )

    def test_hold_pattern_does_not_drop_rows_or_timestamps(self):
        x = np.broadcast_to(np.arange(31, dtype=np.float64), (2, 6, 31)).copy()
        x[1] += 100
        times = np.arange(31, dtype=np.int64) * 20
        for period, offset in ((10, None), (3, 1), (2, 1)):
            spec = IMUPerturbation(hold_every=period, hold_offset=offset)
            result = self.apply(spec, x=x, keys=["first", "second"], timestamps=times)
            expected = x.copy()
            indices = list(range(period - 1 if offset is None else offset, 31, period))
            for index in indices:
                expected[:, :, index] = x[:, :, index - 1]
            np.testing.assert_array_equal(result.windows, expected)
            np.testing.assert_array_equal(result.timestamps, times)
            self.assertEqual(result.windows.shape, x.shape)
            self.assertEqual(result.metadata["window_keys"], ["first", "second"])

    def test_time_stress_matches_independent_interpolation_and_keeps_timestamps(self):
        times = np.cumsum(np.linspace(0.01, 0.03, self.x.shape[2]))
        for scale in (0.5, 0.9, 1.1, 2.0):
            spec = IMUPerturbation(time_scale=scale)
            result = self.apply(spec, timestamps=times)
            ticks = np.arange(self.x.shape[2])
            query = np.clip((ticks - 15) * scale + 15, 0, 30)
            expected = np.array(
                [
                    [np.interp(query, ticks, channel) for channel in row]
                    for row in self.x
                ]
            )
            np.testing.assert_allclose(result.windows, expected, atol=1e-15)
            np.testing.assert_array_equal(result.timestamps, times)
        for spec in (
            IMUPerturbation(time_scale=2),
            self.noisy_spec(hold_every=10, rotation_degrees=90),
        ):
            result = self.apply(spec, x=np.ones((1, 6, 1)), keys=["one"])
            self.assertEqual(result.windows.shape, (1, 6, 1))
            self.assertTrue(np.isfinite(result.windows).all())

    def test_metadata_hash_frozen_defensive_copies_and_grid(self):
        reference = [1, 2, 3, 4, 5, 6]
        grid = modest_stress_grid(
            reference_std=reference,
            reference_id="TRAIN manifest v1",
            units=("g", "deg/s"),
        )
        self.assertEqual(tuple(s.name for s in grid), MODEST_STRESS_NAMES)
        self.assertEqual(len({s.parameter_hash for s in grid}), len(grid))
        reference[0] = 999
        self.assertEqual(grid[6].reference_std, (1, 2, 3, 4, 5, 6))
        self.assertEqual(tuple(s.rotation_degrees for s in grid[1:4]), (15, 45, 90))
        self.assertEqual(grid[4].gain, (0.9, 0.9))
        self.assertEqual(grid[5].gain, (1.1, 1.1))
        self.assertEqual(grid[6].noise_fraction, (0.01, 0.01))
        self.assertEqual(grid[7].noise_fraction, (0.05, 0.05))
        self.assertEqual((grid[8].hold_every, grid[8].hold_offset), (10, 9))
        spec = grid[6]
        with self.assertRaises(FrozenInstanceError):
            attribute_name = "name"
            setattr(spec, attribute_name, "changed")
        result = self.apply(spec, seed=12)
        payload = json.dumps(
            spec.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(
            result.metadata["parameter_hash"],
            hashlib.sha256(payload.encode()).hexdigest(),
        )
        self.assertEqual(result.metadata["parameters"]["reference_split"], "TRAIN")
        result.metadata["parameters"]["reference_std"][0] = 100
        assert spec.reference_std is not None
        self.assertEqual(spec.reference_std[0], 1)
        self.assertEqual(json.loads(json.dumps(result.metadata)), result.metadata)
        self.assertEqual(
            self.apply(spec, seed=13).metadata["parameter_hash"], spec.parameter_hash
        )
        self.assertNotEqual(
            self.apply(spec, seed=13).metadata["run_hash"], result.metadata["run_hash"]
        )

    def test_parameter_validation_rejects_bool_fractional_and_extreme_values(self):
        invalid: list[dict[str, Any]] = [
            {"name": ""},
            {"name": True},
            {"name": "bad\nname"},
            {"rotation_axis": (0, 0, 0)},
            {"rotation_axis": (1, True, 0)},
            {"rotation_axis": (1, np.inf, 0)},
            {"rotation_axis": (1, 2)},
            {"rotation_degrees": True},
            {"rotation_degrees": np.bool_(False)},
            {"rotation_degrees": "15"},
            {"rotation_degrees": np.nan},
            {"rotation_degrees": 181},
            {"rotation_degrees": -(10**500)},
            {"gain": (True, 1)},
            {"gain": (0, 1)},
            {"gain": (1, -1)},
            {"gain": (1e100, 1)},
            {"gain": (1,)},
            {"gain": 1.0},
            {"bias": (0, 0, 0, 0, 0, True)},
            {"bias": (1e20,) * 6},
            {"noise_fraction": (True, 0)},
            {"noise_fraction": (-0.1, 0)},
            {"noise_fraction": (1.1, 0)},
            {"noise_fraction": (0.1, 0)},
            {"reference_std": (1,) * 6},
            {"reference_id": "TRAIN"},
            {"reference_std": (1,) * 5, "reference_id": "TRAIN"},
            {"reference_std": (True,) * 6, "reference_id": "TRAIN"},
            {"reference_std": (-1,) * 6, "reference_id": "TRAIN"},
            {"reference_std": (np.inf,) * 6, "reference_id": "TRAIN"},
            {"reference_std": (1,) * 6, "reference_id": " "},
            {"units": None},
            {"units": "native"},
            {"units": ("g", "")},
            {"hold_every": True},
            {"hold_every": 10.0},
            {"hold_every": 1},
            {"hold_every": 2**40},
            {"hold_offset": 1},
            {"hold_every": 10, "hold_offset": 1.5},
            {"hold_every": 10, "hold_offset": False},
            {"hold_every": 10, "hold_offset": 0},
            {"hold_every": 10, "hold_offset": 10},
            {"time_scale": True},
            {"time_scale": 0},
            {"time_scale": 1e200},
        ]
        for kwargs in invalid:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((ValueError, TypeError)),
            ):
                IMUPerturbation(**kwargs)
        with self.assertRaises(TypeError):
            constructor: Any = IMUPerturbation
            constructor(unknown=True)

    def test_call_validation_shapes_dtype_keys_seed_batch_size_timestamps(self):
        spec = IMUPerturbation()
        invalid_arrays = [
            [[[1.0]]],
            np.zeros((2, 3, 10)),
            np.zeros((6, 10)),
            np.zeros((1, 6, 0)),
            np.zeros((1, 6, 3), dtype=int),
            np.zeros((1, 6, 3), dtype=bool),
            np.zeros((1, 6, 3), dtype=complex),
            np.zeros((1, 6, 3), dtype=np.float16),
        ]
        for x in invalid_arrays:
            with (
                self.subTest(shape=getattr(x, "shape", None)),
                self.assertRaises(ValueError),
            ):
                self.apply(spec, x=x)
        for field, values in (
            ("batch_size", (True, 1.5, 0, -1, 4097, 10**100)),
            ("seed", (False, 1.5, -1, 2**64)),
        ):
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self.apply(spec, **{field: value})
        for keys in (
            "abcdefg",
            None,
            [1] * 7,
            [0] * 6,
            [True] + list(range(1, 7)),
            [0.5] + list(range(1, 7)),
            [-1] + list(range(1, 7)),
            [""] + list(range(1, 7)),
        ):
            with self.subTest(keys=keys), self.assertRaises((ValueError, TypeError)):
                self.apply(spec, keys=keys)
        for times in (
            np.zeros(30),
            np.zeros((1, 31)),
            np.zeros(31, dtype=bool),
            np.zeros(31),
            -np.arange(31),
            np.full(31, np.nan),
            np.arange(31, dtype=complex),
        ):
            with self.subTest(times=times.dtype), self.assertRaises(ValueError):
                self.apply(spec, timestamps=times)
        with self.assertRaises(TypeError):
            self.apply({})

    def test_nonfinite_and_overflow_never_modify_inputs(self):
        for value in (np.nan, np.inf, -np.inf):
            x = self.x.copy()
            x[3, 2, 1] = value
            before = x.copy()
            for spec in (IMUPerturbation(), self.noisy_spec()):
                with self.assertRaises(ValueError):
                    self.apply(spec, x=x, batch_size=2)
                np.testing.assert_array_equal(x, before)
        for dtype in (np.float32, np.float64):
            x = np.full((1, 6, 2), np.finfo(dtype).max, dtype=dtype)
            np.testing.assert_array_equal(
                self.apply(IMUPerturbation(), x=x, keys=["extreme"]).windows, x
            )
            with self.assertRaises(ValueError):
                self.apply(IMUPerturbation(gain=(2, 2)), x=x, keys=["extreme"])
            self.assertTrue(np.isfinite(x).all())

    def test_stream_iterator_is_lazy_bounded_and_checks_later_rows_when_read(self):
        x = self.x.copy()
        x[2, 0, 0] = np.nan
        stream = iter_imu_perturbation_batches(
            x, IMUPerturbation(), window_keys=self.keys, batch_size=2
        )
        first = next(stream)
        self.assertEqual(first.windows.shape, (2, 6, 31))
        self.assertEqual(
            (first.metadata["row_start"], first.metadata["row_stop"]), (0, 2)
        )
        with self.assertRaises(ValueError):
            next(stream)
        for n in (0, 1):
            x = np.zeros((n, 6, 1), dtype=np.float32)
            result = self.apply(IMUPerturbation(), x=x, keys=list(range(n)))
            self.assertEqual(result.windows.shape, x.shape)
            self.assertEqual(result.metadata["window_keys"], list(range(n)))
        self.assertEqual(
            list(
                iter_imu_perturbation_batches(
                    np.zeros((0, 6, 2)), IMUPerturbation(), window_keys=[]
                )
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
