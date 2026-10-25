"""Deterministic, evaluation-only stress tests for aligned (N, 6, T) IMU windows.

Channels are acceleration xyz followed by angular-velocity xyz, in caller/native
units. No data loading, fitting, training hooks, or implicit reference estimation.
See docs/imu_perturbations.md for the exact conventions and scientific limits.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MODEST_STRESS_NAMES",
    "IMUPerturbation",
    "PerturbationResult",
    "apply_imu_perturbation",
    "iter_imu_perturbation_batches",
    "modest_stress_grid",
]

_VERSION = "imu-evaluation-stress-v1"
_MAX_INTEGER = 2**64 - 1
_ORDER = ("time_scale", "rotation", "gain", "bias", "noise", "hold")
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
MODEST_STRESS_NAMES = (
    "clean",
    "rotation15",
    "rotation45",
    "rotation90",
    "gain0.9",
    "gain1.1",
    "noise0.01",
    "noise0.05",
    "hold_every10",
)


def _real(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is outside the supported range") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return 0.0 if result == 0 else result


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool or float")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 1024
        or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)
    ):
        raise ValueError(f"{name} must be nonempty text of at most 1024 UTF-8 bytes")
    return value


def _vector(
    value: Any, name: str, size: int, low: float, high: float
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly {size} real values")
    try:
        if len(value) != size:
            raise ValueError(f"{name} must contain exactly {size} real values")
        return tuple(_real(v, name, low, high) for v in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly {size} real values") from exc


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class IMUPerturbation:
    """Validated and frozen parameters; construct before inspecting DEV/probes.

    gain/noise_fraction have two entries (acceleration, angular velocity).
    bias/reference_std have six entries in channel order, in ``units``.
    reference_std is supplied from TRAIN, never computed here; reference_id is
    the caller's immutable TRAIN-reference provenance identifier.
    """

    name: str = "clean"
    rotation_axis: tuple[float, ...] = (1.0, 1.0, 1.0)
    rotation_degrees: float = 0.0
    gain: tuple[float, ...] = (1.0, 1.0)
    bias: tuple[float, ...] = (0.0,) * 6
    units: tuple[str, ...] = ("caller/native", "caller/native")
    noise_fraction: tuple[float, ...] = (0.0, 0.0)
    reference_std: tuple[float, ...] | None = None
    reference_id: str | None = None
    hold_every: int | None = None
    hold_offset: int | None = None
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        axis = _vector(self.rotation_axis, "rotation_axis", 3, -math.inf, math.inf)
        largest = max(abs(v) for v in axis)
        if largest == 0:
            raise ValueError("rotation_axis must be nonzero, even for identity")
        # Normalize safely even for subnormal or very large finite axes.
        scaled = tuple(v / largest for v in axis)
        norm = math.hypot(*scaled)
        object.__setattr__(self, "rotation_axis", tuple(v / norm for v in scaled))
        object.__setattr__(
            self,
            "rotation_degrees",
            _real(self.rotation_degrees, "rotation_degrees", -180, 180),
        )
        for name, size, low, high in (
            ("gain", 2, 1e-6, 10.0),
            ("bias", 6, -1e12, 1e12),
            ("noise_fraction", 2, 0.0, 1.0),
        ):
            object.__setattr__(
                self, name, _vector(getattr(self, name), name, size, low, high)
            )
        if (
            not isinstance(self.units, (tuple, list, np.ndarray))
            or len(self.units) != 2
        ):
            raise ValueError(
                "units must contain acceleration and angular-velocity labels"
            )
        object.__setattr__(self, "units", tuple(_text(v, "units") for v in self.units))
        if self.reference_std is not None:
            object.__setattr__(
                self,
                "reference_std",
                _vector(self.reference_std, "reference_std", 6, 0.0, 1e12),
            )
            object.__setattr__(
                self, "reference_id", _text(self.reference_id, "TRAIN reference_id")
            )
        elif self.reference_id is not None or any(self.noise_fraction):
            raise ValueError("noise/reference_id requires fixed TRAIN reference_std")
        if self.hold_every is None:
            if self.hold_offset is not None:
                raise ValueError("hold_offset requires hold_every")
        else:
            period = _integer(self.hold_every, "hold_every", 2, 2**31 - 1)
            offset = period - 1 if self.hold_offset is None else self.hold_offset
            object.__setattr__(self, "hold_every", period)
            object.__setattr__(
                self, "hold_offset", _integer(offset, "hold_offset", 1, period - 1)
            )
        object.__setattr__(
            self, "time_scale", _real(self.time_scale, "time_scale", 0.5, 2.0)
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.rotation_degrees == 0
            and self.gain == (1.0, 1.0)
            and not any(self.bias)
            and not any(self.noise_fraction)
            and self.hold_every is None
            and self.time_scale == 1
        )

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON-native parameters, including all conventions."""
        return {
            "version": _VERSION,
            "name": self.name,
            "rotation_axis": list(self.rotation_axis),
            "rotation_degrees": self.rotation_degrees,
            "gain": list(self.gain),
            "bias": list(self.bias),
            "units": list(self.units),
            "noise_fraction": list(self.noise_fraction),
            "reference_std": (
                None if self.reference_std is None else list(self.reference_std)
            ),
            "reference_id": self.reference_id,
            "reference_split": "TRAIN" if self.reference_std is not None else None,
            "hold_every": self.hold_every,
            "hold_offset": self.hold_offset,
            "hold_rule": "replace t=offset+k*every with t-1, all six channels",
            "time_scale": self.time_scale,
            "time_rule": "q=clip(center+time_scale*(t-center),0,T-1); linear interpolation",
            "transform_order": list(_ORDER),
            "channel_order": list(_CHANNELS),
        }

    @property
    def parameter_hash(self) -> str:
        """SHA-256 of canonical JSON parameters; independent of batching."""
        return _hash(self.to_dict())


@dataclass(frozen=True)
class PerturbationResult:
    """Detached output arrays and JSON-native audit metadata (no source hashes)."""

    windows: NDArray
    timestamps: NDArray | None
    metadata: dict[str, Any]


def _rotation(spec: IMUPerturbation) -> NDArray:
    x, y, z = spec.rotation_axis
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    angle = math.radians(spec.rotation_degrees)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _validate_inputs(
    windows: NDArray,
    spec: IMUPerturbation,
    window_keys: Sequence[str | int],
    seed: int,
    batch_size: int,
    timestamps: NDArray | None,
) -> tuple[tuple[str | int, ...], int, int]:
    if not isinstance(spec, IMUPerturbation):
        raise TypeError("spec must be a validated IMUPerturbation")
    if (
        not isinstance(windows, np.ndarray)
        or windows.ndim != 3
        or windows.shape[1] != 6
        or windows.shape[2] < 1
        or windows.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
        raise ValueError(
            "windows must be a float32/float64 NumPy array of shape (N,6,T), T>=1"
        )
    seed = _integer(seed, "seed", 0, _MAX_INTEGER)
    batch_size = _integer(batch_size, "batch_size", 1, 4096)
    if isinstance(window_keys, (str, bytes)):
        raise TypeError("window_keys must contain one stable str/int key per window")
    try:
        if len(window_keys) != len(windows):
            raise ValueError("window_keys length must match N")
        keys = tuple(
            _text(k, "window_key")
            if isinstance(k, str)
            else _integer(k, "window_key", 0, _MAX_INTEGER)
            for k in window_keys
        )
    except TypeError as exc:
        raise ValueError(
            "window_keys must be a sequence of stable str/int keys"
        ) from exc
    if len(set(keys)) != len(keys):
        raise ValueError("window_keys must be unique within this call")
    if timestamps is not None and (
        not isinstance(timestamps, np.ndarray)
        or timestamps.dtype.kind not in "fiu"
        or timestamps.shape
        not in ((windows.shape[2],), (len(windows), windows.shape[2]))
    ):
        raise ValueError("timestamps must be numeric NumPy (T,) or (N,T)")
    return keys, seed, batch_size


def _times(timestamps: NDArray | None, start: int, stop: int) -> NDArray | None:
    if timestamps is None:
        return None
    values = timestamps if timestamps.ndim == 1 else timestamps[start:stop]
    if not np.isfinite(values).all() or not np.all(values[..., 1:] > values[..., :-1]):
        raise ValueError(
            "timestamps must be finite and strictly increasing within each window"
        )
    return values.copy()


def _metadata(
    spec: IMUPerturbation,
    seed: int,
    keys: tuple[str | int, ...],
    shape: tuple,
    dtype: np.dtype,
    start: int,
    stop: int,
    has_timestamps: bool,
) -> dict[str, Any]:
    return {
        "parameters": spec.to_dict(),
        "parameter_hash": spec.parameter_hash,
        "run_hash": _hash({"parameter_hash": spec.parameter_hash, "seed": seed}),
        "seed": seed,
        "rng": "PCG64; seed=SHA256(canonical JSON [version, seed, typed window key])",
        "window_keys": list(keys),
        "row_start": start,
        "row_stop": stop,
        "shape": list(shape),
        "dtype": str(dtype),
        "evaluation_only": True,
        "identity": spec.is_identity,
        "timestamps_supplied": has_timestamps,
        "timestamps_preserved": True,
        "window_order_preserved": True,
        "new_independent_data": False,
    }


def _window(
    source: NDArray, spec: IMUPerturbation, rotation: NDArray, seed: int, key: str | int
) -> NDArray:
    if not np.isfinite(source).all():
        raise ValueError("windows must contain only finite values")
    if spec.is_identity:
        # Bit-exact, including signed zeros; no identity arithmetic.
        return source.copy()
    work = source.astype(np.float64, copy=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if spec.time_scale != 1:
                ticks = np.arange(source.shape[1], dtype=np.float64)
                center = (len(ticks) - 1) / 2
                query = np.clip(
                    center + spec.time_scale * (ticks - center), 0, len(ticks) - 1
                )
                # A convex combination avoids subtraction overflow in np.interp.
                left = np.floor(query).astype(np.intp)
                right = np.minimum(left + 1, len(ticks) - 1)
                weight = query - left
                work = work[:, left] * (1 - weight) + work[:, right] * weight
            if spec.rotation_degrees:
                work[:3] = rotation @ work[:3]
                work[3:] = rotation @ work[3:]
            if spec.gain != (1.0, 1.0):
                work *= np.repeat(spec.gain, 3)[:, None]
            if any(spec.bias):
                work += np.asarray(spec.bias)[:, None]
            if any(spec.noise_fraction):
                digest = _hash(
                    [_VERSION, seed, ["str" if isinstance(key, str) else "int", key]]
                )
                rng = np.random.Generator(np.random.PCG64(int(digest, 16)))
                std = np.asarray(spec.reference_std) * np.repeat(spec.noise_fraction, 3)
                work += rng.standard_normal(work.shape) * std[:, None]
            if spec.hold_every is not None:
                assert spec.hold_offset is not None
                held = np.arange(spec.hold_offset, source.shape[1], spec.hold_every)
                work[:, held] = work[:, held - 1]
            result = work.astype(source.dtype)
    except FloatingPointError as exc:
        raise ValueError(
            "perturbation exceeds the finite output/intermediate dtype range"
        ) from exc
    if not np.isfinite(result).all():
        raise ValueError("perturbation must produce finite outputs")
    return result


def _batches(
    windows: NDArray,
    spec: IMUPerturbation,
    keys: tuple[str | int, ...],
    seed: int,
    batch_size: int,
    timestamps: NDArray | None,
) -> Iterator[PerturbationResult]:
    rotation = _rotation(spec)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        times = _times(timestamps, start, stop)
        output = np.empty((stop - start, 6, windows.shape[2]), dtype=windows.dtype)
        for local, index in enumerate(range(start, stop)):
            output[local] = _window(windows[index], spec, rotation, seed, keys[index])
        yield PerturbationResult(
            output,
            times,
            _metadata(
                spec,
                seed,
                keys[start:stop],
                output.shape,
                output.dtype,
                start,
                stop,
                times is not None,
            ),
        )


def iter_imu_perturbation_batches(
    windows: NDArray,
    spec: IMUPerturbation,
    *,
    window_keys: Sequence[str | int],
    seed: int = 0,
    batch_size: int = 128,
    timestamps: NDArray | None = None,
) -> Iterator[PerturbationResult]:
    """Yield <=batch_size windows, preserving order and caller timestamps.

    Signal workspace is O(batch_size*6*T), even for memmap inputs. Key validation
    uses O(N) scalar metadata. Values/timestamps are checked as batches are read;
    a later invalid batch can raise after earlier batches have been yielded.
    """
    keys, seed, batch_size = _validate_inputs(
        windows, spec, window_keys, seed, batch_size, timestamps
    )
    return _batches(windows, spec, keys, seed, batch_size, timestamps)


def apply_imu_perturbation(
    windows: NDArray,
    spec: IMUPerturbation,
    *,
    window_keys: Sequence[str | int],
    seed: int = 0,
    batch_size: int = 128,
    timestamps: NDArray | None = None,
) -> PerturbationResult:
    """Collect the batch iterator into a detached (N,6,T) array plus audit metadata.

    Output storage is O(N*6*T); use the iterator for bounded output storage.
    Preserve stable keys when passing external batches; never reset keys to 0.
    """
    keys, seed, batch_size = _validate_inputs(
        windows, spec, window_keys, seed, batch_size, timestamps
    )
    output = np.empty(windows.shape, dtype=windows.dtype)
    times = (
        None
        if timestamps is None
        else np.empty(timestamps.shape, dtype=timestamps.dtype)
    )
    if times is not None and times.ndim == 1:
        shared_times = _times(timestamps, 0, len(windows))
        assert shared_times is not None
        times[:] = shared_times
    for batch in _batches(windows, spec, keys, seed, batch_size, timestamps):
        start, stop = batch.metadata["row_start"], batch.metadata["row_stop"]
        output[start:stop] = batch.windows
        if times is not None and times.ndim == 2:
            assert batch.timestamps is not None
            times[start:stop] = batch.timestamps
    return PerturbationResult(
        output,
        times,
        _metadata(
            spec,
            seed,
            keys,
            output.shape,
            output.dtype,
            0,
            len(windows),
            times is not None,
        ),
    )


def modest_stress_grid(
    *,
    reference_std: Sequence[float],
    reference_id: str,
    units: tuple[str, ...] = ("caller/native", "caller/native"),
) -> tuple[IMUPerturbation, ...]:
    """Freeze the named grid from caller-supplied TRAIN statistics before DEV use.

    Reference entries are six fixed per-channel standard deviations in native
    units. No selection, fitting, probe statistics, bias/time stress, or random
    angles are hidden in this grid. Every rotation uses normalized (1,1,1).
    """
    reference = _vector(reference_std, "reference_std", 6, 0.0, 1e12)
    return (
        IMUPerturbation(name="clean", units=units),
        *(
            IMUPerturbation(name=f"rotation{d}", rotation_degrees=d, units=units)
            for d in (15, 45, 90)
        ),
        *(
            IMUPerturbation(name=f"gain{g}", gain=(g, g), units=units)
            for g in (0.9, 1.1)
        ),
        *(
            IMUPerturbation(
                name=f"noise{f}",
                noise_fraction=(f, f),
                reference_std=reference,
                reference_id=reference_id,
                units=units,
            )
            for f in (0.01, 0.05)
        ),
        IMUPerturbation(name="hold_every10", hold_every=10, units=units),
    )
