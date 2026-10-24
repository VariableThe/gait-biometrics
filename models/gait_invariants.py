"""Deterministic triad-wise invariants; no fitting, labels, or learned transform.

Each physical vector triad may undergo its own CONSTANT orthogonal coordinate
change throughout a window. Directional temporal descriptors complement scalar
norms. This does not imply invariance to moving sensors or changed placement.
See ``docs/multisensor_features.md`` for definitions, units, and research limits.
"""

from __future__ import annotations

from itertools import pairwise
from numbers import Integral, Real

import numpy as np

FEATURES_PER_VECTOR_GROUP = 64
LAG_SECONDS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
_BAND_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf)
_STAT_NAMES = ("mean", "std", "rms", "min", "max", "q10", "q25", "q50", "q75", "q90")
_SPECTRAL_NAMES = (
    "power_0_0.5hz",
    "power_0.5_1hz",
    "power_1_2hz",
    "power_2_3hz",
    "power_3_5hz",
    "power_5_inf_hz",
    "entropy",
    "dominant_hz",
    "centroid_hz",
)
_PER_GROUP_NAMES = (
    *(f"norm.{name}" for name in _STAT_NAMES),
    *(f"centered_norm.{name}" for name in _STAT_NAMES),
    *(f"difference_norm.{name}" for name in _STAT_NAMES),
    "mean_vector_norm",
    *(f"covariance_eigen_fraction.{i}" for i in range(3)),
    *(f"lag_cosine.{seconds:g}s" for seconds in LAG_SECONDS),
    *(f"centered_lag_correlation.{seconds:g}s" for seconds in LAG_SECONDS),
    *(f"vector_spectrum.{name}" for name in _SPECTRAL_NAMES),
    *(f"norm_spectrum.{name}" for name in _SPECTRAL_NAMES),
)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _groups(n_channels, vector_groups):
    if (
        isinstance(n_channels, (bool, np.bool_))
        or not isinstance(n_channels, Integral)
        or n_channels not in (3, 6, 9)
    ):
        raise ValueError("n_channels must be 3, 6, or 9 physical vector axes")
    if vector_groups is None:
        return tuple(tuple(range(i, i + 3)) for i in range(0, n_channels, 3))
    try:
        groups = tuple(tuple(group) for group in vector_groups)
    except TypeError as exc:
        raise ValueError(
            "vector_groups must be an ordered partition into index triples"
        ) from exc
    indices = [index for group in groups for index in group]
    if (
        any(len(group) != 3 for group in groups)
        or any(
            isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            for index in indices
        )
        or sorted(indices) != list(range(n_channels))
    ):
        raise ValueError(
            "vector_groups must partition every channel once into index triples"
        )
    return groups


def invariant_feature_names(n_channels, *, vector_groups=None):
    """Return the exact group-major column names for C=3, 6, or 9.

    Explicit groups are ordered triples of channel indices and must partition
    all channels. Names include the indices so reordering is visible in metadata.
    """
    groups = _groups(n_channels, vector_groups)
    return tuple(
        f"group{i}[{','.join(str(index) for index in group)}].{name}"
        for i, group in enumerate(groups)
        for name in _PER_GROUP_NAMES
    )


def _stats(values):
    return [
        values.mean(axis=-1),
        values.std(axis=-1),
        np.sqrt(np.mean(values * values, axis=-1)),
        values.min(axis=-1),
        values.max(axis=-1),
        *np.quantile(values, (0.1, 0.25, 0.5, 0.75, 0.9), axis=-1),
    ]


def _ratio(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def _spectrum(power, frequencies):
    # Normalize before summing: this also handles very small nonzero signals.
    peak = power.max(axis=-1, keepdims=True)
    scaled = _ratio(power, peak)
    total = scaled.sum(axis=-1, keepdims=True)
    relative = _ratio(scaled, total)
    features = [
        relative[:, (frequencies >= low) & (frequencies < high)].sum(axis=-1)
        for low, high in pairwise(_BAND_EDGES)
    ]
    entropy = -np.sum(relative * np.log(np.maximum(relative, 1e-300)), axis=-1)
    entropy /= np.log(max(2, power.shape[-1] - 1))
    # Resolve numerical ties toward the lowest bin under coordinate rotations.
    near_peak = scaled >= 1.0 - 64 * np.finfo(np.float64).eps
    dominant = frequencies[np.argmax(near_peak, axis=-1)]
    dominant = np.where(total[:, 0] > 0, dominant, 0.0)
    features.extend((entropy, dominant, np.sum(relative * frequencies, axis=-1)))
    return features


def _extract_group(vectors, rate):
    _, _, length = vectors.shape
    # Work in float64 with one scale per window/triad, avoiding squared-value
    # overflow. Restore input units only for the 31 amplitude-valued features.
    scale = np.max(np.abs(vectors), axis=(1, 2))
    safe_scale = np.where(scale > 0, scale, 1.0)
    unit = vectors / safe_scale[:, None, None]
    mean = unit.mean(axis=-1, keepdims=True)
    centered = unit - mean
    # Constant vectors can acquire roundoff residuals after a coordinate change.
    centered_energy = np.sum(centered * centered, axis=(1, 2))
    raw_energy = np.sum(unit * unit, axis=(1, 2))
    constant = centered_energy <= (32 * np.finfo(np.float64).eps) ** 2 * raw_energy
    centered[constant] = 0.0
    norm = np.linalg.norm(unit, axis=1)
    centered_norm = np.linalg.norm(centered, axis=1)
    difference_norm = np.linalg.norm(np.diff(unit, axis=-1), axis=1)
    features = [
        *(value * scale for value in _stats(norm)),
        *(value * scale for value in _stats(centered_norm)),
        *(value * scale for value in _stats(difference_norm)),
        np.linalg.norm(mean[:, :, 0], axis=1) * scale,
    ]
    covariance = centered @ centered.transpose(0, 2, 1) / length
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    fractions = _ratio(eigenvalues, eigenvalues.sum(axis=-1, keepdims=True))
    features.extend(fractions.T)  # ascending eigenvalue order
    direction = _ratio(unit, norm[:, None, :])
    lags = [
        max(1, min(length - 1, round(seconds * rate)))
        if seconds * rate < length - 1
        else length - 1
        for seconds in LAG_SECONDS
    ]
    for lag in lags:
        cosine = np.sum(direction[..., :-lag] * direction[..., lag:], axis=1)
        features.append(np.mean(np.clip(cosine, -1.0, 1.0), axis=-1))
    for lag in lags:
        left, right = centered[..., :-lag], centered[..., lag:]
        numerator = np.sum(left * right, axis=(1, 2))
        denominator = np.sqrt(
            np.sum(left * left, axis=(1, 2)) * np.sum(right * right, axis=(1, 2))
        )
        features.append(np.clip(_ratio(numerator, denominator), -1.0, 1.0))
    taper = np.hanning(length)
    vector_power = np.sum(np.abs(np.fft.rfft(centered * taper, axis=-1)) ** 2, axis=1)
    norm_centered = norm - norm.mean(axis=-1, keepdims=True)
    norm_constant = np.sum(norm_centered * norm_centered, axis=-1) <= (
        (32 * np.finfo(np.float64).eps) ** 2 * np.sum(norm * norm, axis=-1)
    )
    norm_centered[norm_constant] = 0.0
    norm_power = np.abs(np.fft.rfft(norm_centered * taper, axis=-1)) ** 2
    vector_power[:, 0] = 0.0
    norm_power[:, 0] = 0.0
    frequencies = np.fft.rfftfreq(length) * rate
    features.extend(_spectrum(vector_power, frequencies))
    features.extend(_spectrum(norm_power, frequencies))
    return np.stack(features, axis=-1)


def extract_invariant_features(X, sampling_rate, *, vector_groups=None):
    """Return float32 [N, 64*G] train-independent vector-triad descriptors.

    X is a finite real array [N,C,T>=4], C in {3,6,9}. Default groups are adjacent
    xyz triples; explicit ordered index triples must partition all channels.
    Each triple must describe ONE physical vector in one consistent unit.
    HAR C9 order is body acceleration, gyro, total acceleration; C6 order is
    acceleration, gyro. No unit conversion, resampling, or fitting is performed.

    Features are invariant to a constant proper rotation per triad, including
    independent triad rotations because no cross-triad products are used. Time-
    varying coordinate rotations, placement, translation, time scaling, and
    changed sensor units are not invariances. Zero-energy ratios are zero.
    Float64 calculations are batched in 256 windows; unrepresentable float32
    features raise ValueError instead of clipping or returning infinities.
    """
    if isinstance(sampling_rate, (bool, np.bool_)) or not isinstance(
        sampling_rate, Real
    ):
        raise ValueError("sampling_rate must be finite and positive")  # noqa: TRY004 -- public validation contract
    try:
        rate = float(sampling_rate)
    except (ValueError, OverflowError) as exc:
        raise ValueError("sampling_rate must be finite and positive") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[2] < 4:
        raise ValueError("X must have shape (N, C in {3,6,9}, T >= 4)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("X must contain finite real sensor values")
    if np.any(X > _FLOAT32_MAX) or np.any(X < -_FLOAT32_MAX):
        raise ValueError("sensor magnitudes exceed finite float32 feature range")
    groups = _groups(X.shape[1], vector_groups)
    out = np.empty((len(X), FEATURES_PER_VECTOR_GROUP * len(groups)), dtype=np.float32)
    for start in range(0, len(X), 256):
        batch = X[start : start + 256].astype(np.float64)
        for group_index, group in enumerate(groups):
            features = _extract_group(batch[:, group, :], rate)
            if not np.isfinite(features).all() or np.any(
                np.abs(features) > _FLOAT32_MAX
            ):
                raise ValueError(
                    "sensor magnitudes or sampling_rate exceed finite float32 feature range"
                )
            left = group_index * FEATURES_PER_VECTOR_GROUP
            out[start : start + len(batch), left : left + FEATURES_PER_VECTOR_GROUP] = (
                features
            )
    return out
