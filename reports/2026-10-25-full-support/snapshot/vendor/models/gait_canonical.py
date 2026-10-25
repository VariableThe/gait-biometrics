"""Window-local gravity/PCA frames and sign-pooled fusion336 descriptors.

This is a fixed-window adaptation of IDNet section 3.3, equations (5)--(13),
not an IDNet reproduction. Both native-unit physical triads share one proper
frame. Unidentifiable windows retain their rows and carry no finite features.
See ``docs/gait_canonical.md`` for assumptions and the coverage contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from models.gait_features import extract_gait_features
from models.gait_invariants import extract_invariant_features

WINDOW_SAMPLES = 256
CANONICAL_FEATURE_DIM = 336
GRAVITY_REL_TOL = 1e-6
HORIZONTAL_ENERGY_REL_TOL = 1e-12
HORIZONTAL_GAP_REL_TOL = 1e-6
SPECTRAL_PEAK_REL_TOL = 64 * np.finfo(np.float64).eps
_BATCH_SIZE = 64
_FLOAT64_MAX = float(np.finfo(np.float64).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class CanonicalFrames:
    """Row-aligned frame pairs and dimensionless conditioning diagnostics.

    ``frames[n,s]`` has columns (forward, lateral, gravity). Its transpose
    projects either triad; multiplication by it reconstructs that triad.
    The two frames differ by diag(-1,-1,1) on the right. Their ORDER has no
    physical meaning. Invalid rows contain NaN frames, never fallback axes.
    Undefined horizontal diagnostics (without gravity) are NaN as well.
    """

    frames: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    gravity_ratio: np.ndarray
    horizontal_energy_ratio: np.ndarray
    eigengap_ratio: np.ndarray

    def require_all_valid(self) -> None:
        """Raise on any missing frame, without selecting or deleting rows."""
        missing = int(np.count_nonzero(~self.valid))
        if missing:
            raise ValueError(
                f"unidentifiable canonical frames: {missing}/{len(self.valid)} windows"
            )


@dataclass(frozen=True)
class CanonicalFeatures:
    """Float32 (N,336) features plus geometry for EVERY source row.

    Invalid feature rows are entirely NaN. Callers must preserve those rows
    as non-evaluable/abstained trials in their original coverage denominator.
    Do not impute NaNs, silently select valid trials, or score fallback frames.
    """

    features: np.ndarray
    geometry: CanonicalFrames

    @property
    def valid(self) -> np.ndarray:
        return self.geometry.valid

    def require_all_valid(self) -> np.ndarray:
        """Return the full feature matrix, or raise if any frame is missing."""
        self.geometry.require_all_valid()
        return self.features


def _windows(X):
    try:
        values = np.asarray(X)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be a finite real numeric (N,6,256) array") from exc
    if values.ndim != 3 or values.shape[1:] != (6, WINDOW_SAMPLES):
        raise ValueError("X must have shape (N,6,256): acceleration xyz, gyroscope xyz")
    if values.dtype.kind not in "fiu":
        raise ValueError("X must contain finite real numeric values")
    return values


def _batch(values):
    # Validate in bounded slices, including gyro even though only acceleration
    # estimates the frame. Check before float64 conversion, including longdouble.
    if not np.isfinite(values).all():
        raise ValueError("X must contain finite real numeric values")
    if (
        values.dtype.kind == "f"
        and values.dtype.itemsize > 8
        and (np.any(values > _FLOAT64_MAX) or np.any(values < -_FLOAT64_MAX))
    ):
        raise ValueError("X must be representable as finite float64 values")
    return values.astype(np.float64)


def _rate(sampling_rate):
    if isinstance(sampling_rate, (bool, np.bool_)) or not isinstance(
        sampling_rate, Real
    ):
        raise ValueError("sampling_rate must be finite and positive")  # noqa: TRY004
    try:
        rate = float(sampling_rate)
    except (ValueError, OverflowError) as exc:
        raise ValueError("sampling_rate must be finite and positive") from exc
    # The inherited descriptor forms 1/rate and emits float32 frequencies.
    if not np.isfinite(rate) or not 1.0 / _FLOAT64_MAX <= rate <= _FLOAT32_MAX:
        raise ValueError(
            "sampling_rate and its reciprocal must fit the descriptor's finite range"
        )
    return rate


def estimate_canonical_frames(X) -> CanonicalFrames:
    """Estimate two proper frames per finite real (N,6,256) native-unit X.

    Gravity is mean acceleration. The dominant eigenvector of centered
    horizontal acceleration covariance determines an unoriented forward line.
    No sign is selected using a sensor-axis component. Both signs are returned
    with lateral = gravity cross forward, so both determinants are +1.

    Validity requires mean-norm / acceleration-RMS > 1e-6, horizontal sample
    covariance trace / acceleration mean-square norm > 1e-12, and horizontal
    (largest - second-largest eigenvalue) / largest > 1e-6. These fixed relative
    numerical guards are not learned quality or authentication thresholds.
    Invalid rows remain in place. Nothing from another window is consulted.
    """
    values = _windows(X)
    count = len(values)
    frames = np.full((count, 2, 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    reason = np.full(count, "gravity_unidentifiable", dtype="<U32")
    gravity_ratio = np.zeros(count, dtype=np.float64)
    horizontal_energy_ratio = np.full(count, np.nan, dtype=np.float64)
    eigengap_ratio = np.full(count, np.nan, dtype=np.float64)

    for start in range(0, count, _BATCH_SIZE):
        acceleration = _batch(values[start : start + _BATCH_SIZE])[:, :3]
        # One scalar numerical scale per window, never per axis. It cancels in
        # all diagnostics and is never used to choose an eigenvector's sign.
        scale = np.max(np.abs(acceleration), axis=(1, 2))
        with np.errstate(under="ignore"):
            unit = acceleration / np.where(scale > 0, scale, 1.0)[:, None, None]
        mean = unit.mean(axis=-1)
        mean_norm = np.linalg.norm(mean, axis=1)
        energy = np.mean(np.sum(unit * unit, axis=1), axis=1)
        ratio = np.divide(
            mean_norm, np.sqrt(energy), out=np.zeros_like(energy), where=energy > 0
        )
        gravity_ratio[start : start + len(unit)] = ratio
        local = np.flatnonzero(ratio > GRAVITY_REL_TOL)
        if not len(local):
            continue
        rows = start + local
        up = mean[local] / mean_norm[local, None]
        centered = unit[local] - mean[local, :, None]
        horizontal = centered - up[:, :, None] * np.sum(
            up[:, :, None] * centered, axis=1, keepdims=True
        )
        covariance = horizontal @ horizontal.swapaxes(1, 2) / (WINDOW_SAMPLES - 1)
        covariance = (covariance + covariance.swapaxes(1, 2)) * 0.5
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("canonical horizontal eigendecomposition failed") from exc
        eigenvalues = np.maximum(eigenvalues, 0.0)
        horizontal_ratio = eigenvalues.sum(axis=1) / energy[local]
        leading = eigenvalues[:, -1]
        gap = np.divide(
            leading - eigenvalues[:, -2],
            leading,
            out=np.zeros_like(leading),
            where=leading > 0,
        )
        horizontal_energy_ratio[rows] = horizontal_ratio
        eigengap_ratio[rows] = gap
        strong = horizontal_ratio > HORIZONTAL_ENERGY_REL_TOL
        separated = gap > HORIZONTAL_GAP_REL_TOL
        reason[rows] = "horizontal_unidentifiable"
        reason[rows[strong]] = "horizontal_degenerate"
        chosen = np.flatnonzero(strong & separated)
        if not len(chosen):
            continue
        up = up[chosen]
        forward = eigenvectors[chosen, :, -1]
        # Reproject and orthogonalize roundoff; never break the PCA sign tie.
        forward = forward - up * np.sum(up * forward, axis=1, keepdims=True)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        lateral = np.cross(up, forward)
        lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
        forward = np.cross(lateral, up)
        basis = np.stack((forward, lateral, up), axis=-1)
        rows = rows[chosen]
        frames[rows, 0] = basis
        frames[rows, 1] = basis * np.array([-1.0, -1.0, 1.0])
        valid[rows] = True
        reason[rows] = "valid"

    return CanonicalFrames(
        frames, valid, reason, gravity_ratio, horizontal_energy_ratio, eigengap_ratio
    )


def _stable_axis_features(values, rate):
    """Keep inherited axis features except for the numerically tied peak rule.

    This local adapter deliberately leaves the historical control unchanged.
    Resolve powers within64 float64 eps of the maximum toward the lowest bin,
    matching the existing invariant descriptor's relative peak convention.
    """
    features = extract_gait_features(values, rate)
    norms = np.linalg.norm(values.reshape(len(values), 2, 3, WINDOW_SAMPLES), axis=2)
    channels = np.concatenate((values, norms), axis=1)
    centered = channels - channels.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(WINDOW_SAMPLES), axis=-1)) ** 2
    power[..., 0] = 0.0
    peak = power.max(axis=-1, keepdims=True)
    relative_peak = np.divide(power, peak, out=np.zeros_like(power), where=peak > 0)
    near_peak = relative_peak >= 1.0 - SPECTRAL_PEAK_REL_TOL
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / rate)
    dominant = frequencies[np.argmax(near_peak, axis=-1)]
    dominant = np.where(power.sum(axis=-1) > 1e-24, dominant, 0.0)
    features[:, 26 * np.arange(8) + 24] = dominant
    return features


def extract_canonical_features(X, sampling_rate) -> CanonicalFeatures:
    """Return row-preserving fusion336 pooled over BOTH canonical sign frames.

    For each valid row, project acceleration and gyro with the SAME frame,
    concatenate axis+norm208 (with tolerance-based spectral peak ties) and
    existing vector-invariant128 descriptors,
    then average the two 336D descriptors in float64 before returning float32.
    Pooling happens before any learned metric; there are no fitted parameters,
    labels, resampling, filtering, unit conversions, or amplitude normalization.

    Invalid frames produce NaN rows and explicit diagnostics. Nonfinite input
    or unrepresentable descriptor values raise ValueError. Output storage is
    O(N); all signal/FFT temporaries hold at most 64 windows times two signs.
    """
    rate = _rate(sampling_rate)
    values = _windows(X)
    geometry = estimate_canonical_frames(values)
    features = np.full((len(values), CANONICAL_FEATURE_DIM), np.nan, dtype=np.float32)
    for start in range(0, len(values), _BATCH_SIZE):
        local = np.flatnonzero(geometry.valid[start : start + _BATCH_SIZE])
        if not len(local):
            continue
        rows = start + local
        physical = _batch(values[rows]).reshape(len(rows), 2, 3, WINDOW_SAMPLES)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                canonical = np.einsum(
                    "nsji,ngjt->nsgit", geometry.frames[rows], physical
                ).reshape(2 * len(rows), 6, WINDOW_SAMPLES)
                if not np.isfinite(canonical).all() or np.any(
                    np.abs(canonical) > _FLOAT32_MAX
                ):
                    raise ValueError(
                        "canonical sensor values exceed finite float32 descriptor range"
                    )
                descriptors = np.concatenate(
                    (
                        _stable_axis_features(canonical, rate),
                        extract_invariant_features(canonical, rate),
                    ),
                    axis=1,
                ).reshape(len(rows), 2, CANONICAL_FEATURE_DIM)
                pooled = descriptors.astype(np.float64).mean(axis=1).astype(np.float32)
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError(
                    "canonical descriptor exceeds finite numeric range"
                ) from exc
        if not np.isfinite(pooled).all():
            raise ValueError(
                "canonical descriptor must be finite for every valid frame"
            )
        features[rows] = pooled
    return CanonicalFeatures(features, geometry)
