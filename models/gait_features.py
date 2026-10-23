"""Compact time/spectral diagnostic baseline; no labels or learned encoder.

Feature scaling and optional PCA must be fitted on optimizer-training subjects
only. Configuration selection belongs on validation, never the locked test set.
"""

from __future__ import annotations

from itertools import pairwise
from numbers import Integral, Real

import numpy as np

_LAG_SECONDS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
_BAND_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf)
FEATURES_PER_CHANNEL = 26


def _rate(sampling_rate):
    if (
        isinstance(sampling_rate, (bool, np.bool_))
        or not isinstance(sampling_rate, Real)
        or not np.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be finite and positive")
    return float(sampling_rate)


def _windows(X):
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1] < 1 or X.shape[2] < 4:
        raise ValueError("X must have shape (N, positive C, T >= 4)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("X must contain finite real sensor values")
    return X


def _extract_batch(x, sampling_rate):
    n, channels, length = x.shape
    groups = channels // 3
    if groups:
        # HAR: body acceleration, gyroscope, total acceleration; WISDM: xyz.
        magnitudes = np.linalg.norm(
            x[:, : 3 * groups].reshape(n, groups, 3, length), axis=2
        )
        x = np.concatenate((x, magnitudes), axis=1)
    mean = x.mean(axis=-1)
    centered = x - mean[..., None]
    stats = [
        mean,
        x.std(axis=-1),
        np.sqrt(np.mean(x * x, axis=-1)),
        x.min(axis=-1),
        x.max(axis=-1),
        *np.quantile(x, (0.1, 0.25, 0.5, 0.75, 0.9), axis=-1),
        np.mean(np.abs(np.diff(x, axis=-1)), axis=-1),
    ]
    for seconds in _LAG_SECONDS:
        lag = min(length - 1, max(1, round(seconds * sampling_rate)))
        left, right = centered[..., :-lag], centered[..., lag:]
        energy = np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1))
        corr = np.divide(
            np.sum(left * right, axis=-1),
            energy,
            out=np.zeros_like(energy),
            where=energy > 1e-24,
        )
        stats.append(np.clip(corr, -1.0, 1.0))
    power = np.abs(np.fft.rfft(centered * np.hanning(length), axis=-1)) ** 2
    power[..., 0] = 0.0  # DC is represented by the time-domain mean.
    frequencies = np.fft.rfftfreq(length, d=1.0 / sampling_rate)
    total = power.sum(axis=-1, keepdims=True)
    relative = np.divide(power, total, out=np.zeros_like(power), where=total > 1e-24)
    for low, high in pairwise(_BAND_EDGES):
        stats.append(
            relative[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1)
        )
    entropy = -np.sum(relative * np.log(np.maximum(relative, 1e-30)), axis=-1)
    entropy /= np.log(max(2, power.shape[-1] - 1))
    dominant = frequencies[np.argmax(power, axis=-1)]
    dominant = np.where(total[..., 0] > 1e-24, dominant, 0.0)
    stats.extend((entropy, dominant, np.sum(relative * frequencies, axis=-1)))
    # Channel-major: the 26 features of each original channel, then xyz norms.
    return np.stack(stats, axis=-1).reshape(n, -1)


def extract_gait_features(X, sampling_rate):
    """Return float32 (N, 26 * (C + floor(C/3))) handcrafted features.

    Input is NumPy or torch (N,C,T). Each adjacent xyz triple contributes its
    vector norm in addition to axis features. Batches of 256 bound temporary FFT
    allocations. No identity, fitted state, normalization or random state is used.
    """
    rate = _rate(sampling_rate)
    X = _windows(X)
    out = np.empty(
        (len(X), FEATURES_PER_CHANNEL * (X.shape[1] + X.shape[1] // 3)),
        dtype=np.float32,
    )
    for start in range(0, len(X), 256):
        out[start : start + 256] = _extract_batch(
            X[start : start + 256].astype(np.float64), rate
        )
    if not np.isfinite(out).all():
        raise ValueError("sensor magnitudes exceed finite feature range")
    return out


class GaitFeatureBaseline:
    """Train-only standardization and optional deterministic, unwhitened PCA.

    fit/transform accept windows, not identities. The caller is responsible for
    supplying only optimizer-training subjects to fit. transform never refits.
    State is JSON-safe and can be frozen before validation/test inference.
    """

    def __init__(self, sampling_rate, *, standardize=True, pca_components=None):
        self.sampling_rate = _rate(sampling_rate)
        if not isinstance(standardize, (bool, np.bool_)):
            raise ValueError("standardize must be boolean")  # noqa: TRY004 -- public validation contract
        self.standardize = bool(standardize)
        if pca_components is not None and (
            isinstance(pca_components, (bool, np.bool_))
            or not isinstance(pca_components, Integral)
            or pca_components < 1
        ):
            raise ValueError("pca_components must be None or a positive integer")
        self.pca_components = None if pca_components is None else int(pca_components)

    def fit(self, X):
        X = _windows(X)
        if not len(X):
            raise ValueError("fit requires nonempty optimizer-training windows")
        f = extract_gait_features(X, self.sampling_rate).astype(np.float64)
        if self.pca_components is not None and self.pca_components > min(
            len(f) - 1, f.shape[1]
        ):
            raise ValueError(
                "pca_components must be <= min(n_training_windows - 1, n_features)"
            )
        self.n_channels_in_ = X.shape[1]
        self.window_size_in_ = X.shape[2]
        self.n_fit_windows_ = len(f)
        self.mean_ = f.mean(axis=0) if self.standardize else np.zeros(f.shape[1])
        scale = f.std(axis=0) if self.standardize else np.ones(f.shape[1])
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        f = (f - self.mean_) / self.scale_
        self.pca_mean_ = np.zeros(f.shape[1])
        self.components_ = None
        if self.pca_components is not None:
            self.pca_mean_ = f.mean(axis=0)
            _, _, vh = np.linalg.svd(f - self.pca_mean_, full_matrices=False)
            self.components_ = vh[: self.pca_components].copy()
            # Resolve the otherwise arbitrary sign for stable serialized state.
            for row in self.components_:
                if row[np.argmax(np.abs(row))] < 0:
                    row *= -1
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise ValueError("fit on optimizer-training windows before transform")
        X = _windows(X)
        if X.shape[1:] != (self.n_channels_in_, self.window_size_in_):
            raise ValueError(
                "transform channels and window size must match fitted training data"
            )
        f = (
            extract_gait_features(X, self.sampling_rate).astype(np.float64) - self.mean_
        ) / self.scale_
        if self.components_ is not None:
            f = (f - self.pca_mean_) @ self.components_.T
        f = f.astype(np.float32)
        if not np.isfinite(f).all():
            raise ValueError("transformed features must be finite")
        return f

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        if not hasattr(self, "mean_"):
            raise ValueError("fit before serializing the baseline")
        return {
            "version": 1,
            "sampling_rate": self.sampling_rate,
            "standardize": self.standardize,
            "pca_components": self.pca_components,
            "n_channels_in": self.n_channels_in_,
            "window_size_in": self.window_size_in_,
            "n_fit_windows": self.n_fit_windows_,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "pca_mean": self.pca_mean_.tolist(),
            "components": None
            if self.components_ is None
            else self.components_.tolist(),
            "fit_scope_requirement": "optimizer-training subjects only; caller-enforced",
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported gait feature state version")
        obj = cls(
            state["sampling_rate"],
            standardize=state["standardize"],
            pca_components=state["pca_components"],
        )
        for key, minimum in (
            ("n_channels_in", 1),
            ("window_size_in", 4),
            ("n_fit_windows", 1),
        ):
            value = state[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"invalid {key}")
            setattr(obj, key + "_", int(value))
        dimension = FEATURES_PER_CHANNEL * (
            obj.n_channels_in_ + obj.n_channels_in_ // 3
        )
        for key in ("mean", "scale", "pca_mean"):
            array = np.asarray(state[key], dtype=np.float64)
            if array.shape != (dimension,) or not np.isfinite(array).all():
                raise ValueError(f"invalid {key} shape/values")
            setattr(obj, key + "_", array.copy())
        if np.any(obj.scale_ <= 0):
            raise ValueError("scales must be positive")
        obj.components_ = None
        if obj.pca_components is not None:
            components = np.asarray(state["components"], dtype=np.float64)
            if (
                components.shape != (obj.pca_components, dimension)
                or not np.isfinite(components).all()
                or obj.pca_components > min(obj.n_fit_windows_ - 1, dimension)
            ):
                raise ValueError("invalid PCA components")
            obj.components_ = components.copy()
        elif state["components"] is not None:
            raise ValueError("unexpected PCA components")
        return obj
