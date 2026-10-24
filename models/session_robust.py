"""Exploratory train-only session compensation and lagged triad spectra.

No Torch, sensor-unit conversion, split selection, or evaluation fitting.
See ``docs/session_robust.md`` for definitions and research provenance.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

_FLOOR = 1e-8


def _real_array(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be representable in float64")
    return values


def extract_lagged_covariance_features(X, *, lags=(0, 5, 10, 20)):
    """Return float32 [N, (3*len(lags)+1)*G] for adjacent xyz triads.

    X has shape [N,C,T], C in {3,6,9}. Each lag uses the SAME base times
    0..T-max(lags)-1; at least two base times are required. Covariance is
    channel-centered and divided by (number of base times - 1). Per triad,
    output ascending log eigenvalues of covariance/max(trace,1e-8), floored
    at 1e-8, then log(max(trace,1e-8)). Raw physical units are preserved.
    """
    values = _real_array(X, "X")
    if values.ndim != 3 or values.shape[1] not in (3, 6, 9):
        raise ValueError("X must have shape (N,C,T), C in {3,6,9}")
    try:
        lags = tuple(lags)
    except TypeError as exc:
        raise ValueError("lags must be distinct nonnegative integers") from exc
    if (
        not lags
        or any(
            isinstance(lag, (bool, np.bool_))
            or not isinstance(lag, Integral)
            or lag < 0
            for lag in lags
        )
        or len(set(lags)) != len(lags)
    ):
        raise ValueError("lags must be distinct nonnegative integers")
    length = values.shape[2] - max(lags)
    if length < 2:
        raise ValueError("T-max(lags) must be at least 2")
    width = 3 * len(lags)
    result = np.empty((len(values), (width + 1) * (values.shape[1] // 3)), np.float32)
    log_floor = np.log(_FLOOR)
    # Small batches bound the stacked covariance allocation. Scaling before
    # centering avoids overflow even for finite values near float64's limit.
    for start in range(0, len(values), 128):
        for group in range(values.shape[1] // 3):
            triad = values[start : start + 128, 3 * group : 3 * group + 3]
            stacked = np.concatenate(
                [triad[:, :, lag : lag + length] for lag in lags], axis=1
            )
            magnitude = np.max(np.abs(stacked), axis=(1, 2))
            safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)
            centered = stacked / safe_magnitude[:, None, None]
            # Reference subtraction makes exactly constant channels exactly
            # zero even when their repeated nonbinary mean would round.
            centered -= centered[:, :, :1].copy()
            centered -= centered.mean(axis=2, keepdims=True)
            covariance = (centered @ centered.swapaxes(1, 2)) / (length - 1)
            trace = np.trace(covariance, axis1=1, axis2=2)
            positive = trace > 0
            log_trace = np.full(len(triad), -np.inf)
            log_trace[positive] = np.log(trace[positive]) + 2 * np.log(
                safe_magnitude[positive]
            )
            normalized = covariance / np.where(positive, trace, 1)[:, None, None]
            # cov/max(raw_trace,floor), without ever forming raw_trace.
            normalized *= np.exp(np.minimum(log_trace - log_floor, 0))[:, None, None]
            eigenvalues = np.linalg.eigvalsh(normalized)
            block = np.column_stack(
                (
                    np.log(np.maximum(eigenvalues, _FLOOR)),
                    np.maximum(log_trace, log_floor),
                )
            )
            result[
                start : start + len(triad),
                group * (width + 1) : (group + 1) * (width + 1),
            ] = block
    return result


def _features(values):
    values = _real_array(values, "features")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("features must be a nonempty (N,D) matrix")
    return values


def _integer_vector(values, length, name):
    values = np.asarray(values)
    if values.shape != (length,) or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must have one integer per row")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit in int64")
    return values.astype(np.int64)


class NuisanceProjectedMetric:
    """Train standardization -> paired-day SVD removal -> existing WCCN.

    Supply DEV training identities only, with days exactly 1 and 2 for each
    identity. The API cannot infer split membership; callers enforce it.
    """

    def __init__(self, nuisance_directions=2, shrinkage="auto"):
        if (
            isinstance(nuisance_directions, (bool, np.bool_))
            or not isinstance(nuisance_directions, Integral)
            or nuisance_directions < 0
        ):
            raise ValueError("nuisance_directions must be a nonnegative integer")
        if shrinkage != "auto" and (
            isinstance(shrinkage, (bool, np.bool_))
            or not isinstance(shrinkage, Real)
            or not np.isfinite(shrinkage)
            or not 0 < shrinkage <= 1
        ):
            raise ValueError("shrinkage must be auto or in (0,1]")
        self.nuisance_directions = int(nuisance_directions)
        self.shrinkage = shrinkage

    def fit(self, features, subjects, days):
        from .gait_metric import WithinClassMetric

        values = _features(features)
        subjects = _integer_vector(subjects, len(values), "subjects")
        days = _integer_vector(days, len(values), "days")
        identities = np.unique(subjects)
        if len(identities) < 2 or not np.isin(days, (1, 2)).all():
            raise ValueError("fit requires >=2 identities and days exactly 1 and 2")
        counts = np.array(
            [
                [np.sum((subjects == identity) & (days == day)) for day in (1, 2)]
                for identity in identities
            ],
            dtype=np.int64,
        )
        if (counts == 0).any():
            raise ValueError("every training identity must have both day 1 and day 2")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Bounded coordinates make mean/std safe on large finite inputs.
                magnitude = np.maximum(np.max(np.abs(values), axis=0), 1.0)
                bounded = values / magnitude
                mean = bounded.mean(axis=0) * magnitude
                scale = np.maximum(bounded.std(axis=0) * magnitude, 1e-6)
                standardized = (values / scale) - (mean / scale)
                differences = np.stack(
                    [
                        standardized[(subjects == identity) & (days == 2)].mean(axis=0)
                        - standardized[(subjects == identity) & (days == 1)].mean(
                            axis=0
                        )
                        for identity in identities
                    ]
                )
                _, singular_values, right = np.linalg.svd(
                    differences, full_matrices=False
                )
                tolerance = (
                    (singular_values[0] if singular_values.size else 0)
                    * max(differences.shape)
                    * np.finfo(np.float64).eps
                )
                rank = min(
                    self.nuisance_directions, int(np.sum(singular_values > tolerance))
                )
                basis = right[:rank].copy()
                projected = standardized - (standardized @ basis.T) @ basis
                metric = WithinClassMetric("wccn", self.shrinkage).fit(
                    projected, subjects
                )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("features exceed stable metric fitting range") from exc
        self.mean_, self.scale_, self.basis_, self.metric_ = mean, scale, basis, metric
        self.training_subject_ids_ = identities.tolist()
        self.training_day_counts_ = counts
        self.nuisance_singular_values_ = singular_values
        return self

    def transform(self, features):
        if not hasattr(self, "metric_"):
            raise ValueError("fit on training identities before transforming")
        values = _features(features)
        if values.shape[1] != len(self.mean_):
            raise ValueError("feature dimension differs from fitted training data")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                standardized = values / self.scale_ - self.mean_ / self.scale_
                projected = standardized - (standardized @ self.basis_.T) @ self.basis_
                return self.metric_.transform(projected)
        except FloatingPointError as exc:
            raise ValueError("features exceed stable metric transform range") from exc

    def state_dict(self):
        """Flat numeric arrays, suitable for np.savez / allow_pickle=False."""
        if not hasattr(self, "metric_"):
            raise ValueError("fit before serialization")
        state = {
            "version": np.array(1, dtype=np.int64),
            "nuisance_directions": np.array(self.nuisance_directions, dtype=np.int64),
            "shrinkage": np.array(-1.0 if self.shrinkage == "auto" else self.shrinkage),
            "mean": self.mean_.copy(),
            "scale": self.scale_.copy(),
            "basis": self.basis_.copy(),
            "training_subject_ids": np.asarray(
                self.training_subject_ids_, dtype=np.int64
            ),
            "training_day_counts": self.training_day_counts_.copy(),
            "nuisance_singular_values": self.nuisance_singular_values_.copy(),
            "wccn_shrinkage": np.array(self.metric_.shrinkage_),
        }
        state.update(
            {f"wccn_{key}": value for key, value in self.metric_.state_dict().items()}
        )
        return state

    @classmethod
    def from_state_dict(cls, state):
        """Validate and restore fitted state without refitting any estimator."""
        from .gait_metric import WithinClassMetric

        try:
            arrays = {key: np.asarray(value) for key, value in state.items()}
            if any(
                value.dtype.kind not in "fiu" or not np.isfinite(value).all()
                for value in arrays.values()
            ):
                raise ValueError("state must contain finite numeric arrays")
            for key in (
                "version",
                "nuisance_directions",
                "shrinkage",
                "wccn_shrinkage",
            ):
                if arrays[key].shape != ():
                    raise ValueError("state settings must be scalar")
            if (
                arrays["version"].dtype.kind not in "iu"
                or arrays["version"].item() != 1
                or arrays["nuisance_directions"].dtype.kind not in "iu"
            ):
                raise ValueError("unsupported state version or direction count")
            shrinkage = arrays["shrinkage"].item()
            result = cls(
                arrays["nuisance_directions"].item(),
                "auto" if shrinkage == -1 else shrinkage,
            )
            result.mean_ = _real_array(arrays["mean"], "mean").copy()
            result.scale_ = _real_array(arrays["scale"], "scale").copy()
            result.basis_ = _real_array(arrays["basis"], "basis").copy()
            if (
                result.mean_.ndim != 1
                or result.mean_.size == 0
                or result.scale_.shape != result.mean_.shape
                or (result.scale_ < 1e-6).any()
            ):
                raise ValueError("invalid standardization state")
            dimension = len(result.mean_)
            basis = result.basis_
            if (
                basis.ndim != 2
                or basis.shape[1] != dimension
                or len(basis) > min(result.nuisance_directions, dimension)
                or not np.allclose(
                    basis @ basis.T, np.eye(len(basis)), atol=1e-8, rtol=1e-8
                )
            ):
                raise ValueError("invalid orthonormal nuisance basis")
            ids = arrays["training_subject_ids"]
            if ids.ndim != 1:
                raise ValueError("invalid training IDs")
            ids = _integer_vector(ids, len(ids), "training_subject_ids")
            if len(ids) < 2 or not np.array_equal(ids, np.unique(ids)):
                raise ValueError("training IDs must be sorted, unique, and >=2")
            counts = arrays["training_day_counts"]
            singular_values = arrays["nuisance_singular_values"]
            if (
                counts.dtype.kind not in "iu"
                or counts.shape != (len(ids), 2)
                or (counts <= 0).any()
            ):
                raise ValueError("invalid training day counts")
            if (
                singular_values.shape != (min(len(ids), dimension),)
                or (singular_values < 0).any()
                or (np.diff(singular_values) > 0).any()
            ):
                raise ValueError("invalid nuisance singular values")
            result.training_subject_ids_ = ids.tolist()
            result.training_day_counts_ = counts.copy()
            result.nuisance_singular_values_ = singular_values.astype(np.float64).copy()
            nested = {
                key: arrays[f"wccn_{key}"]
                for key in ("mean", "scale", "projection", "training_subject_ids")
            }
            _integer_vector(
                nested["training_subject_ids"], len(ids), "wccn training IDs"
            )
            result.metric_ = WithinClassMetric.from_state_dict(nested)
            fitted_shrinkage = arrays["wccn_shrinkage"].item()
            if (
                not 0 <= fitted_shrinkage <= 1
                or result.metric_.mean_.shape != (dimension,)
                or result.metric_.training_subject_ids_ != ids.tolist()
            ):
                raise ValueError("inconsistent WCCN state")
            result.metric_.shrinkage = result.shrinkage
            result.metric_.shrinkage_ = float(fitted_shrinkage)
            return result
        except (KeyError, TypeError, OverflowError, IndexError) as exc:
            raise ValueError("invalid nuisance metric state") from exc
