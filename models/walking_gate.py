"""CPU walking gate with fixed features and an explicit source/sensor contract.

Inference needs only NumPy and a numeric JSON artifact. Each output uses one
completed window; no adjacent window, future sample, or test-time adaptation.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np

FEATURE_VERSION = "magnitude_statistics_cadence_v1"
FEATURE_NAMES = (
    "magnitude_mean",
    "magnitude_std",
    "magnitude_q10",
    "magnitude_median",
    "magnitude_q90",
    "magnitude_rms",
    "magnitude_skew",
    "magnitude_excess_kurtosis",
    "magnitude_jerk_mean_abs",
    "magnitude_jerk_rms",
    "dynamic_vector_rms",
    "cadence_peak_hz",
    "cadence_power_fraction",
    "low_cadence_power_fraction",
    "high_cadence_power_fraction",
    "spectral_entropy",
    "autocorrelation_peak",
    "autocorrelation_peak_hz",
)
HYPERPARAMETERS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-8,
    "class_weight": "balanced",
    "random_state": 20261024,
    "standardization": "fit-only population std; std<=1e-12 replaced by 1",
    "threads": 1,
}
DEFAULT_BATCH_SIZE = 64


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def numeric_digest(values):
    values = np.ascontiguousarray(values)
    if values.dtype.kind not in "fiu":
        raise ValueError("numeric arrays required")
    head = canonical({"dtype": values.dtype.str, "shape": list(values.shape)})
    return hashlib.sha256(head + values.tobytes()).hexdigest()


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class SourceContract:
    """Exact equality is required, including source, units and channel order.

    ``source`` identifies the acquisition/preprocessing domain, not an arbitrary
    filename. Matching metadata is necessary, not evidence of domain robustness.
    Only three unscaled total-acceleration axes are supported by these features.
    """

    source: str
    sample_rate: float
    channels: tuple[str, str, str]
    units: tuple[str, str, str]
    window_length: int
    acceleration: str = "total_including_gravity"

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source is required")
        if (
            isinstance(self.sample_rate, (bool, np.bool_))
            or not isinstance(self.sample_rate, Real)
            or not np.isfinite(self.sample_rate)
            or self.sample_rate < 10
        ):
            raise ValueError("sample_rate must be finite and >=10 Hz")
        channels, units = tuple(self.channels), tuple(self.units)
        if (
            len(channels) != 3
            or len(set(channels)) != 3
            or any(not isinstance(c, str) or not c for c in channels)
        ):
            raise ValueError("exactly three named acceleration channels required")
        if units not in (("m/s^2",) * 3, ("wisdm_native_10_per_g",) * 3):
            raise ValueError(
                "three matching acceleration units m/s^2 or wisdm_native_10_per_g required"
            )
        length = _positive_integer(self.window_length, "window_length")
        if length < max(16, int(np.ceil(2 * self.sample_rate)) + 1):
            raise ValueError("window_length must cover more than two seconds")
        if self.acceleration != "total_including_gravity":
            raise ValueError("unscaled total acceleration including gravity required")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "sample_rate", float(self.sample_rate))
        object.__setattr__(self, "window_length", length)

    def to_dict(self):
        state = asdict(self)
        state["channels"], state["units"] = list(self.channels), list(self.units)
        return state

    @classmethod
    def from_dict(cls, state):
        if not isinstance(state, dict) or set(state) != set(cls.__dataclass_fields__):
            raise ValueError("complete source contract required")
        return cls(**state)


def require_contract(actual, expected):
    if not isinstance(actual, SourceContract) or actual != expected:
        raise ValueError(
            "incompatible source/sensor contract; no implicit transfer or resampling"
        )


def binary_labels(activity, activity_names):
    """Use the authoritative name 'Walking'; never assume numeric activity 0."""
    names = list(activity_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
        or names.count("Walking") != 1
    ):
        raise ValueError(
            "unique authoritative activity names including 'Walking' required"
        )
    values = np.asarray(activity)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values >= len(names))
    ):
        raise ValueError("activity labels must index authoritative activity_names")
    return (values == names.index("Walking")).astype(np.int64)


def _windows(X, contract):
    X = np.asarray(X)
    if (
        X.ndim != 3
        or X.shape[1:] != (3, contract.window_length)
        or X.dtype.kind not in "fiu"
    ):
        raise ValueError("windows must have numeric shape (N,3,contract.window_length)")
    return X


def _features_batch(X, contract):
    X = np.ascontiguousarray(X, dtype=np.float64)
    if not np.isfinite(X).all() or np.any(np.abs(X) > 1e6):
        raise ValueError("acceleration must be finite and within +/-1e6 source units")
    magnitude = np.sqrt(np.sum(X * X, axis=1))
    mean, std = magnitude.mean(axis=1), magnitude.std(axis=1)
    centered = magnitude - mean[:, None]
    normalized = centered / np.where(std > 1e-12, std, 1)[:, None]
    quantiles = np.quantile(magnitude, (0.1, 0.5, 0.9), axis=1)
    jerk = np.diff(magnitude, axis=1) * contract.sample_rate
    dynamic = X - X.mean(axis=2, keepdims=True)
    # Hann taper only within this completed window; DC excluded from all ratios.
    power = np.abs(np.fft.rfft(centered * np.hanning(X.shape[2]), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(X.shape[2], 1 / contract.sample_rate)
    power[:, 0] = 0
    total = power.sum(axis=1)
    denominator = np.where(total > 1e-12, total, 1)
    band = (frequencies >= 0.5) & (frequencies <= 4)
    peak_hz = frequencies[band][np.argmax(power[:, band], axis=1)]
    peak_hz = np.where(total > 1e-12, peak_hz, 0)
    distribution = power / denominator[:, None]
    entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-300)), axis=1)
    entropy /= np.log(power.shape[1] - 1)
    lags = np.arange(
        max(1, int(np.ceil(contract.sample_rate / 4))),
        int(np.floor(2 * contract.sample_rate)) + 1,
    )
    # Biased autocorrelation divided by zero-lag energy avoids short-lag bias
    # from pairwise re-normalization. Fixed 0.5--4 Hz search, without tuning.
    energy = np.sum(centered * centered, axis=1)
    correlations = np.stack(
        [
            np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
            / np.where(energy > 1e-12, energy, 1)
            for lag in lags
        ],
        axis=1,
    )
    best_lag = np.argmax(correlations, axis=1)
    peak_correlation = correlations[np.arange(len(X)), best_lag]
    lag_hz = np.where(energy > 1e-12, contract.sample_rate / lags[best_lag], 0)

    def fraction(low, high, include_high=False):
        mask = (frequencies >= low) & (
            (frequencies <= high) if include_high else (frequencies < high)
        )
        # Boolean column indexing can produce Fortran-order arrays when N>1.
        # Fix reduction order so a threshold-boundary decision is unchanged
        # when the same window is scored alone or in a larger batch.
        return np.ascontiguousarray(power[:, mask]).sum(axis=1) / denominator

    result = np.column_stack(
        (
            mean,
            std,
            *quantiles,
            np.sqrt(np.mean(magnitude * magnitude, axis=1)),
            np.mean(normalized**3, axis=1),
            np.mean(normalized**4, axis=1) - 3,
            np.mean(np.abs(jerk), axis=1),
            np.sqrt(np.mean(jerk * jerk, axis=1)),
            np.sqrt(np.mean(np.sum(dynamic * dynamic, axis=1), axis=1)),
            peak_hz,
            fraction(0.5, 4, True),
            fraction(0.5, 1.5),
            fraction(1.5, 4, True),
            entropy,
            peak_correlation,
            lag_hz,
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite fixed features")
    return result


def extract_features(X, *, contract, batch_size=DEFAULT_BATCH_SIZE):
    if not isinstance(contract, SourceContract):
        raise TypeError("explicit SourceContract required")
    X = _windows(X, contract)
    batch_size = _positive_integer(batch_size, "batch_size")
    result = np.empty((len(X), len(FEATURE_NAMES)), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        result[start : start + batch_size] = _features_batch(
            X[start : start + batch_size], contract
        )
    return result


def _feature_matrix(values):
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[1] != len(FEATURE_NAMES)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError("finite fixed-feature matrix required")
    return values.astype(np.float64, copy=False)


def _binary(values, size, both=False):
    values = np.asarray(values)
    if (
        values.shape != (size,)
        or values.dtype.kind not in "iu"
        or not np.isin(values, (0, 1)).all()
        or (both and set(values.tolist()) != {0, 1})
    ):
        raise ValueError(
            "aligned binary labels with both classes required"
            if both
            else "aligned binary labels required"
        )
    return values.astype(np.int64, copy=False)


def _ids(values):
    if isinstance(values, (tuple, list)) and not values:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 1):
        raise ValueError("positive integer real subject IDs required")
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class WalkingGate:
    contract: SourceContract
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float
    fit_provenance: dict

    def __post_init__(self):
        if not isinstance(self.contract, SourceContract):
            raise TypeError("source contract required")
        for key in ("mean", "scale", "coefficients"):
            raw = np.asarray(getattr(self, key))
            if raw.dtype.kind not in "fiu":
                raise ValueError(f"{key} must contain numbers")
            array = np.array(raw, dtype=np.float64, copy=True)
            if array.shape != (len(FEATURE_NAMES),) or not np.isfinite(array).all():
                raise ValueError(f"invalid {key} dimensions/values")
            array.flags.writeable = False
            object.__setattr__(self, key, array)
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")
        for key in ("intercept", "threshold"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"finite numeric {key} required")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        provenance = json.loads(canonical(self.fit_provenance))
        required = {
            "fit_subject_ids",
            "forbidden_subject_ids",
            "n_fit_windows",
            "features_sha256",
            "labels_sha256",
            "fit_calls",
        }
        if not isinstance(provenance, dict) or set(provenance) != required:
            raise ValueError("complete fit provenance required")
        fit_ids, forbidden = (
            _ids(provenance["fit_subject_ids"]),
            _ids(provenance["forbidden_subject_ids"]),
        )
        if (
            not len(fit_ids)
            or set(fit_ids) & set(forbidden)
            or len(set(fit_ids)) != len(fit_ids)
            or len(set(forbidden)) != len(forbidden)
        ):
            raise ValueError("fit and forbidden subjects must be unique and disjoint")
        _positive_integer(provenance["n_fit_windows"], "n_fit_windows")
        if type(provenance["fit_calls"]) is not int or provenance["fit_calls"] != 1:
            raise ValueError("exactly one fit required")
        for name in ("features_sha256", "labels_sha256"):
            value = provenance[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("invalid fit provenance digest")
        object.__setattr__(self, "fit_provenance", provenance)

    @classmethod
    def fit(
        cls,
        X,
        activity,
        *,
        activity_names,
        subjects,
        fit_subject_ids,
        forbidden_subject_ids,
        contract,
    ):
        return cls.fit_features(
            extract_features(X, contract=contract),
            binary_labels(activity, activity_names),
            subjects=subjects,
            fit_subject_ids=fit_subject_ids,
            forbidden_subject_ids=forbidden_subject_ids,
            contract=contract,
        )

    @classmethod
    def fit_features(
        cls,
        features,
        labels,
        *,
        subjects,
        fit_subject_ids,
        forbidden_subject_ids,
        contract,
    ):
        """Fit exactly once on registered fit-only rows of the fixed feature cache."""
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits

        if not isinstance(contract, SourceContract):
            raise TypeError("explicit SourceContract required")
        values = _feature_matrix(features)
        labels = _binary(labels, len(values), both=True)
        subjects, fit_ids, forbidden = (
            _ids(subjects),
            _ids(fit_subject_ids),
            _ids(forbidden_subject_ids),
        )
        if (
            len(subjects) != len(values)
            or not len(fit_ids)
            or len(set(fit_ids)) != len(fit_ids)
            or len(set(forbidden)) != len(forbidden)
            or set(fit_ids) & set(forbidden)
            or not set(subjects) <= set(fit_ids)
        ):
            raise ValueError(
                "leakage guard: fitting rows must belong only to disjoint registered fit IDs"
            )
        mean, std = values.mean(axis=0), values.std(axis=0)
        scale = np.where(std > 1e-12, std, 1.0)
        model = LogisticRegression(
            **{
                k: HYPERPARAMETERS[k]
                for k in (
                    "C",
                    "solver",
                    "max_iter",
                    "tol",
                    "class_weight",
                    "random_state",
                )
            }
        )
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit((values - mean) / scale, labels)
        return cls(
            contract,
            mean,
            scale,
            model.coef_[0],
            float(model.intercept_[0]),
            0.5,
            {
                "fit_subject_ids": sorted(map(int, fit_ids)),
                "forbidden_subject_ids": sorted(map(int, forbidden)),
                "n_fit_windows": len(values),
                "features_sha256": numeric_digest(values),
                "labels_sha256": numeric_digest(labels),
                "fit_calls": 1,
            },
        )

    def with_threshold(self, threshold):
        return replace(self, threshold=threshold)

    def predict_feature_proba(
        self, features, *, contract, batch_size=DEFAULT_BATCH_SIZE
    ):
        """Replay a trusted fixed-feature cache; raw callers should use predict_proba."""
        require_contract(contract, self.contract)
        values = _feature_matrix(features)
        batch_size = _positive_integer(batch_size, "batch_size")
        result = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), batch_size):
            standardized = (values[start : start + batch_size] - self.mean) / self.scale
            logits = np.sum(standardized * self.coefficients, axis=1) + self.intercept
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite standardized features or logits")
            # Stable sigmoid with strict numeric endpoints for threshold=0/1.
            eps = np.finfo(np.float64).eps
            result[start : start + batch_size] = np.clip(
                0.5 + 0.5 * np.tanh(logits / 2), eps, 1 - eps
            )
        return result

    def predict_proba(self, X, *, contract, batch_size=DEFAULT_BATCH_SIZE):
        require_contract(contract, self.contract)
        X = _windows(X, contract)
        batch_size = _positive_integer(batch_size, "batch_size")
        result = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), batch_size):
            features = _features_batch(X[start : start + batch_size], contract)
            result[start : start + batch_size] = self.predict_feature_proba(
                features, contract=contract, batch_size=batch_size
            )
        return result

    def predict(self, X, *, contract, batch_size=DEFAULT_BATCH_SIZE):
        return (
            self.predict_proba(X, contract=contract, batch_size=batch_size)
            >= self.threshold
        ).astype(np.int64)

    def to_dict(self):
        state = {
            "schema": "walking_gate_numeric_v1",
            "contract": self.contract.to_dict(),
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "hyperparameters": dict(HYPERPARAMETERS),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "fit_provenance": json.loads(canonical(self.fit_provenance)),
        }
        return {
            **state,
            "integrity_sha256": hashlib.sha256(canonical(state)).hexdigest(),
        }

    @classmethod
    def from_dict(cls, state):
        keys = {
            "schema",
            "contract",
            "feature_version",
            "feature_names",
            "hyperparameters",
            "mean",
            "scale",
            "coefficients",
            "intercept",
            "threshold",
            "fit_provenance",
            "integrity_sha256",
        }
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("invalid numeric gate schema")
        payload = {k: v for k, v in state.items() if k != "integrity_sha256"}
        if hashlib.sha256(canonical(payload)).hexdigest() != state["integrity_sha256"]:
            raise ValueError("numeric gate integrity mismatch")
        if (
            state["schema"] != "walking_gate_numeric_v1"
            or state["feature_version"] != FEATURE_VERSION
            or state["feature_names"] != list(FEATURE_NAMES)
            or state["hyperparameters"] != HYPERPARAMETERS
        ):
            raise ValueError("unsupported gate feature/hyperparameter schema")
        return cls(
            SourceContract.from_dict(state["contract"]),
            *(
                state[k]
                for k in (
                    "mean",
                    "scale",
                    "coefficients",
                    "intercept",
                    "threshold",
                    "fit_provenance",
                )
            ),
        )

    def save(self, path):
        with Path(path).open("xb") as stream:
            stream.write(canonical(self.to_dict()) + b"\n")

    @classmethod
    def load(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        return cls.from_dict(
            json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
        )


def select_threshold(labels, probabilities):
    """Calibration only: maximize BA, tie by fewer other passes, then larger t."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = _binary(labels, len(probabilities), both=True)
    if (
        probabilities.ndim != 1
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("finite probabilities in [0,1] required")
    order = np.argsort(probabilities, kind="stable")
    sorted_p, sorted_y = probabilities[order], labels[order]
    pos_prefix = np.r_[0, np.cumsum(sorted_y)]
    neg_prefix = np.r_[0, np.cumsum(1 - sorted_y)]
    npos, nneg = int(pos_prefix[-1]), int(neg_prefix[-1])
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best = None
    for threshold in candidates:
        index = np.searchsorted(sorted_p, threshold, side="left")
        tp, tn = npos - int(pos_prefix[index]), int(neg_prefix[index])
        key = (tp * nneg + tn * npos, -(nneg - tn), float(threshold))
        if best is None or key > best[0]:
            best = key, float(threshold)
    assert best is not None  # Candidates always include 0.0 and 1.0.
    return best[1]
