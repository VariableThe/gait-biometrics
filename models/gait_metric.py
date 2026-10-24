"""Train-only regularized within-class covariance metric for gait descriptors.

Adapted from WCCN (Hatch et al., Interspeech 2006), not a reproduction of its
speaker-recognition system: https://doi.org/10.21437/Interspeech.2006-183
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from sklearn.covariance import LedoitWolf


class WithinClassMetric:
    """Standardize features, then suppress training within-identity variation.

    ``fit`` receives optimizer-training identities only. Validation/test feature
    vectors never update centering, scaling, or the learned covariance matrix.
    """

    def __init__(self, mode: str = "wccn", shrinkage: float | str = "auto"):
        if mode not in ("standard", "wccn"):
            raise ValueError("mode must be standard or wccn")
        if not (isinstance(shrinkage, str) and shrinkage == "auto") and (
            isinstance(shrinkage, (bool, np.bool_))
            or not isinstance(shrinkage, Real)
            or not np.isfinite(shrinkage)
            or not 0 < shrinkage <= 1
        ):
            raise ValueError("shrinkage must be auto or a finite real in (0, 1]")
        self.mode, self.shrinkage = mode, shrinkage

    @staticmethod
    def _features(features):
        values = np.asarray(features)
        if values.dtype.kind not in "iuf":
            raise ValueError("features must be real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float64)
        if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
            raise ValueError("features must be a nonempty finite (N,D) matrix")
        return values

    def fit(self, features, subject_ids):
        values = self._features(features)
        subjects = np.asarray(subject_ids)
        if (
            subjects.shape != (len(values),)
            or subjects.dtype.kind not in "iu"
            or np.any(subjects < 0)
            or np.any(subjects > np.iinfo(np.int64).max)
        ):
            raise ValueError(
                "one nonnegative int64-compatible subject ID is required per training row"
            )
        identities, counts = np.unique(subjects, return_counts=True)
        if len(identities) < 2:
            raise ValueError("metric fitting requires at least two training subjects")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = values.mean(axis=0)
            scale = values.std(axis=0).clip(min=1e-6)
            standardized = (values - mean) / scale
        if not all(np.isfinite(x).all() for x in (mean, scale, standardized)):
            raise ValueError("training statistics must remain finite in float64")
        self.mean_, self.scale_ = mean, scale
        self.projection_ = np.eye(values.shape[1])
        self.training_subject_ids_ = identities.tolist()
        if self.mode == "wccn":
            residuals = np.zeros_like(standardized)
            for subject, count in zip(identities, counts):
                selected = subjects == subject
                # Give each training identity equal covariance weight.
                residuals[selected] = (
                    standardized[selected] - standardized[selected].mean(axis=0)
                ) * np.sqrt(len(values) / (len(identities) * count))
            if self.shrinkage == "auto":
                estimator = LedoitWolf(assume_centered=True).fit(residuals)
                covariance = estimator.covariance_
                self.shrinkage_ = float(estimator.shrinkage_)
            else:
                covariance = residuals.T @ residuals / len(values)
                alpha = float(self.shrinkage)
                covariance = (1 - alpha) * covariance + alpha * np.trace(
                    covariance
                ) / len(covariance) * np.eye(len(covariance))
                self.shrinkage_ = alpha
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            floor = max(float(eigenvalues.max()) * 1e-5, 1e-6)
            self.projection_ = (
                eigenvectors * np.maximum(eigenvalues, floor) ** -0.5
            ) @ eigenvectors.T
        return self

    def transform(self, features):
        if not hasattr(self, "projection_"):
            raise ValueError(
                "fit the metric on training identities before transforming"
            )
        values = self._features(features)
        if values.shape[1] != len(self.mean_):
            raise ValueError("feature dimension differs from fitted training data")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = ((values - self.mean_) / self.scale_) @ self.projection_
            norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        if not np.isfinite(transformed).all():
            raise ValueError("transformed descriptors must remain finite in float64")
        unstable = (
            ~np.isfinite(norms) | (norms < np.sqrt(np.finfo(float).tiny))
        ).ravel()
        if unstable.any():
            # Recover finite nonzero directions without changing ordinary
            # normalization arithmetic or imposing a magnitude-dependent floor.
            scale = np.max(np.abs(transformed[unstable]), axis=1, keepdims=True)
            if (scale == 0).any():
                raise ValueError(
                    "a descriptor equals the training center and has no cosine direction"
                )
            transformed[unstable] /= scale
            norms[unstable] = np.linalg.norm(
                transformed[unstable], axis=1, keepdims=True
            )
        return (transformed / norms).astype(np.float32)

    def state_dict(self):
        if not hasattr(self, "projection_"):
            raise ValueError("fit the metric before serialization")
        return {
            "mean": self.mean_.copy(),
            "scale": self.scale_.copy(),
            "projection": self.projection_.copy(),
            "training_subject_ids": np.asarray(
                self.training_subject_ids_, dtype=np.int64
            ),
        }

    @classmethod
    def from_state_dict(cls, state):
        try:
            arrays = [np.asarray(state[key]) for key in ("mean", "scale", "projection")]
            subjects = np.asarray(state["training_subject_ids"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "fitted metric state requires numeric arrays and training IDs"
            ) from exc
        if any(array.dtype.kind not in "iuf" for array in arrays):
            raise ValueError("fitted metric arrays must be real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            mean, scale, projection = [array.astype(np.float64) for array in arrays]
        if (
            mean.ndim != 1
            or not mean.size
            or projection.shape != (mean.size, mean.size)
            or scale.shape != mean.shape
            or (scale <= 0).any()
        ):
            raise ValueError("invalid fitted metric dimensions or scales")
        if not all(np.isfinite(x).all() for x in (mean, scale, projection)):
            raise ValueError("fitted metric state must be finite")
        if (
            subjects.ndim != 1
            or subjects.dtype.kind not in "iu"
            or len(subjects) < 2
            or np.any(subjects < 0)
            or np.any(subjects > np.iinfo(np.int64).max)
            or len(np.unique(subjects)) != len(subjects)
        ):
            raise ValueError(
                "training IDs must be unique nonnegative int64-compatible integers for >=2 subjects"
            )
        result = cls()
        result.mean_, result.scale_, result.projection_ = mean, scale, projection
        result.training_subject_ids_ = subjects.astype(np.int64).tolist()
        return result
