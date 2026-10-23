"""Train-only regularized within-class covariance metric for gait descriptors.

Adapted from WCCN (Hatch et al., Interspeech 2006), not a reproduction of its
speaker-recognition system: https://doi.org/10.21437/Interspeech.2006-183
"""

from __future__ import annotations

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
        if shrinkage != "auto" and not 0 < float(shrinkage) <= 1:
            raise ValueError("shrinkage must be auto or in (0, 1]")
        self.mode, self.shrinkage = mode, shrinkage

    @staticmethod
    def _features(features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
            raise ValueError("features must be a nonempty finite (N,D) matrix")
        return values

    def fit(self, features, subject_ids):
        values = self._features(features)
        subjects = np.asarray(subject_ids)
        if subjects.shape != (len(values),) or subjects.dtype.kind not in "iu":
            raise ValueError("one integer subject ID is required per training row")
        identities, counts = np.unique(subjects, return_counts=True)
        if len(identities) < 2:
            raise ValueError("metric fitting requires at least two training subjects")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0).clip(min=1e-6)
        standardized = (values - self.mean_) / self.scale_
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
        transformed = ((values - self.mean_) / self.scale_) @ self.projection_
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError(
                "a descriptor equals the training center and has no cosine direction"
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
        result = cls()
        result.mean_ = np.asarray(state["mean"], dtype=np.float64).copy()
        result.scale_ = np.asarray(state["scale"], dtype=np.float64).copy()
        result.projection_ = np.asarray(state["projection"], dtype=np.float64).copy()
        result.training_subject_ids_ = np.asarray(
            state["training_subject_ids"], dtype=np.int64
        ).tolist()
        if (
            result.projection_.shape != (len(result.mean_), len(result.mean_))
            or result.scale_.shape != result.mean_.shape
            or (result.scale_ <= 0).any()
        ):
            raise ValueError("invalid fitted metric dimensions or scales")
        if not all(
            np.isfinite(x).all()
            for x in (result.mean_, result.scale_, result.projection_)
        ):
            raise ValueError("fitted metric state must be finite")
        return result
