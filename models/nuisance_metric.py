"""Clean automatic WCCN plus paired, identity-balanced nuisance second moments.

The nuisance term is a mathematical adaptation of WCCN, not a reproduction of
Hatch et al. (2006). Feature extraction and TRAIN-only row pairing belong to
the caller; see ``docs/nuisance_metric.md`` for the frozen study contract.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from models.gait_metric import WithinClassMetric


def _checked_eigh(matrix, name):
    """Validate finite symmetric PSD arithmetic without repairing the matrix."""
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must remain finite in float64")
    magnitude = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 64 * np.finfo(np.float64).eps * len(matrix)
    scaled = matrix / magnitude
    if not np.allclose(scaled, scaled.T, rtol=0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} eigendecomposition must remain finite") from exc
    if not all(np.isfinite(value).all() for value in (eigenvalues, eigenvectors)):
        raise ValueError(f"{name} eigendecomposition must remain finite")
    if eigenvalues.min() / magnitude < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return eigenvalues, eigenvectors


def fit_nuisance_metric(
    features, subject_ids, view_features=None
) -> tuple[WithinClassMetric, dict[str, np.ndarray]]:
    """Fit automatic clean WCCN, optionally adding exactly four paired views.

    ``features`` is finite real ``(N,D)``; ``subject_ids`` supplies nonnegative
    int64-compatible integer IDs for at least two subjects. ``view_features``
    is either None (the exact clean baseline) or finite real ``(N,4,D)`` in
    caller-supplied gain-0.9, gain-1.1, noise-plus, noise-minus order. Views
    never contribute rows to clean statistics or Ledoit-Wolf estimation.

    Return the existing metric type and float64/integer NumPy diagnostics
    (including zero-dimensional scalar arrays). No inputs are modified.
    """
    values = WithinClassMetric._features(features)
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

    views = None
    if view_features is not None:
        views = np.asarray(view_features)
        if views.dtype.kind not in "iuf":
            raise ValueError("view_features must be real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            views = views.astype(np.float64)
        if views.shape != (len(values), 4, values.shape[1]):
            raise ValueError(
                "view_features must have shape (N,4,D) paired with features"
            )
        if not np.isfinite(views).all():
            raise ValueError("view_features must be finite in float64")

    # Keep expression order identical to WithinClassMetric.fit for bitwise
    # clean-arm parity. Estimate C_B directly, before any eigenvalue floor.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = values.mean(axis=0)
        scale = values.std(axis=0).clip(min=1e-6)
        standardized = (values - mean) / scale
    if not all(np.isfinite(value).all() for value in (mean, scale, standardized)):
        raise ValueError("training statistics must remain finite in float64")
    residuals = np.zeros_like(standardized)
    for subject, count in zip(identities, counts):
        selected = subjects == subject
        residuals[selected] = (
            standardized[selected] - standardized[selected].mean(axis=0)
        ) * np.sqrt(len(values) / (len(identities) * count))
    if not np.isfinite(residuals).all():
        raise ValueError("clean residuals must remain finite in float64")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            estimator = LedoitWolf(assume_centered=True).fit(residuals)
    except FloatingPointError as exc:
        raise ValueError("clean covariance must remain finite in float64") from exc
    clean_covariance = estimator.covariance_
    shrinkage = float(estimator.shrinkage_)
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("clean shrinkage must be finite and in [0,1]")
    eigenvalues, eigenvectors = _checked_eigh(clean_covariance, "clean covariance")

    nuisance = np.zeros_like(clean_covariance)
    if views is not None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                delta = (views - values[:, None, :]) / scale
                if not np.isfinite(delta).all():
                    raise ValueError(
                        "paired displacements must remain finite in float64"
                    )
                for subject, count in zip(identities, counts):
                    paired = delta[subjects == subject].reshape(-1, values.shape[1])
                    # Uncentered: retain systematic corruption bias. Each
                    # original identity has equal weight, regardless of n_j.
                    nuisance += (paired.T @ paired) / (count * 4)
                nuisance /= len(identities)
        except FloatingPointError as exc:
            raise ValueError(
                "nuisance second moment must remain finite in float64"
            ) from exc
        _checked_eigh(nuisance, "nuisance second moment")
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = clean_covariance + nuisance
        eigenvalues, eigenvectors = _checked_eigh(covariance, "combined covariance")

    floor = max(float(eigenvalues.max()) * 1e-5, 1e-6)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projection = (
            eigenvectors * np.maximum(eigenvalues, floor) ** -0.5
        ) @ eigenvectors.T
    if not np.isfinite(floor) or not np.isfinite(projection).all():
        raise ValueError("whitening projection must remain finite in float64")

    metric = WithinClassMetric()
    metric.mean_, metric.scale_, metric.projection_ = mean, scale, projection
    metric.training_subject_ids_ = identities.tolist()
    metric.shrinkage_ = shrinkage
    diagnostics = {
        "clean_covariance": clean_covariance.copy(),
        "nuisance_second_moment": nuisance,
        "eigenvalues": eigenvalues,
        "eigenvalue_floor": np.asarray(floor, dtype=np.float64),
        "original_rows": np.asarray(len(values), dtype=np.int64),
        "original_subjects": np.asarray(len(identities), dtype=np.int64),
        "views_per_row": np.asarray(0 if views is None else 4, dtype=np.int64),
    }
    return metric, diagnostics
