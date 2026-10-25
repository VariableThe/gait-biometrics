"""Full and identity-reference log lag covariances, without fitted parameters.

Adjacent xyz triads retain separate physical units. See
``docs/lag_covariance.md`` for feature ordering and research provenance.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

_TRACE_FLOOR = 1e-8
_BATCH_SIZE = 128


def extract_lag_covariance_features(
    X, *, logarithm: bool | np.bool_, lags=(0, 5, 10, 20), shrinkage=0.2
):
    """Return float64 lag covariance features from finite real ``(N,C,T)`` X.

    C must be 3, 6, or 9. Each adjacent xyz triad contributes the row-major
    upper triangle of S (or its symmetric matrix logarithm), with off-diagonal
    entries multiplied by sqrt(2), followed by log(max(trace(G), 1e-8)).
    For d = 3*len(lags), S = (1-shrinkage)*G/max(trace(G),1e-8)
    + shrinkage*I_d/d. G is channel-centered sample covariance, using the
    common base times 0..T-max(lags)-1 and denominator T-max(lags)-1.

    Lags are distinct nonnegative integers, in the supplied order; at least
    two aligned samples are required. Shrinkage is a real scalar in (0,1],
    representable as a positive float64. Booleans are accepted only for
    logarithm. Default output widths are 79, 158, and 237 for C=3,6,9.
    Empty batches are supported. X is never modified; no fitting occurs.
    """
    if not isinstance(logarithm, (bool, np.bool_)):
        # Keep the same ValueError contract as all other invalid settings.
        raise ValueError("logarithm must be a boolean")  # noqa: TRY004
    try:
        values = np.asarray(X)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be a finite real numeric (N,C,T) array") from exc
    if values.ndim != 3 or values.shape[1] not in (3, 6, 9):
        raise ValueError("X must have shape (N,C,T), C in {3,6,9}")
    if values.dtype.kind not in "fiu":
        raise ValueError("X must contain finite real numeric values")
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
    lags = tuple(int(lag) for lag in lags)
    length = values.shape[2] - max(lags)
    if length < 2:
        raise ValueError("T-max(lags) must be at least 2")
    if (
        isinstance(shrinkage, (bool, np.bool_))
        or not isinstance(shrinkage, Real)
        or not 0 < shrinkage <= 1
    ):
        raise ValueError("shrinkage must be a finite real scalar in (0,1]")
    alpha = float(shrinkage)
    if alpha == 0:
        raise ValueError("shrinkage must be representable as a positive float64")

    width = 3 * len(lags)
    groups = values.shape[1] // 3
    rows, columns = np.triu_indices(width)
    off_diagonal = rows != columns
    block_width = len(rows) + 1
    result = np.empty((len(values), groups * block_width), dtype=np.float64)
    log_floor = np.log(_TRACE_FLOOR)

    for start in range(0, len(values), _BATCH_SIZE):
        batch = values[start : start + _BATCH_SIZE]
        if not np.isfinite(batch).all():
            raise ValueError("X must contain finite real numeric values")
        # Convert integers before abs (including int64's minimum). Preserve
        # extended floats until scaling, so a finite raw trace need not fit
        # float64 when its logarithm and normalized covariance do fit.
        if batch.dtype.kind in "iu" or batch.dtype.itemsize < 8:
            batch = batch.astype(np.float64)
        for group in range(groups):
            triad = batch[:, 3 * group : 3 * group + 3]
            stacked = np.concatenate(
                [triad[:, :, lag : lag + length] for lag in lags], axis=1
            )
            magnitude = np.max(np.abs(stacked), axis=(1, 2))
            safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)
            with np.errstate(under="ignore"):
                centered = stacked / safe_magnitude[:, None, None]
                # Reference subtraction makes exactly constant channels zero.
                centered -= centered[:, :, :1].copy()
                centered -= centered.mean(axis=2, keepdims=True)
                # Rescale centered deviations before squaring, too; this also
                # protects small variation in channels with a large DC offset.
                deviation = np.max(np.abs(centered), axis=(1, 2))
                safe_deviation = np.where(deviation > 0, deviation, 1.0)
                centered = np.asarray(
                    centered / safe_deviation[:, None, None], dtype=np.float64
                )
                covariance = (centered @ centered.swapaxes(1, 2)) / (length - 1)
                trace = np.trace(covariance, axis1=1, axis2=2)
                positive = trace > 0
                log_trace = np.full(len(batch), -np.inf)
                log_trace[positive] = (
                    np.log(trace[positive])
                    + 2 * np.asarray(np.log(safe_magnitude[positive]), dtype=np.float64)
                    + 2 * np.asarray(np.log(safe_deviation[positive]), dtype=np.float64)
                )
                normalized = covariance / np.where(positive, trace, 1)[:, None, None]
                # G/max(raw_trace,floor) without forming raw_trace or scale**2.
                normalized *= np.exp(np.minimum(log_trace - log_floor, 0))[
                    :, None, None
                ]
                normalized = (normalized + normalized.swapaxes(1, 2)) * 0.5
                if logarithm:
                    # S and normalized share eigenvectors. Applying the affine
                    # eigenvalue transform in log space preserves even a tiny
                    # positive shrinkage whose alpha/width would underflow.
                    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
                    with np.errstate(divide="ignore"):
                        log_eigenvalues = np.logaddexp(
                            np.log1p(-alpha) + np.log(np.maximum(eigenvalues, 0)),
                            np.log(alpha) - np.log(width),
                        )
                    matrix = (eigenvectors * log_eigenvalues[:, None, :]) @ (
                        eigenvectors.swapaxes(1, 2)
                    )
                    matrix = (matrix + matrix.swapaxes(1, 2)) * 0.5
                else:
                    matrix = (1 - alpha) * normalized
                    diagonal = np.arange(width)
                    matrix[:, diagonal, diagonal] += alpha / width
                block = np.empty((len(batch), block_width), dtype=np.float64)
                block[:, :-1] = matrix[:, rows, columns]
                block[:, :-1][:, off_diagonal] *= np.sqrt(2)
                block[:, -1] = np.maximum(log_trace, log_floor)
            if not np.isfinite(block).all():
                raise ValueError(
                    "lag covariance features are not representable in float64"
                )
            result[
                start : start + len(batch),
                group * block_width : (group + 1) * block_width,
            ] = block
    return result
