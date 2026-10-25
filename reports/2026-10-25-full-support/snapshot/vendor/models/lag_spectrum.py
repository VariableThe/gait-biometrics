"""Statically extracted pure source; provenance in source-origins.json."""
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
