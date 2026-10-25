"""Verbatim reviewed numerical pure components; no parent imports."""
import common as c
import legacy
import numpy as np
from scipy.signal import sosfiltfilt


def magnitudes(window):
    # Preserve original reduction order exactly.
    return np.stack(
        (np.linalg.norm(window[:3], axis=0), np.linalg.norm(window[3:], axis=0))
    )


def filter_modalities(mags):
    return np.stack([sosfiltfilt(c.SOS, row, padtype="odd", padlen=9) for row in mags])


def vector_guard(vector):
    finite = bool(np.isfinite(vector).all())
    if not finite:
        return False, np.array([np.nan, np.nan]), False
    q = float(np.sqrt(np.mean(vector * vector)))
    centered = vector - vector.mean()
    std = float(np.sqrt(np.mean(centered * centered)))
    good = bool(np.isfinite(q) and np.isfinite(std) and q > 0 and std > 1e-12 * q)
    return good, np.array([q, std]), True


def pearson(left, right):
    lg, ls, lf = vector_guard(left)
    rg, rs, rf = vector_guard(right)
    value = float(np.corrcoef(left, right)[0, 1]) if lg and rg else np.nan
    return value, np.stack((ls, rs)), np.array([lg, rg]), np.array([lf, rf])


def joint_correlations(filtered):
    values = np.full((3, 121), np.nan)
    stats = np.full((2, 121, 2, 2), np.nan)
    valid = np.zeros((2, 121, 2), dtype=bool)
    finite = np.zeros_like(valid)
    for ci in range(2):
        for i, lag in enumerate(legacy.LAGS):
            c.CHECK()
            value, stat, good, fin = pearson(filtered[ci, :-lag], filtered[ci, lag:])
            values[ci, i], stats[ci, i], valid[ci, i], finite[ci, i] = (
                value,
                stat,
                good,
                fin,
            )
    both = np.isfinite(values[0]) & np.isfinite(values[1])
    values[2, both] = (values[0, both] + values[1, both]) / 2
    return values, stats, valid, finite


def whole_window(mags, filtered):
    rows = []
    for label, vector in zip(
        ("raw_acc", "raw_gyro", "filtered_acc", "filtered_gyro"),
        np.vstack((mags, filtered)),
    ):
        good, stats, finite = vector_guard(vector)
        rows.append(
            {
                "channel": label,
                "finite": finite,
                "valid": good,
                "q": float(stats[0]) if np.isfinite(stats[0]) else None,
                "std_population": float(stats[1]) if np.isfinite(stats[1]) else None,
                "degenerate": bool(finite and not good),
            }
        )
    return rows
