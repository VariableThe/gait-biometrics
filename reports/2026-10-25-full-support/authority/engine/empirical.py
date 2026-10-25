"""Statically extracted pure source; provenance in source-origins.json."""
from numbers import Real
import numpy as np


def _probability(value, name, *, endpoints=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")  # noqa: TRY004 -- public validation contract
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    valid_range = 0 <= value <= 1 if endpoints else 0 < value < 1
    if not np.isfinite(value) or not valid_range:
        interval = "[0, 1]" if endpoints else "(0, 1)"
        raise ValueError(f"{name} must be in {interval}")
    return value

def _finite_scores(values, name):
    try:
        values = np.asarray(values)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a nonempty real numeric vector")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            values = values.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError, FloatingPointError) as error:
        raise ValueError(f"{name} must be a nonempty finite float64 vector") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite float64 vector")
    return values

def _next_threshold(score):
    # Subnormal nextafter steps and overflow at float64.max are handled below,
    # independently of the caller's NumPy error settings.
    with np.errstate(over="ignore", under="ignore"):
        return float(np.nextafter(score, np.inf))

def empirical_threshold(labels, scores, target_far=0.01):
    """Match the frozen mentor study's finite grouped-ROC calibration rule.

    With default target_far, the entire result equals
    ``scripts.mentor_backend_study.calibrate(labels, scores)`` for valid numeric
    inputs. Select the last whole score group with FAR <= target_far, including
    the finite reject-all point. Equality is accepted. Both binary classes are
    required; target_far is a real scalar in [0, 1], excluding booleans. This is
    an empirical rule with no population FAR or confidence guarantee.
    ROC selection uses float64, and reported decisions retain the original
    scores' dtype, exactly as the frozen function does.
    """
    original_scores = np.asarray(scores)
    values = _finite_scores(original_scores, "scores")
    labels = np.asarray(labels)
    if (
        labels.shape != values.shape
        or labels.dtype.kind not in "biuf"
        or not np.isin(labels, (0, 1)).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("finite scores and both binary classes are required")
    target_far = _probability(target_far, "target_far", endpoints=True)
    reject_all = _next_threshold(values.max())
    if not np.isfinite(reject_all):
        raise ValueError("finite reject-all threshold is required")

    order = np.argsort(-values, kind="stable")
    sorted_scores, positive = values[order], labels[order] == 1
    # Adjacent inequality is the frozen diff != 0 rule without subtraction
    # overflow for scores spanning the finite float64 range.
    ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(values) - 1
    ]
    n_impostor = int(np.count_nonzero(labels == 0))
    n_genuine = int(np.count_nonzero(labels == 1))
    far = np.r_[0.0, np.cumsum(~positive)[ends] / n_impostor]
    thresholds = np.r_[reject_all, sorted_scores[ends]]
    threshold = float(thresholds[np.flatnonzero(far <= target_far)[-1]])
    # Retain the frozen fixed_metrics comparison, including NumPy's scalar
    # promotion behavior for lower-precision score arrays.
    accepted = original_scores >= threshold
    false_accepts = int(np.count_nonzero(accepted & (labels == 0)))
    false_rejects = int(np.count_nonzero(~accepted & (labels == 1)))
    return {
        "threshold": threshold,
        "far": false_accepts / n_impostor,
        "frr": false_rejects / n_genuine,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
        "decision_rule": "score >= threshold",
        "target_far": target_far,
        "selection": "empirical_far_at_most_target",
    }
