"""Pure exploratory tail thresholds; no iid guarantee for clustered gait claims.

The nominal calculation references Tong, Feng and Li (2018), Algorithm 1 and
Proposition 1, https://arxiv.org/abs/1608.03109. Its probability is over repeated
independent calibration samples, not a posterior probability after calibration.
No data, model, protocol or frozen study source is loaded by this module.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.stats import binom

__all__ = [
    "empirical_threshold",
    "nominal_order_statistic_threshold",
    "nominal_violation_bound",
]

_ASSUMPTIONS = (
    (
        "Impostor calibration scores are independent and identically distributed "
        "draws from the deployment impostor-score distribution."
    ),
    (
        "The entire scoring rule, including fitting and model selection, is fixed "
        "independently of the calibration scores."
    ),
    (
        "The exact order-statistic reference assumes continuous scores; rounding "
        "upward and rejecting whole ties can be conservative."
    ),
    (
        "These assumptions are not established for clustered gait claims. "
        "This is a nominal reference only, with no guarantee asserted for these data."
    ),
)


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


def nominal_violation_bound(n, *, allowed_count=0, alpha=0.01):
    """Return ``P[Binomial(n, alpha) <= allowed_count]`` as a Python float.

    This is a nominal repeated-calibration violation bound for the strict
    ``(allowed_count + 1)``-th highest cutoff, under the stated assumptions.
    It is neither an empirical FAR nor a posterior probability. ``n`` must be
    a positive integer and ``allowed_count`` an integer in ``[0, n - 1]``;
    booleans are rejected. Counts must be fixed independently of the scores.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n must be a positive integer")
    if (
        isinstance(allowed_count, (bool, np.bool_))
        or not isinstance(allowed_count, Integral)
        or not 0 <= allowed_count < n
    ):
        raise ValueError("allowed_count must be an integer in [0, n - 1]")
    alpha = _probability(alpha, "alpha")
    return float(binom.cdf(int(allowed_count), int(n), alpha))


def nominal_order_statistic_threshold(impostor_scores, *, alpha=0.01, delta=0.1):
    """Return JSON-safe, explicitly nominal order-statistic diagnostics.

    Choose the largest ``j in [0, n - 1]`` with ``binom.cdf(j,n,alpha) <= delta``.
    ``rank = n - j`` is the one-based ASCENDING rank; threshold is the next
    float64 above that score. Accept with ``score >= threshold``, rejecting
    every tie at the boundary. ``count`` is the actual accepted calibration
    count, while ``allowed_count`` is j; the bound always uses j, even if ties
    reduce count. Inputs are converted to float64 without mutating them.
    Apply the returned threshold to float64 scores as well; NumPy comparisons
    with a lower-precision array can round a Python float threshold downward.

    No feasible rank yields ``insufficient_independent_calibration`` with no
    threshold, rank, count or selected bound. ``best_nominal_violation_bound``
    reports the j=0 reference in all cases. If the next float is infinity,
    ``threshold_not_representable`` retains the nominal rank/bound but has no
    usable threshold or count. Only status ``ok`` supplies a usable threshold.
    None of these diagnostics certifies independence or the supplied claims.
    """
    values = _finite_scores(impostor_scores, "impostor_scores")
    alpha, delta = _probability(alpha, "alpha"), _probability(delta, "delta")
    n = int(values.size)
    best_bound = nominal_violation_bound(n, alpha=alpha)
    result = {
        "status": "insufficient_independent_calibration",
        "threshold": None,
        "count": None,
        "rank": None,
        "n": n,
        "allowed_count": None,
        "alpha": alpha,
        "delta": delta,
        "nominal_violation_bound": None,
        "best_nominal_violation_bound": best_bound,
        "order_statistic": None,
        "decision_rule": "score >= threshold",
        "interpretation": "nominal_iid_reference_only",
        "assumptions": list(_ASSUMPTIONS),
    }
    if best_bound > delta:
        return result

    # The CDF increases in j. n is an infeasible sentinel because delta < 1.
    lo, hi = 0, n
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if binom.cdf(middle, n, alpha) <= delta:
            lo = middle
        else:
            hi = middle
    rank = n - lo
    boundary = float(np.partition(values, rank - 1)[rank - 1])
    result.update(
        status="threshold_not_representable",
        rank=rank,
        allowed_count=lo,
        order_statistic=boundary,
        nominal_violation_bound=nominal_violation_bound(
            n, allowed_count=lo, alpha=alpha
        ),
    )
    threshold = _next_threshold(boundary)
    if not np.isfinite(threshold):
        return result
    result.update(
        status="ok",
        threshold=threshold,
        count=int(np.count_nonzero(values >= threshold)),
    )
    return result


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
