# Pure function copies only: no old module import or old top-level execution.
import math
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")
LAGS = np.arange(80, 201)
GUARD = 1e-12
ATOL = 1e-10


def correlations(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.corrcoef(y[:-int(lag)], y[int(lag):])[0, 1] for lag in LAGS])


def select(rs, rule, lags=LAGS):
    finite = np.isfinite(rs)
    if not finite.any():
        return None, None, [], []
    best = float(np.max(rs[finite]))
    ties = sorted(lags[finite & (rs == best)].tolist())
    near = sorted(lags[finite & (np.abs(rs - best) <= ATOL)].tolist())
    return int(min(ties if rule == RULES[0] else near)), best, ties, near


def correlation_summary(rs, lags=LAGS):
    lag, best, ties, near = select(rs, RULES[1], lags)
    finite = rs[np.isfinite(rs)]
    outside = rs[np.isfinite(rs) & ~np.isin(lags, near)]
    ordered = np.sort(finite)
    return {"finite_correlation_count": int(finite.size), "undefined_correlation_count": int(rs.size - finite.size),
            "max_correlation": best, "exact_max_ties": ties, "all_nearmax_lags": near,
            "lowest_nearmax_lag": lag,
            "correlation_gaps": {"top_minus_second": None if finite.size < 2 else float(ordered[-1] - ordered[-2]),
                                 "max_minus_lowest_nearmax": None if lag is None else best - float(rs[np.flatnonzero(lags == lag)[0]]),
                                 "max_minus_best_outside_nearmax": None if not outside.size else best - float(np.max(outside))}}


def minima_and_pair(y, lag):
    n = len(y)
    minima = [i for i in range(10, n - 10) if y[i] < y[i - 1] and y[i] < y[i + 1]]
    pairs = [(a, b) for a in minima for b in minima if 80 <= b - a <= 200 and abs(b - a - lag) <= 0.1 * lag]
    if not pairs:
        return minima, None, 0
    pair = min(pairs, key=lambda ab: (abs(ab[1] - ab[0] - lag), abs((ab[0] + ab[1]) / 2 - (n - 1) / 2), ab[0], ab[1]))
    return minima, list(pair), len(pairs)


def features_from_pair(magnitudes, pair):
    # Function arithmetic copied from r1 without numerical changes.
    a, b = pair
    knots = np.arange(a, b + 1, dtype=float)
    points = np.linspace(a, b, 200, endpoint=True)
    sampled = CubicSpline(knots, magnitudes[:, a:b + 1], axis=1, bc_type="natural", extrapolate=False)(points)
    centered = sampled - sampled.mean(axis=1, keepdims=True)
    stds = np.sqrt(np.mean(centered * centered, axis=1))
    scales = np.sqrt(np.mean(sampled * sampled, axis=1))
    stats = {"channel_population_std": stds.tolist(), "channel_raw_rms": scales.tolist(), "channel_degenerate": ((scales == 0) | (stds <= GUARD * scales)).tolist()}
    if any(stats["channel_degenerate"]):
        return None, "channel_degenerate", stats
    feature = (centered / stds[:, None]).reshape(400)
    feature /= np.linalg.norm(feature)
    if not np.isfinite(feature).all():
        return None, "nonfinite_features", stats
    stats["feature_norm"] = float(np.linalg.norm(feature))
    stats["feature_channel_means"] = feature.reshape(2, 200).mean(axis=1).tolist()
    return feature, None, stats


def extract(mags, smooth, rs, rule):
    rec = {"selected_lag": None, "raw_maxr": None, "exact_max_ties": [], "strict_minima": [], "pair_bounds_inclusive": None, "pair_length_samples": None, "pair_stored_samples": None, "eligible_pair_count": 0, "failure_reason": None, "finite_features": False, "valid_pre_quality": False, "quality_evaluated": False}
    lag, maxr, ties, _ = select(rs, rule)
    rec.update(selected_lag=lag, raw_maxr=maxr, exact_max_ties=ties)
    if lag is None:
        rec["failure_reason"] = "no_lag"
        return rec, None
    minima, pair, pair_count = minima_and_pair(smooth, lag)
    rec.update(strict_minima=minima, pair_bounds_inclusive=pair, eligible_pair_count=pair_count)
    if pair is None:
        rec["failure_reason"] = "no_eligible_minima_pair"
        return rec, None
    rec.update(pair_length_samples=pair[1] - pair[0], pair_stored_samples=pair[1] - pair[0] + 1)
    feature, failure, stats = features_from_pair(mags, pair)
    rec.update(stats)
    rec.update(failure_reason=failure, finite_features=feature is not None, valid_pre_quality=feature is not None)
    return rec, feature


def alias(value, period):
    if value is None:
        return "unavailable"
    ratio = value / period
    target, label = min(((0.5, "half_stride"), (1.0, "one_stride"), (2.0, "double_stride")), key=lambda x: abs(ratio - x[0]))
    return label if abs(ratio - target) <= 0.1 * target else "other"


def truth(period, phase_index, rec, n):
    events = [period * (k - phase_index / 32) for k in range(-2, math.ceil(n / period) + 3)]
    raw = [v for v in events if 0 <= v <= n - 1]
    interior = [v for v in events if 10 <= v <= n - 11]
    pair = rec["pair_bounds_inclusive"]
    contained = [] if pair is None else [v for v in events if pair[0] <= v <= pair[1]]
    return {"truth_stride_period_samples": period, "truth_events_raw": raw, "truth_events_with_minima_margins": interior,
            "truth_full_anchored_stride_in_raw": len(raw) >= 2, "truth_full_anchored_stride_with_margins": len(interior) >= 2,
            "truth_interval_can_fit_stride_raw": period <= n - 1, "truth_interval_can_fit_stride_with_margins": period <= n - 21,
            "selected_lag_alias": alias(rec["selected_lag"], period), "selected_pair_alias": alias(rec["pair_length_samples"], period),
            "selected_pair_period_ratio": None if pair is None else (pair[1] - pair[0]) / period,
            "truth_anchored_strides_contained_in_selected_pair": max(0, len(contained) - 1)}


def geometric_reference(family, period, phase, n, lag, roots):
    events = []
    for k in range(-2, math.ceil(n/period)+3):
        for ri, root in enumerate(roots[family]):
            t = period*(k+root-phase/32)
            if 10 <= t <= n-11:
                events.append((float(t), k, ri))
    events.sort()
    pairs = []
    if lag is not None:
        for i, a in enumerate(events):
            for b in events[i+1:]:
                delta = period*((b[1]-a[1]) + (roots[family][b[2]]-roots[family][a[2]]))
                if 80 <= delta <= 200 and abs(delta-lag) <= 0.1*lag:
                    pairs.append({"bounds": [a[0], b[0]], "length": float(delta), "event_indices": [events.index(a), events.index(b)]})
    chosen = min(pairs, key=lambda p: (abs(p["length"]-lag), abs(sum(p["bounds"])/2-(n-1)/2), *p["bounds"])) if pairs else None
    return {"minima_times": [v[0] for v in events], "eligible_pairs": pairs, "available_at_selected_lag": bool(pairs),
            "chosen_pair": chosen, "reference_kind": "continuous prescribed raw-magnitude minima; no filtered minima used"}
