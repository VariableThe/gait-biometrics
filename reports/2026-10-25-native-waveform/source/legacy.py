"""Verbatim legacy generator and fixed helper controls; no parent imports."""
import itertools
import warnings
import numpy as np
PERIODS = (80,90,100,110,120,128,140,160,180,200)
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")
LAGS = np.arange(80,201)
GUARD = 1e-12
ATOL = 1e-10


def fixture(family, period, phase_index, n=768):
    # Arithmetic copied verbatim from r1; only the default length is extended.
    theta = 2 * np.pi * (np.arange(n, dtype=float) / period + phase_index / 32)
    s, c = np.sin, np.cos
    zero = np.zeros(n)
    acc = [0.8 * s(theta) + 0.15 * s(2 * theta), 0.6 * c(theta) + 0.1 * s(3 * theta), 9.81 - 1.2 * c(theta) + 0.2 * c(2 * theta)]
    gyro = [25 * s(theta) + 4 * s(2 * theta), 18 * c(theta) + 3 * s(3 * theta), 10 * s(theta + 0.4) + 2 * c(2 * theta)]
    if family == "alternating_step_harmonic":
        acc = [0.3 * s(theta), 0.2 * c(theta), 9.81 - 1.1 * c(2 * theta) - 0.25 * c(theta)]
        gyro = [20 * s(theta) + 5 * s(2 * theta), 15 * c(theta) + 2 * c(2 * theta), 12 + 4 * s(theta + 0.3)]
    elif family == "pure_periodic_balanced_peaks":
        acc = [zero, zero, 9.81 - 1.2 * c(2 * theta)]
        gyro = [zero, zero, 40 - 8 * c(theta)]
    elif family == "weak_degenerate_gyro":
        gyro = [zero, zero, 30 * (1 + 1e-14 * c(theta))]
    elif family != "gravity_periodic_6axis":
        raise ValueError(family)
    return np.asarray(acc + gyro)


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


def fixed_checks():
    periodic = np.tile(np.sin(2*np.pi*np.arange(100)/100), 3)[:256]
    tile = correlations(periodic)
    equal = np.zeros(len(LAGS))
    equal[[20, 120]] = 1.0
    vectors, rows = [], []
    for label, base in (("unfiltered_tile_sine_P100_N256", tile), ("exactly_equal_synthetic_vector", equal)):
        for da, db in itertools.product((-1, 0, 1), repeat=2):
            vector = base.copy()
            for idx, direction in ((20, da), (120, db)):
                if direction:
                    vector[idx] = np.nextafter(vector[idx], np.inf if direction > 0 else -np.inf)
            other = np.ones(len(LAGS), dtype=bool)
            other[[20, 120]] = False
            assert np.array_equal(vector[other], base[other])
            exact = select(vector, RULES[0])
            candidate = select(vector, RULES[1])
            assert candidate[0] == 100 and candidate[3] == [100, 200]
            assert select(vector[::-1], RULES[0], LAGS[::-1]) == exact
            assert select(vector[::-1], RULES[1], LAGS[::-1]) == candidate
            rows.append({"vector_row": len(vectors), "source": label, "ulp_shifts_r100_r200": [da, db],
                         "r100": float(vector[20]), "r200": float(vector[120]),
                         "r100_hex": float(vector[20]).hex(), "r200_hex": float(vector[120]).hex(),
                         "exact_selected_lag": exact[0], "exact_ties": exact[2], "candidate_selected_lag": candidate[0],
                         "candidate_nearmax_lags": candidate[3], "reversal_stable": True, "other_values_bitwise_unchanged": True,
                         "exact_candidate_diverge_expected": exact[0] != candidate[0]})
            vectors.append(vector)
    assert float(tile[20]) == 0.9999999999999999 and float(tile[120]) == 1.0
    assert select(tile, RULES[0])[0] == 200
    return {"checks_pass": True, "all_candidate_choices_stable": True, "all_candidate_nearsets_stable": True,
            "baseline_counterexample": {"r100": float(tile[20]), "r200": float(tile[120]), "exact": 200, "candidate": 100},
            "exact_divergence_count": sum(row["exact_candidate_diverge_expected"] for row in rows), "rows": rows,
            "scope": "These 18 frozen vectors only; no assertion about arbitrary values at the 1e-10 set boundary"}, np.stack(vectors)
