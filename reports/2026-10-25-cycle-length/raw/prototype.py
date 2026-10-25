#!/usr/bin/env python3
"""One frozen, portable, synthetic-only length recreation. Never imports r1.

Freeze once with --freeze-only, then execute once with --run. All output is
relative to this source's own parent. Failed/incomplete attempts cannot resume.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import platform
import resource
import signal
import sys
import time
import warnings

START_CPU = time.process_time()
START_WALL = time.perf_counter()
resource.setrlimit(resource.RLIMIT_CPU, (290, 299))
STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


signal.signal(signal.SIGXCPU, request_stop)

import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.signal import butter, sosfiltfilt

PERIODS = (80, 90, 100, 110, 120, 128, 140, 160, 180, 200)
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")
LENGTHS = (256, 768)
LAGS = np.arange(80, 201)
DIAG_LAGS = np.arange(40, 401)
ANCHORS = np.arange(10, 768 - 410)
DIAG_CHANNELS = ("raw_acc_magnitude", "raw_gyro_magnitude", "filtered_acc_magnitude", "filtered_gyro_magnitude")
GUARD = 1e-12
ATOL = 1e-10
SEED = 20261025
SOS = butter(2, 3.0, btype="lowpass", fs=100.0, output="sos")
REPORT = ROOT / "report.json"
REFERENCE = ROOT / "cached_reference.json"
SOURCE = ROOT / "prototype.py"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def runtime_identity():
    import numpy._core._multiarray_umath as numpy_core
    import scipy.interpolate._cubic as cubic_module
    import scipy.interpolate._ppoly as ppoly_module
    import scipy.signal._sosfilt as sos_module

    files = {"python_executable": sys.executable, "numpy_init": np.__file__,
             "numpy_core_binary": numpy_core.__file__, "scipy_init": scipy.__file__,
             "natural_spline_source": cubic_module.__file__, "ppoly_binary": ppoly_module.__file__,
             "sos_filter_binary": sos_module.__file__}
    # Paths are informational; identity is based on file content, versions and ABI.
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "implementation": sys.implementation.name, "machine": platform.machine(),
            "byteorder": sys.byteorder, "float64_dtype": np.dtype(np.float64).str,
            "file_sha256": {key: file_sha(path) for key, path in files.items()},
            "single_thread_environment": {key: os.environ[key] for key in THREAD_KEYS},
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}


def manifest(reference):
    return {
        "preregistration": "cycle-length-feasibility-r1; ONE exclusive attempt; no in-place retry",
        "seed": SEED, "random_draws": 0,
        "family_equations": reference["family_equations"],
        "theta": "2*np.pi*(np.arange(N,dtype=float)/P+j/32); t origin is sample zero",
        "families_in_order": list(FAMILIES), "periods_samples": list(PERIODS),
        "phases": list(range(32)), "lengths_only": list(LENGTHS), "rules_in_order": list(RULES),
        "fixture_count": 1280, "fixture_length_count": 2560, "record_count": 5120,
        "nested_generation": "Generate six-axis N768 once per fixture; copy its prefix256; independently generate256 solely for bitwise prefix control. Never concatenate windows.",
        "execution_order": "First all 1280 exact N256 records, compare ALL cached keys with exact float bits and expected counts; STOP on any mismatch. Only after success, candidate N256 and both N768 rules plus N768-only diagnostics.",
        "filter": {"order": 2, "cutoff_hz": 3, "fs_hz": 100, "type": "lowpass Butterworth SOS", "padtype": "odd", "padlen": 9, "sos": SOS.tolist()},
        "detector": {"signal": "filtered raw acceleration magnitude only", "dtype": "float64",
                     "lags": LAGS.tolist(), "pearson": "np.corrcoef(y[:-L],y[L:])[0,1]; standard clipping; finite lags only",
                     "exact": "Largest computed value, smallest lag among exact equals",
                     "candidate": {"rtol": 0, "atol": ATOL, "membership": "finite & abs(r-maxr)<=1e-10", "selection": "smallest lag in entire nearmax set; PUTATIVE repeat interval, never anatomical-stride inference"}},
        "minima_pair": {"indices": "10..N-11 inclusive", "minima": "strict immediate-neighbor minima of filtered acceleration magnitude",
                        "pairs": "all ordered minima pairs a<b with 80<=b-a<=200 and abs((b-a)-L)<=0.1*L",
                        "ranking": ["absolute period discrepancy", "distance of pair midpoint to (N-1)/2", "a", "b"]},
        "features": {"channels": "raw acceleration and gyro magnitudes; retain six raw axes in NPZ",
                     "endpoints": "a:b inclusive", "spline": "natural CubicSpline, 200 endpoint-inclusive points each channel",
                     "normalize": "center; population std(ddof0); concatenate acc then gyro; unit L2 norm",
                     "hard_guard": "raw spline RMS==0 OR population std<=1e-12*raw spline RMS for either channel",
                     "invalid": "separate bool valid mask and all-NaN 400-vector; no score sentinel; no auth scores"},
        "quality": {"cutoff": None, "trained": False, "applied": False, "explicit_no_quality_cutoffs": True},
        "diagnostics_768_only": {"channels": list(DIAG_CHANNELS), "lags": DIAG_LAGS.tolist(),
                                 "anchors": "np.arange(10,N-410); 10..357; SAME 348 pairs at EVERY lag",
                                 "pair_count": 348, "calculation": "np.corrcoef stacked four anchors and four shifted channels; take corresponding 4 cross entries; float64",
                                 "nearmax": {"rtol": 0, "atol": ATOL},
                                 "gaps": "max-minus-selected, top-minus-second, max-minus-largest-outside-nearset; all defined with explicit finite counts",
                                 "harmonics": "report P/2,P,2P whenever in 40..400, plus every multiple of analytic observable period <=400 (P/2 for balanced acceleration, P otherwise)",
                                 "extraction_effect": "NONE; never rescue, reject, impute, or change selected extraction"},
        "fixed_numerical_checks": {"tile_sine": "np.tile(np.sin(2*np.pi*np.arange(100)/100),3)[:256]; unfiltered",
                                   "checks": "Both tile-sine correlations and exactly equal artificial maxima vector: all nine Cartesian shifts {-1,0,+1} ULP at lags100,200 only; every other lag unchanged. Also reverse enumeration for every vector.",
                                   "equal_vector": "zeros on 80..200 except r100=r200=1",
                                   "requirements": "candidate lag100 and nearset[100,200] stable for all fixed vectors and reversed enumeration; exact divergence expected and logged",
                                   "boundary_scope": "No robustness claim at arbitrary 1e-10 membership boundaries"},
        "geometric_reference": {"definition": "Continuous local minima of prescribed RAW acceleration norm, independently enumerated; not theta0 stride events and not detected filtered minima",
                                "balanced_and_alternating_minimum_phases": [0, 0.5],
                                "gravity_and_weak": "negative-to-positive roots of a(theta).a'(theta) on fixed 4096 subintervals [-pi/4096,2pi-pi/4096]; brentq xtol1e-14,rtol4eps; positive curvature; enumerate all roots; require exactly one minimum",
                                "enumeration": "times=P*(k+root_phase-j/32); keep [10,N-11]; all continuous minima pairs with same 80..200, +/-10% selected-lag constraints; no snapping, no tolerance rescue",
                                "pair_delta": "same-root difference computed as integer cycle difference*P; otherwise (cycle difference+root phase difference)*P",
                                "report": "all reference minima and eligible pairs, chosen reference pair, availability, phase holes, observed/reference disagreement; geometry is not an extraction guarantee"},
        "truth_reporting": {"events": "P*(k-j/32) in sample interval and interior margins; labels are stipulated",
                            "alias": "nearest of 0.5,1,2 within relative10%; otherwise other; unavailable if absent; metadata only",
                            "flags_separate": ["acceleration harmonic ambiguity", "full six-axis signal half-periodicity", "joint magnitude half-periodicity", "selected interval alias", "finite selected interval alias"],
                            "balanced_limit": "Acceleration alone cannot identify physical P because its period is P/2. Gyro magnitude 40-8*cos(theta) contains P; THIS joint-magnitude fixture is not unidentifiable. Unique nearmax does not imply stride.",
                            "general_limit": "Identical joint magnitudes can in general be ambiguous; no extra generator is implemented or tested"},
        "record_row_mapping": "row=fixture_index*4+length_index*2+rule_index; fixture_index=(family_index*10+period_index)*32+phase; corr_row=fixture_index*2+length_index",
        "comparison_keys": reference["comparison_keys"], "original_hashes": reference["original_hashes"],
        "expected_legacy": {"records": 1280, "finite": 848, "missing_pair": 192, "channel_degenerate": 240, "finite_half_alias": 96},
        "runtime_pins": {"python": "3.14.8", "numpy": "2.5.4", "scipy": "1.18.1"},
        "budget": {"single_cpu_affinity": True, "cuda_off": True, "soft_cpu_seconds": 290, "hard_cpu_seconds": 299, "cooperative_stop_seconds": 270},
        "scope": "Standalone synthetic only. No real TRAIN/DEV/data/archive/CALIB/TEST, fitting, installs, Git, SSH, child agents, noise/gain/conditions/seed/parameter search or old-artifact writes.",
        "literature": {"idnet": {"url": "https://arxiv.org/html/1606.03238", "section": "3", "context": "Eight normalized channels, orientation transform and adaptive templates are absent here; this is not IDNet recreation"},
                       "derawi": {"doi": "10.1109/IIHMSP.2010.84", "year": 2010, "context": "Longer-walk cycle detection does not establish this synthetic window's coverage; DOI metadata supplied in brief and original r1, DOI endpoint unavailable during preparation"}},
        "length_rationale": "Researcher's fixed N768 is first multiple of256 greater than 3*Pmax+2*margin=620; geometric rationale only, never an empirical optimum or gait guarantee",
        "review": "Root independently reviews before any publication or real DEV plan; no publication or real-data authorization inferred",
    }


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


def gravity_derivative(theta):
    s, c = np.sin, np.cos
    a = np.array([0.8*s(theta)+0.15*s(2*theta), 0.6*c(theta)+0.1*s(3*theta), 9.81-1.2*c(theta)+0.2*c(2*theta)])
    d = np.array([0.8*c(theta)+0.3*c(2*theta), -0.6*s(theta)+0.3*c(3*theta), 1.2*s(theta)-0.4*s(2*theta)])
    dd = np.array([-0.8*s(theta)-0.6*s(2*theta), -0.6*c(theta)-0.9*s(3*theta), 1.2*c(theta)-0.8*c(2*theta)])
    return float(a @ d), float(d @ d + a @ dd)


def analytic_minima_roots():
    grid = np.linspace(-np.pi/4096, 2*np.pi-np.pi/4096, 4097)
    derivative = np.array([gravity_derivative(theta)[0] for theta in grid])
    roots = []
    for left, right, dl, dr in zip(grid[:-1], grid[1:], derivative[:-1], derivative[1:]):
        if dl < 0 <= dr:
            root = brentq(lambda theta: gravity_derivative(theta)[0], left, right, xtol=1e-14, rtol=4*np.finfo(float).eps)
            assert gravity_derivative(root)[1] > 0
            roots.append(float((root % (2*np.pi))/(2*np.pi)))
    assert len(roots) == 1, "Prescribed gravity minima enumeration failed; no retuning"
    return {FAMILIES[0]: roots, FAMILIES[1]: [0.0, 0.5], FAMILIES[2]: [0.0, 0.5], FAMILIES[3]: roots}


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


def diagnostic(channels, family, period):
    out = np.empty((4, len(DIAG_LAGS)))
    anchor = channels[:, ANCHORS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, lag in enumerate(DIAG_LAGS):
            matrix = np.corrcoef(np.concatenate((anchor, channels[:, ANCHORS+lag]), axis=0))
            out[:, j] = matrix[np.arange(4), np.arange(4)+4]
    summaries = []
    for ci, name in enumerate(DIAG_CHANNELS):
        row = correlation_summary(out[ci], DIAG_LAGS)
        observable = period/2 if family == FAMILIES[2] and ci in (0, 2) else period
        alternatives = sorted({int(v) for v in (period/2, period, 2*period) if 40 <= v <= 400} |
                              {int(k*observable) for k in range(1, int(400/observable)+1) if 40 <= k*observable <= 400})
        row.update(channel=name, anchor_pair_count=348,
                   harmonic_alternatives=[{"lag": lag, "physical_P_ratio": lag/period,
                                           "correlation": float(out[ci, lag-40]) if np.isfinite(out[ci, lag-40]) else None,
                                           "gap_below_max": None if row["max_correlation"] is None or not np.isfinite(out[ci, lag-40]) else row["max_correlation"]-float(out[ci, lag-40]),
                                           "in_nearmax_set": lag in row["all_nearmax_lags"]} for lag in alternatives],
                   analytic_observable_period_samples=observable,
                   acceleration_harmonic_ambiguity=family == FAMILIES[2] and ci in (0, 2),
                   unique_nearmax_is_stride_evidence=False, extraction_effect="none")
        summaries.append(row)
    return out, summaries


def counts(rows):
    return {"records": len(rows), "finite": sum(r["valid_pre_quality"] for r in rows),
            "missing_pair": sum(r["failure_reason"] == "no_eligible_minima_pair" for r in rows),
            "channel_degenerate": sum(r["failure_reason"] == "channel_degenerate" for r in rows),
            "failures": dict(Counter(r["failure_reason"] or "none" for r in rows)),
            "lag_aliases": dict(Counter(r["selected_lag_alias"] for r in rows)),
            "pair_aliases_all_rows": dict(Counter(r["selected_pair_alias"] for r in rows)),
            "finite_pair_aliases": dict(Counter(r["selected_pair_alias"] for r in rows if r["valid_pre_quality"])),
            "finite_half_alias": sum(r["valid_pre_quality"] and r["selected_pair_alias"] == "half_stride" for r in rows),
            "geometric_available": sum(r.get("geometric_reference", {}).get("available_at_selected_lag", False) for r in rows),
            "geometric_phase_hole_indices": [r["phase_index"] for r in rows if "geometric_reference" in r and not r["geometric_reference"]["available_at_selected_lag"]],
            "no_pair_despite_geometric_available": sum(r["failure_reason"] == "no_eligible_minima_pair" and r.get("geometric_reference", {}).get("available_at_selected_lag", False) for r in rows),
            "observed_pair_despite_geometric_hole": sum(r["pair_bounds_inclusive"] is not None and "geometric_reference" in r and not r["geometric_reference"]["available_at_selected_lag"] for r in rows)}


def summary(rows):
    groups = []
    by_family = []
    by_length_rule = []
    for n in LENGTHS:
        for rule in RULES:
            subset = [r for r in rows if r["length"] == n and r["rule"] == rule]
            by_length_rule.append({"length": n, "rule": rule, **counts(subset)})
            for family in FAMILIES:
                fr = [r for r in subset if r["family"] == family]
                by_family.append({"length": n, "rule": rule, "family": family, **counts(fr)})
                for period in PERIODS:
                    pr = [r for r in fr if r["period_samples"] == period]
                    groups.append({"length": n, "rule": rule, "family": family, "period_samples": period, **counts(pr)})
    return {"records": len(rows), "by_length_rule": by_length_rule, "by_family_length_rule": by_family, "by_family_period_length_rule": groups}


def write_report(report):
    report["timings"] = {"process_cpu_seconds": time.process_time(), "elapsed_after_import_cpu_seconds": time.process_time()-START_CPU,
                         "elapsed_after_import_wall_seconds": time.perf_counter()-START_WALL,
                         "cpu_affinity": sorted(os.sched_getaffinity(0)), "limits": {"soft_cpu_seconds": 290, "hard_cpu_seconds": 299, "cooperative_stop_seconds": 270}}
    temporary = ROOT / "report.pending.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(REPORT)


class BudgetStop(Exception):
    pass


def budget():
    if STOP or time.process_time() > 270:
        raise BudgetStop("Preserve partial attempt at cooperative single-core CPU cap; never retry in place")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    assert platform.python_version() == "3.14.8" and np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1", "Pinned shared environment required; no installation or substitution"
    reference = json.loads(REFERENCE.read_text())
    frozen_manifest = manifest(reference)
    runtime = runtime_identity()
    prereg_files = {name: file_sha(ROOT/name) for name in ("prototype.py", "audit.py", "README.md", "cached_reference.json")}
    if args.freeze_only:
        with REPORT.open("x") as handle:
            report = {"status": "frozen_before_measurement", "freeze_utc": utc(), "manifest": frozen_manifest,
                      "source_sha256": prereg_files["prototype.py"], "manifest_sha256": fingerprint(frozen_manifest),
                      "preregistration_file_sha256": prereg_files, "runtime_identity": runtime, "runtime_sha256": fingerprint(runtime),
                      "measurement_started": False, "record_rows_completed": 0, "diagnostic_rows_completed": 0}
            handle.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key: report[key] for key in ("status", "source_sha256", "manifest_sha256", "runtime_sha256")}), flush=True)
        return
    report = json.loads(REPORT.read_text())
    if (report["status"] != "frozen_before_measurement" or report["preregistration_file_sha256"] != prereg_files or
            report["manifest"] != frozen_manifest or report["manifest_sha256"] != fingerprint(frozen_manifest) or
            report["runtime_sha256"] != fingerprint(runtime)):
        raise SystemExit("Refusing changed freeze/runtime or previously attempted root; preserve all evidence")
    # An exclusive durable marker blocks retries even if killed before first checkpoint.
    with (ROOT/"attempt.started.json").open("x") as marker:
        json.dump({"started_utc": utc(), "source_sha256": prereg_files["prototype.py"], "manifest_sha256": report["manifest_sha256"]}, marker)
    report.update(status="running_incomplete", measurement_started=True, measurement_start_utc=utc(), stage="legacy_reproduction")
    write_report(report)
    arrays = {}
    records, diag_records = [], []
    journal = diag_journal = None
    try:
        (ROOT/"arrays").mkdir()
        specs = {"windows_768": ((1280, 6, 768), np.float64), "extractor_correlations": ((2560, 121), np.float64),
                 "diagnostic_correlations": ((1280, 4, 361), np.float64), "features": ((5120, 400), np.float64),
                 "feature_valid_mask": ((5120,), np.bool_), "record_present_mask": ((5120,), np.bool_),
                 "diagnostic_present_mask": ((1280,), np.bool_), "fixture_present_mask": ((1280,), np.bool_),
                 "correlation_present_mask": ((2560,), np.bool_)}
        for name, (shape, dtype) in specs.items():
            arrays[name] = np.lib.format.open_memmap(ROOT/"arrays"/(name+".npy"), mode="w+", dtype=dtype, shape=shape)
            arrays[name][:] = np.nan if dtype == np.float64 else False
        journal = (ROOT/"records.jsonl").open("x")
        diag_journal = (ROOT/"diagnostics.jsonl").open("x")
        keys = [(family, period, phase) for family in FAMILIES for period in PERIODS for phase in range(32)]
        assert len(reference["records"]) == len(keys) == 1280
        cached = {(r["family"], r["period_samples"], r["phase_index"]): r for r in reference["records"]}
        assert len(cached) == 1280 and set(cached) == set(keys)
        roots = analytic_minima_roots()
        report["analytic_minimum_root_phases_cycles"] = roots
        report["legacy_comparison"] = {"passed": False, "records_compared": 0, "all_cached_fields_compared": True,
                                        "comparison_keys": reference["comparison_keys"], "numeric_comparison": "Float.hex exact; missing keys compared; no tolerance",
                                        "mismatches": [], "bitwise_nested_prefix_checks": 0}

        def flush():
            for array in arrays.values():
                array.flush()
            journal.flush()
            diag_journal.flush()
            report["record_rows_completed"] = len(records)
            report["diagnostic_rows_completed"] = len(diag_records)
            write_report(report)

        def add_record(fi, li, ri, base, feature, mags, smooth, rs):
            family, period, phase = keys[fi]
            n = LENGTHS[li]
            rec = dict(base)
            row = fi*4+li*2+ri
            corrrow = fi*2+li
            rec.update(row_id=row, fixture_index=fi, fixture_key=f"{family}/P{period}/phase{phase:02d}",
                       length=n, rule=RULES[ri], correlation_row=corrrow,
                       family=family, period_samples=period, phase_index=phase, phase_cycles=phase/32)
            rec.update(truth(period, phase, rec, n))
            cs = correlation_summary(rs)
            rec.update(all_nearmax_lags=cs["all_nearmax_lags"], correlation_gaps=cs["correlation_gaps"],
                       finite_correlation_count=cs["finite_correlation_count"], undefined_correlation_count=cs["undefined_correlation_count"],
                       selected_lag_correlation=None if rec["selected_lag"] is None else float(rs[rec["selected_lag"]-80]),
                       selected_interval_interpretation="putative repeat interval; anatomical stride not inferred",
                       status="finite_pre_quality" if feature is not None else rec["failure_reason"],
                       input_finite_count=int(np.isfinite(arrays["windows_768"][fi, :, :n]).sum()), input_sample_value_count=6*n,
                       magnitude_finite_count=int(np.isfinite(mags).sum()), filtered_acc_finite_count=int(np.isfinite(smooth).sum()),
                       guard_evaluated=rec["pair_bounds_inclusive"] is not None,
                       channel_population_std=rec.get("channel_population_std"), channel_raw_rms=rec.get("channel_raw_rms"),
                       channel_degenerate=rec.get("channel_degenerate"), feature_norm=rec.get("feature_norm"),
                       feature_channel_means=rec.get("feature_channel_means"), explicit_no_quality_cutoffs=True,
                       quality_cutoff=None, quality_trained=False, quality_applied=False,
                       acceleration_harmonic_ambiguity=family == FAMILIES[2], full_six_axis_signal_half_periodic=False,
                       joint_magnitudes_half_periodic=False, gyro_contains_physical_P=True,
                       extracted_interval_alias=rec["selected_pair_alias"] not in ("unavailable", "one_stride"),
                       finite_extracted_interval_alias=rec["valid_pre_quality"] and rec["selected_pair_alias"] != "one_stride",
                       unique_nearmax_is_stride_evidence=False,
                       geometric_reference=geometric_reference(family, period, phase, n, rec["selected_lag"], roots))
            assert not arrays["record_present_mask"][row]
            arrays["feature_valid_mask"][row] = feature is not None
            if feature is not None:
                arrays["features"][row] = feature
            arrays["record_present_mask"][row] = True
            records.append(rec)
            journal.write(json.dumps(rec, allow_nan=False, separators=(",", ":"))+"\n")

        # No N768 extraction, candidate extraction or long-window diagnostic before this gate.
        for fi, (family, period, phase) in enumerate(keys):
            budget()
            long = fixture(family, period, phase, 768)
            short = np.ascontiguousarray(long[:, :256])
            independent = fixture(family, period, phase, 256)
            assert short.tobytes() == independent.tobytes(), f"Non-bitwise prefix: {keys[fi]}"
            report["legacy_comparison"]["bitwise_nested_prefix_checks"] += 1
            arrays["windows_768"][fi] = long
            arrays["fixture_present_mask"][fi] = True
            mags = np.stack([np.linalg.norm(short[:3], axis=0), np.linalg.norm(short[3:], axis=0)])
            smooth = sosfiltfilt(SOS, mags[0], padtype="odd", padlen=9)
            rs = correlations(smooth)
            arrays["extractor_correlations"][fi*2] = rs
            arrays["correlation_present_mask"][fi*2] = True
            rec, feature = extract(mags, smooth, rs, RULES[0])
            rec.update(family=family, period_samples=period, phase_index=phase, phase_cycles=phase/32)
            rec.update(truth(period, phase, rec, 256))
            original = cached[keys[fi]]
            # Float.hex catches bit-level numeric changes including signed zero.
            def bits(value):
                if isinstance(value, float):
                    return {"float64_hex": value.hex()}
                if isinstance(value, dict):
                    return {key: bits(val) for key, val in value.items()}
                if isinstance(value, list):
                    return [bits(val) for val in value]
                return value
            mismatch = [key for key in sorted(set(rec)|set(original)) if key not in rec or key not in original or bits(rec[key]) != bits(original[key])]
            if mismatch:
                report["legacy_comparison"]["mismatches"].append({"fixture": list(keys[fi]), "keys": mismatch, "reproduced": rec, "cached": original})
                flush()
                raise AssertionError("Cached N256 exact-rule mismatch; stop before longer execution")
            report["legacy_comparison"]["records_compared"] += 1
            add_record(fi, 0, 0, rec, feature, mags, smooth, rs)
            if phase == 31:
                flush()
                print(json.dumps({"stage": "legacy", "fixtures": fi+1, "cpu_seconds": round(time.process_time(), 3)}), flush=True)
        observed = counts(records)
        expected = frozen_manifest["expected_legacy"]
        assert all(observed[key] == value for key, value in expected.items()), (observed, expected)
        report["legacy_comparison"].update(passed=True, observed_counts=observed,
                                             cached_records_sha256=fingerprint(reference["records"]),
                                             reproduced_cached_fields_sha256=fingerprint([{key: r[key] for key in cached[keys[r["fixture_index"]]]} for r in records]))
        assert report["legacy_comparison"]["cached_records_sha256"] == report["legacy_comparison"]["reproduced_cached_fields_sha256"]
        report["stage"] = "legacy_passed_before_longer_execution"
        flush()
        print(json.dumps({"legacy_passed": True, **{key: observed[key] for key in expected}}), flush=True)
        report["fixed_numerical_checks"], check_vectors = fixed_checks()
        np.save(ROOT/"arrays"/"fixed_check_vectors.npy", check_vectors, allow_pickle=False)
        report["stage"] = "locked_candidate_and_N768"
        flush()
        for fi, (family, period, phase) in enumerate(keys):
            budget()
            for li, n in enumerate(LENGTHS):
                window = np.ascontiguousarray(arrays["windows_768"][fi, :, :n])
                mags = np.stack([np.linalg.norm(window[:3], axis=0), np.linalg.norm(window[3:], axis=0)])
                smooth = sosfiltfilt(SOS, mags[0], padtype="odd", padlen=9)
                if li == 0:
                    rs = np.asarray(arrays["extractor_correlations"][fi*2])
                    rule_indices = (1,)
                else:
                    rs = correlations(smooth)
                    arrays["extractor_correlations"][fi*2+1] = rs
                    arrays["correlation_present_mask"][fi*2+1] = True
                    rule_indices = (0, 1)
                for ri in rule_indices:
                    rec, feature = extract(mags, smooth, rs, RULES[ri])
                    add_record(fi, li, ri, rec, feature, mags, smooth, rs)
                if li == 1:
                    filtered_gyro = sosfiltfilt(SOS, mags[1], padtype="odd", padlen=9)
                    diagnostic_rs, channel_summaries = diagnostic(np.stack([mags[0], mags[1], smooth, filtered_gyro]), family, period)
                    arrays["diagnostic_correlations"][fi] = diagnostic_rs
                    arrays["diagnostic_present_mask"][fi] = True
                    dr = {"fixture_index": fi, "fixture_key": f"{family}/P{period}/phase{phase:02d}", "family": family,
                          "period_samples": period, "phase_index": phase, "length": 768, "channels": channel_summaries,
                          "same_anchor_count": 348, "no_extraction_effect": True, "physical_stride_identified": False}
                    diag_records.append(dr)
                    diag_journal.write(json.dumps(dr, allow_nan=False, separators=(",", ":"))+"\n")
            if phase == 31:
                flush()
                print(json.dumps({"stage": "locked_recreation", "fixtures": fi+1, "records": len(records), "cpu_seconds": round(time.process_time(), 3)}), flush=True)
        budget()
        assert len(records) == 5120 and len(diag_records) == 1280
        assert all(array.all() for name, array in arrays.items() if name.endswith("present_mask"))
        records.sort(key=lambda row: row["row_id"])
        assert [row["row_id"] for row in records] == list(range(5120))
        report["summary"] = summary(records)
        report["diagnostic_summary"] = [{"channel": name, "fixtures": 1280,
                                         "nearmax_set_size_counts": dict(Counter(len(row["channels"][ci]["all_nearmax_lags"]) for row in diag_records)),
                                         "unique_nearmax_is_stride_evidence": False} for ci, name in enumerate(DIAG_CHANNELS)]
        report["scientific_limits"] = [
            "Only four noiseless analytic fixtures, two preregistered lengths and two rules; these counts do not estimate gait, subject, session, authentication or real-DEV performance.",
            "N768's 620-sample geometric rationale is not an empirical optimum or sufficient condition for correct stride extraction or finite features.",
            "Balanced acceleration has period P/2; physical stride is unidentifiable from acceleration alone. Its gyro magnitude has period P, so joint magnitudes in THIS fixture are not unidentifiable.",
            "In general identical joint magnitudes can be ambiguous, but no additional generator was planned or tested. Full-signal and extracted-interval aliases are separately flagged.",
            "Nearmax selection stabilizes only the fixed numerical checks. A unique nearmax set, high correlation, or finite feature does not identify anatomical stride.",
            "Long-window harmonic diagnostics use common fixed anchors and never rescue extraction. Filter boundaries can change observed minima relative to independent raw-minimum geometry.",
            "Hard near-flat gyro guards remain failures without imputation. Aliases remain extraction successes when features are finite; no quality cutoff is trained or applied.",
            "IDNet section3's eight channels, orientation transform and adaptive templates are absent; this two-magnitude experiment is not an IDNet reproduction. Derawi paper is 2010 DOI10.1109/IIHMSP.2010.84.",
            "Independent root review is required before publication or any real DEV plan. No such work is performed here.",
        ]
        report["stage"] = "lossless_packaging"
        flush()
        mapping = np.array([[fi, LENGTHS[li], ri, fi*2+li] for fi in range(1280) for li in range(2) for ri in range(2)], dtype=np.int64)
        correlation_mapping = np.array([[fi, n] for fi in range(1280) for n in LENGTHS], dtype=np.int64)
        fixture_key_strings = np.array([f"{f}/P{p}/phase{j:02d}" for f,p,j in keys])
        extra = {"record_mapping": mapping, "record_mapping_columns": np.array(["fixture_index", "length", "rule_index", "correlation_row"]),
                 "correlation_mapping": correlation_mapping, "correlation_mapping_columns": np.array(["fixture_index", "length"]),
                 "fixture_keys": fixture_key_strings, "rules": np.array(RULES), "extractor_lags": LAGS,
                 "diagnostic_lags": DIAG_LAGS, "diagnostic_anchors": ANCHORS, "diagnostic_channels": np.array(DIAG_CHANNELS),
                 "fixed_check_vectors": check_vectors, "fixed_check_lags": LAGS}
        with (ROOT/"audit_arrays.npz").open("xb") as handle:
            np.savez(handle, **arrays, **extra)
        journal.close()
        diag_journal.close()
        artifact_names = ["records.jsonl", "diagnostics.jsonl", "audit_arrays.npz", "attempt.started.json"]
        report["artifact_sha256"] = {name: file_sha(ROOT/name) for name in artifact_names}
        report["array_sha256"] = {path.name: file_sha(path) for path in sorted((ROOT/"arrays").glob("*.npy"))}
        report["storage"] = {"npz": "audit_arrays.npz", "records": "records.jsonl", "diagnostics": "diagnostics.jsonl",
                             "mapping": "NPZ record_mapping indexed by row_id; correlations indexed by correlation_row; diagnostic arrays indexed by fixture_index",
                             "partial_arrays": "arrays/*.npy preserve all allocated partial state; present masks distinguish unmeasured values",
                             "dtype": "lossless float64; invalid features NaN with separate bool mask; no pickle; no scores"}
        scientific_fields = ["source_sha256", "manifest_sha256", "preregistration_file_sha256", "runtime_sha256", "legacy_comparison",
                             "analytic_minimum_root_phases_cycles", "fixed_numerical_checks", "summary", "diagnostic_summary", "scientific_limits", "artifact_sha256", "array_sha256", "storage"]
        report["scientific_fingerprint_fields"] = scientific_fields
        report["scientific_fingerprint_sha256"] = fingerprint({key: report[key] for key in scientific_fields})
        report.update(status="complete", stage="complete", completed_utc=utc(), record_rows_completed=5120, diagnostic_rows_completed=1280)
        write_report(report)
        lines = ["# Locked synthetic length recreation", "", "Completed one exclusive frozen attempt. Root review precedes publication or any real DEV plan.", "",
                 "Legacy control: all 1,280 cached records, all cached fields, match exactly; 1,280 six-axis nested prefixes are bitwise identical.", "",
                 "| N | Rule | Finite / 1280 | Missing pair | Degenerate gyro | Finite half aliases |", "|---|---|---:|---:|---:|---:|"]
        for item in report["summary"]["by_length_rule"]:
            lines.append(f"| {item['length']} | {item['rule']} | {item['finite']} | {item['missing_pair']} | {item['channel_degenerate']} | {item['finite_half_alias']} |")
        lines += ["", f"Run process CPU: {report['timings']['process_cpu_seconds']:.6f}s; wall after imports: {report['timings']['elapsed_after_import_wall_seconds']:.6f}s.",
                  "", "All 5,120 record rows and 1,280 four-channel long-window diagnostics are retained. All fixed numerical checks pass; exact-rule divergences are expected.", "",
                  *["- "+limit for limit in report["scientific_limits"]], "", "Hashes and scientific fingerprint are in report.json. Named lossless audit_arrays.npz includes six raw axes, correlations, vectors, masks and row maps. audit.py performs saved-artifact checks without new extraction.", ""]
        with (ROOT/"RESULTS.md").open("x") as handle:
            handle.write("\n".join(lines))
        print(json.dumps({"status": "complete", "summary": report["summary"]["by_length_rule"], "cpu_seconds": report["timings"]["process_cpu_seconds"], "scientific_fingerprint_sha256": report["scientific_fingerprint_sha256"]}), flush=True)
    except BaseException as error:
        report.update(status="incomplete_cpu_cap" if isinstance(error, BudgetStop) else "failed_preserved", exception={"type": type(error).__name__, "message": str(error)},
                      record_rows_completed=len(records), diagnostic_rows_completed=len(diag_records), summary=summary(records))
        for array in arrays.values():
            array.flush()
        if journal is not None and not journal.closed:
            journal.flush()
        if diag_journal is not None and not diag_journal.closed:
            diag_journal.flush()
        write_report(report)
        raise


if __name__ == "__main__":
    main()
