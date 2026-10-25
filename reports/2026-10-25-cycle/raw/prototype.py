#!/usr/bin/env python3
"""Frozen, synthetic-only cyclemag400 feasibility test. No dataset or fit code.

Run --freeze-only once, inspect the manifest, then --run. The latter refuses a
changed source/manifest and preserves partial results on a CPU budget signal.
All reads are this source, its report, interpreter/library code and /proc runtime
metadata; all writes are report.json beside this source. No network/subprocesses.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
import warnings

CPU_START = time.process_time()
WALL_START = time.perf_counter()
resource.setrlimit(resource.RLIMIT_CPU, (290, 299))
STOP_REQUESTED = False


def cpu_signal(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGXCPU, cpu_signal)

import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

ROOT = Path("/home/sherlock/Code/github/VariableThe/artifacts/cycle-feasibility-r1")
SOURCE = ROOT / "prototype.py"
REPORT = ROOT / "report.json"
PERIODS = (80, 90, 100, 110, 120, 128, 140, 160, 180, 200)
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
LAGS = np.arange(80, 201)
SEED = 20261025
SAMPLE_COUNT = 256
GUARD = 1e-12
SOS = butter(2, 3.0, btype="lowpass", fs=100.0, output="sos")

# Every fixture and every diagnostic/control is declared before any measurements.
# phase_index is j in theta = 2*pi*(n/P+j/32), n=0,...,255.
# theta=0 mod 2*pi is the stipulated physical stride boundary, even where
# acceleration alone has a shorter observable period. No gait labels are inferred.
MANIFEST = {
    "seed": SEED,
    "periods_samples": list(PERIODS),
    "phases": {"count": 32, "cycles": [j / 32 for j in range(32)]},
    "grid_count": 1280,
    "family_equations": {
        "gravity_periodic_6axis": {
            "acc": ["0.8*sin(theta)+0.15*sin(2*theta)", "0.6*cos(theta)+0.1*sin(3*theta)", "9.81-1.2*cos(theta)+0.2*cos(2*theta)"],
            "gyro": ["25*sin(theta)+4*sin(2*theta)", "18*cos(theta)+3*sin(3*theta)", "10*sin(theta+0.4)+2*cos(2*theta)"],
        },
        "alternating_step_harmonic": {
            "acc": ["0.3*sin(theta)", "0.2*cos(theta)", "9.81-1.1*cos(2*theta)-0.25*cos(theta)"],
            "gyro": ["20*sin(theta)+5*sin(2*theta)", "15*cos(theta)+2*cos(2*theta)", "12+4*sin(theta+0.3)"],
        },
        "pure_periodic_balanced_peaks": {
            "acc": ["0", "0", "9.81-1.2*cos(2*theta)"],
            "gyro": ["0", "0", "40-8*cos(theta)"],
            "interpretation": "Physical stride P remains observable in gyro; acceleration has exactly balanced steps and period P/2.",
        },
        "weak_degenerate_gyro": {
            "acc": ["0.8*sin(theta)+0.15*sin(2*theta)", "0.6*cos(theta)+0.1*sin(3*theta)", "9.81-1.2*cos(theta)+0.2*cos(2*theta)"],
            "gyro": ["0", "0", "30*(1+1e-14*cos(theta))"],
            "interpretation": "Physical stride P; almost constant gyro magnitude should fail the frozen channel guard if a pair exists.",
        },
    },
    "truth": "Continuous event times P*(k-j/32). Count full anchored strides via pairs of consecutive events in [0,255] and [10,245].",
    "alias_reporting_only": "Lag/pair ratios nearest one of 0.5,1,2 within relative 10%; otherwise other. This tolerance NEVER changes detection, ties, gates or features.",
    "controls": {
        "base": {"family": "gravity_periodic_6axis", "period_samples": 100, "phase_index": 7},
        "transforms": ["gain_0.9_both_sensors", "gain_1.1_both_sensors", "common_rotation_30deg_axis_1_2_3", "common_rotation_90deg_axis_1_2_3"],
        "feature_comparison_tolerance": 1e-10,
        "invariance_comparison_only": "Report exact detector changes as well as tolerant feature comparisons; never tolerate lag ties.",
        "batch": [["gravity_periodic_6axis", 100, 7], ["alternating_step_harmonic", 160, 11], ["pure_periodic_balanced_peaks", 180, 19], ["weak_degenerate_gyro", 120, 3]],
        "reorder": "Reverse fixed batch; invert permutation; require bitwise identical records and features.",
        "future": "Embed base in 6x512 stream; replace samples 256:512 using RNG(seed) only; process prefix 0:256 before/after; require bitwise identity.",
        "degenerates": ["all_zero", "constant_gravity_constant_gyro", "periodic_acc_zero_gyro", "periodic_acc_constant_gyro", "nonfinite_one_acc_sample"],
        "direct_feature_guard": ["both_constant", "zero_gyro", "relative_gyro_1e-14", "relative_gyro_1e-9"],
        "exact_periodic_pearson": "Tile fixed sin(2*pi*arange(100)/100) to 256; compare unfiltered corrcoef/dot Pearson at 100/200 and all lags. Separate diagnostic, no extra grid or phase search.",
        "handcrafted_minima": "Flat signal and minima at 20,120,135,235 with selected L=100; expect flat none and pair [20,120] by midpoint tie/indices.",
    },
    "roc_toys": {
        "all_reject": {"genuine": [None] * 4, "impostor": [None] * 4},
        "poor_coverage_perfect_conditional": {"genuine": [0.9, 0.8] + [None] * 8, "impostor": [0.2, 0.1] + [None] * 8},
        "crossing_gated": {"genuine": [0.9, 0.6, 0.3, None], "impostor": [0.8, 0.5, 0.2, None]},
    },
    "quality_cutoff": {"own_9_train": None, "own_12_train": None, "estimated": False, "applied": False,
                       "future_rule": "max(0, identity-balanced fifth percentile of TRAIN maxr), fitted separately on each own TRAIN split; weighted quantile convention must be preregistered."},
    "scope": "Synthetic physical fixtures only; no real DEV, TRAIN, raw data, archive, NPZ, model fits, installs, Git, SSH or children.",
}

ASSUMPTIONS = [
    "Input float64, row order ax,ay,az,gx,gy,gz; 100Hz; native m/s^2 and deg/s. No noise/error search.",
    "Detector uses RAW acceleration norm, second-order 3Hz Butterworth sosfiltfilt(padtype='odd',padlen=9). Lag Pearson is np.corrcoef(y[:-L],y[L:])[0,1] with its standard [-1,1] roundoff clipping; undefined lags excluded, all undefined means no_lag. No variance tolerance at detector and no near-tie tolerance.",
    "Max Pearson chooses exact floating maximum then smallest L. Strict minima use immediate neighbors and candidate indices 10 through 245 inclusive. All (not just adjacent) minima pairs considered. Eligible if 80<=b-a<=200 and abs((b-a)-L)<=0.1*L; ordered by (abs((b-a)-L),abs((a+b)/2-127.5),a,b).",
    "No TRAIN quality cutoff is applied, including no guessed zero cutoff; extraction success is pre-quality and is an upper bound on complete-pipeline coverage.",
    "Extract RAW magnitudes over a:b inclusive, natural cubic spline to 200 endpoint-inclusive points/channel. Center and divide population std(ddof=0), concatenate acceleration then gyro and unit normalize. Guard rejects if raw spline RMS==0 or centered population std<=1e-12*raw spline RMS for either channel. RMS is a frozen interpretation of the otherwise unspecified relative scale.",
    "Truth/alias 10% tolerance is evaluation metadata only; full-pair length is b-a samples, inclusive stored sample count is b-a+1. Ground-truth events are continuous, not snapped to measured minima.",
    "No label means no finite lag here; fixture physical labels are reporting-only and never passed to extraction. This does not resolve ambiguities in an eventual real-data protocol.",
    "All fixed fixtures have equal reporting weight; their fractions are synthetic stress counts, never estimates of actual gait/DEV rates.",
    "ROC higher score means accept, score>=threshold only for valid claims. Invalid scores are JSON null with explicit masks. Only distinct valid scores plus reject-all are threshold states; terminal means all VALID scores accepted, not artificial accept-all.",
    "ROC EER is exact equality or linear crossing interpolation of adjacent attainable threshold states (randomized threshold convention). If no crossing, EER=null with terminal FAR/FRR. Conditional EER is explicitly secondary and null if either valid class is empty.",
]

SOURCES = [
    {"url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html", "verified": "Official docs: fs makes Wn Hz; Butterworth cutoff gain 1/sqrt(2); SOS output. Online page identifies 1.18.0; execution is installed 1.18.1."},
    {"url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html", "verified": "Official docs: forward/backward SOS, odd extension, explicit pad length; second-order default formula gives 9."},
    {"url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html", "verified": "Official docs: natural imposes zero endpoint second derivatives; not the default boundary condition."},
    {"url": "https://arxiv.org/html/1606.03238", "sections": "3.1-3.4", "verified": "IDNet uses a one-second template, correlation-distance cycle extraction and 0.9 old + 0.1 new template updates. It transforms axes using gravity/PCA and normalizes eight channels to 200 points. This two-magnitude prototype is not full IDNet."},
    {"url": "https://www.researchgate.net/publication/221566748_Improved_Cycle_Detection_for_Accelerometer_Based_Gait_Authentication", "verified": "Author-uploaded primary full text: observed 80-140 samples/cycle at 100Hz; 20m walks, 12 sessions, repeated cycle detection on longer walks. Publication October 2010, DOI 10.1109/IIHMSP.2010.84, rather than the supplied 2009 attribution. These observations do not guarantee cycle availability in 256-sample windows."},
    {"url": "https://www.ntnu.no/ansatte/mohammad.derawi", "verified": "Author institutional bibliography corroborates the paper's 2010 publication year. Author-site PDF URL was inaccessible; primary full text was accessible in the author's ResearchGate upload."},
]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_report(report):
    report["runtime"] = {
        "process_cpu_seconds": time.process_time(),
        "measurement_cpu_seconds": time.process_time() - CPU_START,
        "measurement_wall_seconds": time.perf_counter() - WALL_START,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_limit_seconds": {"soft": 290, "hard": 299, "cooperative_stop": 280},
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "single_thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def fixture(family, period, phase_index, n=SAMPLE_COUNT):
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


def pearson_lags(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.corrcoef(y[:-int(lag)], y[int(lag):])[0, 1] for lag in LAGS])


def select_lag(rs):
    finite = np.isfinite(rs)
    if not finite.any():
        return None, None, []
    best = float(np.max(rs[finite]))
    ties = LAGS[finite & (rs == best)].tolist()
    return int(min(ties)), best, ties


def minima_and_pair(y, lag):
    minima = [i for i in range(10, 246) if y[i] < y[i - 1] and y[i] < y[i + 1]]
    pairs = [(a, b) for a in minima for b in minima if 80 <= b - a <= 200 and abs(b - a - lag) <= 0.1 * lag]
    if not pairs:
        return minima, None, 0
    pair = min(pairs, key=lambda ab: (abs(ab[1] - ab[0] - lag), abs((ab[0] + ab[1]) / 2 - 127.5), ab[0], ab[1]))
    return minima, list(pair), len(pairs)


def features_from_pair(magnitudes, pair):
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


def extract(window):
    rec = {"selected_lag": None, "raw_maxr": None, "exact_max_ties": [], "strict_minima": [], "pair_bounds_inclusive": None, "pair_length_samples": None, "pair_stored_samples": None, "eligible_pair_count": 0, "failure_reason": None, "finite_features": False, "valid_pre_quality": False, "quality_evaluated": False}
    if window.shape != (6, 256):
        rec["failure_reason"] = "bad_shape"
        return rec, None
    if not np.isfinite(window).all():
        rec["failure_reason"] = "nonfinite_input"
        return rec, None
    mags = np.stack([np.linalg.norm(window[:3], axis=0), np.linalg.norm(window[3:], axis=0)])
    smooth = sosfiltfilt(SOS, mags[0], padtype="odd", padlen=9)
    lag, maxr, ties = select_lag(pearson_lags(smooth))
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


def truth(period, phase_index, rec):
    events = [period * (k - phase_index / 32) for k in range(-2, 7)]
    raw = [v for v in events if 0 <= v <= 255]
    interior = [v for v in events if 10 <= v <= 245]
    pair = rec["pair_bounds_inclusive"]
    contained = [] if pair is None else [v for v in events if pair[0] <= v <= pair[1]]
    return {"truth_stride_period_samples": period, "truth_events_raw": raw, "truth_events_with_minima_margins": interior,
            "truth_full_anchored_stride_in_raw": len(raw) >= 2, "truth_full_anchored_stride_with_margins": len(interior) >= 2,
            "truth_interval_can_fit_stride_raw": period <= 255, "truth_interval_can_fit_stride_with_margins": period <= 235,
            "selected_lag_alias": alias(rec["selected_lag"], period), "selected_pair_alias": alias(rec["pair_length_samples"], period),
            "selected_pair_period_ratio": None if pair is None else (pair[1] - pair[0]) / period,
            "truth_anchored_strides_contained_in_selected_pair": max(0, len(contained) - 1)}


def manual_filter(y):
    """Independent odd extension, steady-state initial states and DF-II recurrence."""
    b0, b1, b2, a0, a1, a2 = SOS[0]
    assert a0 == 1.0
    extended = np.r_[2 * y[0] - y[9:0:-1], y, 2 * y[-1] - y[-2:-11:-1]]
    dc = (b0 + b1 + b2) / (1 + a1 + a2)

    def forward(x):
        z1, z2 = (dc - b0) * x[0], (b2 - a2 * dc) * x[0]
        out = np.empty_like(x)
        for i, value in enumerate(x):
            result = b0 * value + z1
            z1, z2 = b1 * value - a1 * result + z2, b2 * value - a2 * result
            out[i] = result
        return out

    return forward(forward(extended)[::-1])[::-1][9:-9]


def references_and_ties():
    # Hand bilinear transform of H(s)=1/(s^2+sqrt(2)*s+1).
    k = math.tan(math.pi * 3 / 100)
    den = 1 + math.sqrt(2) * k + k * k
    hand_sos = np.array([[k * k / den, 2 * k * k / den, k * k / den, 1, 2 * (k * k - 1) / den, (1 - math.sqrt(2) * k + k * k) / den]])
    base = fixture("gravity_periodic_6axis", 100, 7)
    mag = np.linalg.norm(base[:3], axis=0)
    smooth = sosfiltfilt(SOS, mag, padtype="odd", padlen=9)
    coefficient_error = float(np.max(np.abs(hand_sos - SOS)))
    manual_error = float(np.max(np.abs(manual_filter(mag) - smooth)))
    dc_error = float(np.max(np.abs(sosfiltfilt(SOS, np.full(256, 9.81), padtype="odd", padlen=9) - 9.81)))
    omega = 2 * math.pi * 3 / 100
    z = np.exp(-1j * omega)
    h = (SOS[0, 0] + SOS[0, 1] * z + SOS[0, 2] * z * z) / (1 + SOS[0, 4] * z + SOS[0, 5] * z * z)
    rs = pearson_lags(smooth)
    long_ref = []
    for lag in LAGS:
        left, right = smooth[:-int(lag)].astype(np.longdouble), smooth[int(lag):].astype(np.longdouble)
        left, right = left - left.mean(), right - right.mean()
        long_ref.append(float(np.sum(left * right) / np.sqrt(np.sum(left * left) * np.sum(right * right))))
    pearson_error = float(np.max(np.abs(rs - long_ref)))
    # Fixed pair-order test: [20,120] and [135,235] have equal midpoint
    # distance 57.5 from 127.5. The first indices resolve the tie.
    y = np.ones(256)
    y[[20, 120, 135, 235]] = 0
    minima, pair, count = minima_and_pair(y, 100)
    flat = minima_and_pair(np.ones(256), 100)
    tie_rs = np.zeros(121)
    tie_rs[100 - 80] = tie_rs[200 - 80] = 1
    tie_selection = select_lag(tie_rs)
    # Natural spline reference: f(x)=x^3/6 on [0,1];
    # f(x)=-u^3/6+u^2/2+u/2+1/6 on [1,2], u=x-1,
    # interpolates [0,1/6,1] with endpoint second derivatives zero.
    knots = np.array([0., 1., 2.])
    values = np.array([0., 1 / 6, 1.])
    spline = CubicSpline(knots, values, bc_type="natural")
    q = np.array([0., .5, 1., 1.5, 2.])
    u = q - 1
    hand = np.where(q <= 1, q ** 3 / 6, -u ** 3 / 6 + u ** 2 / 2 + u / 2 + 1 / 6)
    spline_error = float(np.max(np.abs(spline(q) - hand)))
    periodic = np.tile(np.sin(2 * np.pi * np.arange(100) / 100), 3)[:256]
    periodic_rs = pearson_lags(periodic)
    dot_rs = []
    for lag in LAGS:
        a, b = periodic[:-int(lag)], periodic[int(lag):]
        a, b = a - a.mean(), b - b.mean()
        dot_rs.append(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    dot_rs = np.array(dot_rs)
    return {
        "hand_bilinear_sos": hand_sos.tolist(), "scipy_sos": SOS.tolist(), "max_coefficient_error": coefficient_error,
        "manual_df2_odd_forward_backward_max_error": manual_error, "constant_dc_max_error": dc_error,
        "cutoff_single_pass_gain": float(abs(h)), "cutoff_two_pass_gain": float(abs(h) ** 2),
        "pearson_vs_longdouble_max_error": pearson_error,
        "minima_reference": {"minima": minima, "pair": pair, "eligible_pairs": count, "flat_minima": flat[0], "flat_pair": flat[1]},
        "exact_tie_reference": {"lag": tie_selection[0], "maxr": tie_selection[1], "ties": tie_selection[2]},
        "natural_spline_hand_max_error": spline_error, "natural_spline_endpoint_second_derivatives": spline(knots[[0, -1]], 2).tolist(),
        "checks_pass": bool(coefficient_error < 1e-14 and manual_error < 1e-11 and dc_error < 1e-11 and pearson_error < 1e-12 and spline_error < 1e-14 and pair == [20, 120] and flat[1] is None and tie_selection[0] == 100),
        "fixed_unfiltered_periodic_diagnostic": {
            "not_a_pipeline_grid_result": True, "exact_period": 100,
            "corrcoef": {"selected_lag": select_lag(periodic_rs)[0], "r100": float(periodic_rs[20]), "r200": float(periodic_rs[120]), "ties": select_lag(periodic_rs)[2]},
            "centered_dot_reference": {"selected_lag": select_lag(dot_rs)[0], "r100": float(dot_rs[20]), "r200": float(dot_rs[120]), "ties": select_lag(dot_rs)[2]},
            "condition": "In exact arithmetic r100=r200=1. Any floating computation with r200>r100 and no larger competitor chooses 200, since only exact ties prefer 100. No near-tie tolerance was used.",
        },
    }


def same_output(a, b):
    ra, fa = a
    rb, fb = b
    return ra == rb and ((fa is None and fb is None) or (fa is not None and fb is not None and np.array_equal(fa, fb)))


def controls():
    x = fixture("gravity_periodic_6axis", 100, 7)
    baseline = extract(x)
    axis = np.array([1., 2., 3.])
    axis /= np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    transforms = [("gain_0.9_both_sensors", x * .9, None), ("gain_1.1_both_sensors", x * 1.1, None)]
    for degrees in (30, 90):
        angle = np.deg2rad(degrees)
        rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
        rotated = np.vstack([rotation @ x[:3], rotation @ x[3:]])
        transforms.append((f"common_rotation_{degrees}deg_axis_1_2_3", rotated, rotation))
    invariance = []
    for name, changed, rotation in transforms:
        rec, feat = extract(changed)
        distance = None if baseline[1] is None or feat is None else float(np.linalg.norm(baseline[1] - feat))
        invariance.append({"name": name, "record": rec, "same_lag": rec["selected_lag"] == baseline[0]["selected_lag"],
                           "same_pair": rec["pair_bounds_inclusive"] == baseline[0]["pair_bounds_inclusive"], "maxr_delta": None if rec["raw_maxr"] is None else rec["raw_maxr"] - baseline[0]["raw_maxr"],
                           "feature_l2_difference": distance, "feature_close_1e-10": distance is not None and distance <= 1e-10,
                           "rotation_determinant": None if rotation is None else float(np.linalg.det(rotation)),
                           "rotation_orthogonality_error": None if rotation is None else float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))})
    specs = MANIFEST["controls"]["batch"]
    windows = np.stack([fixture(*spec) for spec in specs])
    single = [extract(window.copy()) for window in windows]
    batch = [extract(window) for window in windows]
    reversed_results = [extract(window) for window in windows[::-1]][::-1]
    stream = fixture("gravity_periodic_6axis", 100, 7, n=512)
    future_before = extract(stream[:, :256])
    rng = np.random.default_rng(SEED)
    stream[:, 256:] = rng.standard_normal((6, 256)) * 100
    future_after = extract(stream[:, :256])
    zero = np.zeros((6, 256))
    const = zero.copy()
    const[2], const[5] = 9.81, 30
    zero_gyro = x.copy()
    zero_gyro[3:] = 0
    const_gyro = x.copy()
    const_gyro[3:] = 0
    const_gyro[5] = 30
    nonfinite = x.copy()
    nonfinite[0, 100] = np.nan
    degenerates = [{"name": name, "record": extract(window)[0]} for name, window in zip(MANIFEST["controls"]["degenerates"], [zero, const, zero_gyro, const_gyro, nonfinite])]
    n = np.arange(256)
    acc = 9.81 - np.cos(2 * np.pi * n / 100)
    guard_cases = [np.stack([np.ones(256) * 9.81, np.ones(256) * 30]), np.stack([acc, np.zeros(256)]), np.stack([acc, 30 * (1 + 1e-14 * np.cos(2 * np.pi * n / 100))]), np.stack([acc, 30 * (1 + 1e-9 * np.cos(2 * np.pi * n / 100))])]
    direct = []
    for name, mags in zip(MANIFEST["controls"]["direct_feature_guard"], guard_cases):
        feat, failure, stats = features_from_pair(mags, [50, 150])
        direct.append({"name": name, "failure_reason": failure, "finite_features": feat is not None, **stats})
    return {"baseline": baseline[0], "invariance": invariance,
            "batch_equals_individual_bitwise": all(same_output(a, b) for a, b in zip(single, batch)),
            "reordering_invariant_bitwise": all(same_output(a, b) for a, b in zip(single, reversed_results)),
            "future_outside_window_independent_bitwise": same_output(future_before, future_after),
            "future_scope": "Outside-window changes only. sosfiltfilt uses both directions INSIDE the completed window; it is not causal sample-by-sample.",
            "degenerates": degenerates, "direct_feature_guard": direct,
            "proof": "For common proper R, norm(Rv)=norm(v). Positive sensor gain scales each magnitude and its spline/std together, so standardized features are invariant in exact arithmetic, provided unchanged pair and guard outcome. Magnitudes discard directional information; channel normalization discards amplitude."}


def roc(scores_g, scores_i):
    ng, ni = len(scores_g), len(scores_i)
    valid_g = [s for s in scores_g if s is not None]
    valid_i = [s for s in scores_i if s is not None]
    scores = sorted(set(valid_g + valid_i), reverse=True)
    states = [{"threshold": None, "threshold_kind": "reject_all", "far": 0., "frr": 1., "tpr": 0., "accepted_genuine": 0, "accepted_impostor": 0}]
    for score in scores:
        ag = sum(s >= score for s in valid_g)
        ai = sum(s >= score for s in valid_i)
        states.append({"threshold": score, "threshold_kind": "valid_score", "far": ai / ni, "frr": 1 - ag / ng, "tpr": ag / ng, "accepted_genuine": ag, "accepted_impostor": ai})
    eer, kind, bracket = None, "undefined_no_crossing", None
    for idx, state in enumerate(states):
        delta = state["far"] - state["frr"]
        if delta == 0:
            eer, kind, bracket = state["far"], "exact_threshold_state", [idx, idx]
            break
        if idx and (states[idx - 1]["far"] - states[idx - 1]["frr"]) * delta < 0:
            prev = states[idx - 1]
            d0 = prev["far"] - prev["frr"]
            mix = -d0 / (delta - d0)
            eer = prev["far"] + mix * (state["far"] - prev["far"])
            kind, bracket = "linear_crossing_interpolation", [idx - 1, idx]
            break
    return {"denominators": {"genuine": ng, "impostor": ni}, "valid_score_counts": {"genuine": len(valid_g), "impostor": len(valid_i)},
            "coverage": {"genuine": len(valid_g) / ng, "impostor": len(valid_i) / ni},
            "states": states, "eer": eer, "eer_status": kind, "eer_state_bracket": bracket,
            "terminal": states[-1], "invalid_claims_always_rejected": True}


def failure_roc_examples():
    examples = {}
    for name, spec in MANIFEST["roc_toys"].items():
        genuine, impostor = spec["genuine"], spec["impostor"]
        vg, vi = [s is not None for s in genuine], [s is not None for s in impostor]
        cg, ci = [s for s in genuine if s is not None], [s for s in impostor if s is not None]
        examples[name] = {"claims": {"genuine": {"scores": genuine, "validity_mask": vg}, "impostor": {"scores": impostor, "validity_mask": vi}},
                          "failure_inclusive": roc(genuine, impostor), "conditional_secondary": roc(cg, ci) if cg and ci else {"eer": None, "eer_status": "undefined_empty_valid_class"}}
    # Explicit enrollment/probe windows -> complete Cartesian claim set.
    # E1 has one failed enrollment; P1 has one failed probe; neither is dropped.
    enroll = [[True, True], [True, False]]
    probes = [[True, True], [True, False]]
    claims = []
    for claimant in range(2):
        for probe_identity in range(2):
            valid = all(enroll[claimant]) and all(probes[probe_identity])
            claims.append({"claimant": claimant, "probe_identity": probe_identity, "genuine": claimant == probe_identity,
                           "enrollment_windows": enroll[claimant], "probe_windows": probes[probe_identity],
                           "valid": valid, "score": (0.9 if claimant == probe_identity else 0.1) if valid else None})
    examples["all_window_policy_cartesian_claims"] = {"claims": claims, "failure_inclusive": roc([c["score"] for c in claims if c["genuine"]], [c["score"] for c in claims if not c["genuine"]]), "all_original_claims_retained": len(claims) == 4}
    return examples


def analytical_evidence():
    q = .05
    return {
        "coverage_derivation": "For a fixed stride boundary and arbitrary phase, an interval of duration D contains consecutive stride events for every phase only if D>=2*tau. Nominal D=2.56 gives tau<=1.28s. Sample endpoints actually span 255/100=2.55s, giving <=1.275s; margins [10,245] span 235/100=2.35s, giving <=1.175s. This is an ideal geometry guarantee, not sufficient for strict filtered minima, correct alias choice or finite channels.",
        "ideal_continuous_uniform_phase_coverage": [{"period_samples": p, "raw_0_255": max(0., min(1., 255 / p - 1)), "margins_10_245": max(0., min(1., 235 / p - 1))} for p in PERIODS],
        "enrollment_failure": [{"q": q, "m": m, "independent_failure_probability": 1 - (1 - q) ** m,
                                "sharp_dependence_lower_bound": q, "sharp_dependence_upper_bound": min(1., m * q)} for m in (1, 2, 10, 20)],
        "dependence_assumptions": "Union failure under identical marginal q: independence gives 1-(1-q)^m. Perfectly coincident failures attain q; disjoint failures attain min(1,m*q). Bounds require only per-window marginal q. These are illustrations, not actual TRAIN/DEV rates; overlapping windows need not be independent.",
        "claim_failure": "With independent enrollment m and probe k windows of common failure q, claim invalidity is 1-(1-q)^(m+k). Shared enrollment failures correlate many claims. No observed q is estimated here.",
        "roc_algebra": "For genuine invalid fraction fg and impostor invalid fraction fi, FRR_full=fg+(1-fg)*FRR_valid and FAR_full=(1-fi)*FAR_valid. At terminal valid-score acceptance, FRR=fg and FAR=1-fi. Starting reject-all has FAR-FRR=-1; a crossing can exist only if fg<=1-fi (equivalently genuine valid coverage+impostor valid coverage>=1). A threshold attaining equality may still require interpolation across tied-score jumps.",
    }


def summarize(records):
    summaries = []
    for family in FAMILIES:
        for period in PERIODS:
            rows = [r for r in records if r["family"] == family and r["period_samples"] == period]
            summaries.append({"family": family, "period_samples": period, "n": len(rows), "valid_pre_quality": sum(r["valid_pre_quality"] for r in rows),
                              "truth_raw_full_stride_count": sum(r["truth_full_anchored_stride_in_raw"] for r in rows),
                              "truth_margin_full_stride_count": sum(r["truth_full_anchored_stride_with_margins"] for r in rows),
                              "failure_reasons": dict(Counter(r["failure_reason"] or "none" for r in rows)),
                              "lag_alias_counts": dict(Counter(r["selected_lag_alias"] for r in rows)),
                              "pair_alias_counts": dict(Counter(r["selected_pair_alias"] for r in rows))})
    return {"n": len(records), "valid_pre_quality": sum(r["valid_pre_quality"] for r in records),
            "failure_reasons": dict(Counter(r["failure_reason"] or "none" for r in records)),
            "lag_alias_counts": dict(Counter(r["selected_lag_alias"] for r in records)),
            "pair_alias_counts": dict(Counter(r["selected_pair_alias"] for r in records)),
            "by_family_period": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(__file__).resolve() != SOURCE:
        raise SystemExit("This bounded prototype must run at its authorized path.")
    if scipy.__version__ != "1.18.1":
        raise SystemExit("Requires existing SciPy 1.18.1; never installs or substitutes.")
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest_sha = fingerprint(MANIFEST)
    if args.freeze_only:
        if REPORT.exists():
            raise SystemExit("Refusing to overwrite an existing freeze/attempt.")
        report = {"status": "frozen_before_measurement", "source_sha256": source_sha, "manifest_sha256": manifest_sha,
                  "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "manifest": MANIFEST,
                  "assumptions": ASSUMPTIONS, "sources": SOURCES, "records": []}
        write_report(report)
        print(json.dumps({"status": report["status"], "source_sha256": source_sha, "manifest_sha256": manifest_sha, "grid_count": 1280}))
        return
    report = json.loads(REPORT.read_text())
    if report["status"] != "frozen_before_measurement" or report["source_sha256"] != source_sha or report["manifest_sha256"] != manifest_sha or report["manifest"] != MANIFEST:
        raise SystemExit("Freeze mismatch or attempt already made; preserve existing evidence.")
    report["status"] = "running_incomplete"
    report["measurement_start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_report(report)
    try:
        report["analytical_evidence"] = analytical_evidence()
        report["reference_checks"] = references_and_ties()
        report["controls"] = controls()
        report["roc_examples"] = failure_roc_examples()
        write_report(report)
        for family in FAMILIES:
            for period in PERIODS:
                for phase_index in range(32):
                    if STOP_REQUESTED or time.process_time() > 280:
                        report["status"] = "incomplete_cpu_cap"
                        report["summary"] = summarize(report["records"])
                        write_report(report)
                        print(json.dumps({"status": report["status"], "records_preserved": len(report["records"])}))
                        return
                    record, _ = extract(fixture(family, period, phase_index))
                    record.update(family=family, period_samples=period, phase_index=phase_index, phase_cycles=phase_index / 32)
                    record.update(truth(period, phase_index, record))
                    report["records"].append(record)
                write_report(report)
                print(json.dumps({"family": family, "period": period, "completed": len(report["records"]), "cpu_seconds": round(time.process_time(), 3)}), flush=True)
        report["summary"] = summarize(report["records"])
        report["recommendation"] = {
            "production_ready": False, "decision": "Preregistration redesign before production or any authorized real-data experiment.",
            "reasons": ["256-sample phase coverage and minima margins cannot guarantee a stride across the proposed duration range.",
                        "Acceleration periodicity is not uniquely physical stride periodicity; exact floating ties and finite-window boundaries require a declared alias policy.",
                        "The all-window validity gate can dominate claim rejection; quality cutoff cannot be evaluated or calibrated synthetically.",
                        "Conditional scores/ROC omit an operational coverage penalty and may conceal a nonexistent failure-inclusive EER."],
            "required_controls": ["Canonical362 on every original claim with its original full coverage and preprocessing frozen.",
                                  "Matched Canonical362 control subjected to exactly the cycle system's failed enrollment/probe/quality masks and same original claim denominators; compare representation at matched selection while separately retaining the full-coverage canonical operational result.",
                                  "Record separate genuine/impostor enrollment/probe/quality failures, coverage, full ROC terminal rates, EER status and conditional metrics; preserve identity/session weighting and separate own9/own12 TRAIN calibration."],
            "possible_preregistered_alternatives_not_implemented": ["Aggregate available valid enrollment windows with a declared minimum count; retain every original claimant/claim and reject if below count.",
                                                                  "Use a fixed canonical fallback for extraction failures as an explicitly new composite pipeline, retaining every original claim.",
                                                                  "Collect a fixed longer acquisition or declare a repeat/abstention protocol with original attempts retained and failure/latency denominators reported.",
                                                                  "Treat failures as an explicit reject/abstain state with coverage and terminal rates; never delete difficult trials or create a numeric score sentinel."],
            "scope_limit": "None of these alternatives is selected, implemented, trained or validated here; the original detector and policy were tested unchanged apart from frozen implementation assumptions."}
        report["status"] = "complete"
        write_report(report)
        print(json.dumps({"status": report["status"], "n": report["summary"]["n"], "valid_pre_quality": report["summary"]["valid_pre_quality"], "cpu_seconds": report["runtime"]["process_cpu_seconds"]}))
    except BaseException as error:
        report["status"] = "incomplete_exception"
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
        report["summary"] = summarize(report["records"])
        write_report(report)
        raise


if __name__ == "__main__":
    main()
