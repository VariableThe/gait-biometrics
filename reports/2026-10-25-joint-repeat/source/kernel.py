"""Sole joint-Pearson candidate, unchanged legacy extraction, and logging diagnostics."""

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


def compatible_record(base, feature, item, rule_index, window, mags, smooth, rs, roots):
    """Exact complete parent length-study schema. NEW record fields added outside this function."""
    family, period, phase = item["family"], item["period_samples"], item["phase_index"]
    fi, li, n = (
        item["source_fixture_index"],
        item["source_length_index"],
        item["length"],
    )
    rec = dict(base)
    rec.update(
        row_id=fi * 4 + li * 2 + rule_index,
        fixture_index=fi,
        fixture_key=f"{family}/P{period}/phase{phase:02d}",
        length=n,
        rule=legacy.RULES[rule_index],
        correlation_row=fi * 2 + li,
        family=family,
        period_samples=period,
        phase_index=phase,
        phase_cycles=phase / 32,
    )
    rec.update(legacy.truth(period, phase, rec, n))
    cs = legacy.correlation_summary(rs)
    rec.update(
        all_nearmax_lags=cs["all_nearmax_lags"],
        correlation_gaps=cs["correlation_gaps"],
        finite_correlation_count=cs["finite_correlation_count"],
        undefined_correlation_count=cs["undefined_correlation_count"],
        selected_lag_correlation=None
        if rec["selected_lag"] is None
        else float(rs[rec["selected_lag"] - 80]),
        selected_interval_interpretation="putative repeat interval; anatomical stride not inferred",
        status="finite_pre_quality" if feature is not None else rec["failure_reason"],
        input_finite_count=int(np.isfinite(window).sum()),
        input_sample_value_count=6 * n,
        magnitude_finite_count=int(np.isfinite(mags).sum()),
        filtered_acc_finite_count=int(np.isfinite(smooth).sum()),
        guard_evaluated=rec["pair_bounds_inclusive"] is not None,
        channel_population_std=rec.get("channel_population_std"),
        channel_raw_rms=rec.get("channel_raw_rms"),
        channel_degenerate=rec.get("channel_degenerate"),
        feature_norm=rec.get("feature_norm"),
        feature_channel_means=rec.get("feature_channel_means"),
        explicit_no_quality_cutoffs=True,
        quality_cutoff=None,
        quality_trained=False,
        quality_applied=False,
        acceleration_harmonic_ambiguity=family == legacy.FAMILIES[2],
        full_six_axis_signal_half_periodic=False,
        joint_magnitudes_half_periodic=False,
        gyro_contains_physical_P=True,
        extracted_interval_alias=rec["selected_pair_alias"]
        not in ("unavailable", "one_stride"),
        finite_extracted_interval_alias=rec["valid_pre_quality"]
        and rec["selected_pair_alias"] != "one_stride",
        unique_nearmax_is_stride_evidence=False,
        geometric_reference=legacy.geometric_reference(
            family, period, phase, n, rec["selected_lag"], roots
        ),
    )
    return rec


def record(item, arm_index, base, feature, rs, whole, modalities=None):
    result = dict(base)
    period = item["period_samples"]
    lag, duration = result["selected_lag"], result["pair_length_samples"]
    cs = legacy.correlation_summary(rs)
    result.update(
        input_index=item["input_index"],
        row_id=item["input_index"] * 3 + arm_index,
        key=item["key"],
        arm=c.ARMS[arm_index],
        **{
            k: item[k]
            for k in (
                "kind",
                "family",
                "length",
                "period_samples",
                "phase_index",
                "condition",
                "transformation",
            )
        },
        all_nearmax_lags=cs["all_nearmax_lags"],
        correlation_gaps=cs["correlation_gaps"],
        finite_correlation_count=cs["finite_correlation_count"],
        undefined_correlation_count=cs["undefined_correlation_count"],
        selected_lag_correlation=None if lag is None else float(rs[lag - 80]),
        selected_lag_alias=legacy.alias(lag, period),
        selected_pair_alias=legacy.alias(duration, period),
        lag_error_samples=None if lag is None else lag - period,
        pair_error_samples=None if duration is None else duration - period,
        lag_resolves_declared_P_within_one_sample=bool(
            lag is not None and abs(lag - period) <= 1
        ),
        finite_one_P_extraction=bool(
            feature is not None and legacy.alias(duration, period) == "one_stride"
        ),
        whole_window_modalities=whole,
        whole_window_modality_degenerate=any(r["degenerate"] for r in whole[2:]),
        quality_cutoff=None,
        quality_evaluated=False,
        quality_trained=False,
        quality_applied=False,
        selected_interval_interpretation="putative repeat interval; anatomical stride not inferred",
        noise_only=item["condition"] == "both_replaced_4_acrms_noise",
        noise_only_false_valid_output=bool(
            item["condition"] == "both_replaced_4_acrms_noise" and feature is not None
        ),
        both_modalities_half_period=item["condition"] == "both_half_period",
        both_half_available_joint_repeat_samples=period / 2
        if item["condition"] == "both_half_period"
        else None,
        anatomical_period_inferred=False,
    )
    if modalities is not None and lag is not None:
        result["selected_modality_correlations"] = [
            None if not np.isfinite(v) else float(v) for v in modalities[:2, lag - 80]
        ]
    else:
        result["selected_modality_correlations"] = None
    return result


def diagnostic(mags, filtered):
    vectors = np.vstack((mags, filtered))
    n = mags.shape[1]
    anchors = np.arange(n - 200)
    values = np.full((6, 161), np.nan)
    stats = np.full((4, 161, 2, 2), np.nan)
    valid = np.zeros((4, 161, 2), dtype=bool)
    finite = np.zeros_like(valid)
    for ci, vector in enumerate(vectors):
        for i, lag in enumerate(c.DIAG_LAGS):
            c.CHECK()
            value, stat, good, fin = pearson(vector[anchors], vector[anchors + lag])
            values[ci, i], stats[ci, i], valid[ci, i], finite[ci, i] = (
                value,
                stat,
                good,
                fin,
            )
    for out, a, b in ((4, 0, 1), (5, 2, 3)):
        good = np.isfinite(values[a]) & np.isfinite(values[b])
        values[out, good] = (values[a, good] + values[b, good]) / 2
    summaries = [legacy.correlation_summary(row, c.DIAG_LAGS) for row in values]
    return values, stats, valid, finite, summaries
