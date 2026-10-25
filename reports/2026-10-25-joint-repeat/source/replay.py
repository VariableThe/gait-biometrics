"""Fit-free saved-array replay, complete summaries and invariance logs. No generators."""

from collections import Counter

import common as c
import legacy
import kernel
import evidence
import fixtures
import numpy as np


def histogram(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def tally(rows):
    return {
        "inputs": len(rows),
        "finite": sum(r["valid_pre_quality"] for r in rows),
        "finite_one_P": sum(r["finite_one_P_extraction"] for r in rows),
        "lag_resolved_within_one": sum(
            r["lag_resolves_declared_P_within_one_sample"] for r in rows
        ),
        "missing_pair": sum(
            r["failure_reason"] == "no_eligible_minima_pair" for r in rows
        ),
        "channel_degenerate": sum(
            r["failure_reason"] == "channel_degenerate" for r in rows
        ),
        "whole_window_modality_degenerate": sum(
            r["whole_window_modality_degenerate"] for r in rows
        ),
        "undefined_correlation_lags": sum(
            r["undefined_correlation_count"] for r in rows
        ),
        "undefined_all_lags": sum(r["finite_correlation_count"] == 0 for r in rows),
        "failures": histogram(r["failure_reason"] or "none" for r in rows),
        "selected_lag_aliases": histogram(r["selected_lag_alias"] for r in rows),
        "selected_pair_aliases": histogram(r["selected_pair_alias"] for r in rows),
        "finite_pair_aliases": histogram(
            r["selected_pair_alias"] for r in rows if r["valid_pre_quality"]
        ),
        "lag_errors": histogram(r["lag_error_samples"] for r in rows),
        "pair_errors": histogram(r["pair_error_samples"] for r in rows),
        "noise_only_inputs": sum(r["noise_only"] for r in rows),
        "noise_only_false_valid_outputs": sum(
            r["noise_only_false_valid_output"] for r in rows
        ),
        "quality_cutoff": None,
    }


def transition(row):
    if row["failure_reason"] == "no_eligible_minima_pair":
        return "missing_pair"
    if row["failure_reason"] == "channel_degenerate" or (
        not row["valid_pre_quality"] and row["whole_window_modality_degenerate"]
    ):
        return "degenerate"
    if not row["valid_pre_quality"]:
        return "other_failure"
    return "one_P" if row["selected_pair_alias"] == "one_stride" else "other_alias"


def summary(rows, arrays, inventory):
    by_length_arm, groups, primary, half_transitions, legacy_changes, invariance = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    lookup = {(r["input_index"], r["arm"]): r for r in rows}
    legacy_items = {
        (x["family"], x["period_samples"], x["phase_index"], x["length"]): x
        for x in inventory
        if x["kind"] == "legacy"
    }
    for n in c.LENGTHS:
        for arm in c.ARMS:
            subset = [r for r in rows if r["length"] == n and r["arm"] == arm]
            by_length_arm.append({"length": n, "arm": arm, **tally(subset)})
            for kind in ("legacy", "stress", "invariance"):
                kr = [r for r in subset if r["kind"] == kind]
                labels = legacy.FAMILIES if kind != "stress" else c.CONDITIONS
                field = "family" if kind != "stress" else "condition"
                for label in labels:
                    group = [r for r in kr if r[field] == label]
                    groups.append(
                        {
                            "length": n,
                            "arm": arm,
                            "kind": kind,
                            field: label,
                            **tally(group),
                        }
                    )
            panel = [
                r
                for r in subset
                if r["kind"] == "legacy"
                and r["family"] == legacy.FAMILIES[2]
                and r["period_samples"] in (160, 180, 200)
            ]
            c.require(len(panel) == 96, "Primary denominator mismatch")
            primary.append({"length": n, "arm": arm, "denominator": 96, **tally(panel)})
        for arm in c.ARMS[:2]:
            old = [
                r
                for r in rows
                if r["length"] == n and r["arm"] == arm and r["kind"] == "legacy"
            ]
            halves = [
                r
                for r in old
                if r["valid_pre_quality"] and r["selected_pair_alias"] == "half_stride"
            ]
            c.require(
                len(old) == 1280 and len(halves) == 96,
                "Legacy transition denominator mismatch",
            )
            for base in old:
                cand = lookup[(base["input_index"], "candidate_joint")]
                legacy_changes.append(
                    {
                        "input_index": base["input_index"],
                        "key": base["key"],
                        "length": n,
                        "control_arm": arm,
                        "control_finite_one_P": base["finite_one_P_extraction"],
                        "candidate_finite_one_P": cand["finite_one_P_extraction"],
                        "control_lag_error": base["lag_error_samples"],
                        "candidate_lag_error": cand["lag_error_samples"],
                        "control_pair_error": base["pair_error_samples"],
                        "candidate_pair_error": cand["pair_error_samples"],
                    }
                )
            events = [
                {
                    "input_index": r["input_index"],
                    "key": r["key"],
                    "outcome": transition(
                        lookup[(r["input_index"], "candidate_joint")]
                    ),
                    "selected_lag": lookup[(r["input_index"], "candidate_joint")][
                        "selected_lag"
                    ],
                    "pair": lookup[(r["input_index"], "candidate_joint")][
                        "pair_bounds_inclusive"
                    ],
                    "lag_resolved_within_one": lookup[
                        (r["input_index"], "candidate_joint")
                    ]["lag_resolves_declared_P_within_one_sample"],
                }
                for r in halves
            ]
            half_transitions.append(
                {
                    "length": n,
                    "control_arm": arm,
                    "denominator": 96,
                    "outcomes": histogram(e["outcome"] for e in events),
                    "events": events,
                }
            )
    for item in inventory:
        if item["kind"] != "invariance":
            continue
        baseitem = legacy_items[
            (item["family"], 160, item["phase_index"], item["length"])
        ]
        for arm in c.ARMS:
            base, transformed = (
                lookup[(baseitem["input_index"], arm)],
                lookup[(item["input_index"], arm)],
            )
            valid = base["valid_pre_quality"] and transformed["valid_pre_quality"]
            bvec, tvec = (
                arrays["features"][base["row_id"]],
                arrays["features"][transformed["row_id"]],
            )
            invariance.append(
                {
                    "input_index": item["input_index"],
                    "base_input_index": baseitem["input_index"],
                    "key": item["key"],
                    "arm": arm,
                    "length": item["length"],
                    "selected_lag_equal": base["selected_lag"]
                    == transformed["selected_lag"],
                    "pair_equal": base["pair_bounds_inclusive"]
                    == transformed["pair_bounds_inclusive"],
                    "validity_equal": base["valid_pre_quality"]
                    == transformed["valid_pre_quality"],
                    "feature_bits_equal": bvec.tobytes() == tvec.tobytes(),
                    "feature_max_abs_delta": float(np.max(np.abs(bvec - tvec)))
                    if valid
                    else None,
                    "feature_close_atol_1e-10_rtol0": bool(
                        np.allclose(bvec, tvec, atol=1e-10, rtol=0)
                    )
                    if valid
                    else None,
                    "repair_applied": False,
                }
            )
    return {
        "by_length_arm": by_length_arm,
        "groups": groups,
        "primary_96_by_length_arm": primary,
        "original_half_transitions": half_transitions,
        "legacy_1280_coverage_and_errors": legacy_changes,
        "invariance": invariance,
        "all_inputs_retained": True,
        "quality_cutoff": None,
        "claims": "Synthetic repeat feasibility only; no gait accuracy, anatomical stride, real-data protocol or DEV improvement claim; N768 DIAGNOSTIC ONLY",
    }


def verify(rows, arrays, inventory, diagnostics, *, schema, full=True):
    """Production counts are fixed. Recompute only SAVED evidence; no generators/filter/fit."""
    c.require(
        c.typed(inventory) == c.typed(fixtures.keys()), "Production inventory mismatch"
    )
    return _verify(
        rows, arrays, inventory, diagnostics, schema=schema, full=full, inputs=2880
    )


def verify_tiny(rows, arrays, inventory, diagnostics, *, schema, full=True):
    """Explicit 1..6 non-primary synthetic test inputs; identical evidence checks, no AST edits."""
    c.require(1 <= len(inventory) <= 6, "Tiny replay requires 1..6 inputs")
    known = {r["key"]: r for r in fixtures.keys()}  # Metadata only.
    for i, item in enumerate(inventory):
        c.require(item["key"] in known, "Unknown tiny input metadata")
        c.require(
            c.typed(item) == c.typed({**known[item["key"]], "input_index": i}),
            "Tiny inventory metadata mismatch",
        )
        c.require(
            not (
                item["kind"] == "legacy"
                and item["family"] == legacy.FAMILIES[2]
                and item["period_samples"] in (160, 180, 200)
            ),
            "Primary panel prohibited in tiny replay",
        )
    c.require(
        len({r["key"] for r in inventory}) == len(inventory), "Duplicate tiny keys"
    )
    return _verify(
        rows,
        arrays,
        inventory,
        diagnostics,
        schema=schema,
        full=full,
        inputs=len(inventory),
    )


def _verify(rows, arrays, inventory, diagnostics, *, schema, full, inputs):
    c.require(
        len(rows) == inputs * 3
        and len(inventory) == inputs
        and len(diagnostics) == inputs,
        "Incomplete saved artifact",
    )
    evidence.validate_arrays(arrays, inventory, schema)
    c.require(
        arrays["input_present"].all()
        and arrays["record_present"].all()
        and arrays["joint_present"].all()
        and arrays["diagnostic_present"].all(),
        "Missing present mask",
    )
    byrow = {r["row_id"]: r for r in rows}
    c.require(len(byrow) == inputs * 3, "Duplicate row IDs")
    c.require(
        c.typed([r["row_id"] for r in rows]) == c.typed(list(range(inputs * 3))),
        "Record ordering/row mapping mismatch",
    )
    for item in inventory:
        c.CHECK()
        i, n = item["input_index"], item["length"]
        c.require(str(arrays["keys"][i]) == item["key"], "Saved key mapping mismatch")
        c.require(int(arrays["lengths"][i]) == n, "Saved length mismatch")
        for label in ("windows", "magnitudes", "filtered"):
            c.require(
                np.isnan(arrays[label][i, :, n:]).all(), label + " padding mismatch"
            )
        window = np.array(arrays["windows"][i, :, :n], order="C", copy=True)
        c.require(np.isfinite(window).all(), "Nonfinite raw six-axis prefix")
        mags, filtered = arrays["magnitudes"][i, :, :n], arrays["filtered"][i, :, :n]
        c.require(
            kernel.magnitudes(window).tobytes() == mags.tobytes(),
            "Raw six-axis/magnitude norm bits mismatch",
        )
        c.require(
            np.isfinite(mags).all() and np.isfinite(filtered).all(),
            "Nonfinite supplied synthetic input",
        )
        old = arrays["extraction_correlations"][i, 0]
        joint = arrays["extraction_correlations"][i, 1:]
        c.require(
            not np.isinf(old).any()
            and bool((np.abs(old[np.isfinite(old)]) <= 1).all()),
            "Historical Pearson range mismatch",
        )
        evidence.guard_relationships(
            arrays["joint_guard_stats"][i],
            arrays["joint_guard_valid"][i],
            arrays["joint_guard_finite"][i],
            joint[:2],
            "joint",
        )
        dc = arrays["diagnostic_correlations"][i]
        evidence.guard_relationships(
            arrays["diagnostic_guard_stats"][i],
            arrays["diagnostic_guard_valid"][i],
            arrays["diagnostic_guard_finite"][i],
            dc[:4],
            "diagnostic",
        )
        # All diagnostic prefix vectors use the same common anchors, independent of lag.
        prefix_stats = arrays["diagnostic_guard_stats"][i, :, :, 0, :]
        c.require(
            np.array_equal(
                prefix_stats, np.broadcast_to(prefix_stats[:, :1], prefix_stats.shape)
            ),
            "Diagnostic common-prefix stats mismatch",
        )
        evidence.mean_relationship(dc[0], dc[1], dc[4], "raw diagnostic")
        evidence.mean_relationship(dc[2], dc[3], dc[5], "filtered diagnostic")
        c.require(
            np.array_equal(
                arrays["extraction_correlation_finite"][i],
                np.isfinite(arrays["extraction_correlations"][i]),
            ),
            "Extraction finite-mask mismatch",
        )
        c.require(
            np.array_equal(
                arrays["diagnostic_finite"][i],
                np.isfinite(arrays["diagnostic_correlations"][i]),
            ),
            "Diagnostic finite-mask mismatch",
        )
        # Both undefined modality coefficients must propagate, with no reweighting.
        evidence.mean_relationship(joint[0], joint[1], joint[2], "extraction joint")
        whole = kernel.whole_window(mags, filtered)
        for ai, arm in enumerate(c.ARMS):
            rowid = i * 3 + ai
            row = byrow[rowid]
            rs = old if ai < 2 else joint[2]
            c.require(
                row["input_index"] == i
                and row["key"] == item["key"]
                and row["arm"] == arm,
                "Record key/arm mismatch",
            )
            c.require(
                row["quality_cutoff"] is None
                and not row["quality_evaluated"]
                and not row["quality_applied"],
                "Quality fit appeared",
            )
            base, feature = legacy.extract(
                mags, filtered[0], rs, legacy.RULES[0 if ai == 0 else 1]
            )
            replayed = kernel.record(item, ai, base, feature, rs, whole, joint)
            c.require(
                c.typed(replayed) == c.typed(row),
                f"Saved record replay mismatch row{rowid}",
            )
            vector = np.full(400, np.nan) if feature is None else feature
            c.require(
                vector.tobytes() == arrays["features"][rowid].tobytes(),
                "Saved feature float bits mismatch",
            )
            c.require(
                bool(arrays["feature_valid"][rowid]) == (feature is not None),
                "Saved feature mask mismatch",
            )
            pair = (
                [-1, -1]
                if base["pair_bounds_inclusive"] is None
                else base["pair_bounds_inclusive"]
            )
            c.require(
                np.array_equal(arrays["selected_pairs"][rowid], pair),
                "Saved pair mismatch",
            )
            c.require(
                int(arrays["selected_lags"][rowid])
                == (-1 if base["selected_lag"] is None else base["selected_lag"]),
                "Saved lag mismatch",
            )
        summaries = [legacy.correlation_summary(rs, c.DIAG_LAGS) for rs in dc]
        c.require(
            c.typed(diagnostics[i])
            == c.typed(evidence.diagnostic_record(item, summaries)),
            "Diagnostic complete metadata/summary mismatch",
        )
        if full:
            # Correlation/guard identity checks use saved arrays; no new input generation.
            c.require(
                legacy.correlations(filtered[0]).tobytes() == old.tobytes(),
                "Saved legacy correlation bits mismatch",
            )
            actual, stats, valid, finite = kernel.joint_correlations(filtered)
            for label, left, right in (
                ("joint", actual, joint),
                ("joint_stats", stats, arrays["joint_guard_stats"][i]),
                ("joint_valid", valid, arrays["joint_guard_valid"][i]),
                ("joint_finite", finite, arrays["joint_guard_finite"][i]),
            ):
                c.require(
                    left.tobytes() == right.tobytes(),
                    "Saved " + label + " float bits/mask mismatch",
                )
            actual, stats, valid, finite, _ = kernel.diagnostic(mags, filtered)
            for label, left, right in (
                ("diag", actual, arrays["diagnostic_correlations"][i]),
                ("diag_stats", stats, arrays["diagnostic_guard_stats"][i]),
                ("diag_valid", valid, arrays["diagnostic_guard_valid"][i]),
                ("diag_finite", finite, arrays["diagnostic_guard_finite"][i]),
            ):
                c.require(
                    left.tobytes() == right.tobytes(),
                    "Saved " + label + " float bits/mask mismatch",
                )
    return evidence.audit_metadata(full, inputs)
