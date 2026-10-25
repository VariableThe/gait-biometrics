#!/usr/bin/env python3
"""Read-only saved-array audit; never imports prototype or re-extracts fixtures.

POST-RUN saved-array review: preserved original audit stops at integer-key fingerprint serialization. This copy documents that defect; no new measurements.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

from collections import Counter
import hashlib
import json
import time

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    report = json.loads((ROOT/"report.json").read_text())
    check(report["status"] == "complete", "Run is incomplete; preserve root")
    for name, expected in report["preregistration_file_sha256"].items():
        check(sha(ROOT/name) == expected, f"Preregistration changed: {name}")
    for name, expected in report["artifact_sha256"].items():
        check(sha(ROOT/name) == expected, f"Artifact changed: {name}")
    for name, expected in report["array_sha256"].items():
        check(sha(ROOT/"arrays"/name) == expected, f"Partial array changed: {name}")
    check(hashlib.sha256(canonical(report["manifest"])).hexdigest() == report["manifest_sha256"], "Manifest hash")
    check(hashlib.sha256(canonical(report["runtime_identity"])).hexdigest() == report["runtime_sha256"], "Runtime hash")
    scientific = {key: report[key] for key in report["scientific_fingerprint_fields"]}
    # Post-run forensic handling of the preserved integer-key serialization defect.
    note = json.loads((ROOT/"fingerprint_serialization_note.json").read_text())
    check(sha(ROOT/"report.json") == note["report_sha256"], "Preserved report hash")
    check(hashlib.sha256(canonical(scientific)).hexdigest() == note["canonical_saved_JSON_fingerprint"], "Portable saved-JSON scientific hash")
    recovered = json.loads(json.dumps(scientific))
    for row in recovered["diagnostic_summary"]:
        row["nearmax_set_size_counts"] = {int(key): value for key, value in row["nearmax_set_size_counts"].items()}
    check(hashlib.sha256(canonical(recovered)).hexdigest() == report["scientific_fingerprint_sha256"], "Original in-memory hash recovered exactly")
    records = [json.loads(line) for line in (ROOT/"records.jsonl").read_text().splitlines()]
    diagnostics = [json.loads(line) for line in (ROOT/"diagnostics.jsonl").read_text().splitlines()]
    records.sort(key=lambda row: row["row_id"])
    check(len(records) == 5120 and [row["row_id"] for row in records] == list(range(5120)), "Record rows")
    check(len(diagnostics) == 1280 and [row["fixture_index"] for row in diagnostics] == list(range(1280)), "Diagnostic rows")
    cached = json.loads((ROOT/"cached_reference.json").read_text())
    legacy = [r for r in records if r["length"] == 256 and r["rule"] == report["manifest"]["rules_in_order"][0]]
    legacy_projection = [{key: r[key] for key in old} for r, old in zip(legacy, cached["records"])]
    check(canonical(legacy_projection) == canonical(cached["records"]), "All legacy fields exact")
    with np.load(ROOT/"audit_arrays.npz", allow_pickle=False) as archive:
        corr = archive["extractor_correlations"]
        diag_corr = archive["diagnostic_correlations"]
        features = archive["features"]
        valid = archive["feature_valid_mask"]
        mapping = archive["record_mapping"]
        lags = archive["extractor_lags"]
        dlags = archive["diagnostic_lags"]
        check(features.shape == (5120, 400) and corr.shape == (2560, 121) and diag_corr.shape == (1280,4,361), "Array shapes")
        check(archive["windows_768"].shape == (1280,6,768), "All six axes preserved")
        check(np.array_equal(archive["diagnostic_anchors"], np.arange(10,358)), "Common 348 anchors")
        for name in ("record_present_mask", "diagnostic_present_mask", "fixture_present_mask", "correlation_present_mask"):
            check(archive[name].all(), f"Incomplete present mask: {name}")
        check(np.isfinite(features[valid]).all() and np.isnan(features[~valid]).all(), "Feature NaNs/mask")
        check(np.max(np.abs(np.linalg.norm(features[valid], axis=1)-1)) < 1e-14, "Unit feature norms")
        for row in records:
            ri = row["row_id"]
            fi, n, rule, cr = map(int, mapping[ri])
            check((fi, n, cr) == (row["fixture_index"], row["length"], row["correlation_row"]), f"Row map {ri}")
            check(str(archive["rules"][rule]) == row["rule"], f"Rule map {ri}")
            check(str(archive["fixture_keys"][fi]) == row["fixture_key"], f"Fixture map {ri}")
            rs = corr[cr]
            finite = np.isfinite(rs)
            check(finite.any(), f"No finite correlation in fixture {ri}")
            best = float(np.max(rs[finite]))
            exact = lags[finite & (rs == best)].tolist()
            near = lags[finite & (np.abs(rs-best) <= 1e-10)].tolist()
            check(exact == row["exact_max_ties"] and near == row["all_nearmax_lags"], f"Correlation sets {ri}")
            check(min(exact if rule == 0 else near) == row["selected_lag"], f"Selection {ri}")
            check(best == row["raw_maxr"], f"Maximum {ri}")
            minima, lag = row["strict_minima"], row["selected_lag"]
            check(all(10 <= m <= n-11 for m in minima), f"Minima margins {ri}")
            pairs = [(a,b) for a in minima for b in minima if 80 <= b-a <= 200 and abs(b-a-lag) <= 0.1*lag]
            pair = min(pairs, key=lambda ab: (abs(ab[1]-ab[0]-lag), abs(sum(ab)/2-(n-1)/2), *ab)) if pairs else None
            check((None if pair is None else list(pair)) == row["pair_bounds_inclusive"], f"Pair rank {ri}")
            check(len(pairs) == row["eligible_pair_count"], f"Pair count {ri}")
            check(bool(valid[ri]) == row["valid_pre_quality"] == row["finite_features"], f"Validity {ri}")
            if row["guard_evaluated"]:
                rms = np.array(row["channel_raw_rms"])
                std = np.array(row["channel_population_std"])
                degenerate = (rms == 0) | (std <= 1e-12*rms)
                check(degenerate.tolist() == row["channel_degenerate"], f"Guard {ri}")
                check(bool(degenerate.any()) == (row["failure_reason"] == "channel_degenerate"), f"Guard failure {ri}")
            check(not row["quality_evaluated"] and not row["quality_trained"] and not row["quality_applied"] and row["quality_cutoff"] is None and row["explicit_no_quality_cutoffs"], f"Quality scope {ri}")
            check(not row["joint_magnitudes_half_periodic"] and not row["full_six_axis_signal_half_periodic"], f"Signal ambiguity flags {ri}")
            check(row["extracted_interval_alias"] == (row["selected_pair_alias"] not in ("unavailable", "one_stride")), f"Alias separated {ri}")
            geometry = row["geometric_reference"]
            check(geometry["available_at_selected_lag"] == bool(geometry["eligible_pairs"]), f"Geometry availability {ri}")
            check(all(10 <= t <= n-11 for t in geometry["minima_times"]), f"Reference margins {ri}")
        for fi, dr in enumerate(diagnostics):
            for ci, channel in enumerate(dr["channels"]):
                rs = diag_corr[fi,ci]
                finite = np.isfinite(rs)
                best = float(np.max(rs[finite]))
                check(dlags[finite & (np.abs(rs-best) <= 1e-10)].tolist() == channel["all_nearmax_lags"], f"Diagnostic nearsets {fi}/{ci}")
                check(dlags[finite & (rs == best)].tolist() == channel["exact_max_ties"], f"Diagnostic exact ties {fi}/{ci}")
                check(channel["anchor_pair_count"] == 348 and channel["extraction_effect"] == "none", f"Diagnostic scope {fi}/{ci}")
                for harmonic in channel["harmonic_alternatives"]:
                    check(float(rs[harmonic["lag"]-40]) == harmonic["correlation"], f"Harmonic mapping {fi}/{ci}")
        checks = report["fixed_numerical_checks"]
        vectors = archive["fixed_check_vectors"]
        check(vectors.shape == (18,121) and checks["checks_pass"], "Fixed vectors")
        for row, vector in zip(checks["rows"], vectors):
            best = np.max(vector)
            near = lags[np.abs(vector-best) <= 1e-10].tolist()
            check(near == [100,200] and row["candidate_selected_lag"] == 100, "Fixed candidate stability")
            check(float(vector[20]).hex() == row["r100_hex"] and float(vector[120]).hex() == row["r200_hex"], "ULP values")
    for group in report["summary"]["by_length_rule"]:
        subset = [r for r in records if r["length"] == group["length"] and r["rule"] == group["rule"]]
        check(len(subset) == group["records"] == 1280, "Group denominator")
        check(sum(r["valid_pre_quality"] for r in subset) == group["finite"], "Finite count")
        check(dict(Counter(r["failure_reason"] or "none" for r in subset)) == group["failures"], "Failure counts")
        check(sum(r["valid_pre_quality"] and r["selected_pair_alias"] == "half_stride" for r in subset) == group["finite_half_alias"], "Finite half aliases")
    result = {"status": "pass_with_documented_fingerprint_serialization_defect", "new_measurements": False, "canonical_saved_JSON_fingerprint": note["canonical_saved_JSON_fingerprint"], "records_audited": 5120, "diagnostic_channels_audited": 5120,
              "legacy_records_all_fields_exact": 1280, "fixed_vectors_audited": 18,
              "scientific_fingerprint_sha256": report["scientific_fingerprint_sha256"],
              "report_sha256": sha(ROOT/"report.json"), "cpu_seconds": time.process_time()-start_cpu,
              "wall_seconds": time.perf_counter()-start_wall}
    with (ROOT/"postrun_audit.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
