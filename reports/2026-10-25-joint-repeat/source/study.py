#!/usr/bin/env python3
"""Prepare/freeze/ONE execute/saved-evidence verify. Implementer runs preparation only."""

# Stage-FIRST grammar is checked using only the built-in sys module. Keep in sync with run.sh.
import sys

if __name__ == "__main__":
    _argv = sys.argv[1:]
    if _argv in (["-h"], ["--help"]):
        print(
            "usage: study.py {prepare|check-preparation|freeze|execute|selftest|verify [--replay-correlations]}"
        )
        raise SystemExit(0)
    if not (
        (
            len(_argv) == 1
            and _argv[0]
            in (
                "prepare",
                "check-preparation",
                "freeze",
                "execute",
                "selftest",
                "verify",
            )
        )
        or _argv == ["verify", "--replay-correlations"]
    ):
        sys.stderr.write(
            "stage must come first; --replay-correlations is accepted only after verify\n"
        )
        raise SystemExit(2)


# Guard helpers use only sys and builtin file operations; no stdlib/local imports.
def _process_identity(pid):
    with open("/proc/" + str(pid) + "/stat") as handle:
        stat = handle.read(8192)
    head, tail = stat.rsplit(") ", 1)
    fields = tail.split()
    return int(head.split(" ", 1)[0]), int(fields[1]), fields[19], fields[0]


def _attempt_text(route, pid, starttime):
    return (
        '{"automatic_retry":false,"cpu_budget_seconds":600,"launcher_pid":'
        + str(pid)
        + ',"launcher_starttime_ticks":"'
        + starttime
        + '","route":"'
        + route
        + '","schema":2,"status":"exclusive_attempt_consumed","wall_budget_seconds":900}\n'
    )


def _live_ancestor(pid, starttime, worker):
    cursor, seen = worker[1], {worker[0]}
    for _ in range(128):
        if cursor <= 0 or cursor in seen:
            break
        seen.add(cursor)
        current = _process_identity(cursor)
        if current[3] in ("Z", "X", "x"):
            break
        if current[0] == pid:
            if current[2] == starttime:
                return
            break
        cursor = current[1]
    raise RuntimeError(
        "Shell reservation requires the same live launcher ancestor/starttime"
    )


# Valid execute reserves and consumes ONE worker before ANY further Python imports.
_STAGE = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else ""
if _STAGE == "execute":
    _reservation_root = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
    _shell_token = sys._xoptions.get("joint_run_reserved")
    _marker_path = _reservation_root + "/attempt.started.json"
    if _shell_token is None:
        with open(_marker_path, "x") as _attempt:
            _worker = _process_identity("self")
            _expected_marker = _attempt_text("direct", _worker[0], _worker[2])
            _attempt.write(_expected_marker)
    else:
        # Shell handoff is explicit interpreter state, never inferred from a freely set env var.
        _parts = _shell_token.split(":") if type(_shell_token) is str else []
        if len(_parts) != 2 or not all(
            p.isascii() and p.isdecimal() and p[0] != "0" for p in _parts
        ):
            raise RuntimeError("Invalid shell reservation token")
        _launcher_pid, _launcher_start = int(_parts[0]), _parts[1]
        _expected_marker = _attempt_text("shell", _launcher_pid, _launcher_start)
        with open(_marker_path) as _attempt:
            if _attempt.read() != _expected_marker:
                raise RuntimeError("Shell reservation marker missing or mismatched")
        _worker = _process_identity("self")
        _live_ancestor(_launcher_pid, _launcher_start, _worker)
    with open(_reservation_root + "/worker.started.json", "x") as _receipt:
        _receipt.write(
            '{"attempt":'
            + _expected_marker.strip()
            + ',"schema":1,"status":"exclusive_worker_started","worker_pid":'
            + str(_worker[0])
            + ',"worker_starttime_ticks":"'
            + _worker[2]
            + '"}\n'
        )
    if _shell_token is not None:
        _live_ancestor(_launcher_pid, _launcher_start, _worker)

import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if _STAGE == "execute":
    if _ROOT != Path(_reservation_root).resolve():
        raise RuntimeError("Execute via a source symlink is unsupported")
    # Bound scientific import/startup too. Default signals terminate and preserve the reservation.
    import resource
    import signal

    resource.setrlimit(resource.RLIMIT_CPU, (599, 600))
    _launch_elapsed = max(
        0.0, time.time() - float(os.environ.get("JOINT_RUN_LAUNCH_EPOCH", time.time()))
    )
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 900 - _launch_elapsed))

from collections import Counter

import common as c
import fixtures
import evidence
import kernel
import legacy
import replay
import selftest
import numpy as np
from scipy.signal import sosfiltfilt

PREPARATION_INPUTS = (
    "manifest.proposed.json",
    "keys.json",
    "parent_index.json",
    "runtime.preparation.json",
    "preparation.json",
    "selftests.json",
    "sources.json",
    "fingerprint_erratum.json",
)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def validate_parent_saved_hashes():
    parent = c.ROOT.parent / "cycle-length-feasibility-r1"
    report = json.loads((parent / "report.json").read_text())
    c.require(
        c.sha(parent / "report.json")
        == "bd890b2065009c961c7f8dc53624d85435e50ea44186d86465323673b45761da",
        "Unexpected frozen length report",
    )
    c.require(
        c.sha(parent / "prototype.py") == report["source_sha256"],
        "Length source differs from report",
    )
    for name, digest in report["artifact_sha256"].items():
        c.require(
            c.sha(parent / name) == digest, "Parent artifact hash mismatch " + name
        )
    for name, digest in report["array_sha256"].items():
        c.require(
            c.sha(parent / "arrays" / name) == digest,
            "Parent array hash mismatch " + name,
        )
    return report


def prepare():
    c.require(
        not (c.ROOT / "freeze.json").exists()
        and not (c.ROOT / "attempt.started.json").exists(),
        "Already frozen/attempted",
    )
    timings = {}
    with c.stage(timings, "preparation_validation"):
        provenance = c.provenance_check()
        parent_report = validate_parent_saved_hashes()
        parents = c.parent_snapshot()
        runtime = c.runtime_identity()
        tests = c.load_json("selftests.json")
        c.require(tests["status"] == "pass", "Tiny selftests missing or failed")
        c.require(
            tests["systems_regressions"]["status"] == "pass",
            "Review regressions missing or failed",
        )
        c.require(
            tests["source_files_sha256"] == c.source_index(),
            "Tiny checks refer to different sources; rerun only the tiny checks before preparation",
        )
        inventory = fixtures.keys()  # Metadata only; no generator/extraction call.
        proposal = fixtures.manifest()
        proposal["family_equations"] = parent_report["manifest"]["family_equations"]
        proposal["brief_sha256"] = c.sha(c.ROOT / "BRIEF.md")
        proposal["runtime_pins"] = c.EXPECTED_VERSIONS
        proposal["parents_sha256"] = parents
        proposal["copied_function_provenance_sha256"] = c.sha(
            c.ROOT / "copy_provenance.json"
        )
        proposal["inventory_sha256"] = c.fingerprint(inventory)
        proposal["prefreeze_systems_revision"] = {
            "stage_first_grammar": True,
            "single_use_worker_receipt": True,
            "shell_handoff": "PID:starttime ticks; exact attempt binding; live ancestor walk bounded at 128; receipt before further imports; persisted verification does not require launcher alive",
            "raw_magnitude_bit_binding": True,
            "default_guard_stat_mask_consistency": True,
            "full_mode_recomputes_correlations_and_stats": True,
            "array_schema_fingerprint": c.fingerprint(evidence.array_schema(inventory)),
            "exact_inventory_exclusion": "artifact_index.json at root only",
            "report_binding_files": list(evidence.BINDING_FILES),
            "trust_policy": evidence.TRUST_POLICY,
            "runtime_coverage_policy": runtime["coverage_policy"],
            "scientific_choices_changed": False,
        }
    c.save_json("manifest.proposed.json", proposal)
    c.save_json("keys.json", inventory)
    c.save_json("parent_index.json", parents)
    c.save_json("runtime.preparation.json", runtime)
    note = json.loads(
        (
            c.ROOT.parent
            / "cycle-length-feasibility-r1/fingerprint_serialization_note.json"
        ).read_text()
    )
    c.save_json(
        "fingerprint_erratum.json",
        {
            "parent_note": note,
            "new_measurement": False,
            "parent_files_modified": False,
            "handling": "Historical combined in-memory hash is not a valid portable JSON fingerprint; separately retain original report/file hashes and strict-string-key/type-preserving canonical hashes here. No old files repaired.",
        },
    )
    c.save_json(
        "sources.json",
        {
            "files": c.source_index(),
            "local_dependency_closure": {
                "study.py": [
                    "common.py",
                    "evidence.py",
                    "fixtures.py",
                    "kernel.py",
                    "legacy.py",
                    "replay.py",
                    "selftest.py",
                ],
                "selftest.py": [
                    "common.py",
                    "fixtures.py",
                    "kernel.py",
                    "legacy.py",
                    "plateau_helpers.py",
                    "regression.py",
                ],
                "fixtures.py": ["common.py", "legacy.py"],
                "kernel.py": ["common.py", "legacy.py"],
                "replay.py": [
                    "common.py",
                    "evidence.py",
                    "fixtures.py",
                    "kernel.py",
                    "legacy.py",
                ],
                "evidence.py": ["common.py"],
                "regression.py": [
                    "common.py",
                    "evidence.py",
                    "fixtures.py",
                    "kernel.py",
                    "legacy.py",
                    "replay.py",
                ],
                "plateau_helpers.py": ["legacy.py"],
            },
            "source_top_level_parent_imports": False,
        },
    )
    c.save_json(
        "preparation.json",
        {
            "status": "prepared_for_root_review_NOT_FROZEN_NOT_MEASURED",
            "invocation": ["artifacts/joint-repeat-feasibility-r1/run.sh", "prepare"],
            "utc": c.utc(),
            "timings": timings,
            "process": c.elapsed(),
            "tiny_selftest_cpu_seconds": tests["cpu_seconds"],
            "tiny_selftest_wall_seconds": tests["wall_seconds"],
            "copied_functions": provenance,
            "full_grid_extractions": 0,
            "candidate_grid_extractions": 0,
            "primary_panel_extractions": 0,
            "runtime_sha256": c.fingerprint(runtime),
            "manifest_sha256": c.fingerprint(proposal),
            "sources_sha256": c.fingerprint(c.load_json("sources.json")),
            "root_review_required": True,
            "unresolved_brief_ambiguities": proposal["unresolved_brief_ambiguities"],
        },
    )
    emit(
        "prepared",
        inputs_metadata=len(inventory),
        full_grid_extractions=0,
        freeze_performed=False,
        **c.elapsed(),
    )


def check_preparation():
    c.require(
        c.source_index() == c.load_json("sources.json")["files"],
        "Source closure changed since preparation",
    )
    c.provenance_check()
    validate_parent_saved_hashes()
    c.require(
        c.parent_snapshot() == c.load_json("parent_index.json"),
        "Parent snapshot changed",
    )
    runtime = c.validate_runtime(c.load_json("runtime.preparation.json"))
    c.require(
        c.load_json("preparation.json")["runtime_sha256"] == c.fingerprint(runtime),
        "Preparation runtime fingerprint mismatch",
    )
    proposal = c.load_json("manifest.proposed.json")
    c.require(proposal["brief_sha256"] == c.sha(c.ROOT / "BRIEF.md"), "Brief changed")
    c.require(
        proposal["inventory_sha256"] == c.fingerprint(c.load_json("keys.json")),
        "Inventory changed",
    )
    c.require(
        c.typed(c.load_json("keys.json")) == c.typed(fixtures.keys()),
        "Inventory implementation changed",
    )
    c.require(
        c.load_json("preparation.json")["manifest_sha256"] == c.fingerprint(proposal),
        "Proposal changed",
    )
    c.require(
        c.load_json("preparation.json")["sources_sha256"]
        == c.fingerprint(c.load_json("sources.json")),
        "Source index changed",
    )
    return proposal, runtime


def freeze():
    c.require(
        not (c.ROOT / "attempt.started.json").exists(), "Attempt already consumed"
    )
    proposal, runtime = check_preparation()
    c.save_json(
        "freeze.json",
        {
            "status": "frozen_by_explicit_root_invocation",
            "utc": c.utc(),
            "manifest": proposal,
            "manifest_sha256": c.fingerprint(proposal),
            "source_files_sha256": c.source_index(),
            "preparation_files_sha256": {
                n: c.sha(c.ROOT / n) for n in PREPARATION_INPUTS
            },
            "runtime": runtime,
            "runtime_sha256": c.fingerprint(runtime),
            "parents_sha256": c.load_json("parent_index.json"),
            "exclusive_attempts": 1,
            "measurement_performed": False,
        },
    )
    emit("frozen", measurement_performed=False)


def validate_freeze():
    frozen = c.load_json("freeze.json")
    proposal, runtime = check_preparation()
    c.require(
        frozen["manifest_sha256"] == c.fingerprint(proposal), "Frozen manifest mismatch"
    )
    c.require(
        c.typed(frozen["manifest"]) == c.typed(proposal),
        "Frozen manifest content mismatch",
    )
    c.require(
        frozen["source_files_sha256"] == c.source_index(), "Frozen source changed"
    )
    c.require(
        frozen["preparation_files_sha256"]
        == {n: c.sha(c.ROOT / n) for n in PREPARATION_INPUTS},
        "Frozen preparation changed",
    )
    c.require(
        frozen["runtime_sha256"] == c.fingerprint(runtime), "Frozen runtime mismatch"
    )
    c.require(
        c.typed(frozen["runtime"]) == c.typed(runtime),
        "Frozen runtime content mismatch",
    )
    c.require(frozen["parents_sha256"] == c.parent_snapshot(), "Frozen parents changed")
    return frozen


def allocate():
    directory = c.ROOT / "arrays"
    directory.mkdir()
    specs = evidence.storage_specs()
    arrays = {}
    for name, (shape, dtype) in specs.items():
        c.CHECK()
        value = np.lib.format.open_memmap(
            directory / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
        )
        value[:] = (
            np.nan
            if np.issubdtype(dtype, np.floating)
            else (-1 if np.issubdtype(dtype, np.integer) else False)
        )
        arrays[name] = value
    c.save_json("array_schema.json", evidence.array_schema(c.load_json("keys.json")))
    return arrays


def baseline_gate(arrays, inventory):
    parent = c.ROOT.parent / "cycle-length-feasibility-r1"
    report = json.loads((parent / "report.json").read_text())
    expected = {
        r["row_id"]: r
        for r in map(json.loads, (parent / "records.jsonl").read_text().splitlines())
    }
    c.require(len(expected) == 5120, "Cached controls missing")
    original = json.loads((parent / "cached_reference.json").read_text())
    original_lookup = {
        (r["family"], r["period_samples"], r["phase_index"]): r
        for r in original["records"]
    }
    cache = {
        name: np.load(
            parent / "arrays" / (name + ".npy"), mmap_mode="r", allow_pickle=False
        )
        for name in (
            "windows_768",
            "extractor_correlations",
            "features",
            "feature_valid_mask",
            "fixture_present_mask",
            "correlation_present_mask",
            "record_present_mask",
        )
    }
    bases = {}
    counts = {(n, ri): Counter() for n in c.LENGTHS for ri in range(2)}
    reproduced_hash = __import__("hashlib").sha256()
    cached_hash = __import__("hashlib").sha256()
    with (c.ROOT / "baseline_reproduced.jsonl").open("xb") as journal:
        for item in inventory[:2560]:
            c.CHECK()
            i, n, fi, li = (
                item["input_index"],
                item["length"],
                item["source_fixture_index"],
                item["source_length_index"],
            )
            window = fixtures.generate(item)
            c.require(
                cache["fixture_present_mask"][fi]
                and cache["correlation_present_mask"][i],
                "Cached fixture/corr absent",
            )
            c.require(
                window.tobytes() == cache["windows_768"][fi, :, :n].tobytes(),
                "Legacy saved generator/prefix mismatch",
            )
            mags = kernel.magnitudes(window)
            smooth = sosfiltfilt(c.SOS, mags[0], padtype="odd", padlen=9)
            rs = legacy.correlations(smooth)
            c.require(
                rs.tobytes() == cache["extractor_correlations"][i].tobytes(),
                "Legacy correlation float-bit mismatch",
            )
            arrays["windows"][i, :, :n] = window
            arrays["magnitudes"][i, :, :n] = mags
            arrays["filtered"][i, 0, :n] = smooth
            arrays["extraction_correlations"][i, 0] = rs
            arrays["input_present"][i] = True
            for ri, rule in enumerate(legacy.RULES):
                c.CHECK()
                base, feature = legacy.extract(mags, smooth, rs, rule)
                record = kernel.compatible_record(
                    base,
                    feature,
                    item,
                    ri,
                    window,
                    mags,
                    smooth,
                    rs,
                    report["analytic_minimum_root_phases_cycles"],
                )
                parentid = fi * 4 + li * 2 + ri
                c.require(
                    cache["record_present_mask"][parentid], "Cached record absent"
                )
                target = expected[parentid]
                if c.typed(record) != c.typed(target):
                    c.save_json(
                        "baseline_mismatch.json",
                        {
                            "input": item,
                            "rule": rule,
                            "reproduced": record,
                            "cached": target,
                        },
                    )
                    raise RuntimeError(
                        "ALL cached-field/type/float-bit control gate failed"
                    )
                if li == 0 and ri == 0:
                    old = original_lookup[
                        (item["family"], item["period_samples"], item["phase_index"])
                    ]
                    c.require(
                        c.typed({k: record[k] for k in old}) == c.typed(old),
                        "Original r1 cached-field projection failed",
                    )
                vector = np.full(400, np.nan) if feature is None else feature
                c.require(
                    vector.tobytes() == cache["features"][parentid].tobytes(),
                    "Legacy feature float bits mismatch",
                )
                c.require(
                    bool(cache["feature_valid_mask"][parentid])
                    == (feature is not None),
                    "Legacy feature validity mismatch",
                )
                arrays["features"][i * 3 + ri] = vector
                arrays["feature_valid"][i * 3 + ri] = feature is not None
                arrays["baseline_present"][i, ri] = True
                bases[(i, ri)] = base
                encoded = c.canonical(record) + b"\n"
                journal.write(encoded)
                reproduced_hash.update(encoded)
                cached_hash.update(c.canonical(target) + b"\n")
                group = counts[(n, ri)]
                group["finite"] += int(feature is not None)
                group["missing_pair"] += int(
                    base["failure_reason"] == "no_eligible_minima_pair"
                )
                group["channel_degenerate"] += int(
                    base["failure_reason"] == "channel_degenerate"
                )
                group["finite_half_alias"] += int(
                    feature is not None
                    and record["selected_pair_alias"] == "half_stride"
                )
            if (i + 1) % 128 == 0:
                journal.flush()
                emit("baseline", inputs=i + 1, **c.elapsed())
    expected_counts = fixtures.manifest()["expected_control_counts"]
    for (n, ri), observed in counts.items():
        c.require(
            dict(observed) == expected_counts[str(n)],
            f"Baseline counts mismatch N{n},rule{ri}",
        )
    c.require(arrays["baseline_present"].all(), "Baseline present mask incomplete")
    c.require(
        reproduced_hash.hexdigest() == cached_hash.hexdigest(),
        "Baseline combined hash mismatch",
    )
    result = {
        "status": "pass",
        "inputs": 2560,
        "records": 5120,
        "before_any_grid_candidate": True,
        "all_cached_fields_types_float_bits": True,
        "all_saved_generator_prefix_correlation_feature_validity_bits": True,
        "counts": [
            {"length": n, "rule": legacy.RULES[ri], **dict(counts[(n, ri)])}
            for n in c.LENGTHS
            for ri in range(2)
        ],
        "canonical_sorted_string_keys": True,
        "reproduced_jsonl_sha256": reproduced_hash.hexdigest(),
        "cached_ordered_jsonl_sha256": cached_hash.hexdigest(),
    }
    c.save_json("baseline_gate.json", result)
    return bases, result


def run_candidate(arrays, inventory, bases):
    records, diagnostics = [], []
    with (
        (c.ROOT / "records.jsonl").open("xb") as journal,
        (c.ROOT / "diagnostics.jsonl").open("xb") as djournal,
    ):
        for item in inventory:
            c.CHECK()
            i, n = item["input_index"], item["length"]
            if item["kind"] == "legacy":
                mags = np.asarray(arrays["magnitudes"][i, :, :n])
                filtered = np.stack(
                    (
                        arrays["filtered"][i, 0, :n],
                        sosfiltfilt(c.SOS, mags[1], padtype="odd", padlen=9),
                    )
                )
                arrays["filtered"][i, :, :n] = filtered
            else:
                window = fixtures.generate(item)
                mags = kernel.magnitudes(window)
                filtered = kernel.filter_modalities(mags)
                arrays["windows"][i, :, :n] = window
                arrays["magnitudes"][i, :, :n] = mags
                arrays["filtered"][i, :, :n] = filtered
                arrays["extraction_correlations"][i, 0] = legacy.correlations(
                    filtered[0]
                )
                arrays["input_present"][i] = True
            joint, stats, valid, finite = kernel.joint_correlations(filtered)
            arrays["extraction_correlations"][i, 1:] = joint
            arrays["extraction_correlation_finite"][i] = np.isfinite(
                arrays["extraction_correlations"][i]
            )
            arrays["joint_guard_stats"][i] = stats
            arrays["joint_guard_valid"][i] = valid
            arrays["joint_guard_finite"][i] = finite
            arrays["joint_present"][i] = True
            whole = kernel.whole_window(mags, filtered)
            for ai, arm in enumerate(c.ARMS):
                rs = arrays["extraction_correlations"][i, 0] if ai < 2 else joint[2]
                if item["kind"] == "legacy" and ai < 2:
                    base = bases[(i, ai)]
                    feature = (
                        np.asarray(arrays["features"][i * 3 + ai])
                        if arrays["feature_valid"][i * 3 + ai]
                        else None
                    )
                else:
                    base, feature = legacy.extract(
                        mags, filtered[0], rs, legacy.RULES[0 if ai == 0 else 1]
                    )
                row = kernel.record(item, ai, base, feature, rs, whole, joint)
                rowid = row["row_id"]
                arrays["features"][rowid] = (
                    np.full(400, np.nan) if feature is None else feature
                )
                arrays["feature_valid"][rowid] = feature is not None
                arrays["selected_pairs"][rowid] = (
                    [-1, -1]
                    if base["pair_bounds_inclusive"] is None
                    else base["pair_bounds_inclusive"]
                )
                arrays["selected_lags"][rowid] = (
                    -1 if base["selected_lag"] is None else base["selected_lag"]
                )
                arrays["record_present"][rowid] = True
                records.append(row)
                journal.write(c.canonical(row) + b"\n")
            values, stats, valid, finite, summaries = kernel.diagnostic(mags, filtered)
            arrays["diagnostic_correlations"][i] = values
            arrays["diagnostic_finite"][i] = np.isfinite(values)
            arrays["diagnostic_guard_stats"][i] = stats
            arrays["diagnostic_guard_valid"][i] = valid
            arrays["diagnostic_guard_finite"][i] = finite
            arrays["diagnostic_present"][i] = True
            diag = evidence.diagnostic_record(item, summaries)
            diagnostics.append(diag)
            djournal.write(c.canonical(diag) + b"\n")
            if (i + 1) % 32 == 0:
                journal.flush()
                djournal.flush()
                emit(
                    "candidate_and_diagnostics",
                    inputs=i + 1,
                    records=len(records),
                    **c.elapsed(),
                )
    return records, diagnostics


def output_index():
    return evidence.file_inventory(c.ROOT)


def execute():
    c.require(
        _STAGE == "execute" and (c.ROOT / "attempt.started.json").is_file(),
        "Execute requires the reserved CLI entrypoint",
    )
    evidence.validate_attempt(c.ROOT)
    c.arm_budget()
    timings = {}
    arrays = {}
    state = {
        "status": "failed",
        "stage": "validation",
        "exclusive_attempt_consumed": True,
    }
    try:
        with c.stage(timings, "validation_and_allocation"):
            frozen = validate_freeze()
            inventory = c.load_json("keys.json")
            arrays = allocate()
        state["stage"] = "all_legacy_controls_before_candidate"
        with c.stage(timings, "baseline"):
            bases, gate = baseline_gate(arrays, inventory)
        emit("baseline_gate_passed", rows=5120, **c.elapsed())
        state["stage"] = "fixed_checks_after_baseline"
        with c.stage(timings, "fixed_checks"):
            tests = selftest.run()
            c.save_json("measurement_fixed_checks.json", tests)
        state["stage"] = "candidate_and_diagnostics"
        with c.stage(timings, "candidate_and_diagnostics"):
            rows, diagnostics = run_candidate(arrays, inventory, bases)
        del bases
        state["stage"] = "saved_evidence_verification_and_summary"
        extras = evidence.metadata_arrays(inventory)
        for name, array in extras.items():
            with (c.ROOT / "arrays" / (name + ".npy")).open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
        arrays.update(extras)
        with c.stage(timings, "verification"):
            # No second correlation sweep within the exclusive measurement.
            audit = replay.verify(
                rows,
                arrays,
                inventory,
                diagnostics,
                schema=c.load_json("array_schema.json"),
                full=False,
            )
            scientific = replay.summary(rows, arrays, inventory)
            c.save_json("saved_evidence_audit.json", audit)
            c.save_json("summary.json", scientific)
        state["stage"] = "lossless_packaging"
        with c.stage(timings, "packaging"):
            for value in arrays.values():
                if hasattr(value, "flush"):
                    value.flush()
            with (c.ROOT / "evidence.npz").open("xb") as handle:
                np.savez(handle, **arrays)
            c.require(
                c.parent_snapshot() == frozen["parents_sha256"],
                "Parents changed during execution",
            )
            arrays_sha = {
                p.name: c.sha(p) for p in sorted((c.ROOT / "arrays").glob("*.npy"))
            }
        report = {
            "status": "complete",
            "attempt_binding": evidence.validate_attempt(c.ROOT),
            "baseline_gate": gate,
            "scientific_summary_sha256": c.fingerprint(scientific),
            "manifest_sha256": frozen["manifest_sha256"],
            "runtime_sha256": frozen["runtime_sha256"],
            "source_files_sha256": frozen["source_files_sha256"],
            "array_files_sha256": arrays_sha,
            "binding_files_sha256": evidence.binding_hashes(c.ROOT),
            "records_sha256": c.sha(c.ROOT / "records.jsonl"),
            "diagnostics_sha256": c.sha(c.ROOT / "diagnostics.jsonl"),
            "evidence_sha256": c.sha(c.ROOT / "evidence.npz"),
            "timings": timings,
            "process": c.elapsed(),
            "preparation": c.load_json("preparation.json"),
            "saved_evidence_verification": audit,
            "no_real_data_protocol_nomination": True,
            "N768": "DIAGNOSTIC ONLY",
            "quality_cutoff": None,
        }
        c.CHECK()
        c.save_json("report.json", report)
        c.require(
            c.fingerprint(c.load_json("summary.json"))
            == report["scientific_summary_sha256"],
            "Post-save strict fingerprint mismatch",
        )
        state.update(status="complete", stage="saved_and_verified")
        c.save_json("run_state.json", state)
        c.save_json(
            "timings.json",
            {
                "stages": timings,
                "process": c.elapsed(),
                "preparation": c.load_json("preparation.json")["process"],
            },
        )
        emit("complete", records=8640, inputs=2880, **c.elapsed())
        c.save_json("artifact_index.json", evidence.index_value(c.ROOT))
    except BaseException as exc:
        # Partial memmaps and all journals remain. Token is never removed and no retry is offered.
        state.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            timings=timings,
            process=c.elapsed(),
        )
        for value in arrays.values():
            if hasattr(value, "flush"):
                value.flush()
        if (c.ROOT / "run_state.json").exists():
            c.replace_state("run_state.json", state)
        else:
            c.save_json("run_state.json", state)
        emit("failed_attempt_consumed", **state)
        raise


def verify(full):
    # Read-only verification: no writes, no new fixtures, no filter or model/quality fit.
    frozen = validate_freeze()
    evidence.validate_index(c.ROOT)
    rows = list(map(c.decode_json, (c.ROOT / "records.jsonl").read_text().splitlines()))
    diagnostics = list(
        map(c.decode_json, (c.ROOT / "diagnostics.jsonl").read_text().splitlines())
    )
    inventory = c.load_json("keys.json")
    arrays = {
        p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
        for p in sorted((c.ROOT / "arrays").glob("*.npy"))
    }
    report = c.load_json("report.json")
    c.require(report["status"] == "complete", "Exclusive attempt incomplete")
    c.require(
        c.load_json("run_state.json")["status"] == "complete",
        "Exclusive attempt ended in failure",
    )
    c.require(
        c.fingerprint(c.load_json("summary.json"))
        == report["scientific_summary_sha256"],
        "Summary strict fingerprint mismatch",
    )
    evidence.validate_report_bindings(report, frozen, arrays)
    # NPZ is independently matched to every saved standalone array, without pickle.
    with np.load(c.ROOT / "evidence.npz", allow_pickle=False) as saved:
        c.require(
            len(saved.files) == len(arrays) and set(saved.files) == set(arrays),
            "NPZ dependency/array closure mismatch",
        )
        for name, value in arrays.items():
            loaded = saved[name]
            c.require(
                value.shape == loaded.shape
                and value.dtype == loaded.dtype
                and value.tobytes() == loaded.tobytes(),
                "NPZ differs " + name,
            )
    audit = replay.verify(
        rows,
        arrays,
        inventory,
        diagnostics,
        schema=c.load_json("array_schema.json"),
        full=full,
    )
    scientific = replay.summary(rows, arrays, inventory)
    c.require(
        c.typed(scientific) == c.typed(c.load_json("summary.json")),
        "Saved summary replay mismatch",
    )
    c.require(
        c.parent_snapshot() == frozen["parents_sha256"], "Parent changed during verify"
    )
    emit("verify", **audit, **c.elapsed())


def main():
    if _STAGE == "prepare":
        prepare()
    elif _STAGE == "check-preparation":
        check_preparation()
        emit("preparation_checked", freeze_performed=False, measurement_performed=False)
    elif _STAGE == "selftest":
        selftest.main()
    elif _STAGE == "freeze":
        freeze()
    elif _STAGE == "execute":
        execute()
    else:
        verify(sys.argv[2:] == ["--replay-correlations"])


if __name__ == "__main__":
    main()
