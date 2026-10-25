"""Exact saved-evidence schemas and artifact bindings. No generators or correlations."""

from pathlib import Path

import common as c
import numpy as np

TRUST_POLICY = (
    "Local consistency under reviewed source and independently retained freeze/index digests; "
    "the root index excludes only itself. Mutable local sources, freeze, reports and index "
    "cannot authenticate one another against an owner rewriting all trust anchors."
)
BINDING_FILES = (
    "attempt.started.json",
    "worker.started.json",
    "manifest.proposed.json",
    "keys.json",
    "parent_index.json",
    "runtime.preparation.json",
    "preparation.json",
    "selftests.json",
    "sources.json",
    "fingerprint_erratum.json",
    "freeze.json",
    "baseline_gate.json",
    "baseline_reproduced.jsonl",
    "measurement_fixed_checks.json",
    "array_schema.json",
    "saved_evidence_audit.json",
    "summary.json",
)


def storage_specs(inputs=2880, legacy_inputs=2560):
    return {
        "windows": ((inputs, 6, 768), np.float64),
        "magnitudes": ((inputs, 2, 768), np.float64),
        "filtered": ((inputs, 2, 768), np.float64),
        "extraction_correlations": ((inputs, 4, 121), np.float64),
        "extraction_correlation_finite": ((inputs, 4, 121), np.bool_),
        "joint_guard_stats": ((inputs, 2, 121, 2, 2), np.float64),
        "joint_guard_valid": ((inputs, 2, 121, 2), np.bool_),
        "joint_guard_finite": ((inputs, 2, 121, 2), np.bool_),
        "diagnostic_correlations": ((inputs, 6, 161), np.float64),
        "diagnostic_finite": ((inputs, 6, 161), np.bool_),
        "diagnostic_guard_stats": ((inputs, 4, 161, 2, 2), np.float64),
        "diagnostic_guard_valid": ((inputs, 4, 161, 2), np.bool_),
        "diagnostic_guard_finite": ((inputs, 4, 161, 2), np.bool_),
        "features": ((inputs * 3, 400), np.float64),
        "feature_valid": ((inputs * 3,), np.bool_),
        "selected_pairs": ((inputs * 3, 2), np.int64),
        "selected_lags": ((inputs * 3,), np.int64),
        "input_present": ((inputs,), np.bool_),
        "record_present": ((inputs * 3,), np.bool_),
        "joint_present": ((inputs,), np.bool_),
        "diagnostic_present": ((inputs,), np.bool_),
        "baseline_present": ((legacy_inputs, 2), np.bool_),
    }


def metadata_arrays(inventory):
    return {
        "keys": np.array([k["key"] for k in inventory]),
        "lengths": np.array([k["length"] for k in inventory], dtype=np.int64),
        "arms": np.array(c.ARMS),
        "lags": np.arange(80, 201, dtype=np.int64),
        "diagnostic_lags": np.arange(40, 201, dtype=np.int64),
        "diagnostic_anchors_256": np.arange(56, dtype=np.int64),
        "diagnostic_anchors_768": np.arange(568, dtype=np.int64),
        "row_mapping": np.array(
            [[i, ai] for i in range(len(inventory)) for ai in range(3)], dtype=np.int64
        ),
    }


def array_schema(inventory):
    specs = storage_specs(len(inventory), sum(k["kind"] == "legacy" for k in inventory))
    specs.update({k: (v.shape, v.dtype) for k, v in metadata_arrays(inventory).items()})
    return {
        "schema_version": 2,
        "arrays": {
            name: {"shape": list(shape), "dtype": np.dtype(dtype).str}
            for name, (shape, dtype) in specs.items()
        },
        "padding": "Windows/magnitudes/filtered padded NaN beyond actual N; complete evidence requires all present masks true",
        "extraction_channels": [
            "historical_acc",
            "guarded_acc",
            "guarded_gyro",
            "joint",
        ],
        "diagnostic_channels": [
            "raw_acc",
            "raw_gyro",
            "filtered_acc",
            "filtered_gyro",
            "raw_joint",
            "filtered_joint",
        ],
        "guard_stats_axes": "input,modality_or_channel,lag,vector(prefix_or_suffix),stat(q_or_std)",
        "correlation_rule": "controls both share historical_acc; candidate uses joint",
        "row_mapping": "row=input_index*3+arm_index; key inventory keys.json",
        "diagnostic_anchors": "0..N-201 inclusive, N-200 pairs at every lag",
    }


def validate_arrays(arrays, inventory, schema):
    expected = array_schema(inventory)
    c.require(c.typed(schema) == c.typed(expected), "Array schema mismatch")
    c.require(set(arrays) == set(expected["arrays"]), "Array closure mismatch")
    for name, spec in expected["arrays"].items():
        value = arrays[name]
        c.require(
            isinstance(value, np.ndarray)
            and list(value.shape) == spec["shape"]
            and value.dtype.str == spec["dtype"],
            "Array dtype/shape mismatch " + name,
        )
    for name, expected_array in metadata_arrays(inventory).items():
        c.require(
            arrays[name].tobytes() == expected_array.tobytes(),
            "Fixed array mapping/axis mismatch " + name,
        )
    for name in (
        "input_present",
        "record_present",
        "joint_present",
        "diagnostic_present",
        "baseline_present",
    ):
        c.require(bool(arrays[name].all()), "Missing present mask " + name)


def diagnostic_record(item, summaries):
    n = item["length"]
    return {
        "input_index": item["input_index"],
        "key": item["key"],
        "length": n,
        "anchor_count": n - 200,
        "anchors_first": 0,
        "anchors_last": n - 201,
        "lags_first": 40,
        "lags_last": 200,
        "channels": summaries,
        "extraction_effect": "none",
    }


def guard_relationships(stats, valid, finite, coefficients, label):
    # Saved raw/magnitude/filtered prefixes are finite. These masks describe those exact vectors.
    c.require(bool(finite.all()), label + " vector finite masks mismatch")
    c.require(
        bool(np.isfinite(stats).all()) and bool((stats >= 0).all()),
        label + " invalid RMS/std statistics",
    )
    q, std = stats[..., 0], stats[..., 1]
    expected = finite & np.isfinite(q) & np.isfinite(std) & (q > 0) & (std > 1e-12 * q)
    c.require(np.array_equal(valid, expected), label + " guard threshold/mask mismatch")
    c.require(not np.isinf(coefficients).any(), label + " infinite correlation")
    c.require(
        np.array_equal(np.isfinite(coefficients), expected.all(axis=-1)),
        label + " guard/correlation finite mismatch",
    )
    c.require(
        bool((np.abs(coefficients[np.isfinite(coefficients)]) <= 1).all()),
        label + " Pearson range mismatch",
    )


def mean_relationship(left, right, actual, label):
    expected = np.full(actual.shape, np.nan)
    good = np.isfinite(left) & np.isfinite(right)
    expected[good] = (left[good] + right[good]) / 2
    c.require(
        actual.tobytes() == expected.tobytes(),
        label + " mean/no-fallback bits mismatch",
    )


def audit_metadata(full, inputs):
    return {
        "status": "pass",
        "records": inputs * 3,
        "inputs": inputs,
        "no_generated_inputs": True,
        "no_filter_reexecution": True,
        "raw_magnitude_bits_bound": True,
        "array_schema_and_metadata_checked": True,
        "guard_stat_mask_relationships_checked": True,
        "full_correlations_replayed": full,
        "saved_feature_and_pair_replay": True,
        "quality_cutoff": None,
    }


def file_inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        c.require(not path.is_symlink(), "Symlinks forbidden in artifact inventory")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name != "artifact_index.json":
                result[name] = c.sha(path)
    return result


def index_value(root):
    return {
        "files_sha256": file_inventory(root),
        "excludes_self": True,
        "canonical_strict_string_keys": True,
        "trust_policy": TRUST_POLICY,
    }


def validate_index(root):
    path = Path(root) / "artifact_index.json"
    value = c.decode_json(path.read_bytes())
    c.require(
        path.read_bytes() == c.canonical(value) + b"\n", "Noncanonical artifact index"
    )
    c.require(
        c.typed(value) == c.typed(index_value(root)),
        "Artifact index exact inventory/hash/schema mismatch",
    )
    if any(
        (Path(root) / name).exists()
        for name in ("attempt.started.json", "worker.started.json", "report.json")
    ):
        validate_attempt(root)
    return value["files_sha256"]


def binding_hashes(root):
    return {name: c.sha(Path(root) / name) for name in BINDING_FILES}


def validate_attempt(root):
    """Validate persisted identities/bindings, never the lifetime of a deceased launcher."""
    values = {}
    for name in ("attempt.started.json", "worker.started.json"):
        data = (Path(root) / name).read_bytes()
        value = c.decode_json(data)
        c.require(
            data == c.canonical(value) + b"\n", "Noncanonical attempt/worker receipt"
        )
        values[name] = value
    attempt, worker = values["attempt.started.json"], values["worker.started.json"]
    c.require(
        type(attempt) is dict
        and set(attempt)
        == {
            "automatic_retry",
            "cpu_budget_seconds",
            "launcher_pid",
            "launcher_starttime_ticks",
            "route",
            "schema",
            "status",
            "wall_budget_seconds",
        },
        "Attempt schema fields mismatch",
    )
    c.require(
        type(worker) is dict
        and set(worker)
        == {
            "attempt",
            "schema",
            "status",
            "worker_pid",
            "worker_starttime_ticks",
        },
        "Worker receipt schema fields mismatch",
    )
    for value, pid_key, ticks_key in (
        (attempt, "launcher_pid", "launcher_starttime_ticks"),
        (worker, "worker_pid", "worker_starttime_ticks"),
    ):
        pid, ticks = value[pid_key], value[ticks_key]
        c.require(type(pid) is int and pid > 0, "Invalid persisted process PID")
        c.require(
            type(ticks) is str
            and ticks.isascii()
            and ticks.isdecimal()
            and ticks[0] != "0",
            "Invalid persisted process starttime",
        )
    expected = {
        "automatic_retry": False,
        "cpu_budget_seconds": 600,
        "launcher_pid": attempt["launcher_pid"],
        "launcher_starttime_ticks": attempt["launcher_starttime_ticks"],
        "route": attempt["route"],
        "schema": 2,
        "status": "exclusive_attempt_consumed",
        "wall_budget_seconds": 900,
    }
    c.require(c.typed(attempt) == c.typed(expected), "Attempt schema values mismatch")
    c.require(attempt["route"] in ("shell", "direct"), "Unknown attempt route")
    expected_worker = {
        "attempt": attempt,
        "schema": 1,
        "status": "exclusive_worker_started",
        "worker_pid": worker["worker_pid"],
        "worker_starttime_ticks": worker["worker_starttime_ticks"],
    }
    c.require(
        c.typed(worker) == c.typed(expected_worker),
        "Worker receipt attempt/schema mismatch",
    )
    if attempt["route"] == "direct":
        c.require(
            worker["worker_pid"] == attempt["launcher_pid"]
            and worker["worker_starttime_ticks"] == attempt["launcher_starttime_ticks"],
            "Direct worker identity differs from attempt",
        )
    else:
        c.require(
            worker["worker_pid"] != attempt["launcher_pid"],
            "Shell worker must descend from launcher",
        )
    return values


def validate_report_bindings(report, frozen, arrays):
    root = c.ROOT
    expected = {
        "attempt_binding": validate_attempt(root),
        "scientific_summary_sha256": c.fingerprint(c.load_json("summary.json")),
        "manifest_sha256": frozen["manifest_sha256"],
        "runtime_sha256": frozen["runtime_sha256"],
        "source_files_sha256": frozen["source_files_sha256"],
        "array_files_sha256": {
            name + ".npy": c.sha(root / "arrays" / (name + ".npy"))
            for name in sorted(arrays)
        },
        "records_sha256": c.sha(root / "records.jsonl"),
        "diagnostics_sha256": c.sha(root / "diagnostics.jsonl"),
        "evidence_sha256": c.sha(root / "evidence.npz"),
        "binding_files_sha256": binding_hashes(root),
        "baseline_gate": c.load_json("baseline_gate.json"),
        "preparation": c.load_json("preparation.json"),
        "saved_evidence_verification": c.load_json("saved_evidence_audit.json"),
    }
    for name, value in expected.items():
        c.require(
            name in report and c.typed(report[name]) == c.typed(value),
            "Report artifact binding mismatch " + name,
        )
    c.require(
        c.typed(report["saved_evidence_verification"])
        == c.typed(audit_metadata(False, len(arrays["keys"]))),
        "Saved default audit mismatch",
    )
    c.require(
        c.typed(frozen["runtime"]) == c.typed(c.load_json("runtime.preparation.json")),
        "Frozen runtime content mismatch",
    )
    c.require(
        frozen["runtime_sha256"] == c.fingerprint(frozen["runtime"]),
        "Frozen runtime fingerprint mismatch",
    )
    c.require(
        c.typed(report["source_files_sha256"])
        == c.typed(c.load_json("sources.json")["files"]),
        "Report source index mismatch",
    )
