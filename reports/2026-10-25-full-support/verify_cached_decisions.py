"""Replay frozen cached decisions only; never import a study entrypoint or fit."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib
import struct
import sys
import time
from pathlib import Path

import common as c

ANCHORS = {
    "sourceIndex": "dd8d55e9723e33c46e357bed6efe8e228fa00c4d2d1d44b6f177b94586aaf673",
    "runIndex": "cd2ba188ac37358afaf0c83a0be851831d6dd337eb811e3b5175f2ac4b1a9897",
    "freeze": "e3776afc6f29d9cb88c1247e006dd82b41b7d3db499eb63bb6f16e843f8acb1a",
    "results": "1ac851845eedbf43f4fdff53af6327423712321bf6dc240b5fc27509aebee41a",
    "report": "93c3363f8138b3a84bf87ed9e740587b87d9e8a687c0602eacc4942bc602a831",
    "preflight": "c873b83b0eca2db0ef2d222c96969b76b4490a070e776eee3faee96576c4e8af",
    "Rgate": "36b2d5cc74dcd58127222dfde5009db560f4bdad3a8b9d24a623d5dd3b4d4bfd",
}
CONDITIONS = (
    "clean",
    "rotation30",
    "rotation90",
    "gain09",
    "gain11",
    "noise01",
    "noise05",
)
CALC_NAMES = (
    "pool",
    "cutoff",
    "row",
    "descriptive_summaries",
    "exact_difference",
    "comparison",
)
CALC_HEADER = '''"""Exact pure functions extracted from frozen kernel.py; no fitting imports."""
import hashlib
from fractions import Fraction
import common as c
import numpy as np
from engine.empirical import empirical_threshold
from engine.historical_metrics import report
from engine.robustness_selection import ranking_metrics, summarize_nuisance_study, CONDITIONS


'''
ENGINE_FILES = (
    "__init__.py",
    "empirical.py",
    "historical_metrics.py",
    "calibration_selection.py",
    "robustness_selection.py",
)
PUBLIC_FILES = {
    "README.md",
    "audit.md",
    "summary.json",
    "typed-thresholds.json",
    "scope.json",
    "publication-provenance.json",
    "review-status.json",
    "common.py",
    "verify_cached_decisions.py",
    "test_cached_decisions.py",
    "requirements.txt",
    "authority/calculations.py",
    "docs/robustness_selection.md",
    "docs/gait_canonical.md",
    "docs/santos_dataset.md",
    "docs/cross_session_acquisition.md",
    "docs/source_provenance.json",
    "environment/pyproject.toml",
    "environment/uv.lock",
    "environment/requirements.txt",
    "environment/requirements-dev.txt",
} | {"authority/engine/" + name for name in ENGINE_FILES}


def calculation_source(kernel):
    tree = ast.parse(kernel)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    segments = []
    for name in CALC_NAMES:
        segment = ast.get_source_segment(kernel, functions[name])
        assert segment is not None
        segments.append(segment)
    return CALC_HEADER + "\n\n\n".join(segments) + "\n"


def original_bytes(bundle, kind, name):
    path = bundle / c.mapped_path(kind, name)
    raw = path.read_bytes()
    return gzip.decompress(raw) if name.endswith(".json") else raw


def pinned_index(bundle, kind, expected):
    raw = original_bytes(bundle, kind, "index.json")
    c.require(
        hashlib.sha256(raw).hexdigest() == expected, "Immutable original index anchor"
    )
    index = c.decode(raw)
    c.same(index["schema"], 1, "original index schema")
    c.same(index["excludes_only"], "index.json", "original index exclusion")
    return index["files"]


def check_copy(bundle, kind, name, record):
    raw = original_bytes(bundle, kind, name)
    c.same(
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        record,
        "immutable copy " + name,
    )


def threshold_record(fold, arm, record):
    value = record["threshold"]
    c.require(type(value) is float, "Frozen threshold must retain float type")
    return {
        "fold": fold,
        "arm": arm,
        "record": record,
        "float_hex": value.hex(),
        "float64_be_hex": struct.pack(">d", value).hex(),
    }


def unique_records(records, fields, expected, description):
    indexed = {}
    for record in records:
        key = c.fingerprint([record[f] for f in fields])
        c.require(key not in indexed, "Duplicate " + description)
        indexed[key] = record
    c.same(
        sorted(indexed),
        sorted(c.fingerprint(list(k)) for k in expected),
        "complete typed " + description,
    )
    return {k: indexed[c.fingerprint(list(k))] for k in expected}


def load_score(bundle, record, graph, np):
    artifact = record["artifact"]
    name = artifact["path"]
    c.require(
        name.startswith("scores/") and name.endswith(".npz"),
        "Only current score NPZ allowed",
    )
    path = bundle / "evidence" / c.relative(name)
    c.same(
        {"bytes": path.stat().st_size, "sha256": c.sha(path)},
        {k: artifact[k] for k in ("bytes", "sha256")},
        "score bytes",
    )
    with np.load(path, allow_pickle=False) as archive:
        c.same(sorted(archive.files), ["labels", "scores"], "exact NPZ keys")
        c.require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ key")
        arrays = {k: archive[k] for k in archive.files}
    c.same(
        {k: {"dtype": a.dtype.str, "shape": list(a.shape)} for k, a in arrays.items()},
        artifact["schema"],
        "score dtype/shape",
    )
    c.require(
        arrays["labels"].dtype == np.dtype("<i8")
        and arrays["scores"].dtype == np.dtype("<f8"),
        "Unchanged int64/float64 caches required",
    )
    c.require(
        arrays["scores"].ndim == 1 and np.isfinite(arrays["scores"]).all(),
        "Finite one-dimensional scores required",
    )
    c.same(arrays["labels"].tolist(), graph["labels"], "every ordered label")
    c.same(
        list(arrays["scores"].shape),
        [len(graph["labels"])],
        "every ordered claim score",
    )
    metadata = record["metadata"]
    for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids"):
        c.same(metadata[key], graph[key], "every ordered " + key)
    for key in ("n_genuine", "n_impostor"):
        c.same(metadata["coverage"][key], graph[key], "exact coverage denominator")
    return {**arrays, "metadata": metadata}


def rebuild_graph(trials, ids):
    ids = sorted(ids)
    enrollment = {
        str(s): sorted(t for person, day, t in trials if person == s and day == 1)
        for s in ids
    }
    probes = {
        str(s): sorted(t for person, day, t in trials if person == s and day == 2)
        for s in ids
    }
    rows = [
        (int(claim == person), claim, person, trial)
        for person in ids
        for trial in probes[str(person)]
        for claim in ids
    ]
    return {
        **{
            key: [row[i] for row in rows]
            for i, key in enumerate(
                ("labels", "claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
            )
        },
        "enrollment_trials": enrollment,
        "probe_trials": probes,
        "n_genuine": sum(row[0] for row in rows),
        "n_impostor": sum(1 - row[0] for row in rows),
    }


def verify(bundle):
    c.enforce_environment()
    started_cpu, started_wall = time.process_time(), time.monotonic()
    inventory = c.check_inventory(bundle)
    source_index = pinned_index(bundle, "source", ANCHORS["sourceIndex"])
    run_index = pinned_index(bundle, "run", ANCHORS["runIndex"])
    for name in sorted(source_index):
        if c.source_selected(name):
            check_copy(bundle, "source", name, source_index[name])
    result = c.decode(original_bytes(bundle, "run", "results.json"))
    reference = c.decode(original_bytes(bundle, "run", "R-restore-gate.json"))
    current = [*reference["inner"], *result["scores"]]
    c.require(len(current) == 660, "Exactly 660 current score caches required")
    score_names = {r["artifact"]["path"] for r in current}
    c.require(len(score_names) == 660, "Duplicate score cache path")
    selected_run = {
        n for n in run_index if "/" not in n and n.endswith(".json")
    } | score_names
    for name in sorted(selected_run):
        c.require(name in run_index, "Unregistered score path")
        check_copy(bundle, "run", name, run_index[name])
    expected_evidence = {c.mapped_path("run", n) for n in selected_run | {"index.json"}}
    c.same(
        sorted(n for n in inventory["files"] if n.startswith("evidence/")),
        sorted(expected_evidence),
        "no missing/extra evidence",
    )
    expected_source = {
        c.mapped_path("source", n) for n in source_index if c.source_selected(n)
    } | {"snapshot/index.json.gz"}
    c.same(
        sorted(n for n in inventory["files"] if n.startswith("snapshot/")),
        sorted(expected_source),
        "exact immutable source snapshot selection",
    )
    c.same(
        sorted(inventory["files"]),
        sorted(expected_source | expected_evidence | PUBLIC_FILES),
        "unknown publication entries refused even after rehash",
    )
    scope = c.read(bundle / "scope.json")
    c.same(scope["original_anchors"], ANCHORS, "published original anchors")
    for kind, original, selected in (
        ("source", source_index, {n for n in source_index if c.source_selected(n)}),
        ("run", run_index, selected_run),
    ):
        c.same(
            scope["original_inventory"][kind],
            {
                "copied": {
                    name: {**original[name], "published": c.mapped_path(kind, name)}
                    for name in sorted(selected)
                },
                "omitted": {
                    name: original[name] for name in sorted(set(original) - selected)
                },
                "original_index_excludes_only": "index.json",
            },
            "complete retained/omitted original inventory",
        )
    for key, name in (
        ("freeze", "freeze.json"),
        ("results", "results.json"),
        ("report", "report.json"),
        ("preflight", "preflight.json"),
        ("Rgate", "R-restore-gate.json"),
    ):
        c.same(run_index[name]["sha256"], ANCHORS[key], "supplied study anchor " + key)
    metadata = c.read(bundle / "snapshot/metadata.json.gz")
    registration = c.read(bundle / "evidence/registration.json.gz")
    frozen = c.read(bundle / "evidence/freeze.json.gz")
    c.same(
        registration["claim_graphs"], metadata["coverage"], "registered claim graphs"
    )
    c.same(
        registration["fit_assignments"],
        metadata["fit_assignments"],
        "registered TRAIN owners",
    )
    c.same(
        registration["metadata_sha256"],
        source_index["metadata.json"]["sha256"],
        "registered metadata",
    )
    c.same(
        frozen["preparation_index_sha256"],
        ANCHORS["sourceIndex"],
        "frozen source index",
    )
    for key, filename in (
        ("freeze_sha256", "freeze.json"),
        ("registration_sha256", "registration.json"),
        ("R_gate_sha256", "R-restore-gate.json"),
        ("preflight_sha256", "preflight.json"),
        ("attempt_sha256", "attempt.started.json"),
        ("worker_receipt_sha256", "worker.started.json"),
    ):
        c.same(result[key], run_index[filename]["sha256"], "result crosslink " + key)
    c.same(
        reference["original_results_sha256"],
        frozen["reference_files"]["results.json"]["sha256"],
        "original R results binding",
    )
    c.same(
        [
            reference["clean_gate_passed"],
            reference["fit_calls"],
            reference["historical_runtime_equivalence_claimed"],
        ],
        [True, 0, False],
        "R restore scope",
    )
    trials = metadata["trial_units"]
    c.require(
        len(trials) == len({tuple(t) for t in trials}) == 299,
        "299 unchanged trials required",
    )
    for fold in metadata["folds"]:
        expected = {
            "inner": [
                rebuild_graph(trials, inner["held_ids"]) for inner in fold["inner"]
            ],
            "validation": rebuild_graph(trials, fold["validation_ids"]),
        }
        c.same(
            metadata["coverage"][str(fold["fold"])],
            expected,
            "source-trial-derived complete graphs",
        )
    # Only these pure calculation modules are executable; snapshot entrypoints stay inert.
    for name in ENGINE_FILES:
        path = bundle / "authority/engine" / name
        c.same(
            path.read_bytes().hex(),
            (bundle / "snapshot/vendor/engine" / name).read_bytes().hex(),
            "unchanged pure vendor authority",
        )
    c.same(
        (bundle / "authority/calculations.py").read_text(),
        calculation_source((bundle / "snapshot/kernel.py").read_text()),
        "exact extracted functions",
    )
    sys.path.insert(0, str(bundle / "authority"))
    np = importlib.import_module("numpy")
    calc = importlib.import_module("calculations")
    c.same(np.__version__, "2.5.4", "pinned NumPy version")
    expected_inner = [
        (f, arm, role)
        for f in range(20)
        for arm in ("R", "H", "F")
        for role in range(4)
    ]
    expected_outer = [
        (f, arm, condition)
        for f in range(20)
        for arm in ("R", "H", "F")
        for condition in CONDITIONS
    ]
    inner = unique_records(
        [r for r in current if "role" in r],
        ("fold", "arm", "role"),
        expected_inner,
        "inner assignments",
    )
    outer = unique_records(
        [r for r in current if "condition" in r],
        ("fold", "arm", "condition"),
        expected_outer,
        "outer assignments",
    )
    historical = unique_records(
        reference["historical_outer"],
        ("fold", "condition"),
        [(f, cond) for f in range(20) for cond in CONDITIONS],
        "historical provenance",
    )
    rows, typed_thresholds, decisions = [], [], []
    for fold in metadata["folds"]:
        f = fold["fold"]
        c.same(
            c.read(bundle / f"evidence/cutoffs_{f:02}.json.gz"),
            result["cutoffs"][str(f)],
            "persisted both H/F cutoffs",
        )
        for arm in ("R", "H", "F"):
            threshold = (
                reference["cutoffs"][str(f)]
                if arm == "R"
                else result["cutoffs"][str(f)][arm]
            )
            pools = [
                load_score(
                    bundle,
                    inner[f, arm, role],
                    metadata["coverage"][str(f)]["inner"][role],
                    np,
                )
                for role in range(4)
            ]
            c.same(
                calc.cutoff(pools), threshold, "pooled policy C exact cutoff/ties/types"
            )
            typed_thresholds.append(threshold_record(f, arm, threshold))
            for condition in CONDITIONS:
                record = outer[f, arm, condition]
                cached = load_score(
                    bundle, record, metadata["coverage"][str(f)]["validation"], np
                )
                row = calc.row(f, arm, condition, cached, threshold["threshold"])
                rows.append(row)
                decisions.append(
                    {
                        "fold": f,
                        "arm": arm,
                        "condition": condition,
                        "accepted": (
                            cached["scores"] >= threshold["threshold"]
                        ).tolist(),
                        "decision_sha256": row["decision_sha256"],
                    }
                )
                if arm == "R":
                    old = historical[f, condition]
                    c.same(
                        old["paired_gate_ground_truth"],
                        condition not in ("noise01", "noise05"),
                        "historical noise scope",
                    )
                    if condition not in ("noise01", "noise05"):
                        c.same(
                            record["metadata"],
                            old["metadata"],
                            "nonnoise R historical metadata parity",
                        )
    key = lambda r: (r["fold"], r["arm"], r["condition"])
    c.same(
        sorted(rows, key=key),
        sorted(result["rows"], key=key),
        "all cached fixed/ranking/person/pair metrics and decisions",
    )
    c.same(
        [r for r in rows if r["arm"] == "R" and r["condition"] == "clean"],
        reference["rows"],
        "R clean restored gate rows",
    )
    summary = calc.comparison(rows, metadata)
    c.same(
        summary,
        result["summary"],
        "exact primary, all-arm summaries, secondary and all 33 gates",
    )
    report = c.read(bundle / "evidence/report.json.gz")
    c.same(report["summary"], summary, "report summary")
    c.same(c.read(bundle / "summary.json"), summary, "public summary")
    c.same(
        c.read(bundle / "typed-thresholds.json"),
        typed_thresholds,
        "typed IEEE thresholds",
    )
    ledger = c.read(bundle / "evidence/fit-ledger.json.gz")
    c.same(
        [
            ledger["actual_candidate_fit_entries"],
            ledger["R_fit_entries"],
            len(ledger["calls"]),
        ],
        [200, 0, 200],
        "recorded actual-fit budget",
    )
    c.same(
        ledger["both_cutoffs_before_either_refit"], result["cutoffs"], "cutoff ledger"
    )
    expected_events = []
    calls = iter(ledger["calls"])
    for fold in metadata["folds"]:
        f = fold["fold"]
        for arm in ("H", "F"):
            for role in range(4):
                call = next(calls)
                c.same(
                    [call["fold"], call["arm"], call["role"]],
                    [f, arm, role],
                    "inner fit ordering",
                )
                expected_events.append(
                    {"event": "fit_entered", "fold": f, "arm": arm, "role": role}
                )
        expected_events.append(
            {
                "event": "both_cutoffs_frozen",
                "fold": f,
                "sha256": c.fingerprint(result["cutoffs"][str(f)]),
            }
        )
        for arm in ("H", "F"):
            call = next(calls)
            c.same(
                [call["fold"], call["arm"], call["role"]],
                [f, arm, "refit"],
                "refit ordering",
            )
            expected_events.append(
                {"event": "fit_entered", "fold": f, "arm": arm, "role": "refit"}
            )
    c.same(ledger["events"], expected_events, "both cutoffs before either refit")
    assignments = unique_records(
        metadata["fit_assignments"],
        ("fold", "arm", "role"),
        [
            (f, a, role)
            for f in range(20)
            for a in ("H", "F")
            for role in (*range(4), "refit")
        ],
        "fit assignments",
    )
    for call in ledger["calls"]:
        assignment = assignments[call["fold"], call["arm"], call["role"]]
        c.same(call["status"], "fit_completed", "completed candidate fit")
        for field in ("fold", "arm", "role", "ids", "unit_indices", "training_units"):
            c.same(call[field], assignment[field], "fit provenance " + field)
    return {
        "verified": True,
        "scope": "cached-decision/threshold/metric/gate replay only",
        "fit_calls": 0,
        "archive_reads": 0,
        "raw_feature_reads": 0,
        "model_reads": 0,
        "score_sets": 660,
        "inner_score_sets": 240,
        "outer_score_sets": 420,
        "cutoffs": 60,
        "gates": 33,
        "failed_gates": sorted(
            k for k, v in summary["replacement"]["gates"].items() if not v
        ),
        "nominee": summary["nominee"],
        "primary": summary["primary"],
        "full_verification": "outside_cached_replay_scope",
        "concrete_review": "outside_cached_replay_scope",
        "source_index_sha256": ANCHORS["sourceIndex"],
        "run_index_sha256": ANCHORS["runIndex"],
        "bundle_inventory_sha256": c.sha(bundle / "inventory.json"),
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "rows": rows,
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--out", required=True, help="fresh absolute directory outside the bundle"
    )
    args = parser.parse_args()
    bundle, out = c.absolute(args.bundle), c.absolute(args.out)
    c.require(
        not out.exists(), "Output already exists; failed attempts remain consumed"
    )
    c.require(
        not out.is_relative_to(bundle) and not bundle.is_relative_to(out),
        "Output must be separate from bundle",
    )
    c.require(out.parent.is_dir(), "Output parent must exist")
    c.enforce_environment()
    out.mkdir()
    try:
        result = verify(bundle)
        c.write(out / "rows.json", result.pop("rows"))
        c.write(out / "decisions.json", result.pop("decisions"))
        c.write(out / "verification.json", result)
        c.write(
            out / "inventory.json",
            {
                "schema": 1,
                "excludes_only": "inventory.json",
                "files": c.file_index(out),
            },
        )
        print(c.canonical(result).decode())
    except Exception as error:
        c.write(
            out / "failure.json",
            {
                "status": "FAILED_NO_RETRY_IN_PLACE",
                "error": str(error),
                "type": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
