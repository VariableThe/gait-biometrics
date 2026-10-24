"""Fixed, exploratory TRAIN-only descriptor ablation on HAR and WISDM v1.

No selection, calibration, TEST evaluation, neural training, or GPU work. Run as
``python -m scripts.invariant_transfer_study --data-root ... --out-dir ...``.
Importing this module requires NumPy, but never imports Torch or dataset code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("har", "wisdm_v1")
CANDIDATES = ("axis", "invariant", "fusion")
FOLD_PARAMS = {"calibration_subjects": 4, "n_folds": 3, "seed": 2026}
PAIR_PARAMS = {"n_pairs": 10000, "seed": 42}
WINDOW_SIZE = 128
STRIDE = 64
MIN_WALKING_WINDOWS = 2


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def array_fingerprint(value, dtype):
    """Hash canonical little-endian C-order values plus dtype and shape.

    Normalization here means array serialization, never per-axis scaling of
    sensor signals. Signed floating zeros are canonicalized; NaNs are rejected.
    """
    values = np.array(_numpy(value), dtype=np.dtype(dtype).newbyteorder("<"), order="C")
    if values.dtype.kind == "f":
        if not np.isfinite(values).all():
            raise ValueError("fingerprints require finite float arrays")
        values[values == 0] = 0.0
    metadata = {"dtype": values.dtype.str, "shape": list(values.shape)}
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(values).cast("B"))
    return {**metadata, "sha256": digest.hexdigest()}


def input_fingerprint(x, subjects, activities):
    parts = {
        "X": array_fingerprint(x, "<f4"),
        "subjects": array_fingerprint(subjects, "<i8"),
        "activities": array_fingerprint(activities, "<i8"),
    }
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "arrays": parts}


def code_fingerprint():
    # A conservative superset captures lazy/transitive imports of the existing
    # evaluator and adapter, without importing them or inspecting dataset files.
    paths = {Path(__file__).resolve()}
    for folder in ("data", "engine", "models"):
        paths.update((ROOT / folder).rglob("*.py"))
    package_init = ROOT / "scripts" / "__init__.py"
    if package_init.exists():
        paths.add(package_init)
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "sha256": aggregate,
        "files": hashes,
        "scope": "runner, scripts package initializer, all data/engine/models Python sources",
    }


def _runtime():
    """Keep Torch-bearing adapter/evaluator imports behind explicit execution."""
    import torch
    from threadpoolctl import threadpool_limits

    from data.adapter import load_windows
    from engine.evaluate import verification_metrics
    from engine.subject_protocol import fold_summary, make_subject_folds
    from models.gait_features import extract_gait_features
    from models.gait_invariants import extract_invariant_features
    from models.gait_metric import WithinClassMetric

    torch.set_num_threads(1)
    return {
        "load": load_windows,
        "folds": make_subject_folds,
        "summary": fold_summary,
        "axis": extract_gait_features,
        "invariant": extract_invariant_features,
        "metric": WithinClassMetric,
        "verify": verification_metrics,
        "thread_limits": threadpool_limits,
    }


def _counts(subjects):
    ids, counts = np.unique(subjects, return_counts=True)
    return {str(int(subject)): int(count) for subject, count in zip(ids, counts)}


def _cohort(subjects, ids):
    selected = subjects[np.isin(subjects, ids)]
    return {
        "subject_ids": list(map(int, ids)),
        "n_subjects": len(ids),
        "n_windows": len(selected),
        "windows_per_subject": _counts(selected),
    }


def _partition_comparison(reference, current):
    """Expose membership changes against the unfiltered scaling-style split."""

    def changes(before, after):
        return {
            "removed_subject_ids": sorted(set(before) - set(after)),
            "added_subject_ids": sorted(set(after) - set(before)),
        }

    return {
        "reference": "same helper/settings on all walking TRAIN identities; provenance only",
        "reference_unfiltered_subject_protocol": reference,
        "membership_changed": reference != current,
        "calibration": changes(
            reference["calibration_subject_ids"], current["calibration_subject_ids"]
        ),
        "development": changes(
            reference["development_subject_ids"], current["development_subject_ids"]
        ),
        "folds": [
            {
                "fold": after["fold"],
                "train": changes(
                    before["train_subject_ids"], after["train_subject_ids"]
                ),
                "validation": changes(
                    before["val_subject_ids"], after["val_subject_ids"]
                ),
            }
            for before, after in zip(reference["folds"], current["folds"])
        ],
        "comparison_limit": (
            "Same seed/helper does not guarantee the previous scaling cohorts after eligibility filtering; "
            "the reference is a reconstruction, not an audit of historical run artifacts"
        ),
    }


def _prepare(source, data, runtime):
    if data.y_subject is None:
        raise ValueError("authentic subject labels are required")
    x = _numpy(data.X)
    subjects, activities = _numpy(data.y_subject), _numpy(data.y_activity)
    expected_channels = {"har": 9, "wisdm_v1": 3}[source]
    if x.ndim != 3 or x.shape[1:] != (expected_channels, WINDOW_SIZE):
        raise ValueError("unexpected source channel count or window length")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("source windows must be finite real arrays")
    for values in (subjects, activities):
        if (
            values.shape != (len(x),)
            or values.dtype.kind not in "iu"
            or (values < 0).any()
        ):
            raise ValueError(
                "one nonnegative integer subject/activity ID per window required"
            )
    rate = float(data.sampling_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    full_fingerprint = input_fingerprint(x, subjects, activities)
    walking = activities == 0
    x, subjects, activities = x[walking], subjects[walking], activities[walking]
    observed = sorted(map(int, np.unique(subjects)))
    walking_counts = _counts(subjects)
    retained_ids = [
        subject
        for subject in observed
        if walking_counts[str(subject)] >= MIN_WALKING_WINDOWS
    ]
    excluded_ids = [
        subject
        for subject in observed
        if walking_counts[str(subject)] < MIN_WALKING_WINDOWS
    ]
    eligible = np.isin(subjects, retained_ids)
    # Determine pair eligibility before reserving calibration or building DEV
    # folds. This never modifies source arrays or the global ID mapping.
    protocol = runtime["folds"](subjects[eligible], **FOLD_PARAMS)
    reference_protocol = (
        runtime["folds"](subjects, **FOLD_PARAMS) if excluded_ids else protocol
    )
    development = np.isin(subjects, protocol["development_subject_ids"])
    for cohort in protocol["folds"]:
        # Pair sampling requires distinct genuine windows for every subject.
        for key in ("train_subject_ids", "val_subject_ids"):
            fold_counts = _counts(subjects[np.isin(subjects, cohort[key])])
            if (
                len(fold_counts) != len(cohort[key])
                or min(fold_counts.values()) < MIN_WALKING_WINDOWS
            ):
                raise ValueError(
                    "each development subject needs at least two walking windows"
                )
    original_ids = (
        {str(subject): int(data.subject_ids[subject]) for subject in observed}
        if source == "wisdm_v1"
        else {str(subject): subject + 1 for subject in observed}
    )
    metadata = {
        "source": source,
        "split": "train",
        "walking_activity_id": 0,
        "sampling_rate": rate,
        "input_shape": [len(walking), expected_channels, WINDOW_SIZE],
        "source_train_subject_ids": sorted(map(int, np.unique(_numpy(data.y_subject)))),
        "source_train_windows_per_subject": _counts(_numpy(data.y_subject)),
        "source_train_fingerprint": full_fingerprint,
        "walking_fingerprint": input_fingerprint(x, subjects, activities),
        "eligible_walking_fingerprint": input_fingerprint(
            x[eligible], subjects[eligible], activities[eligible]
        ),
        "development_fingerprint": input_fingerprint(
            x[development], subjects[development], activities[development]
        ),
        "walking_source_row_indices": np.flatnonzero(walking).tolist(),
        "development_source_row_indices": np.flatnonzero(walking)[development].tolist(),
        "encoded_to_original_walking_subject_ids": original_ids,
        "walking": _cohort(subjects, observed),
        "walking_eligibility": {
            "minimum_windows_per_subject": MIN_WALKING_WINDOWS,
            "rule": "retain walking identities with at least two windows before subject folds",
            "retained": _cohort(subjects, retained_ids),
            "excluded": _cohort(subjects, excluded_ids),
            "exclusions": [
                {
                    "subject_id": subject,
                    "original_subject_id": original_ids[str(subject)],
                    "n_windows": walking_counts[str(subject)],
                    "reason": "fewer_than_two_walking_windows",
                }
                for subject in excluded_ids
            ],
            "retained_source_row_indices": np.flatnonzero(walking)[eligible].tolist(),
            "excluded_source_row_indices": np.flatnonzero(walking)[~eligible].tolist(),
        },
        "development": _cohort(subjects, protocol["development_subject_ids"]),
        "reserved_calibration": {
            **_cohort(subjects, protocol["calibration_subject_ids"]),
            "use": "provenance only; excluded before feature extraction and never fitted or scored",
        },
        "subject_protocol": protocol,
        "subject_protocol_comparison": _partition_comparison(
            reference_protocol, protocol
        ),
        "folds": [
            {
                "fold": fold["fold"],
                "train": _cohort(subjects, fold["train_subject_ids"]),
                "validation": _cohort(subjects, fold["val_subject_ids"]),
            }
            for fold in protocol["folds"]
        ],
    }
    return {
        "X": x[development],
        "subjects": subjects[development],
        "activities": activities[development],
        "sampling_rate": rate,
        "metadata": metadata,
    }


def _write_new(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _versions():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _evaluate(prepared, runtime):
    x, subjects, activities = (prepared[key] for key in ("X", "subjects", "activities"))
    rate = prepared["sampling_rate"]
    timings, features = {}, {}
    for name in ("axis", "invariant"):
        started = time.perf_counter()
        features[name] = runtime[name](x, rate)
        timings[name] = time.perf_counter() - started
    started = time.perf_counter()
    features["fusion"] = np.concatenate(
        (features["axis"], features["invariant"]), axis=1
    )
    timings["fusion_concatenation"] = time.perf_counter() - started
    rows = []
    for name in CANDIDATES:
        values = features[name]
        records = []
        for fold in prepared["metadata"]["folds"]:
            training = np.isin(subjects, fold["train"]["subject_ids"])
            held_out = np.isin(subjects, fold["validation"]["subject_ids"])
            # Slicing is excluded from timings. Each candidate gets a fresh fit.
            train_values, val_values = values[training], values[held_out]
            metric = runtime["metric"]("wccn", "auto")
            started = time.perf_counter()
            metric.fit(train_values, subjects[training])
            fit_seconds = time.perf_counter() - started
            if sorted(metric.training_subject_ids_) != fold["train"]["subject_ids"]:
                raise ValueError(
                    "fitted metric identities differ from the registered training fold"
                )
            started = time.perf_counter()
            embeddings = metric.transform(val_values)
            transform_seconds = time.perf_counter() - started
            started = time.perf_counter()
            metrics = runtime["verify"](
                embeddings,
                subjects[held_out],
                activity_ids=activities[held_out],
                **PAIR_PARAMS,
            )
            score_seconds = time.perf_counter() - started
            records.append(
                {
                    **fold,
                    "seed": PAIR_PARAMS["seed"],
                    "metrics": metrics,
                    "metric_parameters": {
                        "mode": "wccn",
                        "shrinkage": "auto",
                        "fitted_shrinkage": float(metric.shrinkage_),
                        "feature_dimension": int(values.shape[1]),
                        "stored_numeric_values": int(
                            values.shape[1] ** 2 + 2 * values.shape[1]
                        ),
                        "parameter_count_semantics": "dense projection plus mean and scale; excludes IDs",
                    },
                    "timing_seconds": {
                        "metric_fit": fit_seconds,
                        "validation_metric_transform": transform_seconds,
                        "pair_sampling_scoring_and_metrics": score_seconds,
                    },
                }
            )
        summary = runtime["summary"](
            [
                {
                    "fold": row["fold"],
                    "seed": row["seed"],
                    "eer": row["metrics"]["eer"],
                    "roc_auc": row["metrics"]["roc_auc"],
                }
                for row in records
            ]
        )
        rows.append(
            {
                "source": prepared["metadata"]["source"],
                "candidate": name,
                "feature_dimension": int(values.shape[1]),
                "folds": records,
                "summary": summary,
                "feature_extraction_seconds_shared_across_folds": (
                    timings[name]
                    if name != "fusion"
                    else timings["axis"]
                    + timings["invariant"]
                    + timings["fusion_concatenation"]
                ),
            }
        )
    return {"feature_timing_seconds": timings, "rows": rows}


def run(data_root: Path, out_dir: Path, source: str = "all"):
    """Write an exclusive pre-fit audit and every fixed candidate/fold result."""
    if source not in (*SOURCES, "all"):
        raise ValueError("source must be har, wisdm_v1, or all")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    code = code_fingerprint()
    runtime = _runtime()
    selected = SOURCES if source == "all" else (source,)
    prepared = []
    for name in selected:
        data = runtime["load"](
            source=name,
            split="train",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            data_root=str(Path(data_root)),
        )
        prepared.append(_prepare(name, data, runtime))
    protocol = {
        "study": "invariant_transfer_exploratory_dev_v1",
        "interpretation": "within-source DEV descriptor ablation; no cross-dataset transfer is measured",
        "prior_test_inspection": True,
        "test_loaded": False,
        "candidate_selection": False,
        "calibration_performed": False,
        "deployment_ready": False,
        "candidates": list(CANDIDATES),
        "metric": {"mode": "wccn", "shrinkage": "auto"},
        "fold_parameters": FOLD_PARAMS,
        "minimum_walking_windows_per_subject_before_folds": MIN_WALKING_WINDOWS,
        "pair_parameters": PAIR_PARAMS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "pair_protocol": "same-activity walking window pairs; existing verification_metrics",
        "comparison_limit": "not comparable to Santos cross-day trial/template claim scores",
        "code": code,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": _versions(),
            "numeric_thread_limit": 1,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "sources": [item["metadata"] for item in prepared],
        "timing_semantics": (
            "Single CPU wall observations, no warmup/repetitions; extraction shared across folds. "
            "Metric inference excludes extraction, loading, hashing and pair operations. "
            "Tiny pair operations and import/cache/order effects make timings descriptive, "
            "not throughput or speedup evidence. Fusion reuses both feature blocks."
        ),
    }
    protocol_path = out_dir / "pre_fit_protocol.json"
    _write_new(
        protocol_path, protocol
    )  # Persist all input/code hashes before any fit/extraction.
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    evaluations = []
    with runtime["thread_limits"](limits=1):
        for item in prepared:
            evaluations.append(_evaluate(item, runtime))
    if code_fingerprint() != code:
        raise ValueError(
            "source code changed during the study; discard this incomplete output"
        )
    result = {
        "study": protocol["study"],
        "exploratory_dev_only": True,
        "pre_fit_protocol_sha256": protocol_sha,
        "rows": [row for evaluation in evaluations for row in evaluation["rows"]],
        "source_feature_timings": {
            item["metadata"]["source"]: evaluation["feature_timing_seconds"]
            for item, evaluation in zip(prepared, evaluations)
        },
        "uncertainty": "equal-fold means and sample SD (ddof=1); dependent folds, no CI or fresh confirmation",
    }
    _write_new(out_dir / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", choices=(*SOURCES, "all"), default="all")
    args = parser.parse_args()
    result = run(args.data_root, args.out_dir, args.source)
    print(json.dumps({"out_dir": str(args.out_dir), "n_rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
