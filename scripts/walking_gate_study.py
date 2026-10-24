"""One registered WISDM TRAIN-only walking-gate fit and cache-only verification.

Invoke as ``python -m scripts.walking_gate_study``. Real-data execution is an
explicit run-stage action; importing this module and verify never load datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from models.walking_gate import (
    DEFAULT_BATCH_SIZE,
    FEATURE_NAMES,
    FEATURE_VERSION,
    HYPERPARAMETERS,
    SourceContract,
    WalkingGate,
    binary_labels,
    canonical,
    extract_features,
    numeric_digest,
    select_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "models/walking_gate.py",
    "scripts/walking_gate_study.py",
    "tests/test_walking_gate.py",
    "docs/walking_gate.md",
    "data/adapter.py",
    "data/window_provenance.py",
)
RAW_SHA256 = "9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e"
ACTIVITY_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
CONTRACT = SourceContract(
    "wisdm_v1", 20.0, ("acc_x", "acc_y", "acc_z"), ("wisdm_native_10_per_g",) * 3, 128
)
STRIDE = 64
SPLIT_SEED = 20261024
ROLES = ("fit", "calibration", "validation")
CACHE_KEYS = {
    "features",
    "activity",
    "labels",
    "subjects",
    "role",
    "start_rows",
    "stop_rows",
    "segment_ids",
    "first_timestamps_ns",
    "last_timestamps_ns",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_bytes(),
        object_pairs_hook=unique,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"invalid JSON {v}")),
    )


def same(actual, expected, name):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"{name} differs from registration/replay")


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def versions():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")
        },
    }


def roles():
    """Mirror adapter's official-ID source split, then split TRAIN independently.

    No features, labels, validation results, or fit outcomes affect these roles.
    Balanced here means subject counts, not post-hoc activity stratification.
    """
    source_order = np.random.default_rng(42).permutation(np.arange(1, 37))
    train, test = np.sort(source_order[:28]), np.sort(source_order[28:])
    order = np.random.default_rng(SPLIT_SEED).permutation(train)
    return {
        "source_train": train.tolist(),
        "source_test_never_windowed": test.tolist(),
        "fit": sorted(order[:16].tolist()),
        "calibration": sorted(order[16:22].tolist()),
        "validation": sorted(order[22:].tolist()),
    }


def plan():
    return {
        "protocol": "wisdm_v1_train_only_walking_gate_v1",
        "contract": CONTRACT.to_dict(),
        "raw_units": "unconverted AR v1.1 values; source raw_about metadata defines10=1g=9.81m/s^2; never relabel native values as exact SI acceleration",
        "stride": STRIDE,
        "source_partition": "adapter sorted real IDs 1..36; default_rng(42); 28 TRAIN / 8 TEST",
        "development_partition": "default_rng(20261024) of sorted TRAIN; 16 fit / 6 calibration / 6 validation",
        "roles": roles(),
        "activity_names": ACTIVITY_NAMES,
        "positive_activity_name": "Walking",
        "negative_activity_names": ACTIVITY_NAMES[1:],
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "hyperparameters": dict(HYPERPARAMETERS),
        "fit_budget": 1,
        "threshold_rule": "maximize calibration window-balanced accuracy; ties fewer other passes, then highest threshold; candidates unique probabilities plus 0 and 1; walking iff p>=t",
        "validation": "held DEV subjects scored once after threshold frozen; no selection or refit",
        "batch_size": DEFAULT_BATCH_SIZE,
        "performance_repeats": 5,
        "performance_scope": "completed validation windows, one warmup; features+standardization+sigmoid; excludes load/fit/I/O/imports; perf_counter and process_time",
        "causality": "output only at completed window; 128 samples at nominal20Hz; no lookahead, smoothing, or online adaptation",
        "forbidden": "no TEST window load, no Santos reads, no GPU, no hyperparameter grid",
        "interpretation": "exploratory held-subject DEV activity detection within WISDM; no cross-device, placement, session, or calibrated-probability guarantee",
    }


def load_training(data_root):
    """Only dataset-reading boundary. The caller must explicitly choose run.

    The existing provenance adapter parses the common raw text, filters source
    TRAIN subjects, then windows only those subjects. TEST windows are never
    requested or materialized. Source-wide raw hashing/parsing is not hidden.
    """
    from data.adapter import WISDM_V1_ACTIVITY_NAMES, WISDM_V1_RAW_SHA256
    from data.window_provenance import load_windows_with_provenance

    same(WISDM_V1_ACTIVITY_NAMES, ACTIVITY_NAMES, "authoritative activity names")
    same(WISDM_V1_RAW_SHA256, RAW_SHA256, "official source constant")
    path = Path(data_root) / "wisdm" / "WISDM_ar_v1.1_raw.txt"
    if not path.is_file():
        path = path.parent / "WISDM_ar_v1.1" / path.name
    source_hash = digest(path)
    same(source_hash, RAW_SHA256, "official raw source fingerprint")
    data = load_windows_with_provenance(
        "wisdm_v1",
        "train",
        window_size=CONTRACT.window_length,
        stride=STRIDE,
        data_root=str(data_root),
        include_timestamps=True,
    )
    same(data.activity_names, ACTIVITY_NAMES, "loaded activity names")
    same(data.sampling_rate, CONTRACT.sample_rate, "native sampling rate")
    same(data.subject_ids, list(range(1, 37)), "global real source identities")
    if data.y_subject is None:
        raise ValueError("real subject labels required")
    subject_index = data.y_subject.numpy()
    if (
        subject_index.dtype.kind not in "iu"
        or np.any(subject_index < 0)
        or np.any(subject_index >= len(data.subject_ids))
    ):
        raise ValueError("invalid source subject indices")
    same(data.provenance.source_sha256, source_hash, "parsed source fingerprint")
    same(digest(path), source_hash, "source fingerprint after parsing")
    result = {
        "X": data.X.numpy(),
        "activity": data.y_activity.numpy(),
        "subjects": np.asarray(data.subject_ids, dtype=np.int64)[subject_index],
        "source": {
            "source": "wisdm_v1",
            "path": str(path.resolve()),
            "sha256": source_hash,
            "bytes": path.stat().st_size,
            "contract": CONTRACT.to_dict(),
            "adapter_split": "train",
            "raw_reader_scope": "common raw source parsed; only source TRAIN windowed",
        },
    }
    for name in (
        "start_rows",
        "stop_rows",
        "segment_ids",
        "first_timestamps_ns",
        "last_timestamps_ns",
    ):
        result[name] = getattr(data.provenance, name)
    return result


def array_manifest(cache):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": numeric_digest(value),
        }
        for key, value in sorted(cache.items())
    }


def _validate_source(source):
    if not isinstance(source, dict):
        raise TypeError("registered source manifest required")
    same(source["source"], "wisdm_v1", "source name")
    same(source["sha256"], RAW_SHA256, "official raw fingerprint")
    same(source["contract"], CONTRACT.to_dict(), "source contract")
    same(source["adapter_split"], "train", "TRAIN-only loader guard")


def validate_cache(cache):
    if set(cache) != CACHE_KEYS:
        raise ValueError("complete numeric feature/provenance cache required")
    n = len(cache["features"])
    if (
        cache["features"].shape != (n, len(FEATURE_NAMES))
        or cache["features"].dtype != np.dtype("float64")
        or not np.isfinite(cache["features"]).all()
        or not n
    ):
        raise ValueError("finite nonempty float64 fixed feature cache required")
    for key in CACHE_KEYS - {"features"}:
        values = cache[key]
        if values.shape != (n,) or values.dtype != np.dtype("int64"):
            raise ValueError(f"aligned int64 cache array required: {key}")
    same(
        cache["labels"].tolist(),
        binary_labels(cache["activity"], ACTIVITY_NAMES).tolist(),
        "authoritative binary labels",
    )
    allocation = roles()
    if not set(cache["subjects"].tolist()) <= set(allocation["source_train"]):
        raise ValueError("leakage guard: only source TRAIN subjects allowed")
    for code, name in enumerate(ROLES):
        mask = np.isin(cache["subjects"], allocation[name])
        if not np.array_equal(cache["role"] == code, mask):
            raise ValueError("leakage guard: subject roles differ from fixed split")
        if set(cache["labels"][mask].tolist()) != {0, 1}:
            raise ValueError(
                f"fixed {name} role must contain both classes; no resplitting"
            )
    if not np.isin(cache["role"], (0, 1, 2)).all():
        raise ValueError("unknown role code")
    if (
        np.any(cache["start_rows"] < 0)
        or np.any(cache["segment_ids"] < 0)
        or np.any(cache["segment_ids"] > cache["start_rows"])
        or np.any(cache["stop_rows"] - cache["start_rows"] != CONTRACT.window_length)
        or np.any(cache["first_timestamps_ns"] <= 0)
        or np.any(cache["last_timestamps_ns"] < cache["first_timestamps_ns"])
    ):
        raise ValueError("invalid raw window support provenance")
    # In source order the only overlapping raw supports must be same-person,
    # same-role, same-activity windows from the same contiguous segment.
    order = np.argsort(cache["start_rows"], kind="stable")
    start, stop = cache["start_rows"][order], cache["stop_rows"][order]
    if len(start) > 1:
        if np.any(np.diff(start) <= 0):
            raise ValueError("duplicate raw window support")
        overlap = start[1:] < stop[:-1]
        for key in ("subjects", "role", "activity", "segment_ids"):
            if np.any((cache[key][order][1:] != cache[key][order][:-1]) & overlap):
                raise ValueError(
                    "leakage guard: overlapping support crosses a boundary"
                )


def cache_from_training(data):
    _validate_source(data["source"])
    cache = {
        name: np.asarray(data[name], dtype=np.int64)
        for name in (
            "activity",
            "subjects",
            "start_rows",
            "stop_rows",
            "segment_ids",
            "first_timestamps_ns",
            "last_timestamps_ns",
        )
    }
    cache["features"] = extract_features(data["X"], contract=CONTRACT)
    cache["labels"] = binary_labels(cache["activity"], ACTIVITY_NAMES)
    cache["role"] = np.full(len(cache["subjects"]), -1, dtype=np.int64)
    for code, name in enumerate(ROLES):
        cache["role"][np.isin(cache["subjects"], roles()[name])] = code
    validate_cache(cache)
    return cache


def inventory(cache):
    result = {}
    for code, name in enumerate(ROLES):
        mask = cache["role"] == code
        people = {}
        for subject in roles()[name]:
            person = mask & (cache["subjects"] == subject)
            people[str(subject)] = {
                "windows": int(person.sum()),
                "walking": int(cache["labels"][person].sum()),
                "other": int((1 - cache["labels"][person]).sum()),
            }
        result[name] = {
            "windows": int(mask.sum()),
            "walking": int(cache["labels"][mask].sum()),
            "other": int((1 - cache["labels"][mask]).sum()),
            "per_person": people,
            "activity_counts": {
                activity: int(np.sum(mask & (cache["activity"] == index)))
                for index, activity in enumerate(ACTIVITY_NAMES)
            },
        }
    return result


def confusion(labels, probabilities, threshold):
    labels, predicted = np.asarray(labels) == 1, np.asarray(probabilities) >= threshold
    tp, fn = int(np.sum(labels & predicted)), int(np.sum(labels & ~predicted))
    fp, tn = int(np.sum(~labels & predicted)), int(np.sum(~labels & ~predicted))
    npos, nneg = tp + fn, fp + tn
    return {
        "windows": npos + nneg,
        "walking": npos,
        "other": nneg,
        "true_walking": tp,
        "missed_walking": fn,
        "other_passed": fp,
        "other_rejected": tn,
        "walking_recall": tp / npos if npos else None,
        "walking_miss_rate": fn / npos if npos else None,
        "other_pass_rate": fp / nneg if nneg else None,
        "other_rejection_rate": tn / nneg if nneg else None,
        "balanced_accuracy": (tp / npos + tn / nneg) / 2 if npos and nneg else None,
    }


def report(cache, code, probabilities, threshold):
    mask = cache["role"] == code
    labels, subjects, activity = (
        cache[key][mask] for key in ("labels", "subjects", "activity")
    )
    return {
        "pooled": confusion(labels, probabilities, threshold),
        "per_person": {
            str(s): confusion(
                labels[subjects == s], probabilities[subjects == s], threshold
            )
            for s in roles()[ROLES[code]]
        },
        "per_activity": {
            name: confusion(
                labels[activity == i], probabilities[activity == i], threshold
            )
            for i, name in enumerate(ACTIVITY_NAMES)
        },
    }


def save_cache(path, values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **values)
    return {"path": Path(path).name, "sha256": digest(path)}


def checked_path(root, artifact, filename):
    same(sorted(artifact), ["path", "sha256"], "artifact descriptor keys")
    same(artifact["path"], filename, "artifact filename")
    path = root / filename
    if path.is_symlink():
        raise ValueError("artifact symlinks refused")
    same(digest(path), artifact["sha256"], f"{filename} integrity")
    return path


def read_cache(path):
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("duplicate numeric cache arrays")
        return {key: saved[key] for key in saved.files}


def performance(gate, X):
    gate.predict_proba(X, contract=CONTRACT)  # fixed warmup, no state updates
    wall, cpu = [], []
    for _ in range(plan()["performance_repeats"]):
        before_wall, before_cpu = time.perf_counter(), time.process_time()
        gate.predict_proba(X, contract=CONTRACT, batch_size=DEFAULT_BATCH_SIZE)
        cpu.append(time.process_time() - before_cpu)
        wall.append(time.perf_counter() - before_wall)
    seconds = float(np.median(wall))
    return {
        "scope": plan()["performance_scope"],
        "windows_per_repeat": len(X),
        "repeats": len(wall),
        "batch_size": DEFAULT_BATCH_SIZE,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "median_wall_seconds_per_window": seconds / len(X),
        "median_windows_per_second": len(X) / seconds,
        "nominal_window_seconds": CONTRACT.window_length / CONTRACT.sample_rate,
        "sample_span_seconds": (CONTRACT.window_length - 1) / CONTRACT.sample_rate,
        "lookahead_samples": 0,
        "threads": 1,
    }


def run(args):
    from threadpoolctl import threadpool_limits

    if getattr(args, "threads", 1) != 1:
        raise ValueError("this registered study requires threads=1")
    root = Path(args.out_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("fresh out-root required; no overwrite or resume") from exc
    with threadpool_limits(limits=1):
        data = load_training(args.data_root)
        cache = cache_from_training(data)
        feature_artifact = save_cache(root / "features.npz", cache)
        registration = {
            "plan": plan(),
            "source_files": source_hashes(),
            "versions": versions(),
            "data": data["source"],
            "input_windows_sha256": numeric_digest(np.asarray(data["X"])),
            "inventory": inventory(cache),
            "arrays": array_manifest(cache),
            "features": feature_artifact,
            "provenance_coordinate": "zero-based valid raw record ordinals; [start_rows,stop_rows); first/last included timestamp_ns",
        }
        # All contracts, data/cache hashes, subject roles, versions and settings
        # are registered before the single optimizer fit begins.
        write_new(root / "registration.json", registration)
        fit = cache["role"] == 0
        gate = WalkingGate.fit_features(
            cache["features"][fit],
            cache["labels"][fit],
            subjects=cache["subjects"][fit],
            fit_subject_ids=roles()["fit"],
            contract=CONTRACT,
            forbidden_subject_ids=sorted(
                set(roles()["source_train"]) - set(roles()["fit"])
                | set(roles()["source_test_never_windowed"])
            ),
        )
        calibration = gate.predict_feature_proba(
            cache["features"][cache["role"] == 1], contract=CONTRACT
        )
        threshold = select_threshold(cache["labels"][cache["role"] == 1], calibration)
        gate = gate.with_threshold(threshold)
        # First held-validation scoring happens only after threshold is frozen.
        validation = gate.predict_feature_proba(
            cache["features"][cache["role"] == 2], contract=CONTRACT
        )
        gate.save(root / "model.json")
        scores = save_cache(
            root / "scores.npz", {"calibration": calibration, "validation": validation}
        )
        validation_X = np.asarray(data["X"])[cache["role"] == 2]
        np.testing.assert_array_equal(
            gate.predict_proba(validation_X, contract=CONTRACT),
            validation,
        )
        result = {
            "registration_sha256": digest(root / "registration.json"),
            "fit_calls": 1,
            "model": {"path": "model.json", "sha256": digest(root / "model.json")},
            "scores": scores,
            "threshold": threshold,
            "calibration": report(cache, 1, calibration, threshold),
            "validation": report(cache, 2, validation, threshold),
            "performance": performance(gate, validation_X),
        }
        write_new(root / "results.json", result)
    return result


def verify(args):
    """Replay registration and numeric caches only; never load data or fit."""
    from threadpoolctl import threadpool_limits

    root = Path(args.out_root)
    registration, result = (
        read_json(root / "registration.json"),
        read_json(root / "results.json"),
    )
    same(registration["plan"], plan(), "fixed protocol")
    same(registration["source_files"], source_hashes(), "source-file fingerprints")
    same(registration["versions"], versions(), "runtime versions")
    same(
        result["registration_sha256"],
        digest(root / "registration.json"),
        "registration integrity",
    )
    same(result["fit_calls"], 1, "fit budget")
    _validate_source(registration["data"])
    cache = read_cache(checked_path(root, registration["features"], "features.npz"))
    validate_cache(cache)
    same(array_manifest(cache), registration["arrays"], "registered numeric arrays")
    same(inventory(cache), registration["inventory"], "role/class/person inventory")
    gate = WalkingGate.load(checked_path(root, result["model"], "model.json"))
    same(gate.contract.to_dict(), CONTRACT.to_dict(), "model source contract")
    fit = cache["role"] == 0
    expected_provenance = {
        "fit_subject_ids": roles()["fit"],
        "forbidden_subject_ids": sorted(
            set(roles()["source_train"]) - set(roles()["fit"])
            | set(roles()["source_test_never_windowed"])
        ),
        "n_fit_windows": int(fit.sum()),
        "features_sha256": numeric_digest(cache["features"][fit]),
        "labels_sha256": numeric_digest(cache["labels"][fit]),
        "fit_calls": 1,
    }
    same(gate.fit_provenance, expected_provenance, "fit-only provenance")
    mean, std = cache["features"][fit].mean(axis=0), cache["features"][fit].std(axis=0)
    np.testing.assert_array_equal(gate.mean, mean)
    np.testing.assert_array_equal(gate.scale, np.where(std > 1e-12, std, 1.0))
    scores = read_cache(checked_path(root, result["scores"], "scores.npz"))
    same(sorted(scores), ["calibration", "validation"], "score cache keys")
    with threadpool_limits(limits=1):
        for code, name in ((1, "calibration"), (2, "validation")):
            fresh = gate.predict_feature_proba(
                cache["features"][cache["role"] == code], contract=CONTRACT
            )
            if scores[name].dtype != np.dtype("float64") or not np.array_equal(
                fresh, scores[name]
            ):
                raise ValueError(f"{name} numeric score cache differs")
        threshold = select_threshold(
            cache["labels"][cache["role"] == 1], scores["calibration"]
        )
        same(threshold, gate.threshold, "calibration-only threshold")
        same(threshold, result["threshold"], "reported frozen threshold")
        for code, name in ((1, "calibration"), (2, "validation")):
            same(
                report(cache, code, scores[name], threshold),
                result[name],
                f"{name} confusion report",
            )
    # Timings cannot be replayed from a feature cache; check their declared
    # scope/counts and internal arithmetic without claiming to remeasure them.
    perf = result["performance"]
    same(perf["scope"], plan()["performance_scope"], "timing scope")
    same(
        perf["windows_per_repeat"], int(np.sum(cache["role"] == 2)), "timing population"
    )
    for name, expected in (
        ("repeats", 5),
        ("batch_size", DEFAULT_BATCH_SIZE),
        ("lookahead_samples", 0),
        ("threads", 1),
        ("nominal_window_seconds", 6.4),
        ("sample_span_seconds", 6.35),
    ):
        same(perf[name], expected, f"timing {name}")
    for key in ("wall_seconds", "cpu_seconds"):
        values = np.asarray(perf[key])
        if (
            values.shape != (5,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError("invalid recorded performance durations")
    median = float(np.median(perf["wall_seconds"]))
    same(
        perf["median_wall_seconds_per_window"],
        median / perf["windows_per_repeat"],
        "timing per-window arithmetic",
    )
    same(
        perf["median_windows_per_second"],
        perf["windows_per_repeat"] / median,
        "timing throughput arithmetic",
    )
    verified = {
        "verified": True,
        "new_fits": 0,
        "registered_fits": 1,
        "registration_sha256": digest(root / "registration.json"),
        "results_sha256": digest(root / "results.json"),
        "threshold": threshold,
        "validation": result["validation"],
        "scope": "numeric cache replay; no dataset reads, refit, or timing remeasurement",
    }
    write_new(root / "verification.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, help="required only for run; never read by verify"
    )
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    if args.stage == "run" and args.data_root is None:
        parser.error("--data-root is required for run")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    print(canonical(run(args) if args.stage == "run" else verify(args)).decode())


if __name__ == "__main__":
    main()
