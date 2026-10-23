"""Opt-in Santos descriptor study: development CV -> calibration lock -> test.

No data is loaded at import. The only test-value loader is in evaluate_stage,
after all frozen inputs have been verified and an exclusive attempt is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import roc_auc_score

from models.gait_features import extract_gait_features
from models.gait_metric import WithinClassMetric

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_IDS = (7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17)
CALIBRATION_IDS = (21, 18, 23, 12, 30)
TEST_IDS = (26, 28, 22, 15, 6)
ALL_IDS = tuple(sorted((*DEV_IDS, *CALIBRATION_IDS, *TEST_IDS)))
CANDIDATES = (
    {"name": "standard", "mode": "standard", "shrinkage": "auto"},
    {"name": "wccn_auto", "mode": "wccn", "shrinkage": "auto"},
    {"name": "wccn_0.2", "mode": "wccn", "shrinkage": 0.2},
)
ARCHIVE_SIZE = 1_523_321_221
ARCHIVE_MD5 = "35adf0860202f2aefa37560dc0748ebd"
ARCHIVE_SHA256 = "ed944fa88a98ce1058327ff4b3c8f919fbf58626fc7e3c5256360ee080a3a304"
_MEMBER = re.compile(
    r"(?:^|/)processed_data/user(?P<folder>\d{2})/"
    r"capture_user(?P<subject>\d{2})_day(?P<day>[12])_"
    r"(?P<trial>\d{4})_imu_walk\.csv$"
)
_ARRAY_KEYS = ("X", "subjects", "days", "trials", "starts")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, value: Any) -> None:
    """Exclusive creation, including for locks and failed test attempts."""
    with path.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def development_folds() -> list[dict]:
    return [
        {
            "fold": fold,
            "train_subject_ids": [
                subject for subject in DEV_IDS if subject not in held
            ],
            "val_subject_ids": list(held),
        }
        for fold, held in enumerate(
            DEV_IDS[offset : offset + 3] for offset in range(0, 15, 3)
        )
    ]


def protocol_config() -> dict:
    return {
        "protocol": "santos_v7_cross_day_descriptor_v1",
        "dataset_doi": "10.6084/m9.figshare.14727231.v7",
        "license": "CC BY 4.0",
        "development_subject_ids": list(DEV_IDS),
        "calibration_subject_ids": list(CALIBRATION_IDS),
        "test_subject_ids": list(TEST_IDS),
        "folds": development_folds(),
        "candidates": list(CANDIDATES),
        "selection": "minimum equal-fold mean EER; ties use fixed candidate order",
        "sampling_rate_hz": 100,
        "channels": ["acc x", "acc y", "acc z"],
        "units": "m/s^2",
        "processing": "processed_data MCU walking only; no resampling",
        "official_csv_schema": [
            "frame",
            "acc x",
            "acc y",
            "acc z",
            "gyro x",
            "gyro y",
            "gyro z",
        ],
        "channel_selection": "select only the three declared acceleration columns; no gyro features",
        "placement_side": "unknown; pooled leg placements",
        "window_size": 256,
        "stride": 256,
        "tails": "discard; never cross trial/day/subject boundaries",
        "descriptor": "models.gait_features.extract_gait_features; 104 values at 100 Hz",
        "fit_population": "fold training identities both days; refit on all 15 DEV only",
        "aggregation": "unit windows -> unit trial means -> unit equal-trial day1 template",
        "probe_unit": "each day2 unit trial mean",
        "eligibility": "at least one nonzero eligible trial on each day; >=2 eligible identities",
        "zero_norm": "window fails; trial/template cancellation excluded and reported; epsilon=1e-12",
        "score": "clip(cosine,-1,1) - 1; larger means more similar",
        "claims": "exhaustive directed claims within eligible held-out population",
        "decision_rule": "score >= threshold; tied scores accepted together",
        "calibration_target_far": 0.01,
        "calibration_selection": "engine.evaluate.calibrate_threshold; least restrictive attainable",
        "uncertainty": "descriptive only; repeated subjects/templates/trials are dependent; no CI",
    }


def source_hashes() -> dict[str, str]:
    paths = (
        "scripts/cross_day_verification.py",
        "data/santos.py",
        "models/gait_features.py",
        "models/gait_metric.py",
        "engine/evaluate.py",
        "engine/subject_protocol.py",
        "docs/cross_session_acquisition.md",
        "docs/cross_day_verification.md",
    )
    return {name: digest(PROJECT_ROOT / name) for name in paths}


def audit_archive(path: Path) -> dict:
    """Hash compressed archive bytes and inspect names only, including TEST names."""
    path = Path(path)
    if path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError("official Santos v7 archive byte count differs")
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != ARCHIVE_MD5:
        raise ValueError("official Santos v7 archive MD5 differs")
    if sha256.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("official Santos v7 archive SHA-256 differs")
    members, seen = [], set()
    with ZipFile(path) as archive:
        for info in archive.infolist():
            match = _MEMBER.search(info.filename)
            if match is None:
                continue
            subject, day, trial = (
                int(match[key]) for key in ("subject", "day", "trial")
            )
            if int(match["folder"]) != subject or not 1 <= trial <= 10:
                raise ValueError(
                    "processed MCU filename has inconsistent subject/trial"
                )
            key = (subject, day, trial)
            if key in seen or info.flag_bits & 1:
                raise ValueError("duplicate or encrypted processed MCU recording")
            seen.add(key)
            members.append(
                {
                    "member": info.filename,
                    "subject": subject,
                    "day": day,
                    "trial": trial,
                    "size": info.file_size,
                    "crc32": info.CRC,
                    "compressed_size": info.compress_size,
                }
            )
    if {row["subject"] for row in members} != set(ALL_IDS):
        raise ValueError(
            "archive processed MCU inventory differs from authentic 25 IDs"
        )
    return {
        "size": path.stat().st_size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "members": sorted(members, key=lambda row: row["member"]),
        "missing_recordings": [
            {"subject": subject, "day": day, "trial": trial}
            for subject in ALL_IDS
            for day in (1, 2)
            for trial in range(1, 11)
            if (subject, day, trial) not in seen
        ],
    }


def selected_member_hashes(path: Path, audit: dict, subject_ids) -> list[dict]:
    """Open only the explicitly permitted cohort's members, never TEST before lock."""
    allowed = set(subject_ids)
    result = []
    with ZipFile(path) as archive:
        for row in audit["members"]:
            if row["subject"] in allowed:
                with archive.open(row["member"]) as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                result.append({**row, "sha256": checksum})
    return result


def load_windows(archive_path: Path, subject_ids) -> dict:
    from data.santos import load_santos_windows

    data = load_santos_windows(archive_path, subject_ids, window_size=256, stride=256)
    validate_windows(data, subject_ids)
    return data


def validate_windows(data: dict, requested_ids) -> None:
    missing = {*_ARRAY_KEYS, "metadata"} - set(data)
    if missing:
        raise ValueError(f"adapter fields missing: {sorted(missing)}")
    x = np.asarray(data["X"])
    if x.ndim != 3 or x.shape[1:] != (3, 256) or x.dtype.kind not in "fiu":
        raise ValueError("expected acceleration windows X[N,3,256]")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite acceleration windows")
    for key in _ARRAY_KEYS[1:]:
        values = np.asarray(data[key])
        if values.shape != (len(x),) or values.dtype.kind not in "iu":
            raise ValueError(f"{key} must be one integer per window")
    if not set(map(int, data["subjects"])).issubset(set(requested_ids)):
        raise ValueError("adapter returned identities outside the requested cohort")
    if (
        not np.isin(data["days"], (1, 2)).all()
        or not np.isin(data["trials"], range(1, 11)).all()
    ):
        raise ValueError("invalid day/trial metadata")
    if not isinstance(data["metadata"], dict):
        raise ValueError("metadata must be a mapping")  # noqa: TRY004 -- adapter validation contract
    keys = np.column_stack(
        [np.asarray(data[key]) for key in ("subjects", "days", "trials", "starts")]
    )
    if len(np.unique(keys, axis=0)) != len(x):
        raise ValueError("duplicate subject/day/trial/frame windows")
    _canonical(data["metadata"])


def data_fingerprint(data: dict) -> str:
    """Include all arrays, recording indices, headers, exclusions and provenance."""
    checksum = hashlib.sha256()
    for key in sorted(data):
        value = data[key]
        checksum.update(_canonical(key))
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays cannot be fingerprinted")
            checksum.update(
                _canonical({"dtype": value.dtype.str, "shape": value.shape})
            )
            checksum.update(np.ascontiguousarray(value).tobytes())
        else:
            checksum.update(_canonical(value))
    return checksum.hexdigest()


def _unit(values: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(values)
    return None if norm < 1e-12 else values / norm


def cross_day_scores(embeddings, data: dict, requested_ids) -> dict:
    """Day1 enrollment and day2 probes with equal trial, rather than window, weight."""
    validate_windows(data, requested_ids)
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or len(z) != len(data["X"]) or not np.isfinite(z).all():
        raise ValueError("finite embeddings must have one row per window")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("window descriptor has no cosine direction")
    z = z / norms
    subjects, days, trials = (
        np.asarray(data[key]) for key in ("subjects", "days", "trials")
    )
    trial_vectors, excluded, per_subject = {}, [], []
    for subject in requested_ids:
        row = {"subject": int(subject), "days": {}}
        for day in (1, 2):
            selected_trials, missing_trials = [], []
            for trial in range(1, 11):
                selected = (subjects == subject) & (days == day) & (trials == trial)
                count = int(selected.sum())
                if not count:
                    missing_trials.append(trial)
                    continue
                vector = _unit(z[selected].mean(axis=0))
                if vector is None:
                    excluded.append(
                        {
                            "subject": int(subject),
                            "day": day,
                            "trial": trial,
                            "reason": "zero_norm_trial_mean",
                            "windows": count,
                        }
                    )
                else:
                    trial_vectors[(subject, day, trial)] = vector
                    selected_trials.append(trial)
            row["days"][str(day)] = {
                "n_windows": int(((subjects == subject) & (days == day)).sum()),
                "eligible_trials": selected_trials,
                "missing_trials": missing_trials,
            }
        per_subject.append(row)
    templates, eligible = {}, []
    for subject in requested_ids:
        enrollment = [
            v for (s, d, _), v in trial_vectors.items() if s == subject and d == 1
        ]
        probes = [
            v for (s, d, _), v in trial_vectors.items() if s == subject and d == 2
        ]
        template = _unit(np.mean(enrollment, axis=0)) if enrollment else None
        if template is None or not probes:
            excluded.append(
                {"subject": int(subject), "reason": "missing_day_or_zero_norm_template"}
            )
        else:
            templates[subject] = template
            eligible.append(subject)
    if len(eligible) < 2:
        raise ValueError(
            "cross-day verification requires at least two eligible identities"
        )
    labels, scores, claims, probes, probe_trials = [], [], [], [], []
    for subject in eligible:
        for (s, day, trial), vector in trial_vectors.items():
            if s != subject or day != 2:
                continue
            for claim in eligible:
                labels.append(int(claim == subject))
                scores.append(float(np.clip(vector @ templates[claim], -1, 1) - 1))
                claims.append(claim)
                probes.append(subject)
                probe_trials.append(trial)
    n_genuine = sum(labels)
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "claim_subjects": np.asarray(claims, dtype=np.int64),
        "probe_subjects": np.asarray(probes, dtype=np.int64),
        "probe_trials": np.asarray(probe_trials, dtype=np.int64),
        "coverage": {
            "requested_subject_ids": list(requested_ids),
            "eligible_subject_ids": eligible,
            "n_enrollment_identities": len(eligible),
            "n_probe_identities": len(eligible),
            "n_enrollment_trials": sum(
                s in eligible and d == 1 for s, d, _ in trial_vectors
            ),
            "n_probe_trials": n_genuine,
            "n_windows": len(z),
            "n_genuine": n_genuine,
            "n_impostor": len(labels) - n_genuine,
            "far_resolution": 1 / (len(labels) - n_genuine),
            "per_subject": per_subject,
            "exclusions": excluded,
            "adapter_metadata": data["metadata"],
        },
    }


def evaluation_functions():
    # This existing module imports Torch; import it only when a stage computes
    # metrics, never for importing this descriptor script or inspecting its CLI.
    from engine.evaluate import (
        calibrate_threshold,
        compute_eer,
        fixed_threshold_metrics,
    )

    return compute_eer, calibrate_threshold, fixed_threshold_metrics


def _report(scored: dict, threshold: float | None = None) -> dict:
    compute_eer, _, fixed_threshold_metrics = evaluation_functions()
    labels, scores = scored["labels"], scored["scores"]
    result = {
        "eer": float(compute_eer(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "coverage": scored["coverage"],
        "uncertainty": protocol_config()["uncertainty"],
    }
    if threshold is not None:
        from engine.subject_protocol import per_subject_error_rates

        result["fixed_threshold"] = fixed_threshold_metrics(labels, scores, threshold)
        result["per_subject_errors"] = per_subject_error_rates(
            labels,
            scores,
            scored["claim_subjects"],
            scored["probe_subjects"],
            threshold,
        )
    return result


def select_candidate(rows: list[dict]) -> dict:
    order = {candidate["name"]: index for index, candidate in enumerate(CANDIDATES)}
    if len(rows) != len(order) or {row["name"] for row in rows} != set(order):
        raise ValueError(
            "candidate registry must contain exactly the three fixed methods"
        )
    if not all(np.isfinite(row["mean_eer"]) for row in rows):
        raise ValueError("nonfinite candidate EER")
    return min(rows, key=lambda row: (row["mean_eer"], order[row["name"]]))


def _subset(data: dict, subject_ids) -> dict:
    selected = np.isin(data["subjects"], subject_ids)
    return {
        key: value[selected]
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(selected)
        else value
        for key, value in data.items()
    }


def _fitted_metric(path: Path, expected_ids) -> WithinClassMetric:
    with np.load(path, allow_pickle=False) as saved:
        metric = WithinClassMetric.from_state_dict(dict(saved))
    if set(metric.training_subject_ids_) != set(expected_ids):
        raise ValueError("saved metric was fitted on the wrong identities")
    return metric


def _same(actual, expected, what: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{what} changed; use a fresh output root and declare prior inspection"
        )


def _root(args) -> Path:
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def train_stage(args) -> dict:
    root = _root(args)
    if any(
        (root / name).exists()
        for name in (
            "registry.json",
            "metric.npz",
            "protocol_lock.json",
            "test_attempt.json",
            "test_result.json",
        )
    ):
        raise ValueError("study artifacts already exist; use a new output root")
    audit, sources, config = (
        audit_archive(args.archive_path),
        source_hashes(),
        protocol_config(),
    )
    data = load_windows(args.archive_path, DEV_IDS)
    validate_windows(data, DEV_IDS)
    features = extract_gait_features(data["X"], 100)
    rows = []
    for candidate in CANDIDATES:
        fold_results = []
        for fold in development_folds():
            training = np.isin(data["subjects"], fold["train_subject_ids"])
            if set(map(int, data["subjects"][training])) != set(
                fold["train_subject_ids"]
            ):
                raise ValueError("a fold training identity has no development windows")
            metric = WithinClassMetric(candidate["mode"], candidate["shrinkage"]).fit(
                features[training], data["subjects"][training]
            )
            validation = _subset(data, fold["val_subject_ids"])
            validation_features = features[
                np.isin(data["subjects"], fold["val_subject_ids"])
            ]
            scored = cross_day_scores(
                metric.transform(validation_features),
                validation,
                fold["val_subject_ids"],
            )
            fold_results.append({**fold, **_report(scored)})
        rows.append(
            {
                **candidate,
                "folds": fold_results,
                "mean_eer": float(np.mean([row["eer"] for row in fold_results])),
            }
        )
    winner = select_candidate(rows)
    if set(map(int, data["subjects"])) != set(DEV_IDS):
        raise ValueError("final fit requires all fifteen development identities")
    metric = WithinClassMetric(winner["mode"], winner["shrinkage"]).fit(
        features, data["subjects"]
    )
    with (root / "metric.npz").open("xb") as stream:
        np.savez(stream, **metric.state_dict())
    registry = {
        "config": config,
        "source_hashes": sources,
        "archive": audit,
        "development_data_sha256": data_fingerprint(data),
        "development_member_hashes": selected_member_hashes(
            args.archive_path, audit, DEV_IDS
        ),
        "candidates": rows,
        "winner": winner["name"],
        "metric_sha256": digest(root / "metric.npz"),
        "prior_test_inspection": getattr(args, "prior_test_inspection", None),
        "test_values_loaded": False,
    }
    _write_new(root / "registry.json", registry)
    return registry


def _verify_registry(args, root: Path) -> tuple[dict, dict]:
    registry = _read(root / "registry.json")
    _same(protocol_config(), registry["config"], "protocol config")
    _same(source_hashes(), registry["source_hashes"], "source code/protocol")
    audit = audit_archive(args.archive_path)
    _same(audit, registry["archive"], "archive integrity or member inventory")
    _same(digest(root / "metric.npz"), registry["metric_sha256"], "fitted metric")
    _same(
        select_candidate(registry["candidates"])["name"],
        registry["winner"],
        "candidate selection",
    )
    return registry, audit


def lock_stage(args) -> dict:
    root = _root(args)
    if any(
        (root / name).exists()
        for name in ("protocol_lock.json", "test_attempt.json", "test_result.json")
    ):
        raise ValueError("study is already locked or test was attempted")
    registry, audit = _verify_registry(args, root)
    target_far = float(getattr(args, "target_far", 0.01))
    _same(
        target_far,
        registry["config"]["calibration_target_far"],
        "prespecified calibration FAR",
    )
    development = load_windows(args.archive_path, DEV_IDS)
    validate_windows(development, DEV_IDS)
    _same(
        data_fingerprint(development),
        registry["development_data_sha256"],
        "development data/metadata",
    )
    _same(
        selected_member_hashes(args.archive_path, audit, DEV_IDS),
        registry["development_member_hashes"],
        "development members",
    )
    metric = _fitted_metric(root / "metric.npz", DEV_IDS)
    calibration = load_windows(args.archive_path, CALIBRATION_IDS)
    validate_windows(calibration, CALIBRATION_IDS)
    scored = cross_day_scores(
        metric.transform(extract_gait_features(calibration["X"], 100)),
        calibration,
        CALIBRATION_IDS,
    )
    _, calibrate_threshold, _ = evaluation_functions()
    calibrated = calibrate_threshold(
        scored["labels"], scored["scores"], target_far=target_far
    )
    lock = {
        "config": registry["config"],
        "source_hashes": registry["source_hashes"],
        "archive": audit,
        "registry_sha256": digest(root / "registry.json"),
        "metric_sha256": registry["metric_sha256"],
        "winner": registry["winner"],
        "development_data_sha256": registry["development_data_sha256"],
        "calibration_data_sha256": data_fingerprint(calibration),
        "development_member_hashes": registry["development_member_hashes"],
        "calibration_member_hashes": selected_member_hashes(
            args.archive_path, audit, CALIBRATION_IDS
        ),
        "calibration_subject_ids": list(CALIBRATION_IDS),
        "test_subject_ids": list(TEST_IDS),
        "threshold": float(calibrated["threshold"]),
        "calibration": calibrated,
        "calibration_report": _report(scored, calibrated["threshold"]),
        "prior_test_inspection": registry["prior_test_inspection"],
        "test_values_loaded": False,
    }
    lock["lock_sha256"] = _object_hash(lock)
    _write_new(root / "protocol_lock.json", lock)
    return lock


def evaluate_stage(args) -> dict:
    root = _root(args)
    if not (root / "protocol_lock.json").exists():
        raise ValueError("lock the protocol before loading any test values")
    if (root / "test_attempt.json").exists() or (root / "test_result.json").exists():
        raise ValueError(
            "test evaluation already attempted; use a new root declaring prior inspection"
        )
    lock = _read(root / "protocol_lock.json")
    checksum = lock.pop("lock_sha256")
    _same(_object_hash(lock), checksum, "protocol lock")
    lock["lock_sha256"] = checksum
    _same(
        digest(root / "registry.json"), lock["registry_sha256"], "development registry"
    )
    registry, audit = _verify_registry(args, root)
    for key in (
        "config",
        "source_hashes",
        "metric_sha256",
        "winner",
        "development_data_sha256",
        "development_member_hashes",
    ):
        _same(registry[key], lock[key], f"locked {key}")
    _same(audit, lock["archive"], "locked archive")
    _same(
        lock["calibration_subject_ids"], list(CALIBRATION_IDS), "calibration identities"
    )
    _same(lock["test_subject_ids"], list(TEST_IDS), "test identities")
    for ids, prefix in ((DEV_IDS, "development"), (CALIBRATION_IDS, "calibration")):
        data = load_windows(args.archive_path, ids)
        validate_windows(data, ids)
        _same(
            data_fingerprint(data),
            lock[f"{prefix}_data_sha256"],
            f"{prefix} data/metadata",
        )
        _same(
            selected_member_hashes(args.archive_path, audit, ids),
            lock[f"{prefix}_member_hashes"],
            f"{prefix} members",
        )
    metric = _fitted_metric(root / "metric.npz", DEV_IDS)
    # Exclusive create is the final gate: even a failed/crashed test load is
    # an inspection and cannot be retried under this output root.
    _write_new(
        root / "test_attempt.json",
        {
            "lock_sha256": checksum,
            "test_subject_ids": list(TEST_IDS),
            "status": "reserved_before_test_values_loaded; no retry under this root",
        },
    )
    test = load_windows(args.archive_path, TEST_IDS)
    validate_windows(test, TEST_IDS)
    scored = cross_day_scores(
        metric.transform(extract_gait_features(test["X"], 100)), test, TEST_IDS
    )
    result = {
        "protocol": registry["config"]["protocol"],
        "lock_sha256": checksum,
        "winner": registry["winner"],
        "threshold_source": "frozen calibration only",
        "test_data_sha256": data_fingerprint(test),
        "test_member_hashes": selected_member_hashes(
            args.archive_path, audit, TEST_IDS
        ),
        "prior_test_inspection": lock["prior_test_inspection"],
        **_report(scored, lock["threshold"]),
        "claim_scope": "held-out-subject cross-day verification on controlled leg-mounted short walks",
    }
    _write_new(root / "test_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "lock", "evaluate"), required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument(
        "--prior-test-inspection",
        help="Required disclosure for a new study after any prior TEST inspection",
    )
    args = parser.parse_args()
    if (
        args.prior_test_inspection is not None
        and not args.prior_test_inspection.strip()
    ):
        parser.error("--prior-test-inspection must explain the previous inspection")
    result = {"train": train_stage, "lock": lock_stage, "evaluate": evaluate_stage}[
        args.stage
    ](args)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "out_root": str(args.out_root),
                "winner": result["winner"],
            }
        )
    )


if __name__ == "__main__":
    main()
