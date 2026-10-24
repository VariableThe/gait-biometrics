"""Exploratory DEV-only backend selection; no Torch or data imports at import.

Stages: develop (register before fitting), lock (replay then CALIB), evaluate
(one exclusive attempt before TEST payload access). See the companion protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_IDS = (7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17)
CALIBRATION_IDS = (21, 18, 23, 12, 30)
TEST_IDS = (26, 28, 22, 15, 6)
CANDIDATES = (
    "control",
    "snorm",
    "nuisance",
    "lagcov",
    "snorm_nuisance",
    "snorm_lagcov",
)
BASELINE = "oldbaseline"
METHODS = (*CANDIDATES, BASELINE)
SOURCES = (
    "scripts/mentor_backend_study.py",
    "scripts/multisensor_study.py",
    "scripts/cross_day_verification.py",
    "data/santos.py",
    "models/gait_features.py",
    "models/gait_invariants.py",
    "models/gait_metric.py",
    "models/session_robust.py",
    "engine/template_scoring.py",
    "engine/cohort_normalization.py",
    "docs/mentor_backend_study.md",
)


def canonical(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    return json.dumps(
        value, default=convert, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read(path):
    return json.loads(Path(path).read_text())


def same(actual, expected, name):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"{name} differs from frozen evidence")


def development_folds():
    """Four balanced five-fold partitions, retaining 20 unique held triples."""
    held_sets = [list(DEV_IDS[start : start + 3]) for start in range(0, 15, 3)]
    seen = {tuple(sorted(held)) for held in held_sets}
    rng = np.random.default_rng(2026)
    while len(held_sets) < 20:
        permutation = rng.permutation(DEV_IDS)
        partition = [
            tuple(sorted(map(int, permutation[start : start + 3])))
            for start in range(0, 15, 3)
        ]
        if len(set(partition)) != 5 or any(held in seen for held in partition):
            continue
        seen.update(partition)
        held_sets.extend(list(held) for held in partition)
    return [
        {
            "fold": i,
            "train_subject_ids": [s for s in DEV_IDS if s not in held],
            "val_subject_ids": held,
        }
        for i, held in enumerate(held_sets)
    ]


def plan():
    return {
        "protocol": "santos_mentor_backends_exploratory_v1",
        "prior_test_inspection": True,
        "candidate_expansion_after_prior_dev_inspection": True,
        "development_subject_ids": list(DEV_IDS),
        "calibration_subject_ids": list(CALIBRATION_IDS),
        "test_subject_ids": list(TEST_IDS),
        "folds": development_folds(),
        "resampling": {
            "partitions": 4,
            "folds_per_partition": 5,
            "validation_count_per_subject": 4,
            "original_partition": "previous.development_folds() unchanged at folds 0-4",
            "additional_partitions": "np.random.default_rng(2026).permutation(DEV_IDS); chunk into five triples; sort each triple",
            "rejection": "reject the whole permutation if any held triple duplicates an earlier accepted triple; require all 20 unique",
        },
        "candidates": list(CANDIDATES),
        "comparison_only": BASELINE,
        "feature_spec": ["accel_gyro", "fusion", "auto"],
        "feature_dimensions": {"base": 336, "lagcov": 362, "oldbaseline": 104},
        "lags": [0, 5, 10, 20],
        "nuisance_directions": 2,
        "shrinkage": "auto",
        "template": "multi",
        "oldbaseline_template": "mean",
        "snorm_std_floor": 0.001,
        "cohort": "fold TRAIN window embeddings, original subjects/days/trials; final DEV only",
        "fit": "training identities only, both days; final all 15 DEV only",
        "selection": "minimum equal-split mean EER across all 20; fixed candidate order breaks ties",
        "completeness": "all six candidates and baseline in every split; no omission or early stopping",
        "calibration_target_far": 0.01,
        "decision": "score >= threshold; tied scores grouped",
        "test_methods": "winner, control, oldbaseline; separate frozen CALIB thresholds",
        "uncertainty": "overlapping splits descriptive only; no independent-fold CI or 1SE rule",
        "sampling_rate": 100,
        "window_size": 256,
        "stride": 256,
        "replay": "numeric states; cached scores with replay atol=rtol=1e-6; exact cached metric recomputation",
    }


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def _previous():
    from scripts import cross_day_verification

    return cross_day_verification


def load_data(archive, ids):
    from data.santos import load_santos_windows

    data = load_santos_windows(archive, ids, channels="accel_gyro")
    _previous().validate_windows({**data, "X": data["X"][:, :3]}, ids)
    if data["X"].shape[1:] != (6, 256) or set(data["subjects"].tolist()) != set(ids):
        raise ValueError("all requested authentic six-axis identities are required")
    for subject in ids:
        if set(data["days"][data["subjects"] == subject].tolist()) != {1, 2}:
            raise ValueError("each identity must retain both days")
    return data


def features(data):
    from models.session_robust import extract_lagged_covariance_features
    from scripts.multisensor_study import descriptor_features

    base = np.asarray(
        descriptor_features(data, ("accel_gyro", "fusion", "auto")), dtype=np.float64
    )
    lag = np.asarray(extract_lagged_covariance_features(data["X"]), dtype=np.float64)
    old = np.asarray(
        descriptor_features(data, ("accel", "axes", "auto")), dtype=np.float64
    )
    if (base.shape, lag.shape, old.shape) != (
        (len(base), 336),
        (len(base), 26),
        (len(base), 104),
    ):
        raise ValueError("registered descriptor dimensions differ")
    result = {
        "base": base,
        "lagcov": np.concatenate((base, lag), axis=1),
        BASELINE: old,
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("features must be finite")
    return result


def feature_key(method):
    if method not in METHODS:
        raise ValueError("unregistered backend")
    return (
        BASELINE if method == BASELINE else "lagcov" if "lagcov" in method else "base"
    )


def fit_metric(method, values, data, training_ids):
    """Explicit DEV membership gate before any estimator is constructed."""
    if not set(training_ids).issubset(DEV_IDS) or len(set(training_ids)) != len(
        training_ids
    ):
        raise ValueError("fitting accepts unique DEV training IDs only")
    feature_key(method)
    mask = np.isin(data["subjects"], training_ids)
    subjects, days = data["subjects"][mask], data["days"][mask]
    if set(subjects.tolist()) != set(training_ids) or any(
        set(days[subjects == subject].tolist()) != {1, 2} for subject in training_ids
    ):
        raise ValueError("training requires both days for every requested identity")
    if "nuisance" in method:
        from models.session_robust import NuisanceProjectedMetric

        metric = NuisanceProjectedMetric(nuisance_directions=2, shrinkage="auto").fit(
            values[mask], subjects, days
        )
    else:
        from models.gait_metric import WithinClassMetric

        metric = WithinClassMetric("wccn", "auto").fit(values[mask], subjects)
    same(
        sorted(metric.training_subject_ids_),
        sorted(training_ids),
        "fitted training IDs",
    )
    cohort = {
        "embeddings": metric.transform(values[mask]),
        "subjects": subjects,
        "days": days,
        "trials": data["trials"][mask],
    }
    return metric, cohort


def save_npz(path, arrays):
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object serialization is forbidden")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def save_model(path, metric, cohort):
    arrays = {f"metric__{key}": value for key, value in metric.state_dict().items()}
    arrays.update({f"cohort__{key}": value for key, value in cohort.items()})
    save_npz(path, arrays)


def load_model(path, method, training_ids):
    arrays = load_npz(path)
    state = {
        key.removeprefix("metric__"): value
        for key, value in arrays.items()
        if key.startswith("metric__")
    }
    if "nuisance" in method:
        from models.session_robust import NuisanceProjectedMetric

        metric = NuisanceProjectedMetric.from_state_dict(state)
    else:
        from models.gait_metric import WithinClassMetric

        metric = WithinClassMetric.from_state_dict(state)
    same(
        sorted(metric.training_subject_ids_),
        sorted(training_ids),
        "saved metric training IDs",
    )
    cohort = {
        key.removeprefix("cohort__"): value
        for key, value in arrays.items()
        if key.startswith("cohort__")
    }
    same(
        sorted(np.unique(cohort["subjects"]).tolist()),
        sorted(training_ids),
        "frozen cohort IDs",
    )
    return metric, cohort


def score(method, metric, cohort, values, data, ids):
    from engine.template_scoring import score_trials

    mask = np.isin(data["subjects"], ids)
    z = metric.transform(values[mask])
    args = (z, data["subjects"][mask], data["days"][mask], data["trials"][mask])
    if "snorm" in method:
        from engine.cohort_normalization import cohort_normalized_scores

        labels, scores, metadata = cohort_normalized_scores(
            *args,
            cohort_embeddings=cohort["embeddings"],
            cohort_subjects=cohort["subjects"],
            cohort_days=cohort["days"],
            cohort_trials=cohort["trials"],
            template_method="multi",
            std_floor=0.001,
        )
    else:
        labels, scores, metadata = score_trials(
            *args, template_method="mean" if method == BASELINE else "multi"
        )
    return {"labels": labels, "scores": scores, "metadata": metadata}


def roc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != scores.shape
        or not len(labels)
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("finite scores and both binary classes are required")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, positive = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    tp, fp = np.cumsum(positive)[ends], np.cumsum(~positive)[ends]
    far = np.r_[0.0, fp / np.sum(labels == 0)]
    frr = np.r_[1.0, (np.sum(labels == 1) - tp) / np.sum(labels == 1)]
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), sorted_scores[ends]]
    if not np.isfinite(thresholds).all():
        raise ValueError("finite reject-all threshold is required")
    return far, frr, thresholds


def eer(labels, scores):
    far, frr, _ = roc(labels, scores)
    diff = far - frr
    hi = int(np.searchsorted(diff, 0.0))
    if diff[hi] == 0:
        return float(far[hi])
    lo = hi - 1
    return float(far[lo] - diff[lo] / (diff[hi] - diff[lo]) * (far[hi] - far[lo]))


def fixed_metrics(labels, scores, threshold):
    roc(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    labels, scores = np.asarray(labels), np.asarray(scores)
    genuine, impostor, accepted = labels == 1, labels == 0, scores >= threshold
    fa, fr = int(np.sum(accepted & impostor)), int(np.sum(~accepted & genuine))
    return {
        "threshold": float(threshold),
        "far": fa / int(impostor.sum()),
        "frr": fr / int(genuine.sum()),
        "false_accepts": fa,
        "false_rejects": fr,
        "n_genuine": int(genuine.sum()),
        "n_impostor": int(impostor.sum()),
        "decision_rule": "score >= threshold",
    }


def calibrate(labels, scores):
    far, _, thresholds = roc(labels, scores)
    threshold = float(thresholds[np.flatnonzero(far <= 0.01)[-1]])
    return {
        **fixed_metrics(labels, scores, threshold),
        "target_far": 0.01,
        "selection": "empirical_far_at_most_target",
    }


def report(scored, threshold=None):
    labels, scores, metadata = scored["labels"], scored["scores"], scored["metadata"]
    claims, probes = (
        np.asarray(metadata["claim_subject_ids"]),
        np.asarray(metadata["probe_subject_ids"]),
    )
    far, frr, _ = roc(labels, scores)
    result = {
        "eer": eer(labels, scores),
        "roc_auc": float(np.sum(np.diff(far) * ((1 - frr[1:]) + (1 - frr[:-1])) / 2)),
        "coverage": metadata["coverage"],
        "claim_graph_sha256": object_hash(
            {
                key: metadata[key]
                for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
            }
        ),
        "per_subject_eer": {
            str(int(subject)): eer(
                np.asarray(labels)[probes == subject],
                np.asarray(scores)[probes == subject],
            )
            for subject in np.unique(probes)
        },
    }
    if threshold is not None:
        result["fixed_threshold"] = fixed_metrics(labels, scores, threshold)
        accepted, positive = np.asarray(scores) >= threshold, np.asarray(labels) == 1
        result["per_subject_errors"] = {}
        for subject in np.unique(probes):
            genuine = positive & (probes == subject)
            impostor = ~positive & (claims == subject)
            result["per_subject_errors"][str(int(subject))] = {
                "false_rejects": int(np.sum(~accepted & genuine)),
                "n_genuine": int(genuine.sum()),
                "false_accepts": int(np.sum(accepted & impostor)),
                "n_impostor": int(impostor.sum()),
                "frr": float(np.sum(~accepted & genuine) / genuine.sum()),
                "far": float(np.sum(accepted & impostor) / impostor.sum()),
            }
    return result


def matched_graphs(reports):
    if len({result["claim_graph_sha256"] for result in reports}) != 1:
        raise ValueError("methods must retain an identical claim graph")


def select_candidate(rows):
    expected = {(fold, method) for fold in range(20) for method in METHODS}
    actual = [(row["fold"], row["method"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("selection requires complete unique 20 x 7 results")
    for fold in range(20):
        matched_graphs([row["metrics"] for row in rows if row["fold"] == fold])
    means = {
        method: float(
            np.mean([row["metrics"]["eer"] for row in rows if row["method"] == method])
        )
        for method in METHODS
    }
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in means.values()):
        raise ValueError("all split EERs must be valid")
    winner = min(CANDIDATES, key=lambda method: means[method])
    return {
        "winner": winner,
        "mean_eer": means,
        "n_splits": 20,
        "interpretation": "descriptive overlapping splits; baseline excluded from selection",
    }


def _manifest(data):
    return {
        "data_sha256": _previous().data_fingerprint(data),
        "members_sha256": data["metadata"]["members_sha256"],
        "members": [
            {key: row[key] for key in ("member", "subject", "day", "trial", "sha256")}
            for row in data["metadata"]["recordings"]
        ],
    }


def _artifact(root, path):
    return {"path": str(Path(path).relative_to(root)), "sha256": digest(path)}


def _verify_artifact(root, artifact):
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact paths must be relative to study root")
    path = Path(root) / relative
    same(digest(path), artifact["sha256"], "artifact " + str(relative))
    return path


def _registration(args):
    root = Path(args.out_root)
    registration = read(root / "registration.json")
    same(registration["plan"], plan(), "registered plan")
    same(registration["sources"], source_hashes(), "registered sources")
    same(
        registration["archive"],
        _previous().audit_archive(args.archive_path),
        "official archive",
    )
    return root, registration


def develop_stage(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(
            "develop requires an empty study root; failed runs need a new root"
        )
    # Archive audit reads compressed bytes and names, never held-out CSV values.
    archive, sources = _previous().audit_archive(args.archive_path), source_hashes()
    same(
        development_folds()[:5], _previous().development_folds(), "original five folds"
    )
    data = load_data(args.archive_path, DEV_IDS)
    registration = {
        "plan": plan(),
        "sources": sources,
        "archive": archive,
        "development": _manifest(data),
    }
    write_new(root / "registration.json", registration)  # BEFORE features or fit.
    values = features(data)
    cache = root / "development_features.npz"
    save_npz(cache, values)
    rows = []
    for fold in development_folds():
        for method in METHODS:
            metric, cohort = fit_metric(
                method, values[feature_key(method)], data, fold["train_subject_ids"]
            )
            path = root / f"fold_{fold['fold']:02d}_{method}.npz"
            save_model(path, metric, cohort)
            scored = score(
                method,
                metric,
                cohort,
                values[feature_key(method)],
                data,
                fold["val_subject_ids"],
            )
            scorepath = root / f"fold_{fold['fold']:02d}_{method}_scores.npz"
            save_npz(scorepath, {key: scored[key] for key in ("labels", "scores")})
            row = {
                "fold": fold["fold"],
                "method": method,
                "model": _artifact(root, path),
                "scores": _artifact(root, scorepath),
                "metadata": scored["metadata"],
                "metrics": report(scored),
            }
            write_new(root / f"fold_{fold['fold']:02d}_{method}.json", row)
            rows.append(row)
    selection = select_candidate(rows)
    result = {
        "registration_sha256": digest(root / "registration.json"),
        "feature_cache": _artifact(root, cache),
        "rows": rows,
        "selection": selection,
    }
    write_new(root / "development.json", result)
    return selection


def verify_development(root, registration, archive_path):
    evidence = read(root / "development.json")
    same(
        evidence["registration_sha256"],
        digest(root / "registration.json"),
        "registration artifact",
    )
    data = load_data(archive_path, DEV_IDS)
    same(
        _manifest(data),
        registration["development"],
        "original DEV data and selected members",
    )
    values = load_npz(_verify_artifact(root, evidence["feature_cache"]))
    current = features(data)
    for key in ("base", "lagcov", BASELINE):
        if values[key].shape != current[key].shape or not np.array_equal(
            values[key], current[key]
        ):
            raise ValueError("DEV feature cache differs from source descriptors")
    selection = select_candidate(evidence["rows"])
    same(selection, evidence["selection"], "complete development selection")
    for row in evidence["rows"]:
        fold, method = development_folds()[row["fold"]], row["method"]
        metric, cohort = load_model(
            _verify_artifact(root, row["model"]), method, fold["train_subject_ids"]
        )
        verify_cohort(
            metric, cohort, values[feature_key(method)], data, fold["train_subject_ids"]
        )
        replayed = score(
            method,
            metric,
            cohort,
            values[feature_key(method)],
            data,
            fold["val_subject_ids"],
        )
        saved = load_npz(_verify_artifact(root, row["scores"]))
        _compare_scores(replayed, saved)
        same(replayed["metadata"], row["metadata"], "replayed claim metadata")
        same(
            report({**saved, "metadata": row["metadata"]}),
            row["metrics"],
            "recomputed full fold metrics",
        )
    return data, values, evidence


def verify_cohort(metric, cohort, values, data, ids):
    mask = np.isin(data["subjects"], ids)
    for key in ("subjects", "days", "trials"):
        if not np.array_equal(cohort[key], data[key][mask]):
            raise ValueError(
                "cohort must retain actual TRAIN subjects, days and trials"
            )
    replayed = metric.transform(values[mask])
    if (
        replayed.shape != cohort["embeddings"].shape
        or not np.isfinite(cohort["embeddings"]).all()
        or not np.allclose(replayed, cohort["embeddings"], atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("cohort embeddings differ from training model")


def _compare_scores(replayed, saved):
    if (
        not np.array_equal(replayed["labels"], saved["labels"])
        or replayed["scores"].shape != saved["scores"].shape
        or not np.isfinite(saved["scores"]).all()
        or not np.allclose(replayed["scores"], saved["scores"], atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("frozen score replay failed")


def lock_stage(args):
    root, registration = _registration(args)
    if (root / "lock_attempt.json").exists() or (root / "protocol_lock.json").exists():
        raise ValueError("lock already attempted")
    data, values, evidence = verify_development(root, registration, args.archive_path)
    methods = list(
        dict.fromkeys((evidence["selection"]["winner"], "control", BASELINE))
    )
    models = {}
    # Final fitting completes on DEV before any CALIB payload access.
    for method in methods:
        metric, cohort = fit_metric(method, values[feature_key(method)], data, DEV_IDS)
        path = root / f"final_{method}.npz"
        save_model(path, metric, cohort)
        models[method] = _artifact(root, path)
    write_new(
        root / "lock_attempt.json",
        {
            "selection": evidence["selection"],
            "models": models,
            "status": "reserved before CALIB load; thresholds are not a selection criterion",
        },
    )
    calibration = load_data(args.archive_path, CALIBRATION_IDS)
    calvalues, calibration_results = features(calibration), {}
    for method in methods:
        metric, cohort = load_model(
            _verify_artifact(root, models[method]), method, DEV_IDS
        )
        scored = score(
            method,
            metric,
            cohort,
            calvalues[feature_key(method)],
            calibration,
            CALIBRATION_IDS,
        )
        threshold = calibrate(scored["labels"], scored["scores"])
        path = root / f"calibration_{method}_scores.npz"
        save_npz(path, {key: scored[key] for key in ("labels", "scores")})
        calibration_results[method] = {
            "threshold": threshold,
            "metrics": report(scored, threshold["threshold"]),
            "metadata": scored["metadata"],
            "scores": _artifact(root, path),
        }
    matched_graphs([value["metrics"] for value in calibration_results.values()])
    lock = {
        "registration_sha256": digest(root / "registration.json"),
        "development_sha256": digest(root / "development.json"),
        "selection": evidence["selection"],
        "models": models,
        "calibration": _manifest(calibration),
        "calibration_results": calibration_results,
        "prior_test_inspection": True,
        "test_subject_ids": list(TEST_IDS),
    }
    lock["lock_sha256"] = object_hash(lock)
    write_new(root / "protocol_lock.json", lock)
    return {
        "locked": True,
        "selection": lock["selection"],
        "lock_sha256": lock["lock_sha256"],
    }


def evaluate_stage(args):
    root = Path(args.out_root)
    if (root / "test_attempt.json").exists() or (root / "test_result.json").exists():
        raise ValueError("TEST evaluation already attempted; no retry under this root")
    lock = read(root / "protocol_lock.json")
    checksum = lock.pop("lock_sha256")
    same(object_hash(lock), checksum, "protocol lock hash")
    root, registration = _registration(args)
    same(
        lock["registration_sha256"],
        digest(root / "registration.json"),
        "locked registration",
    )
    same(
        lock["development_sha256"],
        digest(root / "development.json"),
        "locked development",
    )
    same(lock["test_subject_ids"], list(TEST_IDS), "locked TEST identities")
    data, values, evidence = verify_development(root, registration, args.archive_path)
    same(lock["selection"], evidence["selection"], "locked selection")
    methods = list(
        dict.fromkeys((evidence["selection"]["winner"], "control", BASELINE))
    )
    same(sorted(lock["models"]), sorted(methods), "locked evaluation methods")
    calibration = load_data(args.archive_path, CALIBRATION_IDS)
    same(_manifest(calibration), lock["calibration"], "frozen CALIB data and members")
    calvalues = features(calibration)
    restored = {}
    for method in methods:
        metric, cohort = load_model(
            _verify_artifact(root, lock["models"][method]), method, DEV_IDS
        )
        verify_cohort(metric, cohort, values[feature_key(method)], data, DEV_IDS)
        restored[method] = metric, cohort
        frozen = lock["calibration_results"][method]
        scored = score(
            method,
            metric,
            cohort,
            calvalues[feature_key(method)],
            calibration,
            CALIBRATION_IDS,
        )
        saved = load_npz(_verify_artifact(root, frozen["scores"]))
        _compare_scores(scored, saved)
        same(scored["metadata"], frozen["metadata"], "frozen calibration metadata")
        same(
            calibrate(saved["labels"], saved["scores"]),
            frozen["threshold"],
            "CALIB threshold",
        )
        same(
            report(
                {**saved, "metadata": frozen["metadata"]},
                frozen["threshold"]["threshold"],
            ),
            frozen["metrics"],
            "full CALIB metrics",
        )
    # Atomic exclusive reservation precedes the only TEST payload loader.
    write_new(
        root / "test_attempt.json",
        {
            "lock_sha256": checksum,
            "prior_test_inspection": True,
            "status": "reserved before TEST payload access; failed attempts are consumed",
        },
    )
    test = load_data(args.archive_path, TEST_IDS)
    testvalues, results = features(test), {}
    for method in methods:
        metric, cohort = restored[method]
        scored = score(
            method, metric, cohort, testvalues[feature_key(method)], test, TEST_IDS
        )
        path = root / f"test_{method}_scores.npz"
        save_npz(path, {key: scored[key] for key in ("labels", "scores")})
        results[method] = {
            "metrics": report(
                scored, lock["calibration_results"][method]["threshold"]["threshold"]
            ),
            "metadata": scored["metadata"],
            "scores": _artifact(root, path),
        }
    matched_graphs([value["metrics"] for value in results.values()])
    result = {
        "protocol": plan()["protocol"],
        "exploratory": True,
        "prior_test_inspection": True,
        "lock_sha256": checksum,
        "selection": lock["selection"],
        "test": _manifest(test),
        "results": results,
        "threshold_source": "each method's own frozen CALIB FAR<=1% threshold",
        "uncertainty": "five historically inspected subjects; dependent claims; descriptive only",
    }
    result["result_sha256"] = object_hash(result)
    write_new(root / "test_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("develop", "lock", "evaluate"), required=True
    )
    args = parser.parse_args()
    result = {"develop": develop_stage, "lock": lock_stage, "evaluate": evaluate_stage}[
        args.stage
    ](args)
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
