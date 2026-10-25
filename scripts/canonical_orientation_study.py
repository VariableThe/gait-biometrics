"""Registered DEV-only canonical orientation comparison; stress is diagnostic."""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np

from engine.calibration_selection import _validate_metrics
from models import gait_canonical as canonical
from models.gait_metric import WithinClassMetric
from scripts import covariance_robustness_study as frozen

prior, crossfit, normalization = frozen.prior, frozen.crossfit, frozen.normalization
ARMS = ("control", "canonical336", "canonical362")
DIMENSIONS = dict(zip(ARMS, (362, 336, 362), strict=True))
CONDITIONS = frozen.CONDITIONS
SOURCES = tuple(
    dict.fromkeys(
        (
            *frozen.SOURCES,
            "models/gait_canonical.py",
            "tests/test_gait_canonical.py",
            "docs/gait_canonical.md",
            "scripts/canonical_orientation_study.py",
            "tests/test_canonical_orientation_study.py",
            "docs/canonical_orientation_study.md",
        )
    )
)
# These helpers do not consult frozen.ARMS. No historical globals are changed.
_store, _replay, _row = frozen._store, frozen._replay, frozen._row
_fit, _subset, score = frozen._fit, frozen._subset, frozen.score
training_scale, perturbed = frozen.training_scale, frozen.perturbed


def budget():
    folds = crossfit.folds()
    return {
        "fits": len(ARMS) * sum(len(f["inner"]) + 1 for f in folds),
        "inner_scores": len(ARMS) * sum(len(f["inner"]) for f in folds),
        "outer_scores": len(ARMS) * len(folds) * len(CONDITIONS),
    }


def plan():
    return {
        "protocol": "santos_dev_canonical_orientation_v1",
        "payload_allowlist": list(prior.DEV_IDS),
        "original_calibration_and_test": "never opened",
        "prior_test_inspection": True,
        "candidate_expansion_after_dev_inspection": True,
        "folds": crossfit.folds(),
        "arms": list(ARMS),
        "dimensions": dict(DIMENSIONS),
        "features": {
            "control": "existing lagcov362, unchanged raw sensor coordinates",
            "canonical336": "extract_canonical_features: equal sign pooling BEFORE metric",
            "canonical362": "canonical336 + ORIGINAL raw-window lag-spectrum26 from control[:,336:]",
            "lags": [0, 5, 10, 20],
            "coordinate_frame": "one window-local gravity/horizontal PCA frame for both native-unit triads; both proper sign frames",
            "canonical_axis_peak_relative_tolerance": canonical.SPECTRAL_PEAK_REL_TOL,
        },
        "fit": "automatic shrinkage identity-balanced WCCN on training identities only",
        "scorer": "equal mean/two-medoid blend; raw unnormalized scores",
        "threshold": "policy C: pooled four disjoint clean held-triple empirical FAR<=1%, then12-person refit",
        "budget": {"fits": 300, "inner_scores": 240, "outer_scores": 420},
        "promotion": "clean only: exact mean FAR<=1%; no more splits FAR>2% than control; strictly lower mean FRR; >=10/15 people no worse; tie FRR,FAR,arm order; otherwise control",
        "diagnostic_conditions": list(CONDITIONS),
        "stress": "probe day2 only; enrollment fixed clean; frozen clean thresholds; no stress calibration or selection",
        "rotation": "common proper rotation for both triads, about x, degrees30/90",
        "gain": [0.9, 1.1],
        "noise": "Gaussian1%/5% of12-person refit TRAIN triad dynamic RMS; repeat scale for xyz; seed20261025, original window keys",
        "units": ["m/s^2"] * 3 + ["degrees/sec"] * 3,
        "window": {"samples": 256, "stride": 256, "sampling_rate_hz": 100},
        "coverage": "source-derived exact claim graphs and ALL clean DEV plus every fold/condition validation geometry registered BEFORE features/fits; abort ANY unidentifiable window; no dropping, imputation or fallback",
        "geometry_guards": {
            "gravity_ratio_strictly_above": canonical.GRAVITY_REL_TOL,
            "horizontal_energy_ratio_strictly_above": canonical.HORIZONTAL_ENERGY_REL_TOL,
            "eigengap_ratio_strictly_above": canonical.HORIZONTAL_GAP_REL_TOL,
            "frozen_before_dev": True,
        },
        "replay": "no fits; runtime/source/member hashes; exact geometry arrays including NaNs and row keys; score atol=rtol=1e-6; exact metadata/cached counts, thresholds and selection; identical fresh/cache decisions at frozen thresholds",
        "origins": {
            "copied_protocol_run_verify_summarize": "scripts/covariance_robustness_study.py",
            "reused_helpers": "scripts/covariance_robustness_study.py:_store,_replay,_row,_fit,_subset,score,training_scale,perturbed",
            "folds_graphs": "scripts/crossfit_threshold_study.py",
            "source_coverage_runtime": "scripts/claimant_normalization_study.py",
            "canonical_geometry_features": "models/gait_canonical.py",
            "source_fingerprints": "all origins and their SOURCES are hashed in registration.sources",
        },
        "interpretation": "exploratory dependent DEV folds after prior TEST inspection; sensitivity diagnostics, not phone relocation or independent confirmation; no population FAR guarantee",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def _registration():
    registered = plan()
    prior.same(budget(), registered["budget"], "preflight fit/score budget")
    return {"plan": registered, "sources": hashes(), **normalization.runtime()}


def load_development(archive):
    # No public ID override and no original CALIB/TEST payload access.
    return frozen.load_development(archive)


def representations(data):
    # Fail closed even when called outside run's geometry preflight.
    pooled = canonical.extract_canonical_features(data["X"], 100).require_all_valid()
    pooled = np.asarray(pooled, dtype=np.float64)
    original = prior.features(data)["lagcov"]
    result = {
        "control": original,
        "canonical336": pooled,
        "canonical362": np.concatenate((pooled, original[:, 336:]), axis=1),
    }
    for arm in ARMS:
        if (
            result[arm].shape != (len(data["X"]), DIMENSIONS[arm])
            or not np.isfinite(result[arm]).all()
        ):
            raise ValueError("registered representation shape/values differ")
    return result


def _array_hash(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(prior.canonical([str(array.dtype), list(array.shape)]))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _row_keys(data):
    count = len(data["X"])
    columns = [np.asarray(data[k]) for k in ("subjects", "days", "trials", "starts")]
    if any(a.shape != (count,) or a.dtype.kind not in "iu" for a in columns):
        raise ValueError("one integer source row key per window required")
    keys = np.column_stack(columns)
    if (
        not count
        or data["X"].shape != (count, 6, 256)
        or len(np.unique(keys, axis=0)) != count
        or not set(map(int, keys[:, 0])).issubset(prior.DEV_IDS)
        or not np.isin(keys[:, 1], (1, 2)).all()
    ):
        raise ValueError("unique authentic DEV (N,6,256) row keys required")
    return keys


def _geometry_record(root, name, data, expected=None):
    keys = _row_keys(data)
    geometry = canonical.estimate_canonical_frames(data["X"])
    arrays = {
        "row_keys": keys,
        "frames": geometry.frames,
        "valid": geometry.valid,
        "reason": geometry.reason,
        "gravity_ratio": geometry.gravity_ratio,
        "horizontal_energy_ratio": geometry.horizontal_energy_ratio,
        "eigengap_ratio": geometry.eigengap_ratio,
    }
    path = root / (name + ".npz")
    if expected is None:
        prior.save_npz(path, arrays)
    else:
        prior.same(expected["artifact"]["path"], path.name, "geometry assignment")
        saved = prior.load_npz(prior._verify_artifact(root, expected["artifact"]))
        if saved.keys() != arrays.keys():
            raise ValueError("geometry manifest fields differ")
        for field, fresh in arrays.items():
            cached = saved[field]
            if (
                cached.dtype != fresh.dtype
                or cached.shape != fresh.shape
                or not np.array_equal(cached, fresh, equal_nan=fresh.dtype.kind == "f")
            ):
                raise ValueError("geometry manifest differs: " + field)
    reasons, counts = np.unique(geometry.reason, return_counts=True)
    record = {
        "artifact": prior._artifact(root, path),
        "rows": len(keys),
        "valid": int(np.count_nonzero(geometry.valid)),
        "invalid": int(np.count_nonzero(~geometry.valid)),
        "reason_counts": dict(zip(map(str, reasons), map(int, counts), strict=True)),
        "row_key_columns": ["subject", "day", "trial", "start"],
        "row_keys_sha256": _array_hash(keys),
        "windows_sha256": _array_hash(data["X"]),
    }
    if expected is not None:
        prior.same(record, expected, "geometry manifest record")
    # Persist diagnostics first, even on failure. The caller completes and
    # registers all contexts before raising, without extracting ANY features.
    try:
        geometry.require_all_valid()
    except ValueError:
        if not record["invalid"]:
            raise
    return record


def geometry_coverage(root, data, coverage, expected=None):
    """Register/replay 141 row-preserving contexts, with no features or fits."""
    clean = _geometry_record(
        root, "geometry_clean", data, None if expected is None else expected["clean"]
    )
    folds, scales = {}, {}
    for fold in crossfit.folds():
        key = str(fold["fold"])
        scale = training_scale(data, fold["refit_ids"])
        scales[key] = scale.tolist()
        validation = _subset(data, fold["validation_ids"])
        keys = _row_keys(validation)
        conditions = {}
        for condition in CONDITIONS:
            changed = perturbed(validation, condition, scale)
            if not np.array_equal(_row_keys(changed), keys) or not np.array_equal(
                changed["X"][validation["days"] == 1],
                validation["X"][validation["days"] == 1],
            ):
                raise ValueError("stress changed source row keys or clean enrollment")
            prior.same(
                crossfit.expected_graph(changed, fold["validation_ids"]),
                coverage[key]["validation"],
                "registered stress source graph",
            )
            conditions[condition] = _geometry_record(
                root,
                f"geometry_fold{fold['fold']:02}_{condition}",
                changed,
                None if expected is None else expected["folds"][key][condition],
            )
        folds[key] = conditions
    result = {
        "clean": clean,
        "folds": folds,
        "scales": scales,
        "scale_train_ids": {str(f["fold"]): f["refit_ids"] for f in crossfit.folds()},
        "contexts": 1 + sum(len(c) for c in folds.values()),
        "invalid_context_windows": clean["invalid"]
        + sum(
            r["invalid"] for conditions in folds.values() for r in conditions.values()
        ),
    }
    if expected is not None:
        prior.same(result, expected, "complete geometry manifest")
    return result


def _require_geometry(geometry):
    if geometry["invalid_context_windows"]:
        raise ValueError(
            "unidentifiable canonical geometry; full diagnostics registered; "
            "abort before features/fits"
        )


def summarize(rows):
    """Frozen covariance-study selection, copied to use this module's arms."""
    expected = {
        (fold, arm, condition)
        for fold in range(20)
        for arm in ARMS
        for condition in CONDITIONS
    }
    keys = [(r["fold"], r["arm"], r["condition"]) for r in rows]
    if (
        any(type(r["fold"]) is not int for r in rows)
        or len(keys) != len(expected)
        or set(keys) != expected
    ):
        raise ValueError("complete unique20x3x7 evidence required")
    parsed = {
        key: _validate_metrics(row, set(map(str, prior.DEV_IDS)))
        for key, row in zip(keys, rows, strict=True)
    }
    for fold in range(20):
        if len({parsed[fold, arm, c].graph for arm in ARMS for c in CONDITIONS}) != 1:
            raise ValueError("all arms/conditions must share claim graphs")
        people = set(parsed[fold, "control", "clean"].people)
        if people != set(map(str, crossfit.folds()[fold]["validation_ids"])):
            raise ValueError("registered validation identities required")
    summaries, exact = {}, {}
    for arm in ARMS:
        summaries[arm] = {}
        for condition in CONDITIONS:
            group = [parsed[fold, arm, condition] for fold in range(20)]
            people = {s: [] for s in map(str, prior.DEV_IDS)}
            for row in group:
                for subject, value in row.people.items():
                    people[subject].append(value)
            if any(len(v) != 4 for v in people.values()):
                raise ValueError("each DEV person must validate four times")
            far = sum((r.far for r in group), Fraction()) / 20
            frr = sum((r.frr for r in group), Fraction()) / 20
            person = {s: sum(v, Fraction()) / 4 for s, v in people.items()}
            exact[arm, condition] = far, frr, person
            summaries[arm][condition] = {
                "far": float(far),
                "frr": float(frr),
                "far_exact": str(far),
                "frr_exact": str(frr),
                "splits_far_above_2pct": sum(r.far > Fraction(1, 50) for r in group),
                "per_person_frr": {s: float(v) for s, v in sorted(person.items())},
                "per_person_frr_exact": {s: str(v) for s, v in sorted(person.items())},
            }
    control = exact["control", "clean"]
    eligible = []
    for arm in ARMS:
        far, frr, people = exact[arm, "clean"]
        n = sum(people[s] <= control[2][s] for s in people)
        summaries[arm]["clean"]["people_no_worse_than_control"] = n
        if (
            arm != "control"
            and far <= Fraction(1, 100)
            and frr < control[1]
            and n >= 10
            and (
                summaries[arm]["clean"]["splits_far_above_2pct"]
                <= summaries["control"]["clean"]["splits_far_above_2pct"]
            )
        ):
            eligible.append(arm)
    winner = (
        min(
            eligible,
            key=lambda a: (exact[a, "clean"][1], exact[a, "clean"][0], ARMS.index(a)),
        )
        if eligible
        else "control"
    )
    return {
        "winner": winner,
        "eligible": eligible,
        "summaries": summaries,
        "stress_used_for_selection": False,
        "interpretation": plan()["interpretation"],
    }


def _model_stats(values, data, ids):
    mask = np.isin(data["subjects"], ids)
    return {
        "train_windows": int(mask.sum()),
        "dimension": int(values.shape[1]),
        "train_rows_sha256": _array_hash(_row_keys(data)[mask]),
        "train_features_sha256": _array_hash(values[mask]),
    }


def _save_model(root, name, metric, values, data, ids, fold, arm, role):
    path = root / (name + "_model.npz")
    # All inference state is lossless, including automatic shrinkage diagnostics.
    state = {**metric.state_dict(), "shrinkage": np.asarray(metric.shrinkage_)}
    prior.save_npz(path, state)
    return {
        "fold": fold,
        "arm": arm,
        "role": role,
        "train_ids": ids,
        "stats": {**_model_stats(values, data, ids), "shrinkage": metric.shrinkage_},
        "artifact": prior._artifact(root, path),
    }


def _inventory(result):
    folds = crossfit.folds()
    expected = (
        {(f["fold"], a, r) for f in folds for a in ARMS for r in (*range(4), "refit")},
        {(f["fold"], a, r) for f in folds for a in ARMS for r in range(4)},
        {(f["fold"], a, c) for f in folds for a in ARMS for c in CONDITIONS},
    )
    indexes, paths = [], []
    for name, field, required in zip(
        ("models", "inner", "outer"),
        ("role", "role", "condition"),
        expected,
        strict=True,
    ):
        records = result[name]
        if any(
            type(r["fold"]) is not int
            or (field == "role" and type(r[field]) is not int and r[field] != "refit")
            for r in records
        ):
            raise ValueError("integer registered artifact assignments required")
        index = {(r["fold"], r["arm"], r[field]): r for r in records}
        if (
            len(records) != len(required)
            or len(index) != len(required)
            or set(index) != required
        ):
            raise ValueError(
                "complete unique registered artifact inventory required: " + name
            )
        for (number, arm, role), record in index.items():
            if name == "outer":
                suffix = "_refit_" + role
            else:
                suffix = "_refit" if role == "refit" else f"_inner{role}"
                if name == "models":
                    suffix += "_model"
            prior.same(
                record["artifact"]["path"],
                f"fold{number:02}_{arm}{suffix}.npz",
                "artifact assignment",
            )
        indexes.append(index)
        paths.extend(r["artifact"]["path"] for r in records)
    prior.same(
        [a["path"] for a in result["threshold_files"]],
        [f"thresholds_{i:02}.json" for i in range(20)],
        "ordered threshold assignments",
    )
    paths.extend(a["path"] for a in result["threshold_files"])
    if len(set(paths)) != len(paths):
        raise ValueError("distinct model/score/threshold artifacts required")
    return indexes


def run(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh empty study root required")
    prior.write_new(root / "registration.json", _registration())
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    geometry = geometry_coverage(root, data, coverage)
    manifest = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "data": prior._manifest(data),
        "source_coverage": data["metadata"],
        "coverage": coverage,
        "geometry": geometry,
    }
    prior.write_new(root / "development.json", manifest)
    _require_geometry(geometry)
    values = representations(data)
    rows, models, inner_records, outer_records = [], [], [], []
    scales = geometry["scales"]
    fit_calls = 0
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        scale = np.asarray(scales[str(number)])
        validation = _subset(data, fold["validation_ids"])
        stressed = {
            c: representations(perturbed(validation, c, scale))
            for c in CONDITIONS
            if c != "clean"
        }
        thresholds = {}
        for arm in ARMS:
            pools = []
            for index, inner in enumerate(fold["inner"]):
                metric = _fit(values[arm], data, inner["train_ids"])
                fit_calls += 1
                name = f"fold{number:02}_{arm}_inner{index}"
                models.append(
                    _save_model(
                        root,
                        name,
                        metric,
                        values[arm],
                        data,
                        inner["train_ids"],
                        number,
                        arm,
                        index,
                    )
                )
                scored = score(
                    metric, values[arm], data, inner["held_ids"], graphs["inner"][index]
                )
                inner_records.append(
                    {
                        "fold": number,
                        "arm": arm,
                        "role": index,
                        **_store(root, name, scored),
                    }
                )
                pools.append(scored)
            pool = crossfit._pool(pools)
            crossfit.validate_graph(pool, graphs["pool"])
            calibration = prior.calibrate(pool["labels"], pool["scores"])
            threshold = calibration["threshold"]
            thresholds[arm] = calibration
            metric = _fit(values[arm], data, fold["refit_ids"])
            fit_calls += 1
            name = f"fold{number:02}_{arm}_refit"
            models.append(
                _save_model(
                    root,
                    name,
                    metric,
                    values[arm],
                    data,
                    fold["refit_ids"],
                    number,
                    arm,
                    "refit",
                )
            )
            for condition in CONDITIONS:
                scored = score(
                    metric,
                    values[arm] if condition == "clean" else stressed[condition][arm],
                    data if condition == "clean" else validation,
                    fold["validation_ids"],
                    graphs["validation"],
                )
                rows.append(_row(number, arm, condition, scored, threshold))
                outer_records.append(
                    {
                        "fold": number,
                        "arm": arm,
                        "condition": condition,
                        **_store(root, name + "_" + condition, scored),
                    }
                )
        prior.write_new(root / f"thresholds_{number:02}.json", thresholds)
        print(f"fold {number + 1}/20: {fit_calls} fits", flush=True)
    result = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "development_sha256": prior.digest(root / "development.json"),
        "models": models,
        "inner": inner_records,
        "outer": outer_records,
        "scales": scales,
        "fit_calls": fit_calls,
        "rows": rows,
        "summary": summarize(rows),
        "threshold_files": [
            prior._artifact(root, root / f"thresholds_{i:02}.json") for i in range(20)
        ],
    }
    prior.same(fit_calls, plan()["budget"]["fits"], "actual fit budget")
    _inventory(result)
    prior.write_new(root / "results.json", result)
    print(prior.canonical(result["summary"]).decode(), flush=True)
    return result["summary"]


def _restore_model(root, record, values, data, ids):
    prior.same(record["train_ids"], ids, "model assignment")
    state = prior.load_npz(prior._verify_artifact(root, record["artifact"]))
    if set(state) != {
        "mean",
        "scale",
        "projection",
        "training_subject_ids",
        "shrinkage",
    }:
        raise ValueError("registered fitted model state fields required")
    restored = WithinClassMetric.from_state_dict(state)
    prior.same(restored.training_subject_ids_, sorted(ids), "fitted training IDs")
    alpha = state["shrinkage"]
    if alpha.shape != () or not np.isfinite(alpha) or not 0 <= float(alpha) <= 1:
        raise ValueError("invalid fitted automatic shrinkage")
    prior.same(
        {**_model_stats(values, data, ids), "shrinkage": float(alpha)},
        record["stats"],
        "assigned TRAIN statistics",
    )
    selected = np.asarray(values[np.isin(data["subjects"], ids)], dtype=np.float64)
    if not np.array_equal(restored.mean_, selected.mean(axis=0)) or not np.array_equal(
        restored.scale_, selected.std(axis=0).clip(min=1e-6)
    ):
        raise ValueError(
            "model centering/scaling differs from assigned TRAIN population"
        )
    return restored


def verify(args):
    root = Path(args.out_root)
    registration = prior.read(root / "registration.json")
    prior.same(registration, _registration(), "registration/runtime")
    result = prior.read(root / "results.json")
    prior.same(
        prior.digest(root / "registration.json"),
        result["registration_sha256"],
        "registration digest",
    )
    prior.same(
        prior.digest(root / "development.json"),
        result["development_sha256"],
        "development digest",
    )
    model_index, inner_index, outer_index = _inventory(result)
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    manifest = prior.read(root / "development.json")
    geometry = geometry_coverage(root, data, coverage, manifest["geometry"])
    prior.same(
        manifest,
        {
            "registration_sha256": result["registration_sha256"],
            "data": prior._manifest(data),
            "source_coverage": data["metadata"],
            "coverage": coverage,
            "geometry": geometry,
        },
        "source DEV manifest",
    )
    _require_geometry(geometry)
    prior.same(geometry["scales"], result["scales"], "TRAIN dynamic RMS")
    values = representations(data)
    thresholds = [
        prior.read(prior._verify_artifact(root, a)) for a in result["threshold_files"]
    ]
    for threshold in thresholds:
        prior.same(sorted(threshold), sorted(ARMS), "threshold arm assignments")
    rows = []
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        scale = np.asarray(geometry["scales"][str(number)])
        validation = _subset(data, fold["validation_ids"])
        stressed = {
            c: representations(perturbed(validation, c, scale))
            for c in CONDITIONS
            if c != "clean"
        }
        for arm in ARMS:
            threshold = thresholds[number][arm]["threshold"]
            pools = []
            for index, inner in enumerate(fold["inner"]):
                metric = _restore_model(
                    root,
                    model_index[number, arm, index],
                    values[arm],
                    data,
                    inner["train_ids"],
                )
                fresh = score(
                    metric, values[arm], data, inner["held_ids"], graphs["inner"][index]
                )
                pools.append(
                    _replay(root, fresh, inner_index[number, arm, index], threshold)
                )
            pool = crossfit._pool(pools)
            crossfit.validate_graph(pool, graphs["pool"])
            prior.same(
                prior.calibrate(pool["labels"], pool["scores"]),
                thresholds[number][arm],
                "frozen pooled threshold",
            )
            metric = _restore_model(
                root,
                model_index[number, arm, "refit"],
                values[arm],
                data,
                fold["refit_ids"],
            )
            for condition in CONDITIONS:
                fresh = score(
                    metric,
                    values[arm] if condition == "clean" else stressed[condition][arm],
                    data if condition == "clean" else validation,
                    fold["validation_ids"],
                    graphs["validation"],
                )
                saved = _replay(
                    root, fresh, outer_index[number, arm, condition], threshold
                )
                rows.append(_row(number, arm, condition, saved, threshold))
        print(f"replayed fold {number + 1}/20", flush=True)
    prior.same(result["fit_calls"], 300, "fit budget")
    prior.same(rows, result["rows"], "all cached counts/metrics")
    prior.same(summarize(rows), result["summary"], "exact selection")
    report = {
        "verified": True,
        "source_count": len(SOURCES),
        "models": 300,
        "inner_comparisons": 240,
        "outer_comparisons": 420,
        "geometry_contexts": geometry["contexts"],
        "results_sha256": prior.digest(root / "results.json"),
        "summary": result["summary"],
    }
    prior.write_new(root / "verification.json", report)
    print(prior.canonical(report).decode(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    (run if args.stage == "run" else verify)(args)


if __name__ == "__main__":
    main()
