"""Frozen DEV-only canonical362 nuisance covariance adaptation (no execution on import)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from models.gait_metric import WithinClassMetric
from models.nuisance_metric import fit_nuisance_metric
from scripts import canonical_orientation_study as canonical

prior, crossfit, normalization = (
    canonical.prior,
    canonical.crossfit,
    canonical.normalization,
)
ARMS = ("baseline", "nuisance")
DIMENSIONS = dict.fromkeys(ARMS, 362)
CONDITIONS = canonical.CONDITIONS
TRAIN_VIEWS = ("gain09", "gain11", "noise_plus", "noise_minus")
TRAIN_SEED = 20261026
SOURCES = tuple(
    dict.fromkeys(
        (
            *canonical.SOURCES,
            "models/nuisance_metric.py",
            "tests/test_nuisance_metric.py",
            "docs/nuisance_metric.md",
            "engine/robustness_selection.py",
            "tests/test_robustness_selection.py",
            "docs/robustness_selection.md",
            "scripts/nuisance_covariance_study.py",
            "tests/test_nuisance_covariance_study.py",
            "docs/nuisance_covariance_study.md",
        )
    )
)
_subset, training_scale, perturbed = (
    canonical._subset,
    canonical.training_scale,
    canonical.perturbed,
)
_store, _replay, score = canonical._store, canonical._replay, canonical.score
_array_hash, _row_keys = canonical._array_hash, canonical._row_keys
calibrate = prior.calibrate
DIAGNOSTICS = (
    "clean_covariance",
    "nuisance_second_moment",
    "eigenvalues",
    "eigenvalue_floor",
    "original_rows",
    "original_subjects",
    "views_per_row",
)


def training_transform_registration():
    """A separate TRAIN contract; evaluation_only metadata is never TRAIN evidence."""
    return {
        "version": "canonical362-train-paired-nuisance-v1",
        "scope": "each model's own nine/twelve TRAIN identities, both days, original rows only",
        "view_order": list(TRAIN_VIEWS),
        "gain": [0.9, 1.1],
        "gain_scope": "all six raw coordinates, including acceleration DC",
        "noise_fraction": 0.01,
        "noise_seed": TRAIN_SEED,
        "noise_pair": "X + .01*s*Z and X - .01*s*Z independently from original float64 X before either float32 cast",
        "rng": "PCG64(int(SHA256(canonical JSON ['imu-evaluation-stress-v1',seed,['str',key]]),16))",
        "keys": "original subject/day/trial/start strings; no positional/reset keys",
        "scale": "existing training_scale(data, own TRAIN IDs); dynamic triad RMS repeated across xyz",
        "arithmetic": "source float32 -> float64 -> source float32 for each view",
        "coefficient": 1,
        "evaluation_only_metadata_reused": False,
        "new_independent_data": False,
    }


def budget():
    folds = crossfit.folds()
    return {
        "fits": len(ARMS) * sum(len(f["inner"]) + 1 for f in folds),
        "inner_scores": len(ARMS) * sum(len(f["inner"]) for f in folds),
        "outer_scores": len(ARMS) * len(folds) * len(CONDITIONS),
    }


def plan():
    return {
        "protocol": "santos_dev_canonical362_nuisance_covariance_v1",
        "canonical_source": "108a559bc1a83355dc85335e9f640786846f894a",
        "inspected_evidence_head": "1c861bfc2104b204c7262d931d8cffa19cc6e676",
        "payload_allowlist": list(prior.DEV_IDS),
        "original_calibration_and_test": "never opened",
        "prior_dev_inspection": True,
        "prior_test_inspection": True,
        "prior_stress_inspection": True,
        "adaptation_to_inspected_stress": True,
        "arms": list(ARMS),
        "dimensions": dict(DIMENSIONS),
        "features": "both arms unchanged historical canonical362: equal-sign canonical336 plus original raw lag-spectrum26; no descriptor search",
        "folds": crossfit.folds(),
        "fit": "clean automatic identity-balanced WCCN C_B; nuisance C_A=C_B+V; original rows alone set mean/std/shrinkage; uncentered identity-balanced paired V with coefficient1, then existing eigenvalue floor/whitener",
        "training_transform": training_transform_registration(),
        "scorer": "raw mean_multi, equal mean/two-medoid enrollment blend",
        "threshold": "policy C: four nine-TRAIN clean held-triple score sets pooled for empirical FAR<=1%; twelve-TRAIN refit with frozen cutoff",
        "budget": {"fits": 200, "inner_scores": 160, "outer_scores": 280},
        "geometry_contexts": {
            "clean_dev": 1,
            "outer_validation": 140,
            "training_views": 400,
            "total": 541,
        },
        "geometry_guards": canonical.plan()["geometry_guards"],
        "coverage": "exact source graphs/original exclusions and all541 assigned context records before ANY features/fits; abort whole study with persisted diagnostics on any invalid window or TRAIN RMS reference; no drop/impute/fallback",
        "invalid_train_scale": "record own nine/twelve TRAIN IDs, raw diagnostic RMS and failed triads; materialize independent clean/rotation/gain contexts; retain undefined TRAIN/validation noise contexts as blocked_by_invalid_train_scale with original keys/source hash, false validity, NaN geometry and no transformed-window hash; persist development.json before abort",
        "conditions": list(CONDITIONS),
        "validation_stress": "inherited seven conditions; day2 only, enrollment bit-identical; seed20261025; own twelve-TRAIN dynamic triad RMS; frozen clean cutoff",
        "selection": {
            "implementation": "engine.robustness_selection.summarize_nuisance_study (sole authority)",
            "clean": "mean FAR<=baseline and<=1%; bad folds<=baseline; FRR<=baseline; pooled rejects<=baseline; >=10/15 person means no worse; no person >1 additional reject or new all-reject; mean exact EER<=baseline",
            "objective": "mean FRR across gain09,gain11,noise01 improves>=.05; none of these conditions worse; >=10/15 person mean stress no worse; max person regression<=.05",
            "safety": "every condition FAR<=1%, bad folds<=baseline, each claimant false-accept count<=baseline; both rotations' decisions identical to own clean; noise05 reported, excluded from improvement objective but included in FAR gates",
        },
        "replay": "no fits; source/runtime guards before archive; all diagnostics lossless; deterministic TRAIN algebra and C+V eigensystem/floor/projection;440 score replays atol=rtol=1e-6 with exact metadata, decisions, cached ranking/counts, pool thresholds and selection",
        "paper_basis": [
            "https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf",
            "https://arxiv.org/html/1706.00527v2",
        ],
        "origins": {
            "canonical_representation_geometry": "scripts/canonical_orientation_study.py",
            "scoring_stress": "scripts/covariance_robustness_study.py",
            "source_coverage_runtime": "scripts/claimant_normalization_study.py",
            "folds_graphs": "scripts/crossfit_threshold_study.py",
            "metric": "models/nuisance_metric.py",
            "selection": "engine/robustness_selection.py",
        },
        "interpretation": "mathematical adaptation, not paper reproduction; exploratory dependent DEV folds adapting to inspected stress, not independent confirmation or deployment; augmentation need not preserve labels; prior nominated canonical is not a deployment adapter; no population FAR guarantee",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def _registration():
    registered = plan()
    prior.same(budget(), registered["budget"], "preflight fit/score budget")
    return {"plan": registered, "sources": hashes(), **normalization.runtime()}


def load_development(archive):
    # Fixed DEV-only loader, no public identity override or original CALIB/TEST.
    return canonical.load_development(archive)


def representations(data):
    return canonical.representations(data)["canonical362"]


def training_transform(data, scale):
    """Four raw TRAIN views, with stable keyed antithetic noise before casting.

    This lower-level transform also supports externally batched/reordered TRAIN
    rows using the same fixed scale. The harness supplies only its own TRAIN rows.
    """
    keys = _row_keys(data)
    source = data["X"]
    scale = np.asarray(scale, dtype=np.float64)
    if source.dtype != np.dtype("float32") or not np.isfinite(source).all():
        raise ValueError("TRAIN views require finite source float32 windows")
    if (
        scale.shape != (6,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.array_equal(scale, np.repeat(scale[[0, 3]], 3))
    ):
        raise ValueError(
            "TRAIN scale must be positive finite triad RMS repeated across xyz"
        )
    values = source.astype(np.float64)
    noise = np.empty_like(values)
    for index, row in enumerate(keys):
        key = "/".join(map(str, map(int, row)))
        digest = hashlib.sha256(
            prior.canonical(["imu-evaluation-stress-v1", TRAIN_SEED, ["str", key]])
        ).hexdigest()
        rng = np.random.Generator(np.random.PCG64(int(digest, 16)))
        noise[index] = 0.01 * scale[:, None] * rng.standard_normal(source.shape[1:])
    with np.errstate(over="raise", invalid="raise"):
        raw = (values * 0.9, values * 1.1, values + noise, values - noise)
        outputs = [view.astype(source.dtype) for view in raw]
    if any(not np.isfinite(view).all() for view in outputs):
        raise ValueError("TRAIN views must remain finite")
    return {
        name: {**data, "X": view}
        for name, view in zip(TRAIN_VIEWS, outputs, strict=True)
    }


def _training_source(data, ids):
    if (
        len(ids) not in (9, 12)
        or len(set(ids)) != len(ids)
        or not set(ids).issubset(prior.DEV_IDS)
    ):
        raise ValueError("own nine/twelve unique DEV TRAIN identities required")
    selected = _subset(data, ids)
    if set(map(int, selected["subjects"])) != set(ids):
        raise ValueError("all assigned TRAIN identities required")
    crossfit.expected_graph(selected, ids)  # Both source days must contribute.
    return selected


def _training_inputs(data, ids):
    selected = _training_source(data, ids)
    scale = training_scale(data, ids)
    return selected, scale, training_transform(selected, scale)


def _roles(fold):
    return [
        *((index, inner["train_ids"]) for index, inner in enumerate(fold["inner"])),
        ("refit", fold["refit_ids"]),
    ]


def _role_name(role):
    return "refit" if role == "refit" else f"inner{role}"


def _scale_values(scale):
    # JSON has no nonfinite numbers. None is diagnostic only, never a reference.
    return [float(value) if np.isfinite(value) else None for value in scale]


def _preflight_scale(data, ids):
    """Keep the public positive-RMS contract; recover invalid values only for diagnostics."""
    try:
        return training_scale(data, ids), None
    except (ValueError, FloatingPointError) as error:
        values = data["X"][np.isin(data["subjects"], ids)].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            values -= values.mean(axis=2, keepdims=True)
            rms = np.asarray(
                [np.sqrt(np.mean(values[:, i : i + 3] ** 2)) for i in (0, 3)]
            )
        failed = [
            {
                "triad": name,
                "rms": float(value) if np.isfinite(value) else None,
                "reason": "nonfinite" if not np.isfinite(value) else "not_positive",
            }
            for name, value in zip(("acceleration", "gyroscope"), rms, strict=True)
            if not np.isfinite(value) or value <= 0
        ]
        if not failed:
            raise  # Do not disguise unrelated source/identity contract failures.
        return np.repeat(rms, 3), {
            "status": "invalid_train_scale",
            "reason": str(error),
            "invalid_triads": failed,
        }


def _blocked_geometry_record(root, name, data, expected=None):
    """Persist row-aligned diagnostics without inventing noise windows or frames."""
    keys = _row_keys(data)
    count = len(keys)
    status = "blocked_by_invalid_train_scale"
    arrays = {
        "row_keys": keys,
        "frames": np.full((count, 2, 3, 3), np.nan, np.float64),
        "valid": np.zeros(count, bool),
        "reason": np.full(count, status),
        **{
            field: np.full(count, np.nan, np.float64)
            for field in ("gravity_ratio", "horizontal_energy_ratio", "eigengap_ratio")
        },
    }
    path = root / (name + ".npz")
    if expected is None:
        prior.save_npz(path, arrays)
    else:
        prior.same(
            expected["artifact"]["path"], path.name, "blocked geometry assignment"
        )
        saved = prior.load_npz(prior._verify_artifact(root, expected["artifact"]))
        if saved.keys() != arrays.keys() or any(
            saved[key].dtype != fresh.dtype
            or saved[key].shape != fresh.shape
            or not np.array_equal(saved[key], fresh, equal_nan=fresh.dtype.kind == "f")
            for key, fresh in arrays.items()
        ):
            raise ValueError("blocked geometry diagnostics differ")
    record = {
        "artifact": prior._artifact(root, path),
        "rows": count,
        "valid": 0,
        "invalid": count,
        "reason_counts": {status: count},
        "row_key_columns": ["subject", "day", "trial", "start"],
        "row_keys_sha256": _array_hash(keys),
        "source_windows_sha256": _array_hash(data["X"]),
        "windows_sha256": None,
        "status": status,
        "materialized": False,
    }
    if expected is not None:
        prior.same(record, expected, "blocked geometry manifest record")
    return record


def geometry_coverage(root, data, coverage, expected=None):
    """Complete all541 assigned records, even when a noise reference is undefined."""
    result: dict = {
        "clean": canonical._geometry_record(
            root,
            "geometry_clean",
            data,
            None if expected is None else expected["clean"],
        ),
        "folds": {},
        "scales": {},
        "scale_train_ids": {str(f["fold"]): f["refit_ids"] for f in crossfit.folds()},
    }
    refit_references = {}
    for fold in crossfit.folds():
        key = str(fold["fold"])
        scale, error = _preflight_scale(data, fold["refit_ids"])
        refit_references[key] = (scale, error)
        result["scales"][key] = _scale_values(scale)
        validation = _subset(data, fold["validation_ids"])
        conditions = {}
        for condition in CONDITIONS:
            blocked = error is not None and condition.startswith("noise")
            changed = (
                validation
                if blocked
                else perturbed(validation, condition, scale if error is None else None)
            )
            if not np.array_equal(
                _row_keys(changed), _row_keys(validation)
            ) or not np.array_equal(
                changed["X"][validation["days"] == 1],
                validation["X"][validation["days"] == 1],
            ):
                raise ValueError("stress changed source row keys or clean enrollment")
            prior.same(
                crossfit.expected_graph(changed, fold["validation_ids"]),
                coverage[key]["validation"],
                "registered stress source graph",
            )
            record = _blocked_geometry_record if blocked else canonical._geometry_record
            conditions[condition] = record(
                root,
                f"geometry_fold{fold['fold']:02}_{condition}",
                changed,
                None if expected is None else expected["folds"][key][condition],
            )
        result["folds"][key] = conditions
    training: dict[str, dict] = {}
    scale_errors = []
    for fold in crossfit.folds():
        number = fold["fold"]
        assignments = {}
        for role, ids in _roles(fold):
            name = _role_name(role)
            selected = _training_source(data, ids)
            scale, error = (
                refit_references[str(number)]
                if role == "refit"
                else _preflight_scale(data, ids)
            )
            if error is None:
                views = training_transform(selected, scale)
            else:
                # These gains are scale-free and use exactly the registered arithmetic.
                values = selected["X"].astype(np.float64)
                with np.errstate(over="raise", invalid="raise"):
                    views = {
                        view: {
                            **selected,
                            "X": (values * gain).astype(selected["X"].dtype),
                        }
                        for view, gain in (("gain09", 0.9), ("gain11", 1.1))
                    }
                scale_errors.append(
                    {
                        "fold": number,
                        "role": role,
                        "train_ids": ids,
                        "scale": _scale_values(scale),
                        **error,
                    }
                )
            graph = coverage[str(number)][
                "cohort_refit" if role == "refit" else "cohort_inner"
            ]
            if role != "refit":
                graph = graph[role]
            records = {}
            for view in TRAIN_VIEWS:
                blocked = view not in views
                changed = selected if blocked else views[view]
                prior.same(
                    _row_keys(changed).tolist(),
                    _row_keys(selected).tolist(),
                    "TRAIN view source keys",
                )
                prior.same(
                    crossfit.expected_graph(changed, ids),
                    graph,
                    "TRAIN view source graph",
                )
                record = (
                    _blocked_geometry_record if blocked else canonical._geometry_record
                )
                records[view] = record(
                    root,
                    f"geometry_fold{number:02}_train_{name}_{view}",
                    changed,
                    None
                    if expected is None
                    else expected["training"][str(number)][name]["views"][view],
                )
            assignments[name] = {
                "fold": number,
                "role": role,
                "arms": list(ARMS),
                "train_ids": ids,
                "scale": _scale_values(scale),
                "clean_row_keys_sha256": _array_hash(_row_keys(selected)),
                "clean_windows_sha256": _array_hash(selected["X"]),
                "views": records,
            }
            if error is not None:
                assignments[name]["scale_status"] = error["status"]
        training[str(number)] = assignments
    result["training"] = training
    records = [
        result["clean"],
        *(r for group in result["folds"].values() for r in group.values()),
        *(
            v
            for group in training.values()
            for assignment in group.values()
            for v in assignment["views"].values()
        ),
    ]
    result["contexts"] = len(records)
    result["invalid_context_windows"] = sum(r["invalid"] for r in records)
    if scale_errors:
        result["status"] = "failed_invalid_train_scale"
        result["scale_errors"] = scale_errors
        result["blocked_contexts"] = sum(
            not r.get("materialized", True) for r in records
        )
        result["materialized_contexts"] = len(records) - result["blocked_contexts"]
    prior.same(result["contexts"], 541, "complete geometry context budget")
    if expected is not None:
        prior.same(result, expected, "complete geometry manifest")
    return result


def _require_geometry(geometry):
    if geometry.get("scale_errors"):
        raise ValueError(
            "invalid TRAIN dynamic triad RMS reference (positive finite required); "
            "full diagnostics registered; abort before features/fits"
        )
    canonical._require_geometry(geometry)


def _training_features(data, ids):
    _selected, _scale, views = _training_inputs(data, ids)
    return np.stack([representations(views[v]) for v in TRAIN_VIEWS], axis=1)


def _row(fold, arm, condition, scored, threshold):
    from engine.robustness_selection import ranking_metrics

    return {
        **canonical._row(fold, arm, condition, scored, threshold),
        "ranking": ranking_metrics(
            scored["labels"], scored["scores"], scored["metadata"]["probe_subject_ids"]
        ),
        "decision_sha256": _array_hash(
            np.asarray(scored["scores"], np.float64) >= threshold
        ),
    }


def summarize(rows):
    from engine.robustness_selection import summarize_nuisance_study

    return summarize_nuisance_study(rows, dev_ids=prior.DEV_IDS, folds=crossfit.folds())


def _model_stats(values, data, ids, views):
    return {
        **canonical._model_stats(values, data, ids),
        "train_view_features_sha256": _array_hash(views),
    }


def _fit(values, data, ids, views, arm):
    if arm not in ARMS:
        raise ValueError("registered metric arm required")
    mask = np.isin(data["subjects"], ids)
    return fit_nuisance_metric(
        values[mask], data["subjects"][mask], views if arm == "nuisance" else None
    )


def _save_model(
    root, name, metric, diagnostics, values, data, ids, views, fold, arm, role
):
    if set(diagnostics) != set(DIAGNOSTICS):
        raise ValueError("complete numeric covariance diagnostics required")
    path = root / (name + "_model.npz")
    state = {
        **metric.state_dict(),
        "shrinkage": np.asarray(metric.shrinkage_, np.float64),
        **{
            "diagnostic_" + key: np.asarray(value) for key, value in diagnostics.items()
        },
    }
    prior.save_npz(path, state)
    return {
        "fold": fold,
        "arm": arm,
        "role": role,
        "train_ids": ids,
        "stats": {
            **_model_stats(values, data, ids, views),
            "shrinkage": metric.shrinkage_,
        },
        "artifact": prior._artifact(root, path),
    }


def _close(actual, expected, name):
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.allclose(actual, expected, atol=1e-12, rtol=1e-10)
    ):
        raise ValueError("model diagnostic algebra differs: " + name)


def _clean_algebra(selected, subjects, shrinkage):
    """Deterministic sums only; never call a covariance estimator or fit in replay."""
    n, d = selected.shape
    mean, std = selected.mean(axis=0), selected.std(axis=0).clip(min=1e-6)
    standardized = (selected - mean) / std
    residual = np.zeros_like(standardized)
    identities, counts = np.unique(subjects, return_counts=True)
    for subject, count in zip(identities, counts, strict=True):
        mask = subjects == subject
        residual[mask] = (
            standardized[mask] - standardized[mask].mean(axis=0)
        ) * np.sqrt(n / (len(identities) * count))
    empirical = residual.T @ residual / n
    trace = np.sum(residual**2, axis=0) / n
    mu = trace.sum() / d
    # Ledoit-Wolf's scalar formula, checked without fitting an estimator.
    squared = np.sum(empirical**2)
    beta = (np.sum(np.sum(residual**2, axis=1) ** 2) / n - squared) / (d * n)
    delta = (squared - 2 * mu * trace.sum() + d * mu**2) / d
    beta = min(beta, delta)
    alpha = 0.0 if beta == 0 else beta / delta
    _close(np.asarray(shrinkage), np.asarray(alpha), "clean automatic shrinkage")
    covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(d)
    return mean, std, covariance


def _restore_model(root, record, values, data, ids, views):
    prior.same(record["train_ids"], ids, "model assignment")
    state = prior.load_npz(prior._verify_artifact(root, record["artifact"]))
    required = {"mean", "scale", "projection", "training_subject_ids", "shrinkage"} | {
        "diagnostic_" + k for k in DIAGNOSTICS
    }
    if set(state) != required or any(a.dtype.kind not in "iuf" for a in state.values()):
        raise ValueError(
            "registered numeric fitted model/diagnostic state fields required"
        )
    restored = WithinClassMetric.from_state_dict(state)
    prior.same(restored.training_subject_ids_, sorted(ids), "fitted training IDs")
    alpha = state["shrinkage"]
    if alpha.shape != () or not np.isfinite(alpha) or not 0 <= float(alpha) <= 1:
        raise ValueError("invalid fitted automatic shrinkage")
    prior.same(
        {**_model_stats(values, data, ids, views), "shrinkage": float(alpha)},
        record["stats"],
        "assigned TRAIN statistics",
    )
    mask = np.isin(data["subjects"], ids)
    selected, subjects = np.asarray(values[mask], np.float64), data["subjects"][mask]
    mean, std, clean = _clean_algebra(selected, subjects, float(alpha))
    if not np.array_equal(restored.mean_, mean) or not np.array_equal(
        restored.scale_, std
    ):
        raise ValueError(
            "model centering/scaling differs from assigned TRAIN population"
        )
    dimension = selected.shape[1]
    diag = {key: state["diagnostic_" + key] for key in DIAGNOSTICS}
    if record["arm"] == "baseline" and np.any(diag["nuisance_second_moment"] != 0):
        raise ValueError("baseline nuisance second moment must be exactly zero")
    for field, expected in (
        ("original_rows", len(selected)),
        ("original_subjects", len(ids)),
        ("views_per_row", 4 if record["arm"] == "nuisance" else 0),
    ):
        value = diag[field]
        if value.shape != () or value.dtype.kind not in "iu" or int(value) != expected:
            raise ValueError("model diagnostic training counts differ: " + field)
    nuisance = np.zeros((dimension, dimension), np.float64)
    if record["arm"] == "nuisance":
        if views.shape != (len(selected), 4, dimension) or not np.isfinite(views).all():
            raise ValueError("paired TRAIN features differ")
        delta = (views - selected[:, None, :]) / std
        for subject in np.unique(subjects):
            paired = delta[subjects == subject].reshape(-1, dimension)
            nuisance += paired.T @ paired / len(paired)
        nuisance /= len(ids)
    for field, expected in (
        ("clean_covariance", clean),
        ("nuisance_second_moment", nuisance),
    ):
        value = diag[field]
        _close(value, expected, field)
        _close(value, value.T, field + " symmetry")
        if np.linalg.eigvalsh(value).min() < -1e-10 * max(
            1.0, float(np.max(np.abs(value)))
        ):
            raise ValueError("model diagnostic covariance must be PSD")
    # Use the saved unfloored C_B+V directly; never invert a floored baseline.
    eigenvalues, vectors = np.linalg.eigh(
        diag["clean_covariance"] + diag["nuisance_second_moment"]
    )
    _close(diag["eigenvalues"], eigenvalues, "C+V eigenvalues")
    floor = max(float(eigenvalues.max()) * 1e-5, 1e-6)
    _close(diag["eigenvalue_floor"], np.asarray(floor), "eigenvalue floor")
    projection = (vectors * np.maximum(eigenvalues, floor) ** -0.5) @ vectors.T
    _close(restored.projection_, projection, "C+V whitening projection")
    restored.shrinkage_ = float(alpha)
    return restored


def _inventory(result):
    required = (
        {
            (f["fold"], a, r)
            for f in crossfit.folds()
            for a in ARMS
            for r, ids in _roles(f)
        },
        {(f["fold"], a, r) for f in crossfit.folds() for a in ARMS for r in range(4)},
        {(f["fold"], a, c) for f in crossfit.folds() for a in ARMS for c in CONDITIONS},
    )
    indexes, paths = [], []
    for name, field, expected in zip(
        ("models", "inner", "outer"),
        ("role", "role", "condition"),
        required,
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
            len(records) != len(expected)
            or len(index) != len(expected)
            or set(index) != expected
        ):
            raise ValueError(
                "complete unique registered artifact inventory required: " + name
            )
        for (number, arm, role), record in index.items():
            suffix = "refit_" + role if name == "outer" else _role_name(role)
            if name == "models":
                suffix += "_model"
            prior.same(
                record["artifact"]["path"],
                f"fold{number:02}_{arm}_{suffix}.npz",
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


def _manifest(root, data, coverage, geometry):
    return {
        "registration_sha256": prior.digest(root / "registration.json"),
        "data": prior._manifest(data),
        "source_coverage": data["metadata"],
        "coverage": coverage,
        "geometry": geometry,
    }


def run(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh empty study root required")
    prior.write_new(root / "registration.json", _registration())
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    geometry = geometry_coverage(root, data, coverage)
    prior.write_new(
        root / "development.json", _manifest(root, data, coverage, geometry)
    )
    _require_geometry(geometry)
    values = representations(data)
    rows, models, inner_records, outer_records = [], [], [], []
    fit_calls = 0
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        pools = {arm: [] for arm in ARMS}
        thresholds = {}
        validation = _subset(data, fold["validation_ids"])
        stressed = {
            c: representations(
                perturbed(validation, c, np.asarray(geometry["scales"][str(number)]))
            )
            for c in CONDITIONS
            if c != "clean"
        }
        for role, ids in _roles(fold):
            if role == "refit":
                # Freeze BOTH clean pooled cutoffs before either refit starts,
                # including before extracting the twelve-TRAIN paired views.
                for arm in ARMS:
                    pool = crossfit._pool(pools[arm])
                    crossfit.validate_graph(pool, graphs["pool"])
                    thresholds[arm] = calibrate(pool["labels"], pool["scores"])
            views = _training_features(data, ids)
            for arm in ARMS:
                metric, diagnostics = _fit(values, data, ids, views, arm)
                fit_calls += 1
                name = f"fold{number:02}_{arm}_{_role_name(role)}"
                models.append(
                    _save_model(
                        root,
                        name,
                        metric,
                        diagnostics,
                        values,
                        data,
                        ids,
                        views,
                        number,
                        arm,
                        role,
                    )
                )
                if role != "refit":
                    inner = fold["inner"][role]
                    scored = score(
                        metric, values, data, inner["held_ids"], graphs["inner"][role]
                    )
                    inner_records.append(
                        {
                            "fold": number,
                            "arm": arm,
                            "role": role,
                            **_store(root, name, scored),
                        }
                    )
                    pools[arm].append(scored)
                    continue
                threshold = thresholds[arm]["threshold"]
                for condition in CONDITIONS:
                    scored = score(
                        metric,
                        values if condition == "clean" else stressed[condition],
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
    prior.same(fit_calls, plan()["budget"]["fits"], "actual fit budget")
    result = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "development_sha256": prior.digest(root / "development.json"),
        "models": models,
        "inner": inner_records,
        "outer": outer_records,
        "scales": geometry["scales"],
        "fit_calls": fit_calls,
        "rows": rows,
        "summary": summarize(rows),
        "threshold_files": [
            prior._artifact(root, root / f"thresholds_{i:02}.json") for i in range(20)
        ],
    }
    _inventory(result)
    prior.write_new(root / "results.json", result)
    print(prior.canonical(result["summary"]).decode(), flush=True)
    return result["summary"]


def verify(args):
    root = Path(args.out_root)
    prior.same(
        prior.read(root / "registration.json"), _registration(), "registration/runtime"
    )
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
    thresholds = [
        prior.read(prior._verify_artifact(root, a)) for a in result["threshold_files"]
    ]
    for threshold in thresholds:
        prior.same(sorted(threshold), sorted(ARMS), "threshold arm assignments")
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    manifest = prior.read(root / "development.json")
    geometry = geometry_coverage(root, data, coverage, manifest["geometry"])
    prior.same(
        manifest, _manifest(root, data, coverage, geometry), "source DEV manifest"
    )
    _require_geometry(geometry)
    prior.same(geometry["scales"], result["scales"], "TRAIN dynamic RMS")
    values = representations(data)
    rows, replays = [], 0
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        validation = _subset(data, fold["validation_ids"])
        stressed = {
            c: representations(
                perturbed(validation, c, np.asarray(geometry["scales"][str(number)]))
            )
            for c in CONDITIONS
            if c != "clean"
        }
        pools = {arm: [] for arm in ARMS}
        for role, ids in _roles(fold):
            views = _training_features(data, ids)
            for arm in ARMS:
                threshold = thresholds[number][arm]["threshold"]
                metric = _restore_model(
                    root, model_index[number, arm, role], values, data, ids, views
                )
                if role != "refit":
                    inner = fold["inner"][role]
                    fresh = score(
                        metric, values, data, inner["held_ids"], graphs["inner"][role]
                    )
                    pools[arm].append(
                        _replay(root, fresh, inner_index[number, arm, role], threshold)
                    )
                    replays += 1
                    continue
                pool = crossfit._pool(pools[arm])
                crossfit.validate_graph(pool, graphs["pool"])
                prior.same(
                    calibrate(pool["labels"], pool["scores"]),
                    thresholds[number][arm],
                    "frozen pooled threshold",
                )
                for condition in CONDITIONS:
                    fresh = score(
                        metric,
                        values if condition == "clean" else stressed[condition],
                        data if condition == "clean" else validation,
                        fold["validation_ids"],
                        graphs["validation"],
                    )
                    saved = _replay(
                        root, fresh, outer_index[number, arm, condition], threshold
                    )
                    rows.append(_row(number, arm, condition, saved, threshold))
                    replays += 1
        print(f"replayed fold {number + 1}/20", flush=True)
    prior.same(result["fit_calls"], 200, "fit budget")
    prior.same(replays, 440, "score replay budget")
    prior.same(rows, result["rows"], "all cached ranking/counts/metrics/decisions")
    prior.same(summarize(rows), result["summary"], "exact selection")
    report = {
        "verified": True,
        "source_count": len(SOURCES),
        "models": 200,
        "inner_comparisons": 160,
        "outer_comparisons": 280,
        "score_replays": replays,
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
