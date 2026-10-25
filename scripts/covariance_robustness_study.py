"""Registered DEV-only covariance comparison and fixed-threshold sensor stresses."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from engine.calibration_selection import _validate_metrics
from models.gait_metric import WithinClassMetric
from scripts import claimant_normalization_study as normalization
from scripts import crossfit_threshold_study as crossfit

prior = crossfit.prior
ARMS = ("control", "full_covariance", "log_covariance")
CONDITIONS = (
    "clean",
    "rotation30",
    "rotation90",
    "gain09",
    "gain11",
    "noise01",
    "noise05",
)
SOURCES = tuple(
    dict.fromkeys(
        (
            *normalization.SOURCES,
            "models/lag_covariance.py",
            "tests/test_lag_covariance.py",
            "docs/lag_covariance.md",
            "data/imu_perturbations.py",
            "tests/test_imu_perturbations.py",
            "docs/imu_perturbations.md",
            "scripts/covariance_robustness_study.py",
            "tests/test_covariance_robustness_study.py",
            "docs/covariance_robustness_study.md",
        )
    )
)


def plan():
    return {
        "protocol": "santos_dev_covariance_robustness_v1",
        "payload_allowlist": list(prior.DEV_IDS),
        "original_calibration_and_test": "never opened",
        "prior_test_inspection": True,
        "candidate_expansion_after_dev_inspection": True,
        "folds": crossfit.folds(),
        "arms": list(ARMS),
        "features": {
            "control": "existing fusion336 + lag spectrum26 =362",
            "full_covariance": "fusion336 + per-triad full regularized covariance and log trace =494",
            "log_covariance": "fusion336 + per-triad full matrix logarithm and log trace =494",
            "lags": [0, 5, 10, 20],
            "shrinkage": 0.2,
            "matrix": "S=.8*G/max(trace(G),1e-8)+.2*I12/12; upper triangle with off-diagonal sqrt(2)",
        },
        "fit": "automatic shrinkage identity-balanced WCCN on training identities only",
        "scorer": "equal mean/two-medoid blend; raw unnormalized scores",
        "threshold": "policy C: pooled four disjoint held-triple empirical FAR<=1%, then12-person refit",
        "budget": {"fits": 300, "inner_scores": 240, "outer_scores": 420},
        "promotion": "clean only: exact mean FAR<=1%; no more splits FAR>2% than control; strictly lower mean FRR; >=10/15 people no worse; tie FRR,FAR,arm order; otherwise control",
        "diagnostic_conditions": list(CONDITIONS),
        "stress": "probe day2 only; enrollment fixed; no threshold recalibration or selection from stress results",
        "rotation": "common proper rotation for both triads, about x, degrees30/90",
        "gain": [0.9, 1.1],
        "noise": "Gaussian1%/5% of12-person refit TRAIN triad dynamic RMS; repeat scale for xyz; seed20261025, original window keys",
        "units": ["m/s^2"] * 3 + ["degrees/sec"] * 3,
        "window": {"samples": 256, "stride": 256, "sampling_rate_hz": 100},
        "coverage": "source-derived exact trial graphs before features/fits; every condition retains every claim",
        "replay": "runtime/source/member hashes; score atol=rtol=1e-6; exact metadata/cached metrics and identical fresh/cache threshold decisions",
        "interpretation": "exploratory dependent DEV folds; synthetic sensitivity, not phone relocation or independent data; no population FAR guarantee",
        "paper_basis": [
            "https://www.esann.org/sites/default/files/proceedings/legacy/es2012-130.pdf",
            "https://arxiv.org/abs/1706.00527",
            "https://arxiv.org/abs/1602.03199",
        ],
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def load_development(archive):
    # The allowlist is fixed in source, with no public subject-ID option.
    return prior.load_data(archive, prior.DEV_IDS)


def representations(data):
    from models.lag_covariance import extract_lag_covariance_features

    existing = prior.features(data)
    result = {"control": existing["lagcov"]}
    for arm, logarithm in (("full_covariance", False), ("log_covariance", True)):
        block = extract_lag_covariance_features(data["X"], logarithm=logarithm)
        result[arm] = np.concatenate((existing["base"], block), axis=1)
    for arm, dimension in zip(ARMS, (362, 494, 494), strict=True):
        if (
            result[arm].shape != (len(data["X"]), dimension)
            or not np.isfinite(result[arm]).all()
        ):
            raise ValueError("registered representation shape/values differ")
    return result


def training_scale(data, ids):
    """Dynamic RMS: center each TRAIN window/channel, then average triad energy."""
    selected = np.isin(data["subjects"], ids)
    if set(map(int, data["subjects"][selected])) != set(ids):
        raise ValueError("all registered training people must contribute")
    values = data["X"][selected].astype(np.float64)
    values -= values.mean(axis=2, keepdims=True)
    scales = [
        float(np.sqrt(np.mean(values[:, start : start + 3] ** 2))) for start in (0, 3)
    ]
    if not all(np.isfinite(s) and s > 0 for s in scales):
        raise ValueError("TRAIN dynamic triad scales must be positive finite")
    return np.repeat(scales, 3)


def perturbed(data, condition, scale):
    """Keep clean enrollment; modify only authentic day2 probe windows."""
    from data.imu_perturbations import IMUPerturbation, apply_imu_perturbation

    if condition not in CONDITIONS:
        raise ValueError("unregistered condition")
    if condition == "clean":
        return data
    selected = data["days"] == 2
    keys = [
        f"{int(s)}/{int(d)}/{int(t)}/{int(start)}"
        for s, d, t, start in zip(
            data["subjects"][selected],
            data["days"][selected],
            data["trials"][selected],
            data["starts"][selected],
            strict=True,
        )
    ]
    kwargs: dict[str, Any] = {"name": condition, "units": ("m/s^2", "degrees/sec")}
    if condition.startswith("rotation"):
        kwargs.update(
            rotation_axis=(1.0, 0.0, 0.0), rotation_degrees=float(condition[8:])
        )
    elif condition.startswith("gain"):
        gain = 0.9 if condition == "gain09" else 1.1
        kwargs["gain"] = (gain, gain)
    else:
        noise = 0.01 if condition == "noise01" else 0.05
        kwargs.update(
            noise_fraction=(noise, noise),
            reference_std=tuple(scale),
            reference_id="TRAIN-dynamic-RMS:" + prior.object_hash(scale),
        )
    transformed = apply_imu_perturbation(
        data["X"][selected],
        IMUPerturbation(**kwargs),
        window_keys=keys,
        seed=20261025,
    )
    result = {**data, "X": data["X"].copy()}
    result["X"][selected] = transformed.windows
    if not np.array_equal(result["X"][~selected], data["X"][~selected]):
        raise ValueError("stress modified enrollment")
    return result


def score(metric, values, data, ids, graph):
    return crossfit.checked_score(metric, values, data, ids, "mean_multi", graph)


def summarize(rows):
    expected = {
        (fold, arm, condition)
        for fold in range(20)
        for arm in ARMS
        for condition in CONDITIONS
    }
    keys = [(r["fold"], r["arm"], r["condition"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("complete unique20x3x7 evidence required")
    parsed = {
        key: _validate_metrics(row, set(map(str, prior.DEV_IDS)))
        for key, row in zip(keys, rows, strict=True)
    }
    for fold in range(20):
        if len({parsed[fold, arm, c].graph for arm in ARMS for c in CONDITIONS}) != 1:
            raise ValueError("all arms/conditions must share claim graphs")
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


def _store(root, name, scored):
    path = root / (name + ".npz")
    prior.save_npz(path, {key: scored[key] for key in ("labels", "scores")})
    return {"artifact": prior._artifact(root, path), "metadata": scored["metadata"]}


def _replay(root, fresh, frozen, threshold):
    cache = prior.load_npz(prior._verify_artifact(root, frozen["artifact"]))
    prior._compare_scores(fresh, cache)
    prior.same(fresh["metadata"], frozen["metadata"], "score metadata")
    if not np.array_equal(
        np.asarray(fresh["scores"], np.float64) >= threshold,
        np.asarray(cache["scores"], np.float64) >= threshold,
    ):
        raise ValueError("fresh/cache decisions changed at frozen threshold")
    return {**cache, "metadata": frozen["metadata"]}


def _row(fold, arm, condition, scored, threshold):
    return {
        "fold": fold,
        "arm": arm,
        "condition": condition,
        "metrics": prior.report(scored, threshold),
        "pair_errors": crossfit.pair_errors(scored, threshold),
    }


def _fit(values, data, ids):
    if not set(ids).issubset(prior.DEV_IDS) or len(set(ids)) != len(ids):
        raise ValueError("unique DEV training IDs required")
    mask = np.isin(data["subjects"], ids)
    if set(map(int, data["subjects"][mask])) != set(ids):
        raise ValueError("all training identities required")
    return WithinClassMetric("wccn", "auto").fit(values[mask], data["subjects"][mask])


def _subset(data, ids):
    mask = np.isin(data["subjects"], ids)
    return {
        k: v[mask]
        if isinstance(v, np.ndarray) and v.ndim and len(v) == len(mask)
        else v
        for k, v in data.items()
    }


def run(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh empty study root required")
    registration = {"plan": plan(), "sources": hashes(), **normalization.runtime()}
    prior.write_new(root / "registration.json", registration)
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    manifest = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "data": prior._manifest(data),
        "coverage": coverage,
    }
    prior.write_new(root / "development.json", manifest)
    values = representations(data)
    rows, models, inner_records, outer_records, scales = [], [], [], [], {}
    fit_calls = 0
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        scale = training_scale(data, fold["refit_ids"])
        scales[str(number)] = scale.tolist()
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
                path = root / (name + "_model.npz")
                prior.save_npz(path, metric.state_dict())
                models.append(
                    {
                        "fold": number,
                        "arm": arm,
                        "role": index,
                        "train_ids": inner["train_ids"],
                        "artifact": prior._artifact(root, path),
                    }
                )
                scored = score(
                    metric, values[arm], data, inner["held_ids"], graphs["inner"][index]
                )
                frozen = _store(root, name, scored)
                inner_records.append(
                    {"fold": number, "arm": arm, "role": index, **frozen}
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
            path = root / (name + "_model.npz")
            prior.save_npz(path, metric.state_dict())
            models.append(
                {
                    "fold": number,
                    "arm": arm,
                    "role": "refit",
                    "train_ids": fold["refit_ids"],
                    "artifact": prior._artifact(root, path),
                }
            )
            for condition in CONDITIONS:
                if condition == "clean":
                    scored = score(
                        metric,
                        values[arm],
                        data,
                        fold["validation_ids"],
                        graphs["validation"],
                    )
                else:
                    scored = score(
                        metric,
                        stressed[condition][arm],
                        validation,
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
    if fit_calls != plan()["budget"]["fits"]:
        raise ValueError("actual fit count differs from registration")
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
    prior.write_new(root / "results.json", result)
    print(prior.canonical(result["summary"]).decode(), flush=True)


def verify(args):
    root = Path(args.out_root)
    registration = prior.read(root / "registration.json")
    prior.same(
        registration,
        {"plan": plan(), "sources": hashes(), **normalization.runtime()},
        "registration/runtime",
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
    data = load_development(args.archive_path)
    coverage = normalization.coverage_plan(data)
    prior.same(
        prior.read(root / "development.json"),
        {
            "registration_sha256": result["registration_sha256"],
            "data": prior._manifest(data),
            "coverage": coverage,
        },
        "source DEV manifest",
    )
    values = representations(data)
    if (
        len(result["models"]) != 300
        or len(result["inner"]) != 240
        or len(result["outer"]) != 420
    ):
        raise ValueError("complete registered artifact inventory required")
    model_index = {(r["fold"], r["arm"], r["role"]): r for r in result["models"]}
    inner_index = {(r["fold"], r["arm"], r["role"]): r for r in result["inner"]}
    outer_index = {(r["fold"], r["arm"], r["condition"]): r for r in result["outer"]}
    if (len(model_index), len(inner_index), len(outer_index)) != (300, 240, 420):
        raise ValueError("duplicate artifact assignments")
    thresholds = [
        prior.read(prior._verify_artifact(root, a)) for a in result["threshold_files"]
    ]
    if len(thresholds) != 20:
        raise ValueError("all threshold artifacts required")
    rows = []
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        scale = training_scale(data, fold["refit_ids"])
        prior.same(scale.tolist(), result["scales"][str(number)], "TRAIN dynamic RMS")
        validation = _subset(data, fold["validation_ids"])
        stressed = {
            c: representations(perturbed(validation, c, scale))
            for c in CONDITIONS
            if c != "clean"
        }
        for arm in ARMS:
            threshold = thresholds[number][arm]["threshold"]

            def model(role, ids, number=number, arm=arm):
                record = model_index[number, arm, role]
                prior.same(record["train_ids"], ids, "model assignment")
                restored = WithinClassMetric.from_state_dict(
                    prior.load_npz(prior._verify_artifact(root, record["artifact"]))
                )
                prior.same(
                    restored.training_subject_ids_, sorted(ids), "fitted training IDs"
                )
                selected = values[arm][np.isin(data["subjects"], ids)]
                if not np.array_equal(
                    restored.mean_, selected.mean(axis=0)
                ) or not np.array_equal(
                    restored.scale_, selected.std(axis=0).clip(min=1e-6)
                ):
                    raise ValueError(
                        "model centering/scaling differs from assigned TRAIN population"
                    )
                return restored

            pools = []
            for index, inner in enumerate(fold["inner"]):
                metric = model(index, inner["train_ids"])
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
            metric = model("refit", fold["refit_ids"])
            for condition in CONDITIONS:
                fresh = score(
                    metric,
                    values[arm] if condition == "clean" else stressed[condition][arm],
                    data if condition == "clean" else validation,
                    fold["validation_ids"],
                    graphs["validation"],
                )
                frozen = _replay(
                    root, fresh, outer_index[number, arm, condition], threshold
                )
                rows.append(_row(number, arm, condition, frozen, threshold))
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
        "results_sha256": prior.digest(root / "results.json"),
        "summary": result["summary"],
    }
    prior.write_new(root / "verification.json", report)
    print(prior.canonical(report).decode(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    (run if args.stage == "run" else verify)(args)


if __name__ == "__main__":
    main()
