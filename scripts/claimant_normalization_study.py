"""Registered DEV-only comparison of raw, claimant Z and symmetric S scores."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np

from engine.calibration_selection import _validate_metrics
from engine.claimant_normalization import normalize_claimant_scores
from engine.tail_calibration import empirical_threshold
from scripts import crossfit_threshold_study as crossfit

prior, secure = crossfit.prior, crossfit.secure
ARMS = ("raw", "z", "s")
SOURCES = tuple(
    dict.fromkeys(
        (
            *crossfit.SOURCES,
            "engine/claimant_normalization.py",
            "scripts/claimant_normalization_study.py",
            "tests/test_claimant_normalization.py",
            "tests/test_claimant_normalization_study.py",
            "docs/claimant_normalization.md",
        )
    )
)


def plan():
    return {
        "protocol": "santos_dev_only_claimant_normalization_v1",
        "prior_test_inspection": True,
        "candidate_expansion_after_prior_dev_inspection": True,
        "payload_allowlist": list(prior.DEV_IDS),
        "original_calibration_and_test": "never opened by this runner",
        "folds": crossfit.folds(),
        "arms": list(ARMS),
        "fixed_representation": "lagcov362; automatic WCCN; equal mean/two-medoid score blend",
        "fixed_threshold_policy": "C: empirical FAR<=1% on four held-triple score pools, then twelve-person metric refit",
        "normalization": "raw; Z from TRAIN day2 trial probes against claimant day1 blend; S averages Z with probe against TRAIN day1 blends",
        "std_floor": 0.001,
        "cohort": "exact metric-training identities only, both days; disjoint from every scored identity; refitted with each metric",
        "dependence": "cohort people also fit WCCN; pooled models differ; no iid, NP, CV+, statistical significance or population guarantees",
        "coverage": "exact source-derived day1 templates/day2 trial claim graphs registered before features or fitting; every arm retains all registered claims",
        "cohort_coverage": "every inner/refit TRAIN day1/day2 trial inventory registered before fitting; transformation cancellation cannot remove trials",
        "replay": "score allclose atol=rtol=1e-6, exact metadata, runtime versions and identical fresh/cached decisions at each frozen threshold",
        "promotion": "exact equal-fold mean FAR<=1%; no more folds FAR>2% than raw; strictly lower mean FRR; at least10/15 people no worse; tie by FRR then FAR then fixed arm order; otherwise retain raw",
        "budget": "100 fresh CPU covariance fits: 20 outer x (4 nine-person inner +1 twelve-person refit), shared across three arms; 60 outer comparisons; no neural/GPU fits",
        "interpretation": "exploratory, overlapping folds; a nomination only, with no new original CALIB/TEST threshold or deployment change",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def runtime():
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn")
        },
        "python": platform.python_version(),
    }


def coverage_plan(data):
    # Policy C uses an empirical threshold. The older study's nominal-rank
    # feasibility requirement belongs exclusively to its policy D.
    result = {}
    for fold in crossfit.folds():
        inner = [
            crossfit.expected_graph(data, item["held_ids"]) for item in fold["inner"]
        ]
        result[str(fold["fold"])] = {
            "inner": inner,
            "pool": crossfit.pooled_graph(inner),
            "validation": crossfit.expected_graph(data, fold["validation_ids"]),
            "cohort_inner": [
                crossfit.expected_graph(data, item["train_ids"])
                for item in fold["inner"]
            ],
            "cohort_refit": crossfit.expected_graph(data, fold["refit_ids"]),
        }
    return result


def summarize(rows):
    expected = {(fold, arm) for fold in range(20) for arm in ARMS}
    actual = [(row["fold"], row["arm"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("complete unique20 x3 normalization comparisons required")
    dev = set(map(str, prior.DEV_IDS))
    parsed = {(row["fold"], row["arm"]): _validate_metrics(row, dev) for row in rows}
    for fold in range(20):
        if len({parsed[fold, arm].graph for arm in ARMS}) != 1:
            raise ValueError("all normalization arms must share each claim graph")
    summaries, exact = {}, {}
    for arm in ARMS:
        group = [parsed[fold, arm] for fold in range(20)]
        people = {subject: [] for subject in dev}
        for row in group:
            for subject, value in row.people.items():
                people[subject].append(value)
        if any(len(values) != 4 for values in people.values()):
            raise ValueError("each DEV person must validate four times")
        far = sum((r.far for r in group), Fraction(0)) / 20
        frr = sum((r.frr for r in group), Fraction(0)) / 20
        person = {s: sum(values, Fraction(0)) / 4 for s, values in people.items()}
        exact[arm] = far, frr, person
        summaries[arm] = {
            "far": float(far),
            "frr": float(frr),
            "far_exact": str(far),
            "frr_exact": str(frr),
            "splits_far_above_2pct": sum(r.far > Fraction(1, 50) for r in group),
            "per_person_mean_frr": {s: float(v) for s, v in sorted(person.items())},
            "per_person_mean_frr_exact": {s: str(v) for s, v in sorted(person.items())},
        }
    eligible = []
    for arm in ARMS:
        far, frr, people = exact[arm]
        no_worse = sum(people[s] <= exact["raw"][2][s] for s in dev)
        summaries[arm]["people_no_worse_than_raw"] = no_worse
        if (
            arm != "raw"
            and far <= Fraction(1, 100)
            and summaries[arm]["splits_far_above_2pct"]
            <= summaries["raw"]["splits_far_above_2pct"]
            and frr < exact["raw"][1]
            and no_worse >= 10
        ):
            eligible.append(arm)
    winner = (
        min(eligible, key=lambda arm: (exact[arm][1], exact[arm][0], ARMS.index(arm)))
        if eligible
        else "raw"
    )
    return {
        "winner": winner,
        "eligible": eligible,
        "summaries": summaries,
        "interpretation": plan()["interpretation"],
    }


def score(metric, cohort, values, data, ids, arm, graph, cohort_graph):
    mask = np.isin(data["subjects"], ids)
    if set(data["subjects"][mask]) != set(ids):
        raise ValueError("all requested scoring people must retain data")
    labels, scores, metadata = normalize_claimant_scores(
        metric.transform(values[mask]),
        *(data[key][mask] for key in ("subjects", "days", "trials")),
        normalization=arm,
        std_floor=0.001,
        **{f"cohort_{key}": value for key, value in cohort.items()},
    )
    scored = {"labels": labels, "scores": scores, "metadata": metadata}
    crossfit.validate_graph(scored, graph)
    if arm != "raw":
        for day, key in ((1, "enrollment_trials"), (2, "probe_trials")):
            actual = {
                str(row["subject"]): row["days"][str(day)]["eligible_trials"]
                for row in metadata["cohort_coverage"]["per_subject"]
            }
            prior.same(actual, cohort_graph[key], "registered TRAIN cohort trials")
    return scored


def _store(root, stem, scored):
    path = root / f"{stem}.npz"
    prior.save_npz(path, {key: scored[key] for key in ("labels", "scores")})
    return {"scores": prior._artifact(root, path), "metadata": scored["metadata"]}


def _replay(root, fresh, frozen):
    cached = prior.load_npz(prior._verify_artifact(root, frozen["scores"]))
    prior._compare_scores(fresh, cached)
    prior.same(fresh["metadata"], frozen["metadata"], "normalization metadata")
    return {**cached, "metadata": frozen["metadata"]}


def _same_decisions(fresh, cached, threshold):
    if not np.array_equal(
        np.asarray(fresh["scores"], dtype=np.float64) >= threshold,
        np.asarray(cached["scores"], dtype=np.float64) >= threshold,
    ):
        raise ValueError("replay changed decisions at the frozen threshold")


def run(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh empty study root required")
    archive, sources = prior._previous().audit_archive(args.archive_path), hashes()
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    coverage = coverage_plan(data)
    prior.write_new(
        root / "registration.json",
        {
            "plan": plan(),
            "sources": sources,
            "archive": archive,
            "development": prior._manifest(data),
            "coverage": coverage,
            **runtime(),
        },
    )
    values = secure._features(data)
    prior.save_npz(root / "features.npz", {"values": values})
    rows, models, fit_calls = [], [], 0
    for fold in crossfit.folds():
        number = fold["fold"]
        graphs = coverage[str(number)]
        inner_models, artifacts = [], []
        for index, inner in enumerate(fold["inner"]):
            fitted = secure._fit(values, data, inner["train_ids"], "auto")
            fit_calls += 1
            path = root / f"fold_{number:02d}_inner_{index}.npz"
            prior.save_model(path, *fitted)
            artifacts.append(prior._artifact(root, path))
            inner_models.append(fitted)
        refit = secure._fit(values, data, fold["refit_ids"], "auto")
        fit_calls += 1
        path = root / f"fold_{number:02d}_refit.npz"
        prior.save_model(path, *refit)
        models.append(
            {"fold": number, "inner": artifacts, "refit": prior._artifact(root, path)}
        )
        for arm in ARMS:
            inner_scores = [
                score(
                    *fitted, values, data, inner["held_ids"], arm, graph, cohort_graph
                )
                for fitted, inner, graph, cohort_graph in zip(
                    inner_models,
                    fold["inner"],
                    graphs["inner"],
                    graphs["cohort_inner"],
                    strict=True,
                )
            ]
            inner_artifacts = [
                _store(root, f"fold_{number:02d}_{arm}_inner_{i}", scored)
                for i, scored in enumerate(inner_scores)
            ]
            pooled = crossfit._pool(inner_scores)
            crossfit.validate_graph(pooled, graphs["pool"])
            threshold = empirical_threshold(pooled["labels"], pooled["scores"])
            scored = score(
                *refit,
                values,
                data,
                fold["validation_ids"],
                arm,
                graphs["validation"],
                graphs["cohort_refit"],
            )
            rows.append(
                {
                    "fold": number,
                    "arm": arm,
                    "inner": inner_artifacts,
                    "threshold": threshold,
                    "pair_errors": crossfit.pair_errors(scored, threshold["threshold"]),
                    **_store(root, f"fold_{number:02d}_{arm}_outer", scored),
                    "metrics": prior.report(scored, threshold["threshold"]),
                }
            )
        print(f"completed outer split {number + 1}/20", flush=True)
    if fit_calls != 100:
        raise ValueError("registered fit budget differs")
    result = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "features": prior._artifact(root, root / "features.npz"),
        "fit_calls": fit_calls,
        "models": models,
        "rows": rows,
        "selection": summarize(rows),
    }
    prior.write_new(root / "results.json", result)
    return result["selection"]


def verify(args):
    root = Path(args.out_root)
    reg, result = (
        prior.read(root / "registration.json"),
        prior.read(root / "results.json"),
    )
    prior.same(reg["plan"], plan(), "normalization plan")
    prior.same(reg["sources"], hashes(), "normalization sources")
    for key, expected in runtime().items():
        prior.same(reg[key], expected, "registered runtime " + key)
    prior.same(
        reg["archive"], prior._previous().audit_archive(args.archive_path), "archive"
    )
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    prior.same(prior._manifest(data), reg["development"], "DEV source manifest")
    prior.same(coverage_plan(data), reg["coverage"], "source claim graphs")
    prior.same(
        prior.digest(root / "registration.json"),
        result["registration_sha256"],
        "registration digest",
    )
    prior.same(
        summarize(result["rows"]), result["selection"], "normalization selection"
    )
    values = prior.load_npz(prior._verify_artifact(root, result["features"]))["values"]
    if not np.array_equal(values, secure._features(data)):
        raise ValueError("registered feature cache differs")
    prior.same(
        [m["fold"] for m in result["models"]], list(range(20)), "model inventory"
    )
    prior.same(result["fit_calls"], 100, "fit budget")
    if sum(len(m["inner"]) + 1 for m in result["models"]) != 100:
        raise ValueError("model inventory differs from registered budget")
    for fold, model_files in zip(crossfit.folds(), result["models"], strict=True):
        graphs = reg["coverage"][str(fold["fold"])]
        fitted = []
        if len(model_files["inner"]) != 4:
            raise ValueError("four inner models required")
        for inner, artifact in zip(fold["inner"], model_files["inner"], strict=True):
            model = prior.load_model(
                prior._verify_artifact(root, artifact), "control", inner["train_ids"]
            )
            prior.verify_cohort(*model, values, data, inner["train_ids"])
            fitted.append(model)
        refit = prior.load_model(
            prior._verify_artifact(root, model_files["refit"]),
            "control",
            fold["refit_ids"],
        )
        prior.verify_cohort(*refit, values, data, fold["refit_ids"])
        for arm in ARMS:
            row = next(
                r
                for r in result["rows"]
                if r["fold"] == fold["fold"] and r["arm"] == arm
            )
            if len(row["inner"]) != 4:
                raise ValueError("four inner score caches required")
            fresh_inner = [
                score(*model, values, data, inner["held_ids"], arm, graph, cohort_graph)
                for model, inner, graph, cohort_graph in zip(
                    fitted,
                    fold["inner"],
                    graphs["inner"],
                    graphs["cohort_inner"],
                    strict=True,
                )
            ]
            cached = [
                _replay(root, fresh, artifact)
                for fresh, artifact in zip(fresh_inner, row["inner"], strict=True)
            ]
            pooled = crossfit._pool(cached)
            crossfit.validate_graph(pooled, graphs["pool"])
            threshold = empirical_threshold(pooled["labels"], pooled["scores"])
            prior.same(threshold, row["threshold"], "frozen-cache threshold")
            for fresh, saved in zip(fresh_inner, cached, strict=True):
                _same_decisions(fresh, saved, threshold["threshold"])
            fresh = score(
                *refit,
                values,
                data,
                fold["validation_ids"],
                arm,
                graphs["validation"],
                graphs["cohort_refit"],
            )
            saved = _replay(root, fresh, row)
            _same_decisions(fresh, saved, threshold["threshold"])
            prior.same(
                prior.report(saved, threshold["threshold"]),
                row["metrics"],
                "cached metrics",
            )
            prior.same(
                crossfit.pair_errors(saved, threshold["threshold"]),
                row["pair_errors"],
                "directed-pair errors",
            )
    validation = {
        "verified": True,
        "fits": 100,
        "outer_comparisons": 60,
        "inner_comparisons": 240,
        "registration_sha256": prior.digest(root / "registration.json"),
        "results_sha256": prior.digest(root / "results.json"),
        "selection": result["selection"],
        "payload_population": "DEV only",
    }
    prior.write_new(root / "validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    args = parser.parse_args()
    print(prior.canonical(run(args) if args.stage == "run" else verify(args)).decode())


if __name__ == "__main__":
    main()
