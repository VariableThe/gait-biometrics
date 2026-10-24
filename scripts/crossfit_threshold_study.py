"""DEV-only calibration-policy comparison; never load original CALIB or TEST."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
from scripts import secure_operating_study as secure

prior = secure.prior
SCORERS = ("mean_multi", "multi")
POLICIES = ("A", "B", "C", "D")
SOURCES = tuple(
    dict.fromkeys(
        (
            *secure.SOURCES,
            "scripts/crossfit_threshold_study.py",
            "engine/tail_calibration.py",
            "engine/calibration_selection.py",
            "docs/crossfit_threshold_study.md",
            "docs/tail_calibration.md",
            "tests/test_crossfit_threshold_study.py",
            "tests/test_tail_calibration.py",
            "tests/test_calibration_selection.py",
        )
    )
)


def folds():
    previous, result = prior.development_folds(), []
    for index, fold in enumerate(previous):
        partition, position = divmod(index, 5)
        validation = fold["val_subject_ids"]
        inner = []
        for offset in range(1, 5):
            held = previous[partition * 5 + (position + offset) % 5]["val_subject_ids"]
            inner.append(
                {
                    "held_ids": held,
                    "train_ids": [
                        s for s in prior.DEV_IDS if s not in (*validation, *held)
                    ],
                }
            )
        result.append(
            {
                "fold": index,
                "validation_ids": validation,
                "inner": inner,
                "refit_ids": [s for s in prior.DEV_IDS if s not in validation],
            }
        )
    return result


def plan():
    return {
        "protocol": "santos_dev_only_crossfit_thresholds_v2",
        "prior_test_inspection": True,
        "candidate_expansion_after_prior_dev_inspection": True,
        "payload_allowlist": list(prior.DEV_IDS),
        "original_calibration_and_test": "never opened by this runner",
        "folds": folds(),
        "scorers": list(SCORERS),
        "policies": list(POLICIES),
        "representation": "fixed lagcov362, automatic WCCN; two existing enrollment scorers",
        "A": "empirical FAR<=1% from next held triple; apply its nine-person fit",
        "B": "empirical FAR<=1% pooled four held triples; apply same nine-person fit as A",
        "C": "same pooled empirical threshold; refit on twelve nonvalidation people",
        "D": "pooled impostor order-statistic threshold alpha=.01 delta=.1; apply twelve-person refit; nominal iid diagnostic only",
        "nominal_assumptions": "claims are dependent and pooled scores use different models; no NP or CV+/jackknife+ guarantee applies",
        "coverage": "exact source-derived trial graphs registered before features/fits; up to120 genuine/240 impostor and24 directed pairs per pool; all scoring exclusions fail against the registered graphs",
        "budget": "100 fresh covariance fits: 20 outer x (4 inner nine-person +1 twelve-person); no reuse or neural/GPU fits",
        "promotion": "all rates/gates/ties use exact fractions of raw counts; blend only; mean validation FAR<=.01; split fraction FAR>.02 no worse than A; mean FRR strictly below A; at least10/15 people have mean FRR no worse; ties minFRR thenFAR then fixed policy order; else retain A",
        "discarded_v1": "100 completed CPU fits excluded: floating-mean ties could promote and trial coverage was assumed; all states and source retained separately",
        "normalization_gate": "diagnostic only: whether top two directed pairs contribute>=half of A/blend false accepts; no normalization arm in this registration",
        "uncertainty": "descriptive dependent splits and claims; no population CI, significance or posterior probability",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def expected_graph(data, ids):
    """Use authentic source labels/trial presence, before features or models."""
    identities = sorted(map(int, ids))
    labels, claims, probes, trials, enrollment, probe_trials = [], [], [], [], {}, {}
    for subject in identities:
        for day, target in ((1, enrollment), (2, probe_trials)):
            mask = (data["subjects"] == subject) & (data["days"] == day)
            available = sorted(set(map(int, data["trials"][mask])))
            if not available:
                raise ValueError("every registered identity needs both source days")
            target[str(subject)] = available
        for trial in probe_trials[str(subject)]:
            for claim in identities:
                labels.append(int(claim == subject))
                claims.append(claim)
                probes.append(subject)
                trials.append(trial)
    return {
        "labels": labels,
        "claim_subject_ids": claims,
        "probe_subject_ids": probes,
        "probe_trial_ids": trials,
        "enrollment_trials": enrollment,
        "probe_trials": probe_trials,
        "n_genuine": sum(labels),
        "n_impostor": len(labels) - sum(labels),
    }


def pooled_graph(graphs):
    keys = ("labels", "claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
    result = {key: [value for graph in graphs for value in graph[key]] for key in keys}
    result.update(
        n_genuine=sum(result["labels"]),
        n_impostor=len(result["labels"]) - sum(result["labels"]),
    )
    pairs = sorted(
        {
            f"{claim}<-{probe}"
            for claim, probe in zip(
                result["claim_subject_ids"], result["probe_subject_ids"], strict=True
            )
            if claim != probe
        }
    )
    result["directed_pairs"] = pairs
    if len(pairs) != 24:
        raise ValueError("four disjoint held triples must retain24 directed pairs")
    return result


def coverage_plan(data):
    from engine.tail_calibration import nominal_violation_bound

    result = {}
    for fold in folds():
        inner = [expected_graph(data, item["held_ids"]) for item in fold["inner"]]
        pooled = pooled_graph(inner)
        if nominal_violation_bound(pooled["n_impostor"], alpha=0.01) > 0.1:
            raise ValueError("source coverage cannot support registered nominal rank")
        result[str(fold["fold"])] = {
            "inner": inner,
            "pool": pooled,
            "validation": expected_graph(data, fold["validation_ids"]),
        }
    return result


def validate_graph(scored, expected):
    prior.same(
        np.asarray(scored["labels"]).tolist(),
        expected["labels"],
        "registered graph labels",
    )
    for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids"):
        prior.same(
            scored["metadata"][key], expected[key], "registered source graph " + key
        )
    if "enrollment_trials" in expected:
        actual = {
            str(row["subject"]): row["enrollment_trial_ids"]
            for row in scored["metadata"]["templates"]
        }
        prior.same(
            actual, expected["enrollment_trials"], "registered source enrollment"
        )


def checked_score(metric, values, data, ids, scorer, expected):
    scored = secure._score(metric, values, data, ids, scorer)
    validate_graph(scored, expected)
    return scored


def _pool(scored):
    metadata = {
        key: [value for item in scored for value in item["metadata"][key]]
        for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
    }
    return {
        "labels": np.concatenate([item["labels"] for item in scored]),
        "scores": np.concatenate([item["scores"] for item in scored]),
        "metadata": metadata,
    }


def thresholds(calibration, pooled):
    from engine.tail_calibration import nominal_order_statistic_threshold

    a = prior.calibrate(calibration["labels"], calibration["scores"])
    b = prior.calibrate(pooled["labels"], pooled["scores"])
    nominal = nominal_order_statistic_threshold(
        pooled["scores"][pooled["labels"] == 0], alpha=0.01, delta=0.1
    )
    if nominal["threshold"] is None:
        raise ValueError(
            "registered nominal threshold is infeasible; preserve failed run"
        )
    return {"A": a, "B": b, "C": b, "D": nominal}


def pair_errors(scored, threshold):
    labels, scores, meta = scored["labels"], scored["scores"], scored["metadata"]
    result = {}
    for claim, probe, label, score in zip(
        meta["claim_subject_ids"],
        meta["probe_subject_ids"],
        labels,
        scores,
        strict=True,
    ):
        if label:
            continue
        key = f"{claim}<-{probe}"
        row = result.setdefault(key, {"claims": 0, "false_accepts": 0})
        row["claims"] += 1
        row["false_accepts"] += int(score >= threshold)
    return result


def summarize(rows):
    from engine.calibration_selection import summarize_calibration

    return summarize_calibration(
        rows, dev_ids=prior.DEV_IDS, scorers=SCORERS, policies=POLICIES
    )


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
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn")
            },
            "python": platform.python_version(),
        },
    )
    values = secure._features(data)
    prior.save_npz(root / "features.npz", {"values": values})
    allrows, outer_artifacts, fit_calls = [], [], 0
    for fold in folds():
        graphs = coverage[str(fold["fold"])]
        inner_models, models = [], []
        for index, inner in enumerate(fold["inner"]):
            metric, cohort = secure._fit(values, data, inner["train_ids"], "auto")
            fit_calls += 1
            path = root / f"fold_{fold['fold']:02d}_inner_{index}.npz"
            prior.save_model(path, metric, cohort)
            models.append(prior._artifact(root, path))
            inner_models.append(metric)
        refit, cohort = secure._fit(values, data, fold["refit_ids"], "auto")
        fit_calls += 1
        path = root / f"fold_{fold['fold']:02d}_refit.npz"
        prior.save_model(path, refit, cohort)
        outer_artifacts.append(
            {
                "fold": fold["fold"],
                "inner": models,
                "refit": prior._artifact(root, path),
            }
        )
        for scorer in SCORERS:
            calibration = [
                checked_score(metric, values, data, inner["held_ids"], scorer, graph)
                for metric, inner, graph in zip(
                    inner_models, fold["inner"], graphs["inner"], strict=True
                )
            ]
            pooled = _pool(calibration)
            validate_graph(pooled, graphs["pool"])
            path = root / f"fold_{fold['fold']:02d}_{scorer}_pool.npz"
            prior.save_npz(path, {key: pooled[key] for key in ("labels", "scores")})
            policies = thresholds(calibration[0], pooled)
            for policy in POLICIES:
                model = inner_models[0] if policy in ("A", "B") else refit
                scored = checked_score(
                    model,
                    values,
                    data,
                    fold["validation_ids"],
                    scorer,
                    graphs["validation"],
                )
                path = root / f"fold_{fold['fold']:02d}_{scorer}_{policy}.npz"
                threshold = policies[policy]["threshold"]
                row = {
                    "fold": fold["fold"],
                    "scorer": scorer,
                    "policy": policy,
                    "threshold": policies[policy],
                    "pool": prior._artifact(
                        root, root / f"fold_{fold['fold']:02d}_{scorer}_pool.npz"
                    ),
                    "pool_metadata": pooled["metadata"],
                    "pair_errors": pair_errors(scored, threshold),
                    **secure._scored_artifact(root, path, scored, threshold),
                }
                allrows.append(row)
        print(f"completed outer split {fold['fold'] + 1}/20", flush=True)
    if fit_calls != 100:
        raise ValueError("registered fit budget differs")
    result = {
        "fit_calls": fit_calls,
        "registration_sha256": prior.digest(root / "registration.json"),
        "features": prior._artifact(root, root / "features.npz"),
        "outer_models": outer_artifacts,
        "rows": allrows,
        "selection": summarize(allrows),
    }
    prior.write_new(root / "results.json", result)
    return result["selection"]


def verify(args):
    root = Path(args.out_root)
    registration = prior.read(root / "registration.json")
    prior.same(registration["plan"], plan(), "registered calibration plan")
    prior.same(registration["sources"], hashes(), "registered calibration source")
    prior.same(
        registration["archive"],
        prior._previous().audit_archive(args.archive_path),
        "official archive",
    )
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    prior.same(prior._manifest(data), registration["development"], "registered DEV")
    coverage = coverage_plan(data)
    prior.same(coverage, registration["coverage"], "registered source coverage")
    result = prior.read(root / "results.json")
    prior.same(
        result["registration_sha256"],
        prior.digest(root / "registration.json"),
        "registration artifact",
    )
    values = prior.load_npz(prior._verify_artifact(root, result["features"]))["values"]
    if not np.array_equal(values, secure._features(data)):
        raise ValueError("DEV feature cache differs")
    prior.same(summarize(result["rows"]), result["selection"], "complete selection")
    if [r["fold"] for r in result["outer_models"]] != list(range(20)):
        raise ValueError("complete ordered model inventory required")
    prior.same(
        result["fit_calls"],
        sum(1 + len(row["inner"]) for row in result["outer_models"]),
        "actual fit/model budget",
    )
    prior.same(result["fit_calls"], 100, "registered fit budget")
    for fold, artifacts in zip(folds(), result["outer_models"], strict=True):
        graphs = coverage[str(fold["fold"])]
        if len(artifacts["inner"]) != 4:
            raise ValueError("four inner models required")
        inner_models = []
        for inner, artifact in zip(fold["inner"], artifacts["inner"], strict=True):
            metric, cohort = prior.load_model(
                prior._verify_artifact(root, artifact), "control", inner["train_ids"]
            )
            prior.verify_cohort(metric, cohort, values, data, inner["train_ids"])
            inner_models.append(metric)
        refit, cohort = prior.load_model(
            prior._verify_artifact(root, artifacts["refit"]),
            "control",
            fold["refit_ids"],
        )
        prior.verify_cohort(refit, cohort, values, data, fold["refit_ids"])
        for scorer in SCORERS:
            calibration = [
                checked_score(metric, values, data, inner["held_ids"], scorer, graph)
                for metric, inner, graph in zip(
                    inner_models, fold["inner"], graphs["inner"], strict=True
                )
            ]
            pooled = _pool(calibration)
            validate_graph(pooled, graphs["pool"])
            group = [
                r
                for r in result["rows"]
                if r["fold"] == fold["fold"] and r["scorer"] == scorer
            ]
            pool_artifact = group[0]["pool"]
            cached = prior.load_npz(prior._verify_artifact(root, pool_artifact))
            prior._compare_scores(pooled, cached)
            size = len(graphs["inner"][0]["labels"])
            first_cached = {key: value[:size] for key, value in cached.items()}
            policies = thresholds(first_cached, cached)
            for row in group:
                prior.same(row["pool"], pool_artifact, "shared frozen pool")
                prior.same(
                    pooled["metadata"], row["pool_metadata"], "pooled claim metadata"
                )
                prior.same(
                    policies[row["policy"]], row["threshold"], "pooled threshold policy"
                )
                metric = inner_models[0] if row["policy"] in ("A", "B") else refit
                scored = checked_score(
                    metric,
                    values,
                    data,
                    fold["validation_ids"],
                    scorer,
                    graphs["validation"],
                )
                saved_outer = secure._replay(
                    scored, row, root, row["threshold"]["threshold"]
                )
                prior.same(
                    pair_errors(
                        {**saved_outer, "metadata": row["metadata"]},
                        row["threshold"]["threshold"],
                    ),
                    row["pair_errors"],
                    "directed-pair errors",
                )
    validation = {
        "verified": True,
        "registration_sha256": prior.digest(root / "registration.json"),
        "results_sha256": prior.digest(root / "results.json"),
        "fits": result["fit_calls"],
        "comparisons": 160,
        "payload_population": "DEV only",
        "selection": result["selection"],
    }
    prior.write_new(root / "validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    args = parser.parse_args()
    print(prior.canonical({"run": run, "verify": verify}[args.stage](args)).decode())


if __name__ == "__main__":
    main()
