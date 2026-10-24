"""DEV-only calibration-policy comparison; never load original CALIB or TEST."""

from __future__ import annotations

import argparse
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
        "protocol": "santos_dev_only_crossfit_thresholds_v1",
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
        "coverage": "pooled calibration: 120 genuine/240 impostor,24 directed person pairs; no cross-triple claims",
        "budget": "100 fresh covariance fits: 20 outer x (4 inner nine-person +1 twelve-person); no reuse or neural/GPU fits",
        "promotion": "blend only; mean validation FAR<=.01; split fraction FAR>.02 no worse than A; mean FRR strictly below A; at least10/15 people have mean FRR no worse; ties minFRR thenFAR then fixed policy order; else retain A",
        "normalization_gate": "diagnostic only: whether top two directed pairs contribute>=half of A/blend false accepts; no normalization arm in this registration",
        "uncertainty": "descriptive dependent splits and claims; no population CI, significance or posterior probability",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


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
    expected = {
        (fold, scorer, policy)
        for fold in range(20)
        for scorer in SCORERS
        for policy in POLICIES
    }
    actual = [(r["fold"], r["scorer"], r["policy"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("complete unique20x2x4 comparisons required")
    for fold in range(20):
        prior.matched_graphs([r["metrics"] for r in rows if r["fold"] == fold])
    summary = {}
    for scorer in SCORERS:
        for policy in POLICIES:
            group = [r for r in rows if r["scorer"] == scorer and r["policy"] == policy]
            people, pairs = {}, {}
            for row in group:
                for subject, errors in row["metrics"]["per_subject_errors"].items():
                    people.setdefault(subject, []).append(errors["frr"])
                for pair, errors in row["pair_errors"].items():
                    total = pairs.setdefault(pair, {"claims": 0, "false_accepts": 0})
                    for key in total:
                        total[key] += errors[key]
            if set(map(int, people)) != set(prior.DEV_IDS) or any(
                len(v) != 4 for v in people.values()
            ):
                raise ValueError("every DEV person must validate exactly four times")
            fixed = [r["metrics"]["fixed_threshold"] for r in group]
            summary[f"{scorer}/{policy}"] = {
                "far": float(np.mean([m["far"] for m in fixed])),
                "frr": float(np.mean([m["frr"] for m in fixed])),
                "splits_far_above_2pct": sum(m["far"] > 0.02 for m in fixed),
                "per_person_mean_frr": {
                    s: float(np.mean(values)) for s, values in people.items()
                },
                "pair_errors": pairs,
            }
    baseline, eligible = summary["mean_multi/A"], []
    for policy in POLICIES:
        result = summary[f"mean_multi/{policy}"]
        no_worse = sum(
            result["per_person_mean_frr"][s]
            <= baseline["per_person_mean_frr"][s] + 1e-12
            for s in baseline["per_person_mean_frr"]
        )
        result["people_no_worse_than_A"] = no_worse
        if (
            policy != "A"
            and result["far"] <= 0.01
            and result["splits_far_above_2pct"] <= baseline["splits_far_above_2pct"]
            and result["frr"] < baseline["frr"]
            and no_worse >= 10
        ):
            eligible.append(policy)
    winner = (
        min(
            eligible,
            key=lambda p: (
                summary[f"mean_multi/{p}"]["frr"],
                summary[f"mean_multi/{p}"]["far"],
                POLICIES.index(p),
            ),
        )
        if eligible
        else "A"
    )
    counts = sorted(
        (v["false_accepts"] for v in baseline["pair_errors"].values()), reverse=True
    )
    total = sum(counts)
    return {
        "winner": winner,
        "promotion_gate_passed": bool(eligible),
        "eligible_policies": eligible,
        "summary": summary,
        "normalization_gate": {
            "false_accepts": total,
            "top_two_pairs_false_accepts": sum(counts[:2]),
            "triggered": bool(total and sum(counts[:2]) >= total / 2),
        },
        "interpretation": "DEV-only descriptive evidence; no new CALIB or TEST evaluation",
    }


def run(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh empty study root required")
    archive, sources = prior._previous().audit_archive(args.archive_path), hashes()
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    prior.write_new(
        root / "registration.json",
        {
            "plan": plan(),
            "sources": sources,
            "archive": archive,
            "development": prior._manifest(data),
        },
    )
    values = secure._features(data)
    prior.save_npz(root / "features.npz", {"values": values})
    allrows, outer_artifacts = [], []
    for fold in folds():
        inner_models, models = [], []
        for index, inner in enumerate(fold["inner"]):
            metric, cohort = secure._fit(values, data, inner["train_ids"], "auto")
            path = root / f"fold_{fold['fold']:02d}_inner_{index}.npz"
            prior.save_model(path, metric, cohort)
            models.append(prior._artifact(root, path))
            inner_models.append(metric)
        refit, cohort = secure._fit(values, data, fold["refit_ids"], "auto")
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
                secure._score(metric, values, data, inner["held_ids"], scorer)
                for metric, inner in zip(inner_models, fold["inner"], strict=True)
            ]
            pooled = _pool(calibration)
            path = root / f"fold_{fold['fold']:02d}_{scorer}_pool.npz"
            prior.save_npz(path, {key: pooled[key] for key in ("labels", "scores")})
            policies = thresholds(calibration[0], pooled)
            for policy in POLICIES:
                model = inner_models[0] if policy in ("A", "B") else refit
                scored = secure._score(
                    model, values, data, fold["validation_ids"], scorer
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
    result = {
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
    for fold, artifacts in zip(folds(), result["outer_models"], strict=True):
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
                secure._score(metric, values, data, inner["held_ids"], scorer)
                for metric, inner in zip(inner_models, fold["inner"], strict=True)
            ]
            pooled = _pool(calibration)
            policies = thresholds(calibration[0], pooled)
            for row in (
                r
                for r in result["rows"]
                if r["fold"] == fold["fold"] and r["scorer"] == scorer
            ):
                prior._compare_scores(
                    pooled, prior.load_npz(prior._verify_artifact(root, row["pool"]))
                )
                prior.same(
                    pooled["metadata"], row["pool_metadata"], "pooled claim metadata"
                )
                prior.same(
                    policies[row["policy"]], row["threshold"], "pooled threshold policy"
                )
                metric = inner_models[0] if row["policy"] in ("A", "B") else refit
                scored = secure._score(
                    metric, values, data, fold["validation_ids"], scorer
                )
                secure._replay(scored, row, root, row["threshold"]["threshold"])
                prior.same(
                    pair_errors(scored, row["threshold"]["threshold"]),
                    row["pair_errors"],
                    "directed-pair errors",
                )
    validation = {
        "verified": True,
        "registration_sha256": prior.digest(root / "registration.json"),
        "results_sha256": prior.digest(root / "results.json"),
        "fits": 100,
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
