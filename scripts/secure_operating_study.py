"""Bounded exploratory enrollment study with identity-disjoint inner calibration.

All selection uses DEV. Original CALIB is opened only after final models freeze;
historically inspected TEST is reserved once and remains exploratory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
from scripts import mentor_backend_study as prior

SHRINKAGES = ("auto", "0.2", "0.5")
SCORERS = ("multi", "mean", "mean_multi", "top2", "lse_0.1")
CANDIDATES = tuple(f"{alpha}/{method}" for alpha in SHRINKAGES for method in SCORERS)
CONTROL = "auto/multi"
SOURCES = tuple(
    dict.fromkeys(
        (
            *prior.SOURCES,
            "scripts/secure_operating_study.py",
            "engine/secure_enrollment.py",
        )
    )
)


def folds():
    """Each person validates four times, calibrates four times, fits twelve."""
    previous = prior.development_folds()
    result = []
    for index, fold in enumerate(previous):
        partition, position = divmod(index, 5)
        calibration = previous[5 * partition + (position + 1) % 5]["val_subject_ids"]
        validation = fold["val_subject_ids"]
        result.append(
            {
                "fold": index,
                "train_subject_ids": [
                    s for s in prior.DEV_IDS if s not in (*calibration, *validation)
                ],
                "calibration_subject_ids": calibration,
                "val_subject_ids": validation,
            }
        )
    return result


def plan():
    return {
        "protocol": "santos_secure_operating_exploratory_v1",
        "prior_test_inspection": True,
        "candidate_expansion_after_prior_dev_inspection": True,
        "dev_ids": list(prior.DEV_IDS),
        "calibration_ids": list(prior.CALIBRATION_IDS),
        "test_ids": list(prior.TEST_IDS),
        "folds": folds(),
        "candidates": list(CANDIDATES),
        "control": CONTROL,
        "features": "fixed 362D six-axis fusion plus lagcov [0,5,10,20]",
        "fit": "identity-balanced WCCN; only the nine inner training identities",
        "inner_threshold": "empirical FAR<=1% on three separate DEV calibration people",
        "selection": "eligible mean validation FAR<=1%; minimum mean validation FRR, then FAR, EER, fixed order; if none eligible, minimum FAR then FRR then EER",
        "coverage": "same exhaustive day1 enrollment/day2 trial claims for all candidates",
        "budget": "20 splits x 3 metrics = 60 fits; 300 candidate comparisons; no neural/GPU training",
        "final_fit": "selected and control on all 15 DEV; separate fixed CALIB thresholds",
        "limits": "three-person inner calibration has only 60 impostor claims; FAR<=1% means zero false accepts there; overlapping splits and all tail rates descriptive only",
    }


def hashes():
    return {name: prior.digest(prior.ROOT / name) for name in SOURCES}


def _features(data):
    return prior.features(data)["lagcov"]


def _fit(values, data, ids, alpha):
    from models.gait_metric import WithinClassMetric

    if (
        alpha not in SHRINKAGES
        or not ids
        or len(set(ids)) != len(ids)
        or not set(ids).issubset(prior.DEV_IDS)
    ):
        raise ValueError(
            "only registered shrinkage and unique DEV training people allowed"
        )
    mask = np.isin(data["subjects"], ids)
    if set(data["subjects"][mask].tolist()) != set(ids):
        raise ValueError("all fit identities must have data")
    metric = WithinClassMetric("wccn", "auto" if alpha == "auto" else float(alpha)).fit(
        values[mask], data["subjects"][mask]
    )
    cohort = {key: data[key][mask] for key in ("subjects", "days", "trials")}
    cohort["embeddings"] = metric.transform(values[mask])
    return metric, cohort


def _score(metric, values, data, ids, method):
    from engine.secure_enrollment import score_secure_trials

    mask = np.isin(data["subjects"], ids)
    if set(data["subjects"][mask].tolist()) != set(ids):
        raise ValueError("all scoring identities must have data")
    labels, scores, metadata = score_secure_trials(
        metric.transform(values[mask]),
        *(data[key][mask] for key in ("subjects", "days", "trials")),
        method=method,
    )
    return {"labels": labels, "scores": scores, "metadata": metadata}


def selection(rows):
    expected = {(i, candidate) for i in range(20) for candidate in CANDIDATES}
    actual = [(row["fold"], row["candidate"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("complete unique 20 x 15 candidate comparisons required")
    for fold in range(20):
        group = [r for r in rows if r["fold"] == fold]
        for role in ("inner_calibration", "validation"):
            prior.matched_graphs([r[role]["metrics"] for r in group])
    summaries = {}
    for candidate in CANDIDATES:
        values = [
            r["validation"]["metrics"] for r in rows if r["candidate"] == candidate
        ]
        summary = {
            name: float(np.mean([v["fixed_threshold"][name] for v in values]))
            for name in ("far", "frr")
        }
        summary["eer"] = float(np.mean([v["eer"] for v in values]))
        summary["mean_worst_person_frr"] = float(
            np.mean(
                [
                    max(s["frr"] for s in v["per_subject_errors"].values())
                    for v in values
                ]
            )
        )
        if not all(
            np.isfinite(value) and 0 <= value <= 1 for value in summary.values()
        ):
            raise ValueError("finite rates in [0,1] required")
        summaries[candidate] = summary
    eligible = [c for c in CANDIDATES if summaries[c]["far"] <= 0.01]
    order = ("frr", "far", "eer") if eligible else ("far", "frr", "eer")
    winner = min(
        eligible or CANDIDATES, key=lambda c: tuple(summaries[c][key] for key in order)
    )
    return {
        "winner": winner,
        "constraint_met": bool(eligible),
        "eligible_candidates": eligible,
        "summaries": summaries,
        "interpretation": "descriptive crossfitted operating points; overlapping DEV splits",
    }


def _scored_artifact(root, path, scored, threshold):
    prior.save_npz(path, {key: scored[key] for key in ("labels", "scores")})
    return {
        "scores": prior._artifact(root, path),
        "metadata": scored["metadata"],
        "metrics": prior.report(scored, threshold),
    }


def _registration(args):
    root = Path(args.out_root)
    reg = prior.read(root / "registration.json")
    prior.same(reg["plan"], plan(), "registered operating study")
    prior.same(reg["sources"], hashes(), "registered operating source")
    prior.same(
        reg["archive"],
        prior._previous().audit_archive(args.archive_path),
        "official archive",
    )
    return root, reg


def develop(args):
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
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn")
            },
        },
    )
    values = _features(data)
    prior.save_npz(root / "features.npz", {"values": values})
    rows = []
    for fold in folds():
        for alpha in SHRINKAGES:
            metric, cohort = _fit(values, data, fold["train_subject_ids"], alpha)
            modelpath = root / f"fold_{fold['fold']:02d}_{alpha}_model.npz"
            prior.save_model(modelpath, metric, cohort)
            for method in SCORERS:
                stem = f"fold_{fold['fold']:02d}_{alpha}_{method}"
                cal = _score(
                    metric, values, data, fold["calibration_subject_ids"], method
                )
                threshold = prior.calibrate(cal["labels"], cal["scores"])
                val = _score(metric, values, data, fold["val_subject_ids"], method)
                row = {
                    "fold": fold["fold"],
                    "candidate": f"{alpha}/{method}",
                    "model": prior._artifact(root, modelpath),
                    "threshold": threshold,
                    "inner_calibration": _scored_artifact(
                        root, root / f"{stem}_cal.npz", cal, threshold["threshold"]
                    ),
                    "validation": _scored_artifact(
                        root, root / f"{stem}_val.npz", val, threshold["threshold"]
                    ),
                }
                rows.append(row)
        print(f"completed split {fold['fold'] + 1}/20", flush=True)
    result = {
        "registration_sha256": prior.digest(root / "registration.json"),
        "features": prior._artifact(root, root / "features.npz"),
        "rows": rows,
        "selection": selection(rows),
    }
    prior.write_new(root / "development.json", result)
    return result["selection"]


def _replay(scored, frozen, root, threshold):
    saved = prior.load_npz(prior._verify_artifact(root, frozen["scores"]))
    prior._compare_scores(scored, saved)
    prior.same(scored["metadata"], frozen["metadata"], "claim metadata")
    prior.same(
        prior.report({**saved, "metadata": frozen["metadata"]}, threshold),
        frozen["metrics"],
        "cached operating metrics",
    )
    return saved


def verify(args):
    root, reg = _registration(args)
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    prior.same(prior._manifest(data), reg["development"], "registered DEV data")
    evidence = prior.read(root / "development.json")
    prior.same(
        evidence["registration_sha256"],
        prior.digest(root / "registration.json"),
        "registered artifact",
    )
    values = prior.load_npz(prior._verify_artifact(root, evidence["features"]))[
        "values"
    ]
    if not np.array_equal(values, _features(data)):
        raise ValueError("DEV feature cache mismatch")
    prior.same(selection(evidence["rows"]), evidence["selection"], "complete selection")
    models = {}
    for row in evidence["rows"]:
        fold = folds()[row["fold"]]
        alpha, method = row["candidate"].split("/")
        key = (fold["fold"], alpha)
        if key not in models:
            path = prior._verify_artifact(root, row["model"])
            metric, cohort = prior.load_model(
                path, "control", fold["train_subject_ids"]
            )
            prior.verify_cohort(metric, cohort, values, data, fold["train_subject_ids"])
            models[key] = metric, row["model"]
        metric, artifact = models[key]
        prior.same(artifact, row["model"], "shared fold metric")
        for role, ids in (
            ("inner_calibration", fold["calibration_subject_ids"]),
            ("validation", fold["val_subject_ids"]),
        ):
            scored = _score(metric, values, data, ids, method)
            saved = _replay(scored, row[role], root, row["threshold"]["threshold"])
            if role == "inner_calibration":
                prior.same(
                    prior.calibrate(saved["labels"], saved["scores"]),
                    row["threshold"],
                    "inner threshold",
                )
    return root, data, values, evidence


def lock(args):
    root = Path(args.out_root)
    if (root / "lock_attempt.json").exists() or (root / "lock.json").exists():
        raise ValueError("lock already attempted")
    root, data, values, evidence = verify(args)
    candidates = list(dict.fromkeys((evidence["selection"]["winner"], CONTROL)))
    models = {}
    for candidate in candidates:
        alpha, method = candidate.split("/")
        metric, cohort = _fit(values, data, list(prior.DEV_IDS), alpha)
        path = root / f"final_{alpha}_{method}.npz"
        prior.save_model(path, metric, cohort)
        models[candidate] = prior._artifact(root, path)
    prior.write_new(
        root / "lock_attempt.json",
        {
            "selection": evidence["selection"],
            "models": models,
            "status": "reserved before original CALIB load; failed attempts consumed",
        },
    )
    calibration = prior.load_data(args.archive_path, prior.CALIBRATION_IDS)
    calvalues, results = _features(calibration), {}
    for candidate in candidates:
        method = candidate.split("/")[1]
        metric, _ = prior.load_model(
            prior._verify_artifact(root, models[candidate]), "control", prior.DEV_IDS
        )
        scored = _score(metric, calvalues, calibration, prior.CALIBRATION_IDS, method)
        threshold = prior.calibrate(scored["labels"], scored["scores"])
        results[candidate] = {
            "threshold": threshold,
            **_scored_artifact(
                root,
                root / f"cal_{candidate.replace('/', '_')}.npz",
                scored,
                threshold["threshold"],
            ),
        }
    prior.matched_graphs([r["metrics"] for r in results.values()])
    result = {
        "selection": evidence["selection"],
        "models": models,
        "calibration": prior._manifest(calibration),
        "calibration_results": results,
        "registration_sha256": prior.digest(root / "registration.json"),
        "development_sha256": prior.digest(root / "development.json"),
        "prior_test_inspection": True,
    }
    result["self_hash"] = prior.object_hash(result)
    prior.write_new(root / "lock.json", result)
    return {
        "winner": evidence["selection"]["winner"],
        "self_hash": result["self_hash"],
        "calibration": {k: v["threshold"] for k, v in results.items()},
    }


def evaluate(args):
    root = Path(args.out_root)
    if (root / "test_attempt.json").exists() or (root / "test_result.json").exists():
        raise ValueError("TEST already attempted; no retry")
    frozen = prior.read(root / "lock.json")
    checksum = frozen.pop("self_hash")
    prior.same(prior.object_hash(frozen), checksum, "operating lock hash")
    root, data, values, evidence = verify(args)
    for name in ("registration", "development"):
        prior.same(
            prior.digest(root / f"{name}.json"),
            frozen[f"{name}_sha256"],
            f"locked {name}",
        )
    prior.same(evidence["selection"], frozen["selection"], "locked selection")
    candidates = list(dict.fromkeys((evidence["selection"]["winner"], CONTROL)))
    prior.same(sorted(frozen["models"]), sorted(candidates), "locked methods")
    prior.same(
        sorted(frozen["calibration_results"]),
        sorted(candidates),
        "locked calibration methods",
    )
    cal = prior.load_data(args.archive_path, prior.CALIBRATION_IDS)
    prior.same(prior._manifest(cal), frozen["calibration"], "locked CALIB data")
    calvalues, models = _features(cal), {}
    for candidate in candidates:
        metric, cohort = prior.load_model(
            prior._verify_artifact(root, frozen["models"][candidate]),
            "control",
            prior.DEV_IDS,
        )
        prior.verify_cohort(metric, cohort, values, data, prior.DEV_IDS)
        models[candidate] = metric
        entry = frozen["calibration_results"][candidate]
        scored = _score(
            metric, calvalues, cal, prior.CALIBRATION_IDS, candidate.split("/")[1]
        )
        saved = _replay(scored, entry, root, entry["threshold"]["threshold"])
        prior.same(
            prior.calibrate(saved["labels"], saved["scores"]),
            entry["threshold"],
            "final calibration threshold",
        )
    prior.write_new(
        root / "test_attempt.json",
        {
            "lock_hash": checksum,
            "prior_test_inspection": True,
            "status": "reserved before TEST load; consumed on failure",
        },
    )
    test = prior.load_data(args.archive_path, prior.TEST_IDS)
    testvalues, results = _features(test), {}
    for candidate, metric in models.items():
        scored = _score(
            metric, testvalues, test, prior.TEST_IDS, candidate.split("/")[1]
        )
        results[candidate] = _scored_artifact(
            root,
            root / f"test_{candidate.replace('/', '_')}.npz",
            scored,
            frozen["calibration_results"][candidate]["threshold"]["threshold"],
        )
    prior.matched_graphs([r["metrics"] for r in results.values()])
    result = {
        "exploratory": True,
        "prior_test_inspection": True,
        "selection": evidence["selection"],
        "lock_hash": checksum,
        "test": prior._manifest(test),
        "results": results,
    }
    result["self_hash"] = prior.object_hash(result)
    prior.write_new(root / "test_result.json", result)
    return {
        "winner": evidence["selection"]["winner"],
        "results": {k: v["metrics"] for k, v in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("develop", "lock", "evaluate"), required=True
    )
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    result = {"develop": develop, "lock": lock, "evaluate": evaluate}[args.stage](args)
    print(prior.canonical(result).decode())


if __name__ == "__main__":
    main()
