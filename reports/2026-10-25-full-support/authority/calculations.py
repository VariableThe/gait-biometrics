"""Exact pure functions extracted from frozen kernel.py; no fitting imports."""
import hashlib
from fractions import Fraction
import common as c
import numpy as np
from engine.empirical import empirical_threshold
from engine.historical_metrics import report
from engine.robustness_selection import ranking_metrics, summarize_nuisance_study, CONDITIONS


def pool(items):
    return {
        "labels": np.concatenate([x["labels"] for x in items]),
        "scores": np.concatenate([x["scores"] for x in items]),
    }


def cutoff(items):
    values = pool(items)
    return empirical_threshold(values["labels"], values["scores"])


def row(fold, arm, condition, scored, threshold):
    labels, scores, metadata = scored["labels"], scored["scores"], scored["metadata"]
    pairs = {}
    for claim, probe, label, value in zip(
        metadata["claim_subject_ids"],
        metadata["probe_subject_ids"],
        labels,
        scores,
        strict=True,
    ):
        if not label:
            item = pairs.setdefault(
                f"{claim}<-{probe}", {"claims": 0, "false_accepts": 0}
            )
            item["claims"] += 1
            item["false_accepts"] += int(value >= threshold)
    return {
        "fold": fold,
        "arm": arm,
        "condition": condition,
        "metrics": report(scored, threshold),
        "pair_errors": pairs,
        "ranking": ranking_metrics(
            labels, scores, np.asarray(metadata["probe_subject_ids"], dtype=np.int64)
        ),
        "decision_sha256": hashlib.sha256(
            np.asarray(scores >= threshold, dtype=np.bool_).tobytes()
        ).hexdigest(),
    }


def descriptive_summaries(rows, metadata):
    """All-arm equal-fold/person/count diagnostics; no H nomination rule."""
    result = {}
    mean = lambda values: sum(values, Fraction()) / len(values)
    for arm in ("R", "H", "F"):
        result[arm] = {}
        for condition in CONDITIONS:
            group = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            fixed = [r["metrics"]["fixed_threshold"] for r in group]
            people, claimants = {}, {}
            for subject in metadata["source_metadata"]["subject_ids"]:
                s = str(subject)
                present = [r for r in group if s in r["metrics"]["per_subject_errors"]]
                c.require(
                    len(present) == 4, "four dependent person appearances required"
                )
                counts = [r["metrics"]["per_subject_errors"][s] for r in present]
                people[s] = {
                    "frr_exact": str(
                        mean(
                            [
                                Fraction(v["false_rejects"], v["n_genuine"])
                                for v in counts
                            ]
                        )
                    ),
                    "eer_exact": str(
                        mean(
                            [
                                Fraction(r["ranking"]["per_subject"][s]["eer_exact"])
                                for r in present
                            ]
                        )
                    ),
                    "auc_exact": str(
                        mean(
                            [
                                Fraction(r["ranking"]["per_subject"][s]["auc_exact"])
                                for r in present
                            ]
                        )
                    ),
                    "repeated_rejects": sum(v["false_rejects"] for v in counts),
                    "repeated_genuine_claims": sum(v["n_genuine"] for v in counts),
                }
                claimants[s] = {
                    "repeated_false_accepts": sum(v["false_accepts"] for v in counts),
                    "repeated_impostor_claims": sum(v["n_impostor"] for v in counts),
                }
            result[arm][condition] = {
                "far_exact": str(
                    mean([Fraction(v["false_accepts"], v["n_impostor"]) for v in fixed])
                ),
                "frr_exact": str(
                    mean([Fraction(v["false_rejects"], v["n_genuine"]) for v in fixed])
                ),
                "eer_exact": str(
                    mean([Fraction(r["ranking"]["eer_exact"]) for r in group])
                ),
                "auc_exact": str(
                    mean([Fraction(r["ranking"]["auc_exact"]) for r in group])
                ),
                "splits_far_above_2pct": sum(
                    Fraction(v["false_accepts"], v["n_impostor"]) > Fraction(1, 50)
                    for v in fixed
                ),
                "repeated_false_accepts": sum(v["false_accepts"] for v in fixed),
                "repeated_false_rejects": sum(v["false_rejects"] for v in fixed),
                "repeated_genuine_claims": sum(v["n_genuine"] for v in fixed),
                "repeated_impostor_claims": sum(v["n_impostor"] for v in fixed),
                "per_person": people,
                "per_claimant": claimants,
            }
    return result


def exact_difference(left, right):
    if type(left) is dict:
        c.require(left.keys() == right.keys(), "matched diagnostic fields required")
        return {key: exact_difference(left[key], right[key]) for key in left}
    if type(left) is str:
        return str(Fraction(left) - Fraction(right))
    c.require(type(left) is type(right) is int, "exact diagnostic counts required")
    return left - right


def comparison(rows, metadata):
    for arm in ("R", "H", "F"):
        group = [r for r in rows if r["arm"] == arm]
        c.require(
            len(group) == 140
            and {(r["fold"], r["condition"]) for r in group}
            == {(f, condition) for f in range(20) for condition in CONDITIONS},
            "complete20x7 arm rows required",
        )
        for condition in CONDITIONS:
            fixed = [
                r["metrics"]["fixed_threshold"]
                for r in group
                if r["condition"] == condition
            ]
            c.require(
                sum(v["n_genuine"] for v in fixed) == 596
                and sum(v["n_impostor"] for v in fixed) == 1192,
                "no partial-success claim denominator",
            )

    def rates(arm, condition):
        group = [
            r["metrics"]["fixed_threshold"]
            for r in rows
            if r["arm"] == arm and r["condition"] == condition
        ]
        return tuple(
            sum((Fraction(v[k], v[n]) for v in group), Fraction()) / 20
            for k, n in (
                ("false_accepts", "n_impostor"),
                ("false_rejects", "n_genuine"),
            )
        )

    far_h, frr_h = rates("H", "clean")
    far_f, frr_f = rates("F", "clean")
    remapped = [
        {**r, "arm": "baseline" if r["arm"] == "R" else "nuisance"}
        for r in rows
        if r["arm"] in ("R", "F")
    ]
    replacement = summarize_nuisance_study(
        remapped,
        dev_ids=metadata["source_metadata"]["subject_ids"],
        folds=metadata["folds"],
    )
    c.require(len(replacement["gates"]) == 33, "authoritative gate inventory changed")
    summaries = descriptive_summaries(rows, metadata)
    secondary = {
        arm + "-R": exact_difference(summaries[arm], summaries["R"])
        for arm in ("H", "F")
    }
    return {
        "primary": {
            "F_minus_H_clean_equalfold_FRR_exact": str(frr_f - frr_h),
            "F_clean_FAR_exact": str(far_f),
            "H_clean_FAR_exact": str(far_h),
            "favors_support": frr_f < frr_h
            and far_f <= far_h
            and far_f <= Fraction(1, 100),
        },
        "replacement": replacement,
        "nominee": "F" if replacement["winner"] == "nuisance" else "R",
        "arm_summaries": summaries,
        "secondary": secondary,
        "H_nomination_permitted": False,
    }
