"""Exact descriptive gates for the predeclared nuisance covariance study.

Counts and tied-score ROC ranks are authoritative. The caller must replay them
against registered source graphs; this module reads no files and fits no model.
Repeated people and folds are dependent, so these are not population guarantees.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from fractions import Fraction

import numpy as np

from engine.calibration_selection import _integer, _subject, _validate_metrics

ARMS = ("baseline", "nuisance")
CONDITIONS = (
    "clean",
    "rotation30",
    "rotation90",
    "gain09",
    "gain11",
    "noise01",
    "noise05",
)
ROBUST_CONDITIONS = ("gain09", "gain11", "noise01")


def _rank_values(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (
        labels.ndim != 1
        or not labels.size
        or scores.shape != labels.shape
        or labels.dtype.kind not in "biu"
        or scores.dtype.kind not in "iuf"
        or not np.isin(labels, (0, 1)).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("finite real scores and both binary classes required")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = scores.astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("finite real scores and both binary classes required")
    return labels, scores


def _ranking(labels, scores):
    """Rational interpolation on identical tied-score ROC endpoints to prior.roc."""
    order = np.argsort(-scores, kind="stable")
    values, positives = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    tp, fp = np.cumsum(positives)[ends], np.cumsum(~positives)[ends]
    ng, ni = int(np.sum(positives)), int(np.sum(~positives))
    far = [Fraction(0), *(Fraction(int(v), ni) for v in fp)]
    frr = [Fraction(1), *(Fraction(ng - int(v), ng) for v in tp)]
    auc = sum(
        (
            (b - a) * (2 - x - y) / 2
            for a, b, x, y in zip(far[:-1], far[1:], frr[:-1], frr[1:], strict=True)
        ),
        Fraction(),
    )
    for index in range(1, len(far)):
        hi = far[index] - frr[index]
        if hi >= 0:
            lo = far[index - 1] - frr[index - 1]
            eer = (
                far[index]
                if hi == 0
                else (far[index - 1] - lo / (hi - lo) * (far[index] - far[index - 1]))
            )
            return {"eer_exact": str(eer), "auc_exact": str(auc)}
    raise ValueError("ROC did not cross equal error")


def ranking_metrics(labels, scores, probe_ids):
    """Exact EER/AUC overall and by probe person; larger scores mean genuine."""
    labels, scores = _rank_values(labels, scores)
    probes = np.asarray(probe_ids)
    if (
        probes.shape != labels.shape
        or probes.dtype.kind not in "iu"
        or np.any(probes < 0)
        or np.any(probes > np.iinfo(np.int64).max)
    ):
        raise ValueError("one nonnegative int64-compatible probe ID per claim required")
    people = {}
    for subject in np.unique(probes):
        selected = probes == subject
        local_labels, local_scores = _rank_values(labels[selected], scores[selected])
        people[str(int(subject))] = _ranking(local_labels, local_scores)
    return {**_ranking(labels, scores), "per_subject": people}


def _fraction(value):
    if not isinstance(value, str):
        raise ValueError("canonical rational strings required")  # noqa: TRY004 -- evidence validation contract
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("canonical rational strings required") from exc
    if str(result) != value or not 0 <= result <= 1:
        raise ValueError("canonical probability rational strings required")
    return result


def _rank_record(record, people):
    if not isinstance(record, Mapping) or set(record) != {
        "eer_exact",
        "auc_exact",
        "per_subject",
    }:
        raise ValueError("complete ranking record required")
    group = record["per_subject"]
    if not isinstance(group, Mapping) or set(group) != set(people):
        raise ValueError("ranking people must match registered validation people")
    result = {k: _fraction(record[k]) for k in ("eer_exact", "auc_exact")}
    result["per_subject"] = {}
    for subject, values in group.items():
        if not isinstance(values, Mapping) or set(values) != {"eer_exact", "auc_exact"}:
            raise ValueError("complete per-person ranking required")
        result["per_subject"][subject] = {k: _fraction(values[k]) for k in values}
    return result


def _mean(values):
    return sum(values, Fraction()) / len(values)


def summarize_nuisance_study(rows, *, dev_ids, folds):
    """Apply all registered clean, person, ranking and robustness safety gates."""
    ids = tuple(_subject(_integer(s, "DEV ID")) for s in dev_ids)
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("fifteen unique DEV identities required")
    folds = list(folds)
    if len(folds) != 20 or any(type(f["fold"]) is not int for f in folds):
        raise ValueError("twenty registered integer folds required")
    validation = {f["fold"]: {_subject(s) for s in f["validation_ids"]} for f in folds}
    if (
        set(validation) != set(range(20))
        or any(len(p) != 3 or not p.issubset(ids) for p in validation.values())
        or any(sum(s in p for p in validation.values()) != 4 for s in ids)
    ):
        raise ValueError("balanced registered validation triples required")
    rows = list(rows)
    expected = {(f, a, c) for f in range(20) for a in ARMS for c in CONDITIONS}
    indexed, parsed, ranks = {}, {}, {}
    for row in rows:
        if (
            type(row["fold"]) is not int
            or row["arm"] not in ARMS
            or row["condition"] not in CONDITIONS
        ):
            raise ValueError("registered fold/arm/condition required")
        key = row["fold"], row["arm"], row["condition"]
        if key in indexed:
            raise ValueError("unique registered rows required")
        metrics = _validate_metrics(row, set(ids))
        if set(metrics.people) != validation[key[0]]:
            raise ValueError("row validation people differ from registered fold")
        pairs = {f"{c}<-{p}" for c in metrics.people for p in metrics.people if c != p}
        if set(metrics.pairs) != pairs:
            raise ValueError("all six directed validation pairs required")
        if not isinstance(row.get("decision_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["decision_sha256"]
        ):
            raise ValueError("decision_sha256 required")
        indexed[key], parsed[key] = row, metrics
        ranks[key] = _rank_record(row["ranking"], metrics.people)
    if set(indexed) != expected:
        raise ValueError("complete unique20x2x7 evidence required")
    for fold in range(20):
        if len({parsed[fold, a, c].graph for a in ARMS for c in CONDITIONS}) != 1:
            raise ValueError("all arms/conditions must share the source claim graph")

    exact, summaries = {}, {a: {} for a in ARMS}
    for arm in ARMS:
        for condition in CONDITIONS:
            keys = [(f, arm, condition) for f in range(20)]
            far = _mean([parsed[k].far for k in keys])
            frr = _mean([parsed[k].frr for k in keys])
            eer = _mean([ranks[k]["eer_exact"] for k in keys])
            auc = _mean([ranks[k]["auc_exact"] for k in keys])
            people, rejects, totals, claimants, person_ranks = {}, {}, {}, {}, {}
            for subject in ids:
                present = [k for k in keys if subject in parsed[k].people]
                people[subject] = _mean([parsed[k].people[subject] for k in present])
                counts = [
                    indexed[k]["metrics"]["per_subject_errors"][subject]
                    for k in present
                ]
                rejects[subject] = sum(int(v["false_rejects"]) for v in counts)
                totals[subject] = sum(int(v["n_genuine"]) for v in counts)
                claimants[subject] = sum(
                    v["false_accepts"]
                    for k in present
                    for p, v in parsed[k].pairs.items()
                    if p.split("<-")[0] == subject
                )
                person_ranks[subject] = {
                    name: _mean(
                        [ranks[k]["per_subject"][subject][name] for k in present]
                    )
                    for name in ("eer_exact", "auc_exact")
                }
            bad = sum(parsed[k].far > Fraction(1, 50) for k in keys)
            exact[arm, condition] = {
                "far": far,
                "frr": frr,
                "eer": eer,
                "auc": auc,
                "people": people,
                "rejects": rejects,
                "totals": totals,
                "claimants": claimants,
                "bad": bad,
            }
            fixed = [indexed[k]["metrics"]["fixed_threshold"] for k in keys]
            summaries[arm][condition] = {
                "far": float(far),
                "frr": float(frr),
                "eer": float(eer),
                "auc": float(auc),
                **{
                    name + "_exact": str(value)
                    for name, value in (
                        ("far", far),
                        ("frr", frr),
                        ("eer", eer),
                        ("auc", auc),
                    )
                },
                "splits_far_above_2pct": bad,
                "repeated_false_accepts": sum(int(v["false_accepts"]) for v in fixed),
                "repeated_false_rejects": sum(rejects.values()),
                "repeated_genuine_claims": sum(totals.values()),
                "repeated_impostor_claims": sum(int(v["n_impostor"]) for v in fixed),
                "per_person_frr_exact": {s: str(v) for s, v in people.items()},
                "per_person_repeated_rejects": rejects,
                "per_person_repeated_genuine_claims": totals,
                "per_claimant_repeated_false_accepts": claimants,
                "per_person_ranking_exact": {
                    s: {k: str(v) for k, v in r.items()}
                    for s, r in person_ranks.items()
                },
            }
    b, a = exact["baseline", "clean"], exact["nuisance", "clean"]
    robust_b = _mean([exact["baseline", c]["frr"] for c in ROBUST_CONDITIONS])
    robust_a = _mean([exact["nuisance", c]["frr"] for c in ROBUST_CONDITIONS])
    person_b = {
        s: _mean([exact["baseline", c]["people"][s] for c in ROBUST_CONDITIONS])
        for s in ids
    }
    person_a = {
        s: _mean([exact["nuisance", c]["people"][s] for c in ROBUST_CONDITIONS])
        for s in ids
    }
    gates = {
        "clean_far_no_worse_and_at_most_1pct": a["far"] <= b["far"]
        and a["far"] <= Fraction(1, 100),
        "clean_fold_tail_no_worse": a["bad"] <= b["bad"],
        "clean_frr_no_worse": a["frr"] <= b["frr"],
        "clean_repeated_rejects_no_worse": sum(a["rejects"].values())
        <= sum(b["rejects"].values()),
        "clean_at_least_10_people_no_worse": sum(
            a["people"][s] <= b["people"][s] for s in ids
        )
        >= 10,
        "clean_no_person_more_than_one_extra_reject": all(
            a["rejects"][s] <= b["rejects"][s] + 1 for s in ids
        ),
        "clean_no_new_completely_rejected_person": all(
            b["rejects"][s] == b["totals"][s] or a["rejects"][s] < a["totals"][s]
            for s in ids
        ),
        "clean_eer_no_worse": a["eer"] <= b["eer"],
        "robust_mean_frr_improves_at_least_5pp": robust_b - robust_a >= Fraction(1, 20),
        "robust_each_condition_frr_no_worse": all(
            exact["nuisance", c]["frr"] <= exact["baseline", c]["frr"]
            for c in ROBUST_CONDITIONS
        ),
        "robust_at_least_10_people_no_worse": sum(
            person_a[s] <= person_b[s] for s in ids
        )
        >= 10,
        "robust_no_person_worsens_more_than_5pp": all(
            person_a[s] - person_b[s] <= Fraction(1, 20) for s in ids
        ),
        "rotations_retain_clean_decisions": all(
            indexed[f, "nuisance", c]["decision_sha256"]
            == indexed[f, "nuisance", "clean"]["decision_sha256"]
            for f in range(20)
            for c in ("rotation30", "rotation90")
        ),
    }
    for condition in CONDITIONS:
        current, baseline = exact["nuisance", condition], exact["baseline", condition]
        gates[condition + "_far_at_most_1pct"] = current["far"] <= Fraction(1, 100)
        gates[condition + "_fold_tail_no_worse"] = current["bad"] <= baseline["bad"]
        gates[condition + "_each_claimant_false_accepts_no_worse"] = all(
            current["claimants"][s] <= baseline["claimants"][s] for s in ids
        )
    eligible = all(gates.values())
    return {
        "winner": "nuisance" if eligible else "baseline",
        "eligible": ["nuisance"] if eligible else [],
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "robust_mean_frr_exact": {"baseline": str(robust_b), "nuisance": str(robust_a)},
        "robust_improvement_exact": str(robust_b - robust_a),
        "summaries": summaries,
        "stress_used_for_selection": True,
        "interpretation": "exploratory reused DEV after stress and TEST inspection; dependent appearances, no population guarantee or independent confirmation",
    }
