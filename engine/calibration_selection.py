"""Exact, descriptive selection for the DEV-only calibration comparison.

This module depends only on the standard library. It reads no files, imports no
study runners, and fits no models. Callers supply the registered fifteen DEV IDs
and twenty folds, with three validation people per fold and four appearances per
person. Each fold must contain both scorers and all four policies on a common
claim graph. Graph hashes and raw counts are caller-provided evidence: checking
them against source-derived coverage and replayed scores is the caller's job.

Only integer decision counts are authoritative; stored FAR/FRR display fields
are ignored. Fold rates receive equal weight, and each person's four fold rates
receive equal weight. Overlapping folds, repeated people and dependent claims
provide descriptive evidence, not independent trials or population guarantees.
The mean_multi scorer is the registered blend used for promotion. The multi
scorer and the baseline's pair concentration are diagnostics only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

__all__ = ["summarize_calibration"]


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004 -- public validation contract
    return value


def _required(record, key):
    if key not in record:
        raise ValueError(f"missing required field: {key}")
    return record[key]


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, excluding bool")
    return int(value)


def _counts(record, errors, denominator):
    n = _integer(_required(record, denominator), denominator, minimum=1)
    k = _integer(_required(record, errors), errors)
    if k > n:
        raise ValueError(f"{errors} must not exceed {denominator}")
    return k, n


def _subject(value):
    if isinstance(value, str):
        try:
            subject = int(value)
        except ValueError as error:
            raise ValueError("subject IDs must be canonical integer strings") from error
        if str(subject) != value:
            raise ValueError("subject IDs must be canonical integer strings")
        return value
    return str(_integer(value, "subject ID"))


def _names(values, name, size, required):
    try:
        values = tuple(values) if not isinstance(values, str) else ()
    except TypeError as error:
        raise ValueError(f"{name} must contain {size} unique names") from error
    if (
        len(values) != size
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != size
        or required not in values
    ):
        raise ValueError(
            f"{name} must contain {size} unique names including {required}"
        )
    return values


@dataclass(frozen=True)
class _Row:
    far: Fraction
    frr: Fraction
    people: dict[str, Fraction]
    pairs: dict[str, dict[str, int]]
    graph: tuple


def _validate_metrics(row, dev_ids):
    metrics = _mapping(_required(row, "metrics"), "metrics")
    graph_hash = _required(metrics, "claim_graph_sha256")
    if not isinstance(graph_hash, str) or not graph_hash.strip():
        raise ValueError("claim_graph_sha256 must be a nonempty string")
    fixed = _mapping(_required(metrics, "fixed_threshold"), "fixed_threshold")
    fa, n_impostor = _counts(fixed, "false_accepts", "n_impostor")
    fr, n_genuine = _counts(fixed, "false_rejects", "n_genuine")

    subjects = _mapping(_required(metrics, "per_subject_errors"), "per_subject_errors")
    people, genuine, impostor = {}, {}, {}
    for key, raw in subjects.items():
        subject = _subject(key)
        if subject not in dev_ids or subject in people:
            raise ValueError("per_subject_errors requires unique expected DEV IDs")
        errors = _mapping(raw, "per_subject_errors entry")
        rejected, n = _counts(errors, "false_rejects", "n_genuine")
        people[subject] = Fraction(rejected, n)
        genuine[subject] = (rejected, n)
        # Prior reports also contain per-claimant impostor counts. If supplied,
        # validate them as well; only genuine counts are needed for person FRR.
        if "false_accepts" in errors or "n_impostor" in errors:
            impostor[subject] = _counts(errors, "false_accepts", "n_impostor")
    if len(people) != 3:
        raise ValueError("each fold must validate exactly three DEV people")
    if (sum(k for k, _ in genuine.values()), sum(n for _, n in genuine.values())) != (
        fr,
        n_genuine,
    ):
        raise ValueError("per-person genuine counts must sum to fixed_threshold counts")

    pairs = {}
    per_claimant = {subject: [0, 0] for subject in people}
    raw_pairs = _mapping(_required(row, "pair_errors"), "pair_errors")
    for pair, raw in raw_pairs.items():
        if not isinstance(pair, str) or len(parts := pair.split("<-")) != 2:
            raise ValueError("pair_errors keys must be directed claimant<-probe IDs")
        claim, probe = map(_subject, parts)
        if claim == probe or claim not in people or probe not in people:
            raise ValueError("pair_errors must join distinct validation people")
        errors = _mapping(raw, "pair_errors entry")
        accepted, n = _counts(errors, "false_accepts", "claims")
        pairs[pair] = {"claims": n, "false_accepts": accepted}
        per_claimant[claim][0] += accepted
        per_claimant[claim][1] += n
    if (
        sum(pair["false_accepts"] for pair in pairs.values()),
        sum(pair["claims"] for pair in pairs.values()),
    ) != (fa, n_impostor):
        raise ValueError("directed-pair counts must sum to fixed_threshold counts")
    if impostor and (
        set(impostor) != set(people)
        or any(impostor[s] != tuple(per_claimant[s]) for s in impostor)
    ):
        raise ValueError("per-person impostor counts must match directed-pair counts")

    graph = (
        graph_hash,
        n_genuine,
        n_impostor,
        tuple(sorted((s, n) for s, (_, n) in genuine.items())),
        tuple(sorted((pair, counts["claims"]) for pair, counts in pairs.items())),
    )
    return _Row(Fraction(fa, n_impostor), Fraction(fr, n_genuine), people, pairs, graph)


def _mean(values):
    return sum(values, Fraction(0)) / len(values)


def _exact(value):
    return f"{value.numerator}/{value.denominator}"


def summarize_calibration(
    rows, *, dev_ids, scorers=("mean_multi", "multi"), policies=("A", "B", "C", "D")
):
    """Return the six JSON-compatible fields of the calibration-study summary.

    Require complete unique 20 x 2 x 4 evidence, common graphs within each fold,
    and fifteen distinct integer DEV IDs validating four times per comparison.
    Counts must be nonnegative integers (excluding booleans), with positive
    denominators and errors no larger than their denominators. Per-person FR
    counts and directed-pair FA counts must partition the fixed-threshold counts.
    Missing or inconsistent evidence raises ValueError; inputs are not mutated.

    Promotion uses exact equal-fold mean FAR <= 1/100, no more folds with exact
    FAR > 1/50 than A, strictly smaller exact equal-fold mean FRR, and at least
    ten of fifteen exact person means no worse than A. Eligible policies are
    ranked by exact FRR, exact FAR, then the supplied policy order. Retain A if
    none qualify. Only mean_multi participates in these decisions.

    Existing float presentation fields are retained. Each summary also has
    far_exact, frr_exact and per_person_mean_frr_exact, whose values are reduced
    "numerator/denominator" strings, including "0/1". No Fraction escapes into
    the returned JSON. Custom orders require two scorers including mean_multi
    and four policies including A.
    """
    scorers = _names(scorers, "scorers", 2, "mean_multi")
    policies = _names(policies, "policies", 4, "A")
    try:
        ids = tuple(_integer(s, "DEV ID") for s in dev_ids)
    except TypeError as error:
        raise ValueError("dev_ids must contain fifteen distinct integers") from error
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("dev_ids must contain fifteen distinct integers")
    subject_order = tuple(map(str, ids))
    dev_set = set(subject_order)

    expected = {
        (fold, scorer, policy)
        for fold in range(20)
        for scorer in scorers
        for policy in policies
    }
    indexed, graphs = {}, {}
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError("complete unique 20 x 2 x 4 comparisons required") from error
    if len(rows) != len(expected):
        raise ValueError("complete unique 20 x 2 x 4 comparisons required")
    for raw in rows:
        row = _mapping(raw, "row")
        fold = _integer(_required(row, "fold"), "fold")
        scorer, policy = _required(row, "scorer"), _required(row, "policy")
        if not isinstance(scorer, str) or not isinstance(policy, str):
            raise ValueError("scorer and policy must be strings")  # noqa: TRY004 -- public validation contract
        key = (fold, scorer, policy)
        if key not in expected or key in indexed:
            raise ValueError("complete unique 20 x 2 x 4 comparisons required")
        parsed = _validate_metrics(row, dev_set)
        if fold in graphs and graphs[fold] != parsed.graph:
            raise ValueError(
                "all scorers and policies must share an identical claim graph"
            )
        graphs[fold] = parsed.graph
        indexed[key] = parsed

    summary, exact = {}, {}
    for scorer in scorers:
        for policy in policies:
            group = [indexed[fold, scorer, policy] for fold in range(20)]
            people = {s: [] for s in subject_order}
            pairs = {}
            for row in group:
                for subject, rate in row.people.items():
                    people[subject].append(rate)
                for pair, counts in row.pairs.items():
                    total = pairs.setdefault(pair, {"claims": 0, "false_accepts": 0})
                    for name in total:
                        total[name] += counts[name]
            if any(len(values) != 4 for values in people.values()):
                raise ValueError("every DEV person must validate exactly four times")
            far, frr = _mean([r.far for r in group]), _mean([r.frr for r in group])
            person_means = {s: _mean(values) for s, values in people.items()}
            bad_splits = sum(r.far > Fraction(1, 50) for r in group)
            name = f"{scorer}/{policy}"
            exact[name] = (far, frr, person_means)
            summary[name] = {
                "far": float(far),
                "frr": float(frr),
                "far_exact": _exact(far),
                "frr_exact": _exact(frr),
                "splits_far_above_2pct": bad_splits,
                "per_person_mean_frr": {s: float(v) for s, v in person_means.items()},
                "per_person_mean_frr_exact": {
                    s: _exact(v) for s, v in person_means.items()
                },
                "pair_errors": pairs,
            }

    baseline = summary["mean_multi/A"]
    _, baseline_frr, baseline_people = exact["mean_multi/A"]
    eligible = []
    for policy in policies:
        name = f"mean_multi/{policy}"
        result = summary[name]
        far, frr, people = exact[name]
        no_worse = sum(people[s] <= baseline_people[s] for s in subject_order)
        result["people_no_worse_than_A"] = no_worse
        if (
            policy != "A"
            and far <= Fraction(1, 100)
            and result["splits_far_above_2pct"] <= baseline["splits_far_above_2pct"]
            and frr < baseline_frr
            and no_worse >= 10
        ):
            eligible.append(policy)
    winner = (
        min(
            eligible,
            key=lambda p: (
                exact[f"mean_multi/{p}"][1],
                exact[f"mean_multi/{p}"][0],
                policies.index(p),
            ),
        )
        if eligible
        else "A"
    )
    counts = sorted(
        (pair["false_accepts"] for pair in baseline["pair_errors"].values()),
        reverse=True,
    )
    total, top_two = sum(counts), sum(counts[:2])
    return {
        "winner": winner,
        "promotion_gate_passed": bool(eligible),
        "eligible_policies": eligible,
        "summary": summary,
        "normalization_gate": {
            "false_accepts": total,
            "top_two_pairs_false_accepts": top_two,
            "triggered": bool(total and Fraction(top_two, total) >= Fraction(1, 2)),
        },
        "interpretation": "DEV-only descriptive evidence; no new CALIB or TEST evaluation",
    }
