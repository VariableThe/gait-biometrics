"""Statically extracted pure source; provenance in source-origins.json."""
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral


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
