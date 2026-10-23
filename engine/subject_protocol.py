"""Subject-disjoint development folds and descriptive verification summaries.

These helpers never load data, fit transforms, choose checkpoints, or inspect
test results. Callers must supply authentic IDs from walking TRAIN windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def _integer(value, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _subject_ids(values, name: str = "subject_ids") -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError(f"{name} must be a 1D array of nonnegative integer IDs")
    return ids


def make_subject_folds(
    subject_ids, calibration_subjects: int = 4, n_folds: int = 3, seed: int = 2026
) -> dict:
    """Reserve calibration identities, then partition development identities.

    Canonicalize to sorted unique authentic encoded IDs before a single seeded
    permutation. Its first ``calibration_subjects`` IDs form a fixed calibration
    cohort; ``np.array_split`` balances the rest across validation folds. Every
    development ID validates exactly once, and each fold trains on all other
    development IDs. Returned ID lists are sorted and retain their input values.

    Each train, validation, and calibration cohort must contain >=2 identities.
    This checks identity counts, not distinct-window or session eligibility.
    The caller must filter the source TRAIN data to walking before calling and
    must keep calibration out of optimizer, epoch, and candidate selection.
    """
    n_calibration = _integer(calibration_subjects, "calibration_subjects", 2)
    n_folds = _integer(n_folds, "n_folds", 2)
    seed = _integer(seed, "seed")
    subjects = np.unique(_subject_ids(subject_ids))
    n_development = len(subjects) - n_calibration
    if n_development < 2 * n_folds:
        raise ValueError(
            f"need at least {n_calibration + 2 * n_folds} unique subjects: "
            f"{n_calibration} calibration and >=2 validation subjects per fold"
        )
    permutation = np.random.default_rng(seed).permutation(subjects)
    calibration = np.sort(permutation[:n_calibration])
    development = np.sort(permutation[n_calibration:])
    validation_groups = np.array_split(permutation[n_calibration:], n_folds)
    folds = []
    for fold, held_out in enumerate(validation_groups):
        training = development[~np.isin(development, held_out)]
        if len(training) < 2 or len(held_out) < 2:
            raise ValueError("every fold needs >=2 training and validation subjects")
        folds.append(
            {
                "fold": fold,
                "train_subject_ids": list(map(int, training)),
                "val_subject_ids": list(map(int, np.sort(held_out))),
            }
        )
    return {
        "protocol": "subject_cv_with_reserved_calibration_v1",
        "seed": seed,
        "n_folds": n_folds,
        "n_calibration_subjects": n_calibration,
        "all_subject_ids": list(map(int, subjects)),
        "development_subject_ids": list(map(int, development)),
        "calibration_subject_ids": list(map(int, calibration)),
        "folds": folds,
        "partition_method": "sorted_unique_then_default_rng_permutation_then_array_split",
        "input_requirement": "authentic encoded subject IDs from walking source TRAIN only",
        "calibration_use": "frozen final model threshold only; never CV or model selection",
    }


def fold_summary(rows) -> dict:
    """Summarize a complete fold x training-seed grid without pair weighting.

    Each row is a mapping with nonnegative integer ``fold`` and ``seed`` plus
    identical finite scalar metric columns (for example, ``eer`` and
    ``roc_auc``). Duplicate runs or incomplete grids fail rather than silently
    weighting a convenient subset. Call separately for every candidate/source.

    Metrics average seeds within each fold, then average folds equally. Fold
    and seed sample SDs are descriptive variation, not standard errors or
    population confidence intervals: CV training cohorts overlap and neither
    folds nor seeds supply independent new people. All input results survive.
    """
    normalized = []
    metric_names = None
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or "fold" not in row or "seed" not in row:
            raise ValueError("each row must be a mapping with fold and seed")
        fold = _integer(row["fold"], "fold")
        seed = _integer(row["seed"], "seed")
        if (fold, seed) in seen:
            raise ValueError(f"duplicate fold/seed result: {(fold, seed)}")
        seen.add((fold, seed))
        names = set(row) - {"fold", "seed"}
        if not names or not all(isinstance(name, str) for name in names):
            raise ValueError("rows need named finite scalar metric columns")
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError("every row must have the same metric columns")
        result = {"fold": fold, "seed": seed}
        for name in sorted(names):
            value = row[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"metric {name!r} must be a finite scalar number")  # noqa: TRY004 -- public validation contract
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"metric {name!r} must be a finite scalar number")
            result[name] = value
        normalized.append(result)
    if not normalized:
        raise ValueError("rows must contain at least one fold/seed result")
    folds = sorted({row["fold"] for row in normalized})
    seeds = sorted({row["seed"] for row in normalized})
    if seen != {(fold, seed) for fold in folds for seed in seeds}:
        raise ValueError("rows must contain a complete fold x seed grid")
    normalized.sort(key=lambda row: (row["fold"], row["seed"]))
    names = sorted(metric_names)

    def average(group):
        means = {}
        for name in names:
            values = np.asarray([row[name] for row in group], dtype=np.float64)
            scale = float(np.max(np.abs(values)))
            means[name] = float(np.mean(values / scale) * scale) if scale else 0.0
        return means

    per_fold = [
        {"fold": fold, "mean": average([r for r in normalized if r["fold"] == fold])}
        for fold in folds
    ]
    per_seed = [
        {"seed": seed, "mean": average([r for r in normalized if r["seed"] == seed])}
        for seed in seeds
    ]

    def sample_sd(groups):
        deviations = {}
        for name in names:
            if len(groups) == 1:
                deviations[name] = None
                continue
            values = np.asarray([group["mean"][name] for group in groups])
            scale = float(np.max(np.abs(values)))
            with np.errstate(over="ignore"):
                sd = float(np.std(values / scale, ddof=1) * scale) if scale else 0.0
            if not np.isfinite(sd):
                raise ValueError(f"metric {name!r} sample SD exceeds finite JSON range")
            deviations[name] = sd
        return deviations

    return {
        "n_folds": len(folds),
        "n_seeds": len(seeds),
        "n_runs": len(normalized),
        "mean": average([row["mean"] for row in per_fold]),
        "per_fold": per_fold,
        "per_seed": per_seed,
        "fold_sample_sd": sample_sd(per_fold),
        "seed_sample_sd": sample_sd(per_seed),
        "all_fold_results": normalized,
        "weighting": "equal training seeds within fold, then equal folds",
        "uncertainty": "descriptive only; dependent folds and seeds are not a population CI",
    }


def per_subject_error_rates(
    labels, scores, claim_subject_ids, probe_subject_ids, threshold: float
) -> dict:
    """Audit a fixed threshold by claimed identity and probe identity.

    All arrays have one entry per scored comparison. For exhaustive enrollment
    scores, tile the sorted enrolled IDs for every probe and repeat each probe
    subject ID for every enrolled ID. Labels must agree with authentic identity
    equality; bootstrap copies must never be relabelled as new identities.

    Claimed-subject FAR measures susceptibility of an enrolled identity to
    others' probes; probe-subject FAR measures its probes' acceptance as others.
    FRR uses genuine claims. Missing denominators give JSON null, never zero.
    Counts reflect the supplied comparisons; exhaustive coverage, independent
    sessions and the provenance of the threshold cannot be verified here.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    claims = _subject_ids(claim_subject_ids, "claim_subject_ids")
    probes = _subject_ids(probe_subject_ids, "probe_subject_ids")
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or not len(labels)
        or not (labels.shape == scores.shape == claims.shape == probes.shape)
    ):
        raise ValueError("all inputs must be equally sized nonempty 1D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary: 1 genuine, 0 impostor")
    if not np.array_equal(labels == 1, claims == probes):
        raise ValueError("labels must match claim/probe subject identity equality")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite scalar number")
    threshold = float(threshold)
    accepted, genuine = scores >= threshold, labels == 1

    def counts(mask):
        n_genuine = int(np.count_nonzero(mask & genuine))
        n_impostor = int(np.count_nonzero(mask & ~genuine))
        false_accepts = int(np.count_nonzero(mask & ~genuine & accepted))
        false_rejects = int(np.count_nonzero(mask & genuine & ~accepted))
        return {
            "n_genuine": n_genuine,
            "n_impostor": n_impostor,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "far": false_accepts / n_impostor if n_impostor else None,
            "frr": false_rejects / n_genuine if n_genuine else None,
        }

    def by_subject(ids):
        subjects = {str(int(sid)): counts(ids == sid) for sid in np.unique(ids)}
        macro = {}
        for metric in ("far", "frr"):
            values = [
                row[metric] for row in subjects.values() if row[metric] is not None
            ]
            macro[metric] = float(np.mean(values)) if values else None
            macro[f"n_subjects_with_{metric}"] = len(values)
        return {"per_subject": subjects, "macro": macro}

    return {
        "threshold": threshold,
        "decision_rule": "score >= threshold",
        "n_comparisons": len(labels),
        "micro": counts(np.ones(len(labels), dtype=bool)),
        "by_claim_subject": by_subject(claims),
        "by_probe_subject": by_subject(probes),
        "threshold_provenance": "caller must supply an independently calibrated fixed threshold",
        "coverage": "supplied comparisons only; exhaustive coverage is not verified",
        "uncertainty": "descriptive counts and rates, not independent-trial confidence bounds",
    }
