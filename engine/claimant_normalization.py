"""Frozen-TRAIN cohort Z/S normalization for the existing enrollment blend."""

from __future__ import annotations

import numpy as np

from engine.cohort_normalization import (
    _floor,
    _probe_vectors,
    _score,
    _stats,
    _templates,
)
from engine.secure_enrollment import score_secure_trials


def _blend_templates(metadata):
    if "mean_templates" not in metadata:
        raise ValueError("the registered blend must retain its mean template")
    return _templates({"templates": metadata["mean_templates"]}), _templates(metadata)


def _blend(probe, mean, multi):
    return 0.5 * _score(probe, mean) + 0.5 * _score(probe, multi)


def _complete_cohort(coverage, subjects, days, trials):
    subjects, days, trials = map(np.asarray, (subjects, days, trials))
    for row in coverage["per_subject"]:
        for day in (1, 2):
            expected = sorted(
                np.unique(trials[(subjects == row["subject"]) & (days == day)]).tolist()
            )
            if row["days"][str(day)]["eligible_trials"] != expected:
                raise ValueError(
                    "normalization requires complete source cohort trial coverage"
                )


def normalize_claimant_scores(
    embeddings,
    subjects,
    days,
    trials,
    *,
    normalization="raw",
    cohort_embeddings=None,
    cohort_subjects=None,
    cohort_days=None,
    cohort_trials=None,
    std_floor=0.001,
):
    """Return labels, scores, metadata on an identical directed claim graph.

    ``z`` standardizes each claimed user's score against frozen TRAIN people's
    day2 trial probes. ``s`` averages that Z score with probe-specific T scores
    against TRAIN day1 templates. Every component scores the same fixed equal
    mean/two-medoid blend. Target day2 probes never fit enrollment or a cohort.
    Cohort IDs must be disjoint from all target IDs; the caller verifies TRAIN
    provenance. Raw output exactly reuses the existing scorer. No thresholds
    are learned here. Normalized scores require their own calibration.
    """
    if normalization not in ("raw", "z", "s"):
        raise ValueError("normalization must be raw, z or s")
    labels, raw, metadata = score_secure_trials(
        embeddings, subjects, days, trials, method="mean_multi"
    )
    if normalization == "raw":
        metadata["normalization"] = "raw"
        return labels, raw, metadata
    floor = _floor(std_floor)
    _, _, cohort_metadata = score_secure_trials(
        cohort_embeddings,
        cohort_subjects,
        cohort_days,
        cohort_trials,
        method="mean_multi",
    )
    coverage, cohort_coverage = metadata["coverage"], cohort_metadata["coverage"]
    target_ids = set(coverage["requested_subject_ids"])
    cohort_ids = cohort_coverage["requested_subject_ids"]
    if target_ids.intersection(cohort_ids):
        raise ValueError("TRAIN cohort IDs must be disjoint from every target ID")
    if cohort_ids != cohort_coverage["eligible_subject_ids"]:
        raise ValueError("every TRAIN cohort identity must retain both source days")
    _complete_cohort(cohort_coverage, cohort_subjects, cohort_days, cohort_trials)
    dimension, probes = _probe_vectors(embeddings, subjects, days, trials, coverage)
    cohort_dimension, cohort_probes = _probe_vectors(
        cohort_embeddings,
        cohort_subjects,
        cohort_days,
        cohort_trials,
        cohort_coverage,
    )
    if dimension != cohort_dimension:
        raise ValueError("target and cohort embedding dimensions must match")
    means, multis = _blend_templates(metadata)
    cohort_means, cohort_multis = _blend_templates(cohort_metadata)
    z_stats = {
        claim: _stats(
            [
                _blend(probe, means[claim], multis[claim])
                for probe in cohort_probes.values()
            ],
            floor,
        )
        for claim in means
    }
    t_stats = (
        {
            key: _stats(
                [
                    _blend(probe, cohort_means[claim], cohort_multis[claim])
                    for claim in cohort_ids
                ],
                floor,
            )
            for key, probe in probes.items()
        }
        if normalization == "s"
        else {}
    )
    scores = np.empty_like(raw, dtype=np.float64)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for index, (value, claim, probe, trial) in enumerate(
                zip(
                    raw,
                    metadata["claim_subject_ids"],
                    metadata["probe_subject_ids"],
                    metadata["probe_trial_ids"],
                    strict=True,
                )
            ):
                z = z_stats[claim]
                score = (value - z["mean"]) / z["denominator"]
                if normalization == "s":
                    t = t_stats[(probe, trial)]
                    score = 0.5 * score + 0.5 * ((value - t["mean"]) / t["denominator"])
                scores[index] = score
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("normalization overflow; use a larger std_floor") from error
    if not np.isfinite(scores).all():
        raise ValueError("normalized scores must remain finite")
    metadata.update(
        base_protocol=metadata["protocol"],
        protocol="cross_day_claimant_blend_normalization_v1",
        normalization=normalization,
        raw_score=metadata["score"],
        raw_scores=raw.tolist(),
        score="(raw-claimant_mean)/claimant_std"
        if normalization == "z"
        else "0.5*(Z+T)",
        score_range="unbounded standardized score",
        cohort_subject_ids=cohort_ids,
        cohort_coverage=cohort_coverage,
        z_stats={str(claim): values for claim, values in z_stats.items()},
        t_stats={
            f"{subject}:{trial}": values for (subject, trial), values in t_stats.items()
        },
        std_floor=floor,
        fit_population="templates from day1 only; frozen TRAIN cohort only; no target cohort fitting",
        cohort_dependence="cohort people also fitted the metric; no independent-score guarantee",
    )
    metadata["calibration"] = {
        "threshold_calibrated": False,
        "required": "separate threshold for normalization and this frozen cohort",
        "candidate": {"normalization": normalization, "std_floor": floor},
    }
    return labels, scores, metadata
