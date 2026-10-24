"""NumPy-only plain S-norm on the unchanged cross-day template claim graph."""

from __future__ import annotations

from numbers import Real

import numpy as np

from engine.template_scoring import _unit, _windows, score_trials


def _floor(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("std_floor must be a positive finite real scalar")  # noqa: TRY004 -- public validation contract
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("std_floor must be float64-compatible") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("std_floor must be a positive finite real scalar")
    return value


def _probe_vectors(embeddings, subjects, days, trials, coverage):
    """Repeat the baseline's unit-window / unit-trial arithmetic exactly."""
    subjects, days, trials = map(np.asarray, (subjects, days, trials))
    windows = _windows(embeddings, len(subjects))
    eligible = set(coverage["eligible_subject_ids"])
    probes = {}
    for row in coverage["per_subject"]:
        subject = row["subject"]
        if subject not in eligible:
            continue
        for trial in row["days"]["2"]["eligible_trials"]:
            mask = (subjects == subject) & (days == 2) & (trials == trial)
            vector = _unit(windows[mask].mean(axis=0))
            if vector is None or not np.isfinite(vector).all():
                raise ValueError("probe reaggregation must retain a finite direction")
            probes[(subject, trial)] = vector
    return windows.shape[1], probes


def _templates(metadata):
    templates = {}
    for row in metadata["templates"]:
        vectors = np.asarray(row["prototype_vectors"], dtype=np.float64)
        if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
            raise ValueError("template prototypes must be nonempty and finite")
        templates[row["subject"]] = vectors
    return templates


def _score(vector, prototypes):
    similarity = (
        vector @ prototypes[0]
        if len(prototypes) == 1
        else np.max(vector @ prototypes.T)
    )
    if not np.isfinite(similarity):
        raise ValueError("cohort cosine score must be finite")
    return float(np.clip(similarity, -1, 1) - 1)


def _stats(scores, floor):
    values = np.asarray(scores, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("normalization needs nonempty finite cohort scores")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("cohort statistics must be finite")
    return {
        "mean": mean,
        "std": std,
        "denominator": max(std, floor),
        "std_floored": std < floor,
        "n_scores": len(values),
    }


def cohort_normalized_scores(
    embeddings,
    subjects,
    days,
    trials,
    *,
    cohort_embeddings,
    cohort_subjects,
    cohort_days,
    cohort_trials,
    template_method="multi",
    std_floor=1e-3,
):
    """Return ``(int64 labels, float64 S-norm scores, JSON-safe metadata)``.

    Both sets use ``score_trials`` with the selected enrollment-only template
    method. Cohort IDs must be disjoint from every supplied target ID and all
    must survive baseline eligibility on both days; each set needs >=2 IDs.
    The caller must supply a frozen DEV-train cohort with authentic identities.

    Z statistics use every eligible cohort day2 trial against a target claim's
    day1 prototypes. T statistics use a target day2 probe against each cohort
    identity's day1 prototypes. Population standard deviations are floored by
    ``std_floor``. Plain S-norm averages these two standardized raw scores.
    No cohort selection, threshold learning, or cross-probe fitting occurs.
    Scores have no cosine range and need independent calibration.
    """
    floor = _floor(std_floor)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            labels, raw_scores, metadata = score_trials(
                embeddings, subjects, days, trials, template_method=template_method
            )
            _, _, cohort_metadata = score_trials(
                cohort_embeddings,
                cohort_subjects,
                cohort_days,
                cohort_trials,
                template_method=template_method,
            )
            coverage = metadata["coverage"]
            cohort_coverage = cohort_metadata["coverage"]
            target_ids = set(coverage["requested_subject_ids"])
            cohort_ids = cohort_coverage["requested_subject_ids"]
            if target_ids.intersection(cohort_ids):
                raise ValueError(
                    "cohort subject IDs must be disjoint from all target IDs"
                )
            if cohort_ids != cohort_coverage["eligible_subject_ids"]:
                raise ValueError(
                    "every cohort ID must have eligible enrollment and probes on both days"
                )
            dimension, probes = _probe_vectors(
                embeddings, subjects, days, trials, coverage
            )
            cohort_dimension, cohort_probes = _probe_vectors(
                cohort_embeddings,
                cohort_subjects,
                cohort_days,
                cohort_trials,
                cohort_coverage,
            )
            if dimension != cohort_dimension:
                raise ValueError("target and cohort embedding dimensions must match")
            templates, cohort_templates = (
                _templates(metadata),
                _templates(cohort_metadata),
            )
            z_stats = {
                claim: _stats(
                    [_score(p, prototypes) for p in cohort_probes.values()], floor
                )
                for claim, prototypes in templates.items()
            }
            t_stats = {
                key: _stats(
                    [_score(probe, cohort_templates[s]) for s in cohort_ids], floor
                )
                for key, probe in probes.items()
            }
            scores = np.empty_like(raw_scores, dtype=np.float64)
            for index, (raw, claim, subject, trial) in enumerate(
                zip(
                    raw_scores,
                    metadata["claim_subject_ids"],
                    metadata["probe_subject_ids"],
                    metadata["probe_trial_ids"],
                    strict=True,
                )
            ):
                enrollment = z_stats[claim]
                probe = t_stats[(subject, trial)]
                z = (raw - enrollment["mean"]) / enrollment["denominator"]
                t = (raw - probe["mean"]) / probe["denominator"]
                scores[index] = 0.5 * z + 0.5 * t
            if not np.isfinite(scores).all():
                raise ValueError("normalized scores must be finite; increase std_floor")
    except FloatingPointError as error:
        raise ValueError(
            "float64 overflow or invalid cohort normalization arithmetic"
        ) from error

    metadata["base_protocol"] = metadata["protocol"]
    metadata["protocol"] = "cross_day_cohort_s_norm_v1"
    metadata["raw_score"] = metadata["score"]
    metadata["raw_scores"] = raw_scores.tolist()
    metadata["score"] = (
        "0.5*((raw-mu_e)/max(std_e,std_floor)+(raw-mu_p)/max(std_p,std_floor))"
    )
    metadata["score_range"] = "unbounded standardized score; not cosine minus one"
    metadata["fit_population"] = (
        "templates: each identity's day1 only; z: frozen cohort day2 trials; "
        "t: current target probe against frozen cohort day1 templates; "
        "no global CALIB/TEST fit or learned threshold; caller enforces DEV-train provenance"
    )
    metadata["calibration"] = {
        "threshold_calibrated": False,
        "required": "own independent calibration for this normalization, template method and frozen cohort",
        "candidate": {
            "normalization": "plain_s_norm",
            "template_method": template_method,
            "probe_method": "mean",
            "std_floor": floor,
            "cohort_subject_ids": cohort_ids,
        },
    }
    metadata["normalization"] = {
        "method": "plain_s_norm",
        "adaptive": False,
        "std_floor": floor,
        "std_ddof": 0,
        "cohort_frozen": True,
        "cohort_provenance": "caller must ensure authentic DEV-train IDs only",
        "cohort_subject_ids": cohort_ids,
        "cohort_templates": cohort_metadata["templates"],
        "cohort_coverage": cohort_coverage,
        "cohort_probe_order": [
            {"subject": subject, "trial": trial} for subject, trial in cohort_probes
        ],
        "z_population": "every eligible cohort day2 unit trial, weighted equally",
        "t_population": "one day1 template score per frozen cohort identity, weighted equally",
        "z_stats": [
            {"subject": subject, **stats} for subject, stats in z_stats.items()
        ],
        "t_stats": [
            {"subject": subject, "trial": trial, **stats}
            for (subject, trial), stats in t_stats.items()
        ],
    }
    return labels, scores, metadata
