"""Exploratory NumPy-only enrollment fusion on one common cross-day graph."""

from __future__ import annotations

import numpy as np

from engine.template_scoring import _unit, _windows, score_trials

_METHODS = ("mean", "multi", "mean_multi", "top2", "lse_0.1")
_TEMPERATURE = 0.1
_GRAPH_KEYS = ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")


def _claim_keys(metadata):
    return zip(*(metadata[key] for key in _GRAPH_KEYS), strict=True)


def _multi_on_common_graph(embeddings, subjects, days, trials, labels, metadata):
    """Reuse the original medoids and scores, joining by authentic claim keys."""
    multi_labels, multi_scores, multi_metadata = score_trials(
        embeddings, subjects, days, trials, template_method="multi"
    )
    positions = {key: index for index, key in enumerate(_claim_keys(multi_metadata))}
    indices = np.asarray(
        [positions[key] for key in _claim_keys(metadata)], dtype=np.intp
    )
    if not np.array_equal(labels, multi_labels[indices]):
        raise ValueError("mean and multi labels must agree on every common claim")
    eligible = set(metadata["coverage"]["eligible_subject_ids"])
    descriptions = [
        row for row in multi_metadata["templates"] if row["subject"] in eligible
    ]
    return multi_scores[indices], descriptions


def _original_trials(embeddings, subjects, days, trials, coverage):
    """Reproduce the shared unit-window, then unit-trial, mean arithmetic."""
    subjects, days, trials = map(np.asarray, (subjects, days, trials))
    windows = _windows(embeddings, len(subjects))
    eligible = set(coverage["eligible_subject_ids"])
    enrollment, probes, descriptions = {}, {}, []
    for row in coverage["per_subject"]:
        subject = row["subject"]
        if subject not in eligible:
            continue
        for day in (1, 2):
            vectors = []
            trial_ids = row["days"][str(day)]["eligible_trials"]
            for trial in trial_ids:
                mask = (subjects == subject) & (days == day) & (trials == trial)
                vector = _unit(windows[mask].mean(axis=0))
                if vector is None or not np.isfinite(vector).all():
                    raise ValueError(
                        f"trial reaggregation lost direction: {(subject, day, trial)}"
                    )
                if day == 1:
                    vectors.append(vector)
                else:
                    probes[(subject, trial)] = vector
            if day == 1:
                enrollment[subject] = np.stack(vectors)
                descriptions.append(
                    {
                        "subject": subject,
                        "enrollment_trial_ids": trial_ids.copy(),
                        "prototype_vectors": enrollment[subject].tolist(),
                        "prototype_trial_ids": trial_ids.copy(),
                        "trial_prototype_indices": list(range(len(trial_ids))),
                        "n_prototypes": len(trial_ids),
                    }
                )
    return enrollment, probes, descriptions


def _logmeanexp(values):
    """Temperature-0.1 logmeanexp; shifting bounds exponential arguments by 0."""
    maximum = np.max(values)
    return float(
        maximum
        + _TEMPERATURE * np.log(np.exp((values - maximum) / _TEMPERATURE).mean())
    )


def score_secure_trials(embeddings, subjects, days, trials, *, method):
    """Return ``(int64 labels, float64 scores, JSON-safe metadata)``.

    Methods: ``mean``, ``multi``, ``mean_multi``, ``top2``, ``lse_0.1``.
    Every candidate uses ``score_trials(..., template_method="mean")`` coverage,
    including its explicit cancellation/missing-day exclusions and its
    ``CoverageError`` when fewer than two identities survive. All claims are
    exhaustive, directed and ordered exactly as that baseline. IDs are never
    relabeled. The original scorer validates every input row before eligibility.

    Only day1 unit trial means enroll; each day2 unit trial mean is one probe.
    Mean/multi reuse the original scorer. Fusion averages their already shifted
    scores. Top2 averages the best min(2, n) original enrollment trial scores;
    LSE uses all n with fixed temperature 0.1 and divides by n inside the log.
    Scores stay on the cosine-minus-one scale [-2, 0], with larger more similar.
    No data loading, threshold fitting, cohort fitting or accuracy claim occurs.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError("method must be mean/multi/mean_multi/top2/lse_0.1")

    labels, mean_scores, metadata = score_trials(
        embeddings, subjects, days, trials, template_method="mean"
    )
    scores = mean_scores
    formula = metadata["score"]
    candidate = {"method": method, "probe_method": "mean"}
    if method in ("multi", "mean_multi"):
        multi_scores, descriptions = _multi_on_common_graph(
            embeddings, subjects, days, trials, labels, metadata
        )
        if method == "mean_multi":
            scores = 0.5 * mean_scores + 0.5 * multi_scores
            metadata["mean_templates"] = metadata["templates"]
            formula = "0.5 * mean_score + 0.5 * multi_score (both cosine minus one)"
            candidate["weights"] = {"mean": 0.5, "multi": 0.5}
        else:
            scores = multi_scores
        metadata["templates"] = descriptions
        metadata["multi_fusion"] = "maximum prototype cosine"
    elif method in ("top2", "lse_0.1"):
        enrollment, probes, descriptions = _original_trials(
            embeddings, subjects, days, trials, metadata["coverage"]
        )
        scores = np.empty_like(mean_scores)
        for index, (claim, subject, trial) in enumerate(_claim_keys(metadata)):
            vector = probes[(subject, trial)]
            values = np.asarray(
                [
                    float(np.clip(vector @ enrolled, -1, 1) - 1)
                    for enrolled in enrollment[claim]
                ],
                dtype=np.float64,
            )
            scores[index] = (
                np.sort(values)[-2:].mean() if method == "top2" else _logmeanexp(values)
            )
        metadata["templates"] = descriptions
        if method == "top2":
            formula = (
                "mean of best min(2,n) original day1 trial cosine-minus-one scores"
            )
        else:
            formula = (
                "m + 0.1 * log(mean(exp((trial_score - m) / 0.1))); m=max(trial_score)"
            )
            candidate["temperature"] = _TEMPERATURE

    metadata["base_protocol"] = metadata["protocol"]
    metadata["protocol"] = "cross_day_secure_enrollment_v1"
    metadata["method"] = method
    metadata["template_method"] = method
    metadata["score"] = formula
    metadata["score_range"] = [-2.0, 0.0]
    metadata["coverage_policy"] = (
        "mean-baseline eligible trials and identities for every method; "
        "all exclusions retained in coverage"
    )
    metadata["calibration"] = {
        "threshold_calibrated": False,
        "required": "separate calibration for every candidate",
        "candidate": candidate,
    }
    return labels, scores, metadata
