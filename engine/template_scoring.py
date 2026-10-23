"""Deterministic enrollment-only cross-day scoring, with no data or Torch imports.

All three candidates require their own independently calibrated threshold.
See docs/template_scoring.md for the small-subject diagnostic's limitations.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
from numbers import Integral, Real

import numpy as np

_EPS = 1e-12
_METHODS = ("mean", "robust", "multi")


class CoverageError(ValueError):
    """Insufficient eligible identities; ``coverage`` retains every exclusion."""

    def __init__(self, coverage):
        super().__init__(
            "cross-day verification requires at least two eligible identities"
        )
        self.coverage = coverage


def _ids(values, name, size=None):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values > np.iinfo(np.int64).max)
        or (size is not None and len(values) != size)
    ):
        raise ValueError(f"{name} must contain one nonnegative int64 ID per row")
    return values.astype(np.int64, copy=False)


def _unit(values):
    norm = np.linalg.norm(values)
    return None if norm < _EPS else values / norm


def _windows(embeddings, size):
    raw = np.asarray(embeddings)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[0] != size:
        raise ValueError(
            "embeddings must be a real numeric matrix with one row per window"
        )
    z = raw.astype(np.float64)
    if not size or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError(
            "embeddings must be nonempty and finite with positive dimension"
        )
    # Preserve the legacy arithmetic for ordinary embeddings, and avoid overflow
    # for finite extreme inputs rather than silently turning them into zeros.
    with np.errstate(over="ignore", under="ignore"):
        norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms < _EPS):
        raise ValueError("window embedding has no cosine direction (norm < 1e-12)")
    overflow = ~np.isfinite(norms[:, 0])
    if overflow.any():
        scaled = z[overflow] / np.max(np.abs(z[overflow]), axis=1, keepdims=True)
        z = z.copy()
        z[overflow] = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[overflow] = 1.0
    return z / norms


def _prototypes(vectors, method):
    """Return unit prototypes, exemplar offsets and deterministic assignments."""
    if method == "mean":
        template = _unit(vectors.mean(axis=0))
        return None if template is None else (template[None, :], [], [0] * len(vectors))
    # Equivalent to acos(dot) for unit vectors, but exactly zero for identical
    # stored vectors, avoiding spurious modes from roundoff near cosine one.
    chords = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    angles = 2 * np.arcsin(np.clip(chords / 2, 0.0, 1.0))
    if method == "robust" or len(vectors) == 1 or np.max(angles) <= _EPS:
        selected = int(np.argmin(angles.sum(axis=1)))
        return vectors[[selected]], [selected], [0] * len(vectors)
    # At most ten trials: enumerate <=45 pairs, minimizing total nearest angular
    # distance. Exact ties retain the lexicographically first trial pair.
    best_pair, best_cost = (0, 1), np.inf
    for pair in combinations(range(len(vectors)), 2):
        cost = float(np.min(angles[:, pair], axis=1).sum())
        if cost < best_cost:
            best_pair, best_cost = pair, cost
    assignments = np.argmin(angles[:, best_pair], axis=1).tolist()
    return vectors[list(best_pair)], list(best_pair), assignments


def score_trials(
    embeddings, subjects, days, trials, *, template_method="mean", probe_method="mean"
):
    """Return ``(int64 labels, float64 scores, JSON-safe metadata)``.

    Normalize windows, then normalize their arithmetic mean per subject/day/
    trial. Day1 trials enroll; each day2 trial is one probe. Trial IDs are 1..10,
    days 1/2, and subject IDs remain authentic nonnegative integers. ``mean``
    averages enrollment trials equally, ``robust`` chooses an angular medoid,
    and ``multi`` chooses up to two exact angular medoids and takes max cosine.
    No day2 values fit a template. Only mean probe aggregation is supported.

    Claims are exhaustive and directed within eligible identities, ordered by
    sorted probe subject, probe trial, then claim subject. Scores are clipped
    cosine minus one; accept iff score >= an independently calibrated threshold.
    Cancellation excludes and reports a trial/template; zero windows fail.
    ``CoverageError.coverage`` is available if fewer than two identities survive.
    """
    if template_method not in _METHODS or probe_method != "mean":
        raise ValueError(
            "template_method must be mean/robust/multi; probe_method must be mean"
        )
    subjects = _ids(subjects, "subjects")
    days = _ids(days, "days", len(subjects))
    trials = _ids(trials, "trials", len(subjects))
    if not np.isin(days, (1, 2)).all() or not np.isin(trials, range(1, 11)).all():
        raise ValueError("days must be 1/2 and trials must be 1..10")
    z = _windows(embeddings, len(subjects))
    requested = list(map(int, np.unique(subjects)))
    trial_vectors, exclusions, per_subject = {}, [], []
    for subject in requested:
        row = {"subject": subject, "days": {}}
        for day in (1, 2):
            eligible_trials, missing_trials = [], []
            for trial in range(1, 11):
                mask = (subjects == subject) & (days == day) & (trials == trial)
                count = int(mask.sum())
                if not count:
                    missing_trials.append(trial)
                    continue
                vector = _unit(z[mask].mean(axis=0))
                if vector is None:
                    exclusions.append(
                        {
                            "subject": subject,
                            "day": day,
                            "trial": trial,
                            "reason": "zero_norm_trial_mean",
                            "windows": count,
                        }
                    )
                else:
                    trial_vectors[(subject, day, trial)] = vector
                    eligible_trials.append(trial)
            row["days"][str(day)] = {
                "n_windows": int(((subjects == subject) & (days == day)).sum()),
                "eligible_trials": eligible_trials,
                "missing_trials": missing_trials,
            }
        per_subject.append(row)
    templates, descriptions = {}, []
    for subject in requested:
        enrollment_trials = [t for s, d, t in trial_vectors if s == subject and d == 1]
        has_probes = any(s == subject and d == 2 for s, d, _ in trial_vectors)
        result = None
        if enrollment_trials:
            vectors = np.stack(
                [trial_vectors[(subject, 1, t)] for t in enrollment_trials]
            )
            result = _prototypes(vectors, template_method)
        if result is None or not has_probes:
            exclusions.append(
                {"subject": subject, "reason": "missing_day_or_zero_norm_template"}
            )
            continue
        prototypes, offsets, assignments = result
        templates[subject] = prototypes
        descriptions.append(
            {
                "subject": subject,
                "enrollment_trial_ids": enrollment_trials,
                "prototype_vectors": prototypes.tolist(),
                "prototype_trial_ids": [enrollment_trials[i] for i in offsets],
                "trial_prototype_indices": assignments,
                "n_prototypes": len(prototypes),
            }
        )
    eligible = list(templates)
    n_probe_trials = sum(s in templates and d == 2 for s, d, _ in trial_vectors)
    n_impostor = n_probe_trials * max(0, len(eligible) - 1)
    coverage = {
        "requested_subject_ids": requested,
        "eligible_subject_ids": eligible,
        "n_enrollment_identities": len(eligible),
        "n_probe_identities": len(eligible),
        "n_enrollment_trials": sum(
            s in templates and d == 1 for s, d, _ in trial_vectors
        ),
        "n_probe_trials": n_probe_trials,
        "n_windows": len(z),
        "per_subject": per_subject,
        "exclusions": exclusions,
    }
    coverage["n_genuine"] = n_probe_trials
    coverage["n_impostor"] = n_impostor
    coverage["far_resolution"] = 1 / n_impostor if n_impostor else None
    if len(eligible) < 2:
        raise CoverageError(coverage)
    labels, scores, claims, probes, probe_trials = [], [], [], [], []
    for subject in eligible:
        for (s, day, trial), vector in trial_vectors.items():
            if s != subject or day != 2:
                continue
            for claim in eligible:
                labels.append(int(claim == subject))
                prototypes = templates[claim]
                similarity = (
                    vector @ prototypes[0]
                    if len(prototypes) == 1
                    else np.max(vector @ prototypes.T)
                )
                scores.append(float(np.clip(similarity, -1, 1) - 1))
                claims.append(claim)
                probes.append(subject)
                probe_trials.append(trial)
    metadata = {
        "protocol": "cross_day_template_scoring_v1",
        "template_method": template_method,
        "probe_method": probe_method,
        "score": "clip(cosine_similarity,-1,1) - 1",
        "score_order": "sorted probe subject, probe trial, then claim subject",
        "claim_subject_ids": claims,
        "probe_subject_ids": probes,
        "probe_trial_ids": probe_trials,
        "coverage": coverage,
        "templates": descriptions,
        "calibration": {
            "threshold_calibrated": False,
            "required": "separate calibration for every candidate",
            "candidate": {
                "template_method": template_method,
                "probe_method": probe_method,
            },
        },
        "fit_population": "day1 enrollment trials only; no probes, cohort transform or thresholds fitted",
        "multi_fusion": "maximum prototype cosine"
        if template_method == "multi"
        else None,
    }
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        metadata,
    )


def _threshold(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")  # noqa: TRY004 -- public validation contract
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite float64-compatible scalar"
        ) from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite float64-compatible scalar")
    return converted


def paired_subject_diagnostics(
    labels,
    baseline_scores,
    new_scores,
    claim_subject_ids,
    probe_subject_ids,
    thresholds,
    *,
    max_exact_subjects=16,
):
    """Describe paired decisions on exactly the same directed claim graph.

    ``thresholds`` must explicitly map ``baseline`` and ``new`` to their own
    frozen, independently calibrated thresholds. This function cannot verify
    that provenance. Exact sign flips concern only per-user genuine acceptance
    differences and require independent subject blocks with sign symmetry under
    the null. Impostor edges share both endpoints: no graph CI or p-value is
    produced. Five subjects give at most 32 sign assignments. Results are
    exploratory/descriptive, never a post-selection confirmatory test.
    """
    raw = np.asarray(labels)
    if (
        raw.ndim != 1
        or raw.dtype.kind not in "iu"
        or not np.isin(raw, (0, 1)).all()
        or not len(raw)
    ):
        raise ValueError("labels must be a nonempty 1D integer binary array")
    claims = _ids(claim_subject_ids, "claim_subject_ids", len(raw))
    probes = _ids(probe_subject_ids, "probe_subject_ids", len(raw))
    if not np.array_equal(raw, (claims == probes).astype(np.int64)):
        raise ValueError("labels must agree with genuine claim/probe identity equality")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {"baseline", "new"}:
        raise ValueError(
            "thresholds must explicitly contain baseline and new calibrated values"
        )
    thresholds = {
        name: _threshold(thresholds[name], name) for name in ("baseline", "new")
    }
    if (
        isinstance(max_exact_subjects, (bool, np.bool_))
        or not isinstance(max_exact_subjects, Integral)
        or not 1 <= max_exact_subjects <= 20
    ):
        raise ValueError("max_exact_subjects must be an integer in 1..20")
    decisions = {}
    for name, values in (("baseline", baseline_scores), ("new", new_scores)):
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or values.shape != raw.shape:
            raise ValueError(
                f"{name}_scores must be finite real values aligned with labels"
            )
        # Compare at the scoring API's precision: NumPy can otherwise downcast
        # a Python threshold to float32 and change a boundary decision.
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}_scores must be finite float64-compatible values")
        decisions[name] = values >= thresholds[name]
    subjects = np.unique(np.concatenate((claims, probes)))
    if len(subjects) < 2:
        raise ValueError("paired diagnostics require at least two subjects")

    def counts(mask):
        n = int(mask.sum())
        result = {"n": n}
        for name, accepted in decisions.items():
            count = int(accepted[mask].sum())
            result[name] = {
                "accepted": count,
                "rejected": n - count,
                "acceptance_rate": count / n if n else None,
            }
        result["acceptance_difference"] = (
            result["new"]["acceptance_rate"] - result["baseline"]["acceptance_rate"]
            if n
            else None
        )
        return result

    rows, differences = [], []
    for subject in subjects:
        genuine = counts((raw == 1) & (probes == subject))
        if not genuine["n"]:
            raise ValueError(
                "every graph subject needs genuine decisions in both candidates"
            )
        difference = genuine["acceptance_difference"]
        differences.append(difference)
        rows.append(
            {
                "subject": int(subject),
                "genuine": genuine,
                "impostor_by_probe": counts((raw == 0) & (probes == subject)),
                "impostor_by_claim": counts((raw == 0) & (claims == subject)),
                "genuine_acceptance_difference": difference,
            }
        )
    differences = np.asarray(differences, dtype=np.float64)
    observed = float(differences.mean())
    reference = {
        "status": "not_enumerated_subject_limit",
        "n_assignments": None,
        "two_sided_tail_fraction": None,
        "assumptions": "independent subject blocks and null invariance to each subject's sign flip",
        "scope": "genuine decisions only; no impostor graph inference",
        "interpretation": "exploratory reference, not a post-selection confirmatory p-value",
    }
    if len(subjects) <= max_exact_subjects:
        # Enumerate all signs, including zero-difference subjects. No RNG or
        # independent-pair approximation. Tolerance only protects floating ties.
        extreme = sum(
            abs(float(np.dot(signs, differences) / len(subjects)))
            >= abs(observed) - 1e-15
            for signs in product((-1, 1), repeat=len(subjects))
        )
        assignments = 2 ** len(subjects)
        reference.update(
            status="exact_enumeration",
            n_assignments=assignments,
            two_sided_tail_fraction=extreme / assignments,
        )
    return {
        "protocol": "paired_subject_diagnostics_v1",
        "thresholds": thresholds,
        "n_subjects": len(subjects),
        "n_comparisons": len(raw),
        "per_subject": rows,
        "genuine_macro_acceptance_difference": observed,
        "n_positive_subject_differences": int((differences > 0).sum()),
        "n_negative_subject_differences": int((differences < 0).sum()),
        "n_zero_subject_differences": int((differences == 0).sum()),
        "exact_sign_flip": reference,
        "uncertainty": "descriptive; directed impostor claims share both identities; no pair-iid CI or bootstrap",
        "threshold_provenance_verified": False,
    }
