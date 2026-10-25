"""Failure-inclusive, tied-score ROC and empirical threshold calibration.

Only NumPy and the standard library are used. No data are read or models fit.
See docs/gated_roc.md for the JSON schema and interpretation limits.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import groupby
from math import isfinite
from numbers import Integral, Real

import numpy as np

__all__ = ["apply_gated_policy", "calibrate_gated_threshold", "gated_roc"]


def _vector(value, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must be an unmasked one-dimensional vector")
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return list(array)


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, excluding bool")  # noqa: TRY004 -- public validation contract
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as finite float64") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be representable as finite float64")
    return result


def _scores_and_valid(scores, valid):
    raw_scores, raw_valid = _vector(scores, "scores"), _vector(valid, "valid")
    if len(raw_scores) != len(raw_valid):
        raise ValueError("scores and valid must be aligned")
    if any(not isinstance(v, (bool, np.bool_)) for v in raw_valid):
        raise ValueError("valid must contain strictly boolean elements")
    values = []
    for score, usable in zip(raw_scores, raw_valid, strict=True):
        if usable:
            values.append(_finite_real(score, "valid score"))
        elif score is not None:
            raise ValueError("invalid scores must be exactly None")
        else:
            values.append(None)
    return values, np.asarray(raw_valid, dtype=np.bool_)


def _claims(labels, scores, valid):
    values, usable = _scores_and_valid(scores, valid)
    raw_labels = _vector(labels, "labels")
    if len(raw_labels) != len(values) or any(
        not isinstance(label, (Integral, bool, np.bool_)) or label not in (0, 1)
        for label in raw_labels
    ):
        raise ValueError("aligned binary integer or boolean labels required")
    genuine = np.asarray(raw_labels, dtype=np.bool_)
    if not genuine.any() or genuine.all():
        raise ValueError("both classes must be present in the original labels")
    return genuine, values, usable


def _rates(far, frr):
    return {
        "far": float(far),
        "far_exact": str(far),
        "frr": float(frr),
        "frr_exact": str(frr),
    }


def _states(genuine, scores, valid):
    ng, ni = int(genuine.sum()), int((~genuine).sum())
    vg, vi = int((genuine & valid).sum()), int((~genuine & valid).sum())
    counts = {
        "n_genuine": ng,
        "n_impostor": ni,
        "n_valid_genuine": vg,
        "n_valid_impostor": vi,
        "n_failed_genuine": ng - vg,
        "n_failed_impostor": ni - vi,
    }

    def state(policy, tp, fp):
        return {
            "policy": policy,
            "counts": {**counts, "tp": tp, "fp": fp, "fn": ng - tp, "tn": ni - fp},
            **_rates(Fraction(fp, ni), Fraction(ng - tp, ng)),
        }

    states = [state({"kind": "reject_all", "threshold": None}, 0, 0)]
    ranked = sorted(
        (
            (score, bool(label))
            for score, label, usable in zip(scores, genuine, valid, strict=True)
            if usable
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    tp = fp = 0
    for threshold, group in groupby(ranked, key=lambda pair: pair[0]):
        for _, label in group:
            tp += int(label)
            fp += int(not label)
        states.append(state({"kind": "score_at_least", "threshold": threshold}, tp, fp))
    return counts, states


def _eer(states, ng, ni):
    result = {
        "status": "undefined_empty_valid_class"
        if not ng or not ni
        else "undefined_no_crossing",
        "eer": None,
        "eer_exact": None,
        "bracket": None,
    }
    if not ng or not ni:
        return result
    previous_far, previous_difference = Fraction(), Fraction(-1)
    for index, state in enumerate(states):
        counts = state["counts"]
        far = Fraction(counts["fp"], ni)
        frr = Fraction(ng - counts["tp"], ng)
        difference = far - frr
        if difference >= 0:
            if difference == 0:
                status, lower, weight, eer = (
                    "exact_threshold_state",
                    index,
                    Fraction(),
                    far,
                )
            else:
                status, lower = "interpolated", index - 1
                weight = -previous_difference / (difference - previous_difference)
                eer = previous_far + weight * (far - previous_far)
            return {
                "status": status,
                "eer": float(eer),
                "eer_exact": str(eer),
                "bracket": {
                    "state_indexes": [lower, index],
                    "policies": [
                        states[lower]["policy"].copy(),
                        state["policy"].copy(),
                    ],
                    "upper_weight": float(weight),
                    "upper_weight_exact": str(weight),
                },
            }
        previous_far, previous_difference = far, difference
    return result


def gated_roc(labels, scores, valid) -> dict:
    """Return original-denominator ROC, coverage, and a separate conditional EER.

    Labels use 1 for genuine and 0 for impostor. Scores are finite real values
    at valid positions and exactly None elsewhere; valid contains strict bools.
    Score ranks and ties use float64 representation. See docs/gated_roc.md.
    """
    genuine, values, usable = _claims(labels, scores, valid)
    counts, states = _states(genuine, values, usable)
    coverage = {}
    for name in ("genuine", "impostor"):
        total, successful = counts[f"n_{name}"], counts[f"n_valid_{name}"]
        fraction = Fraction(successful, total)
        coverage[name] = {
            "valid_fraction": float(fraction),
            "valid_fraction_exact": str(fraction),
            "failure_fraction": float(1 - fraction),
            "failure_fraction_exact": str(1 - fraction),
        }
    return {
        "schema_version": 1,
        "denominator": "all_original_claims",
        "counts": counts,
        "coverage": coverage,
        "states": states,
        "terminal": {"state_index": len(states) - 1, **states[-1]},
        "eer": _eer(states, counts["n_genuine"], counts["n_impostor"]),
        "conditional": {
            "denominator": "valid_claims_only",
            "eer": _eer(states, counts["n_valid_genuine"], counts["n_valid_impostor"]),
        },
    }


def calibrate_gated_threshold(labels, scores, valid, *, target_far=0.01) -> dict:
    """Select the last attainable state satisfying an exact empirical FAR cap.

    The target is interpreted as Fraction(str(target_far)). The caller owns
    held-out calibration isolation; this provides no population FAR guarantee.
    """
    genuine, values, usable = _claims(labels, scores, valid)
    _finite_real(target_far, "target_far")
    if not 0 <= target_far <= 1:
        raise ValueError("target_far must be in [0, 1]")
    target = Fraction(str(target_far))
    _, states = _states(genuine, values, usable)
    selected_index = 0
    for index, state in enumerate(states):
        if Fraction(state["counts"]["fp"], state["counts"]["n_impostor"]) <= target:
            selected_index = index
        else:
            break
    return {
        "schema_version": 1,
        "denominator": "all_original_claims",
        "target_far": float(target),
        "target_far_exact": str(target),
        "selection_contract": (
            "most_permissive_attainable_state_with_empirical_far_at_most_target"
        ),
        "state_index": selected_index,
        **states[selected_index],
    }


def apply_gated_policy(scores, valid, policy) -> np.ndarray:
    """Return a fresh boolean acceptance vector; invalid claims always reject.

    Both vectors are validated even for the explicit reject_all control policy.
    Policies must be dicts with exactly the keys kind and threshold.
    """
    values, usable = _scores_and_valid(scores, valid)
    if not isinstance(policy, dict) or set(policy) != {"kind", "threshold"}:
        raise ValueError("policy must be a dict with exactly kind and threshold")
    kind, threshold = policy["kind"], policy["threshold"]
    if not isinstance(kind, str):
        raise ValueError("policy kind must be a string")  # noqa: TRY004 -- public validation contract
    if kind == "reject_all":
        if threshold is not None:
            raise ValueError("reject_all requires threshold None")
        return np.zeros(len(values), dtype=np.bool_)
    if kind != "score_at_least":
        raise ValueError("unknown policy kind")
    threshold = _finite_real(threshold, "policy threshold")
    return np.asarray(
        [
            bool(score >= threshold) if ok else False
            for score, ok in zip(values, usable, strict=True)
        ],
        dtype=np.bool_,
    )
