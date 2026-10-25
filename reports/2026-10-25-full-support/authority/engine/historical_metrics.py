"""Statically extracted pure source; provenance in source-origins.json."""
import numpy as np
from common import fingerprint as object_hash


def roc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != scores.shape
        or not len(labels)
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("finite scores and both binary classes are required")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, positive = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    tp, fp = np.cumsum(positive)[ends], np.cumsum(~positive)[ends]
    far = np.r_[0.0, fp / np.sum(labels == 0)]
    frr = np.r_[1.0, (np.sum(labels == 1) - tp) / np.sum(labels == 1)]
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), sorted_scores[ends]]
    if not np.isfinite(thresholds).all():
        raise ValueError("finite reject-all threshold is required")
    return far, frr, thresholds

def eer(labels, scores):
    far, frr, _ = roc(labels, scores)
    diff = far - frr
    hi = int(np.searchsorted(diff, 0.0))
    if diff[hi] == 0:
        return float(far[hi])
    lo = hi - 1
    return float(far[lo] - diff[lo] / (diff[hi] - diff[lo]) * (far[hi] - far[lo]))

def fixed_metrics(labels, scores, threshold):
    roc(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    labels, scores = np.asarray(labels), np.asarray(scores)
    genuine, impostor, accepted = labels == 1, labels == 0, scores >= threshold
    fa, fr = int(np.sum(accepted & impostor)), int(np.sum(~accepted & genuine))
    return {
        "threshold": float(threshold),
        "far": fa / int(impostor.sum()),
        "frr": fr / int(genuine.sum()),
        "false_accepts": fa,
        "false_rejects": fr,
        "n_genuine": int(genuine.sum()),
        "n_impostor": int(impostor.sum()),
        "decision_rule": "score >= threshold",
    }

def report(scored, threshold=None):
    labels, scores, metadata = scored["labels"], scored["scores"], scored["metadata"]
    claims, probes = (
        np.asarray(metadata["claim_subject_ids"]),
        np.asarray(metadata["probe_subject_ids"]),
    )
    far, frr, _ = roc(labels, scores)
    result = {
        "eer": eer(labels, scores),
        "roc_auc": float(np.sum(np.diff(far) * ((1 - frr[1:]) + (1 - frr[:-1])) / 2)),
        "coverage": metadata["coverage"],
        "claim_graph_sha256": object_hash(
            {
                key: metadata[key]
                for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
            }
        ),
        "per_subject_eer": {
            str(int(subject)): eer(
                np.asarray(labels)[probes == subject],
                np.asarray(scores)[probes == subject],
            )
            for subject in np.unique(probes)
        },
    }
    if threshold is not None:
        result["fixed_threshold"] = fixed_metrics(labels, scores, threshold)
        accepted, positive = np.asarray(scores) >= threshold, np.asarray(labels) == 1
        result["per_subject_errors"] = {}
        for subject in np.unique(probes):
            genuine = positive & (probes == subject)
            impostor = ~positive & (claims == subject)
            result["per_subject_errors"][str(int(subject))] = {
                "false_rejects": int(np.sum(~accepted & genuine)),
                "n_genuine": int(genuine.sum()),
                "false_accepts": int(np.sum(accepted & impostor)),
                "n_impostor": int(impostor.sum()),
                "frr": float(np.sum(~accepted & genuine) / genuine.sum()),
                "far": float(np.sum(accepted & impostor) / impostor.sum()),
            }
    return result
