"""Opt-in, ordered within-recording enrollment/probe verification.

Call separately for validation and test subjects. Indices refer to the supplied
unshuffled, single-activity source windows; they are not raw sample offsets.
See docs/enrollment_methods.md for the conditional overlap guarantee.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _subjects(value):
    ids = _numpy(value)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("subjects must be a 1D array of nonnegative integer IDs")
    return ids


def _budget(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")  # noqa: TRY004 -- public validation contract
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def enrollment_split(subject_ids, enrollment_windows: int = 5, purge_windows: int = 2):
    """Use each subject's first n windows, skip a fixed gap, probe the rest.

    A subject needs at least n + gap + 1 windows, otherwise all its windows are
    excluded and reported. Returned int64 indices preserve input row order.
    The caller must preserve original source order after activity/subject
    filtering. The function cannot detect shuffling or prove raw nonoverlap.
    A zero purge is allowed for explicit controls and supplies no overlap guard.
    """
    n = _budget(enrollment_windows, "enrollment_windows", 1)
    gap = _budget(purge_windows, "purge_windows", 0)
    ids = _subjects(subject_ids)
    enroll, probe, purged = [], [], []
    included, excluded, counts = [], [], {}
    for subject in np.unique(ids):
        rows = np.flatnonzero(ids == subject)
        eligible = len(rows) >= n + gap + 1
        sid = int(subject)
        counts[str(sid)] = {
            "n_windows": len(rows),
            "eligible": bool(eligible),
            "n_enrollment": n if eligible else 0,
            "n_purged": gap if eligible else 0,
            "n_probes": len(rows) - n - gap if eligible else 0,
        }
        if not eligible:
            excluded.append(sid)
            continue
        included.append(sid)
        enroll.extend(rows[:n])
        purged.extend(rows[n : n + gap])
        probe.extend(rows[n + gap :])
    enroll = np.sort(np.asarray(enroll, dtype=np.int64))
    probe = np.sort(np.asarray(probe, dtype=np.int64))
    coverage = {
        "protocol": "ordered_prefix_enrollment_v1",
        "enrollment_windows": n,
        "purge_windows": gap,
        "n_input_windows": len(ids),
        "n_input_subjects": len(counts),
        "included_subject_ids": included,
        "excluded_subject_ids": excluded,
        "exclusion_reason": "fewer than enrollment_windows + purge_windows + 1 windows",
        "n_enrollment": len(enroll),
        "n_probes": len(probe),
        "n_purged": len(purged),
        "purged_indices": sorted(map(int, purged)),
        "n_excluded_windows": len(ids) - len(enroll) - len(probe) - len(purged),
        "per_subject": counts,
        "ordering_requirement": "unshuffled source order, one activity",
        "overlap_control": "conditional ordered-index purge; raw offsets unavailable",
        "raw_offset_nonoverlap_verified": False,
        "cross_session": False,
    }
    return enroll, probe, coverage


def _indices(value, name, size):
    rows = _numpy(value)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise ValueError(f"{name} must be a nonempty 1D integer index array")
    if np.any(rows < 0) or np.any(rows >= size):
        raise ValueError(f"{name} contains out-of-range indices")
    rows = rows.astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"{name} must not repeat windows")
    return rows


def _unit_rows(values, name):
    # Scaling first avoids overflow for large but finite embeddings.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f"{name} contains a zero-norm row")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def enrollment_scores(embeddings, subjects, enroll_idx, probe_idx):
    """Score every probe against every enrolled subject exactly once.

    Templates are unit-normalized means of unit-normalized enrollment vectors.
    Scores are cosine - 1 in [-2, 0], compatible with engine.evaluate's threshold
    convention (higher is genuine, accept >= threshold). No fitting, calibration
    or random balancing occurs. Output is probe-major, then sorted claimed ID.
    This validates row disjointness; use enrollment_split for the purge policy.
    """
    ids = _subjects(subjects)
    z = _numpy(embeddings)
    if z.dtype.kind not in "iuf":
        raise ValueError("embeddings must be real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        z = z.astype(np.float64)
    if z.ndim != 2 or z.shape[0] != len(ids) or z.shape[1] == 0:
        raise ValueError("embeddings must have shape (len(subjects), positive D)")
    if not np.isfinite(z).all():
        raise ValueError("embeddings must be finite")
    enroll = _indices(enroll_idx, "enroll_idx", len(ids))
    probe = _indices(probe_idx, "probe_idx", len(ids))
    if np.intersect1d(enroll, probe).size:
        raise ValueError("enrollment and probe windows must be disjoint")
    enrolled_ids = np.unique(ids[enroll])
    if len(enrolled_ids) < 2:
        raise ValueError("verification requires at least two enrolled subjects")
    if not np.array_equal(enrolled_ids, np.unique(ids[probe])):
        raise ValueError(
            "every enrolled subject needs probes; every probe needs enrollment"
        )
    enrollment_vectors = _unit_rows(z[enroll], "enrollment embeddings")
    templates = np.stack(
        [enrollment_vectors[ids[enroll] == sid].mean(axis=0) for sid in enrolled_ids]
    )
    if np.any(np.max(np.abs(templates), axis=1) <= 1e-12):
        raise ValueError("enrollment vectors cancel to a zero-norm template")
    templates = _unit_rows(templates, "enrollment templates")
    probes = _unit_rows(z[probe], "probe embeddings")
    scores = (np.clip(probes @ templates.T, -1.0, 1.0) - 1.0).ravel()
    labels = (ids[probe, None] == enrolled_ids[None, :]).astype(np.int64).ravel()
    per_subject = {
        str(int(sid)): {
            "n_enrollment": int(np.sum(ids[enroll] == sid)),
            "n_probes": int(np.sum(ids[probe] == sid)),
            "n_genuine": int(np.sum(ids[probe] == sid)),
            "n_impostor_claims": int(np.sum(ids[probe] != sid)),
        }
        for sid in enrolled_ids
    }
    metadata = {
        "protocol": "ordered_prefix_enrollment_v1",
        "score": "cosine_similarity - 1",
        "score_order": "probe-major, then sorted claimed_subject_ids",
        "template_aggregation": "unit_mean_of_unit_enrollment_vectors",
        "claimed_subject_ids": list(map(int, enrolled_ids)),
        "probe_indices": probe.tolist(),
        "enrollment_indices": enroll.tolist(),
        "template_norms": np.linalg.norm(templates, axis=1).tolist(),
        "n_subjects": len(enrolled_ids),
        "n_enrollment": len(enroll),
        "n_probes": len(probe),
        "n_comparisons": len(scores),
        "n_genuine": int(labels.sum()),
        "n_impostor": int(len(labels) - labels.sum()),
        "n_repeated_comparisons": 0,
        "per_subject": per_subject,
        "cross_session": False,
        "raw_offset_nonoverlap_verified": False,
        "threshold_calibrated": False,
    }
    return labels, scores, metadata
