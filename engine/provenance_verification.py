"""Raw-support-aware, opt-in within-recording verification protocols.

There is no rejection sampling, guessed window purge, session inference, model
fitting, threshold selection, or dataset access in this module.
"""

from __future__ import annotations

import random

import numpy as np

from data.window_provenance import WindowProvenance, _integer, _integer_array


def _aligned_ids(values, n, name):
    ids = _integer_array(values, name)
    if len(ids) != n:
        raise ValueError(f"{name} must have one entry per provenance window")
    return ids


def _require_provenance(provenance):
    if not isinstance(provenance, WindowProvenance):
        raise ValueError("WindowProvenance is required; raw offsets cannot be guessed")  # noqa: TRY004 -- public validation contract


def _add_gap(values, gap):
    # Saturating addition avoids int64 wraparound. A start cannot equal int64's
    # maximum because its exclusive stop must be strictly greater.
    return values + np.minimum(gap, np.iinfo(np.int64).max - values)


def support_conflicts(
    provenance: WindowProvenance, left_indices, right_indices, *, purge_samples: int = 0
) -> np.ndarray:
    """Elementwise shared-support/too-small-gap check for aligned index vectors.

    Touching half-open intervals are safe for a zero gap. A positive gap counts
    valid parsed raw records globally, including across segment boundaries.
    Timestamps and segment labels never establish independent support.
    """
    _require_provenance(provenance)
    gap = _integer(purge_samples, "purge_samples")
    left = _integer_array(left_indices, "left_indices")
    right = _integer_array(right_indices, "right_indices")
    if len(left) != len(right):
        raise ValueError("left_indices and right_indices must be equally sized")
    if np.any(left >= len(provenance)) or np.any(right >= len(provenance)):
        raise ValueError("support check contains out-of-range window indices")
    return (
        provenance.start_rows[left] < _add_gap(provenance.stop_rows[right], gap)
    ) & (provenance.start_rows[right] < _add_gap(provenance.stop_rows[left], gap))


def _safe_against(provenance, references, candidates, gap):
    """Check each candidate against an entire reference union in O(N log N)."""
    if not len(references) or not len(candidates):
        return np.ones(len(candidates), dtype=bool)
    order = references[np.argsort(provenance.start_rows[references], kind="stable")]
    starts = provenance.start_rows[order]
    max_stops = np.maximum.accumulate(provenance.stop_rows[order])
    position = (
        np.searchsorted(
            starts, _add_gap(provenance.stop_rows[candidates], gap), side="left"
        )
        - 1
    )
    safe = position < 0
    active = ~safe
    safe[active] = (
        _add_gap(max_stops[position[active]], gap)
        <= provenance.start_rows[candidates[active]]
    )
    return safe


def provenance_enrollment_split(
    subject_ids,
    provenance: WindowProvenance,
    enrollment_windows: int = 5,
    purge_samples: int = 0,
):
    """Reserve each subject's first n source windows; purge actual raw support.

    Source interval order is recovered even for shuffled subsets. All probe
    intervals must avoid every reserved enrollment prefix, including the raw
    sample gap. Subjects without a surviving probe are excluded completely.
    Returned indices address the supplied arrays, in their original row order.
    No independence between sessions, windows, or subjects is asserted.
    """
    _require_provenance(provenance)
    n = _integer(enrollment_windows, "enrollment_windows", 1)
    gap = _integer(purge_samples, "purge_samples")
    ids = _aligned_ids(subject_ids, len(provenance), "subject_ids")
    groups = {}
    for row, subject in enumerate(ids):
        groups.setdefault(int(subject), []).append(row)
    ordered = {
        subject: np.asarray(rows, dtype=np.int64)[
            np.argsort(provenance.start_rows[rows], kind="stable")
        ]
        for subject, rows in groups.items()
    }
    prefixes = [rows[:n] for rows in ordered.values() if len(rows) > n]
    references = np.concatenate(prefixes) if prefixes else np.empty(0, dtype=np.int64)
    tails = [rows[n:] for rows in ordered.values() if len(rows) > n]
    candidates = np.concatenate(tails) if tails else np.empty(0, dtype=np.int64)
    safe_rows = np.zeros(len(ids), dtype=bool)
    safe_rows[candidates] = _safe_against(provenance, references, candidates, gap)
    enroll, probes, purged = [], [], []
    included, excluded, counts = [], [], {}
    for subject, rows in sorted(ordered.items()):
        tail = rows[n:]
        safe = safe_rows[tail]
        eligible = len(rows) > n and np.any(safe)
        counts[str(subject)] = {
            "n_windows": len(rows),
            "eligible": bool(eligible),
            "n_enrollment": n if eligible else 0,
            "n_probes": int(np.count_nonzero(safe)) if eligible else 0,
            "n_purged": int(np.count_nonzero(~safe)) if eligible else 0,
        }
        if not eligible:
            excluded.append(subject)
            continue
        included.append(subject)
        enroll.extend(rows[:n])
        probes.extend(tail[safe])
        purged.extend(tail[~safe])
    enroll_idx = np.sort(np.asarray(enroll, dtype=np.int64))
    probe_idx = np.sort(np.asarray(probes, dtype=np.int64))
    coverage = {
        "protocol": "raw_support_prefix_enrollment_v1",
        "enrollment_windows": n,
        "purge_samples": gap,
        "purge_unit": "valid parsed raw records between half-open supports",
        "prefix_order": "start_rows ascending, stable input order for ties",
        "reserved_prefix_policy": "all subjects with more than n windows; no re-admission",
        "n_input_windows": len(ids),
        "n_input_subjects": len(groups),
        "included_subject_ids": included,
        "excluded_subject_ids": excluded,
        "exclusion_reason": "insufficient prefix windows or no raw-support-safe probe",
        "n_enrollment": len(enroll_idx),
        "n_probes": len(probe_idx),
        "n_purged": len(purged),
        "purged_indices": sorted(map(int, purged)),
        "n_excluded_windows": len(ids) - len(enroll) - len(probes) - len(purged),
        "per_subject": counts,
        "raw_offset_nonoverlap_verified": True,
        "cross_session": False,
        "provenance": provenance.summary(),
    }
    return enroll_idx, probe_idx, coverage


class _IntervalPairs:
    """Rankable nonoverlapping unordered pairs, with optional subject exclusion.

    Each anchor has a suffix of safe later starts. Subtract its subject's
    positions for impostor pairs. Storage is linear; selecting a rank uses
    bounded binary searches, even when only one admissible pair exists.
    """

    def __init__(self, rows, provenance, gap, exclude_subjects=None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.rows = self.rows[
            np.argsort(provenance.start_rows[self.rows], kind="stable")
        ]
        starts = provenance.start_rows[self.rows]
        self.first = np.searchsorted(
            starts, _add_gap(provenance.stop_rows[self.rows], gap), side="left"
        )
        counts = len(self.rows) - self.first
        self.subjects = None
        self.positions = {}
        if exclude_subjects is not None:
            self.subjects = exclude_subjects[self.rows]
            positions = {}
            for position, subject in enumerate(self.subjects):
                positions.setdefault(int(subject), []).append(position)
            self.positions = {
                subject: np.asarray(p, dtype=np.int64)
                for subject, p in positions.items()
            }
            for p in self.positions.values():
                counts[p] -= len(p) - np.searchsorted(p, self.first[p])
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        self.total = int(self.cumulative[-1]) if len(self.cumulative) else 0

    def pair_at(self, rank):
        if not 0 <= rank < self.total:
            raise ValueError("pair rank is outside the admissible pair population")
        anchor = int(np.searchsorted(self.cumulative, rank, side="right"))
        previous = int(self.cumulative[anchor - 1]) if anchor else 0
        offset = int(rank) - previous
        first = int(self.first[anchor])
        if self.subjects is None:
            other = first + offset
        else:
            banned = self.positions[int(self.subjects[anchor])]
            before = int(np.searchsorted(banned, first))
            low, high = first, len(self.rows) - 1
            while low < high:
                middle = (low + high) // 2
                available = (
                    middle
                    - first
                    + 1
                    - (int(np.searchsorted(banned, middle, side="right")) - before)
                )
                if available > offset:
                    high = middle
                else:
                    low = middle + 1
            other = low
        i, j = int(self.rows[anchor]), int(self.rows[other])
        return min(i, j), max(i, j)


def provenance_verification_pairs(
    subject_ids,
    provenance: WindowProvenance,
    n_pairs: int = 10000,
    seed: int = 42,
    *,
    activity_ids=None,
    purge_samples: int = 0,
):
    """Balance genuine/impostor budgets while excluding shared raw support.

    Genuine sampling is uniform over eligible subjects, then over their safe
    pairs. Impostor sampling is uniform over safe unordered different-subject
    pairs. Both sample WITH replacement and cycle over viable activity strata.
    Pair-index preprocessing/storage is O(N log N)/O(N); returned pairs and
    duplicate-count bookkeeping additionally use O(n_pairs) storage.
    Reused windows and repeated pairs remain dependent observations.
    """
    _require_provenance(provenance)
    count = _integer(n_pairs, "n_pairs", 2)
    gap = _integer(purge_samples, "purge_samples")
    seed = _integer(seed, "seed")
    ids = _aligned_ids(subject_ids, len(provenance), "subject_ids")
    activities = (
        np.zeros(len(ids), dtype=np.int64)
        if activity_ids is None
        else _aligned_ids(activity_ids, len(ids), "activity_ids")
    )
    groups = {}
    for row, (subject, activity) in enumerate(zip(ids, activities)):
        groups.setdefault(int(activity), {}).setdefault(int(subject), []).append(row)
    eligible, available = {}, {}
    for activity, subjects in sorted(groups.items()):
        genuine = {
            subject: _IntervalPairs(rows, provenance, gap)
            for subject, rows in sorted(subjects.items())
        }
        genuine = {subject: index for subject, index in genuine.items() if index.total}
        rows = [row for subject_rows in subjects.values() for row in subject_rows]
        impostor = _IntervalPairs(rows, provenance, gap, exclude_subjects=ids)
        all_genuine = sum(len(r) * (len(r) - 1) // 2 for r in subjects.values())
        all_impostor = len(rows) * (len(rows) - 1) // 2 - all_genuine
        safe_genuine = sum(index.total for index in genuine.values())
        available[str(activity)] = {
            "n_available_genuine_pairs": safe_genuine,
            "n_available_impostor_pairs": impostor.total,
            "n_excluded_raw_support_genuine_pairs": all_genuine - safe_genuine,
            "n_excluded_raw_support_impostor_pairs": all_impostor - impostor.total,
            "eligible_genuine_subject_ids": list(genuine),
            "eligible": bool(genuine and impostor.total),
        }
        if genuine and impostor.total:
            eligible[activity] = (genuine, list(genuine), impostor)
    if not eligible:
        raise ValueError(
            "cannot form raw-support-safe genuine/impostor pairs: need a nonoverlapping "
            "same-subject pair and a nonoverlapping different-subject pair in a shared "
            "activity; enlarge the subset or reduce purge_samples"
        )
    rng = random.Random(seed)
    strata = list(eligible)
    rng.shuffle(strata)
    budgets = (count // 2, count - count // 2)
    pairs, labels = [], []
    for genuine_label, budget in zip((1, 0), budgets):
        for k in range(budget):
            genuine, genuine_subjects, impostor = eligible[strata[k % len(strata)]]
            index = genuine[rng.choice(genuine_subjects)] if genuine_label else impostor
            pairs.append(index.pair_at(rng.randrange(index.total)))
            labels.append(genuine_label)
    pair_array = np.asarray(pairs, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    unique = [
        len(set(map(tuple, pair_array[label_array == label]))) for label in (1, 0)
    ]
    context = {
        "sampling": "genuine_subject_uniform_impostor_pair_uniform_with_replacement",
        "pairing_version": "raw_support_v1",
        "seed": seed,
        "requested_n_pairs": count,
        "n_genuine_pairs": budgets[0],
        "n_impostor_pairs": budgets[1],
        "n_unique_genuine_pairs": unique[0],
        "n_unique_impostor_pairs": unique[1],
        "n_unique_pairs": sum(unique),
        "n_repeated_pairs": count - sum(unique),
        "same_activity_pairs": activity_ids is not None,
        "eligible_activity_ids": list(eligible) if activity_ids is not None else None,
        "excluded_activity_ids": sorted(set(groups) - set(eligible))
        if activity_ids is not None
        else [],
        "per_activity": available,
        "n_subjects": len(np.unique(ids)),
        "n_windows_used": len(np.unique(pair_array)),
        "purge_samples": gap,
        "purge_unit": "valid parsed raw records between half-open supports",
        "raw_offset_nonoverlap_verified": True,
        "pair_independence": "not_independent_windows_and_pairs_can_recur",
        "cross_session": False,
        "provenance": provenance.summary(),
    }
    return pair_array, label_array, context
