"""Synthetic-only raw-coordinate, alignment, and admissible-pair controls."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from itertools import combinations
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data.adapter import (
    WindowData,
    _read_wisdm_v1,
    _wisdm_subject_split,
    build_loader,
    load_windows,
)
from data.window_provenance import (
    ProvenancedWindowData,
    WindowProvenance,
    load_windows_with_provenance,
)
from engine.provenance_verification import (
    _IntervalPairs,
    provenance_enrollment_split,
    provenance_verification_pairs,
    support_conflicts,
)


def metadata(starts, stops, *, segments=None):
    starts = np.asarray(starts, dtype=np.int64)
    return WindowProvenance(
        source_sha256="a" * 64,
        segment_ids=np.zeros(len(starts), dtype=np.int64)
        if segments is None
        else np.asarray(segments, dtype=np.int64),
        start_rows=starts,
        stop_rows=np.asarray(stops, dtype=np.int64),
    )


@pytest.fixture
def raw_source(tmp_path):
    root = tmp_path / "wisdm"
    root.mkdir()
    path = root / "WISDM_ar_v1.1_raw.txt"
    records = []
    valid = 0
    base = 9_007_199_254_740_993
    # Multiple semicolon-delimited records on ONE physical line, with valid
    # ordinals encoded in acc_x to verify provenance independently of the loader.
    for subject in (4, 1, 3, 2):
        for run, times in enumerate(
            (
                [base + j * 50_000_000 for j in range(6)],
                [base + (j + 6) * 50_000_000 for j in range(6)],
                [base + j * 50_000_000 for j in range(6)],  # reset
                [base + (j + 5) * 50_000_000 for j in range(6)],  # duplicate
                [base + 3_000_000_000 + j * 50_000_000 for j in range(6)],
                [base + 4_000_000_000 + j * 50_000_000 for j in range(6)],
            )
        ):
            if run == 1:
                records.extend(("malformed", f"{subject},Walking,{base},nan,0,0"))
            activity = "Jogging" if run == 5 else "Walking"
            for timestamp in times:
                records.append(f"{subject},{activity},{timestamp},{valid},{subject},1")
                valid += 1
    path.write_text(";".join(records) + ";\n", encoding="utf-8")
    return tmp_path, path


def load_fixture(root, split="train", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return load_windows_with_provenance(
            "wisdm_v1", split, window_size=4, stride=2, data_root=str(root), **kwargs
        )


@pytest.mark.parametrize("split", ["train", "test"])
def test_loader_exact_alignment_invalid_gaps_resets_and_integer_timestamps(
    raw_source, split
):
    root, path = raw_source
    data = load_fixture(root, split)
    reference = load_windows("wisdm_v1", split, 4, 2, str(root))
    assert isinstance(data, WindowData)
    assert isinstance(data, ProvenancedWindowData)
    assert torch.equal(data.X, reference.X)
    assert reference.y_subject is not None
    assert torch.equal(data.y_subject, reference.y_subject)
    assert torch.equal(data.y_activity, reference.y_activity)
    assert data.subject_ids == reference.subject_ids == [1, 2, 3, 4]
    assert data.activity_names == reference.activity_names
    assert data.sampling_rate == reference.sampling_rate
    provenance = data.provenance
    assert provenance.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not provenance.summary()["cross_session"]
    frame = _read_wisdm_v1(path)
    assert len(path.read_text().splitlines()) == 1
    np.testing.assert_array_equal(frame._source_row, np.arange(len(frame)))
    assert frame.attrs["invalid_records"] == 8
    for i, (start, stop, segment) in enumerate(
        zip(provenance.start_rows, provenance.stop_rows, provenance.segment_ids)
    ):
        np.testing.assert_array_equal(data.X[i, 0].numpy(), np.arange(start, stop))
        # Every synthetic contiguous run is exactly six VALID records. The
        # malformed records consume no source ordinal but still split a run.
        assert segment == start // 6 * 6
        assert segment <= start < stop <= segment + 6
        assert provenance.first_timestamps_ns[i] == int(frame.timestamp.iloc[start])
        assert provenance.last_timestamps_ns[i] == int(frame.timestamp.iloc[stop - 1])
        assert provenance.first_timestamps_ns[i] % 2 == 1
    assert len(next(iter(build_loader(data, 3, False)))) == 3
    json.dumps(provenance.summary(), allow_nan=False)


def test_subsets_preserve_exact_order_duplicates_labels_and_global_identity_map(
    raw_source,
):
    data = load_fixture(raw_source[0])
    selectors = (
        [5, 1, 5, 0],
        slice(None, None, -2),
        np.arange(len(data.X)) % 3 == 0,
        torch.tensor([3, 0]),
        [],
    )
    for selection in selectors:
        subset = data.take(selection)
        rows = (
            np.arange(len(data.X))[selection]
            if not isinstance(selection, torch.Tensor)
            else selection.numpy()
        )
        np.testing.assert_array_equal(subset.X.numpy(), data.X.numpy()[rows])
        np.testing.assert_array_equal(
            subset.y_subject.numpy(), data.y_subject.numpy()[rows]
        )
        np.testing.assert_array_equal(
            subset.y_activity.numpy(), data.y_activity.numpy()[rows]
        )
        for name in (
            "segment_ids",
            "start_rows",
            "stop_rows",
            "first_timestamps_ns",
            "last_timestamps_ns",
        ):
            np.testing.assert_array_equal(
                getattr(subset.provenance, name), getattr(data.provenance, name)[rows]
            )
        assert subset.subject_ids == data.subject_ids
        assert subset.provenance.source_sha256 == data.provenance.source_sha256
    with pytest.raises(ValueError, match="read-only"):
        data.provenance.start_rows[0] = 42
    for selection in ([True], [0.5], [-1], [len(data.X)], [[0]]):
        with pytest.raises(ValueError):
            data.take(selection)


def test_optional_timestamps_empty_windows_and_nested_official_path(raw_source):
    root, path = raw_source
    nested = path.parent / "WISDM_ar_v1.1"
    nested.mkdir()
    path.rename(nested / path.name)
    data = load_fixture(root, include_timestamps=False)
    assert data.provenance.first_timestamps_ns is None
    assert data.provenance.last_timestamps_ns is None
    assert data.take([0]).provenance.first_timestamps_ns is None
    empty = load_windows_with_provenance("wisdm_v1", "train", 1000, 5, str(root))
    assert empty.X.shape == (0, 3, 1000)
    assert len(empty.provenance) == 0
    assert empty.provenance.first_timestamps_ns is not None
    assert empty.provenance.first_timestamps_ns.dtype == np.int64


def test_source_digest_detects_stale_cache_without_metadata_change(raw_source):
    root, path = raw_source
    load_fixture(root)
    stat = path.stat()
    original = path.read_text()
    modified = original.replace(",0,4,1;", ",9,4,1;", 1)
    assert modified != original and len(modified) == len(original)
    path.write_text(modified)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed since parsing"):
        load_fixture(root)


def test_removed_subject_records_leave_raw_gaps_and_do_not_stitch_runs(tmp_path):
    root = tmp_path / "wisdm"
    root.mkdir()
    kept_ids, removed_ids = _wisdm_subject_split([1, 2])
    kept, removed = int(kept_ids[0]), int(removed_ids[0])
    subjects = [kept, kept, removed, removed, kept, kept]
    records = [
        f"{subject},Walking,{1_000_000_000 + row * 50_000_000},{row},0,1;"
        for row, subject in enumerate(subjects)
    ]
    (root / "WISDM_ar_v1.1_raw.txt").write_text("".join(records))
    data = load_windows_with_provenance("wisdm_v1", "train", 2, 1, str(tmp_path))
    np.testing.assert_array_equal(data.provenance.start_rows, [0, 4])
    np.testing.assert_array_equal(data.provenance.stop_rows, [2, 6])
    np.testing.assert_array_equal(data.provenance.segment_ids, [0, 4])
    assert not len(
        load_windows_with_provenance("wisdm_v1", "train", 3, 1, str(tmp_path)).X
    )


@pytest.mark.parametrize("source", ["har", "wisdm", "unknown"])
def test_unsupported_provenance_fails_before_any_raw_load(source):
    with patch("data.window_provenance._read_wisdm_v1") as reader:
        with pytest.raises(ValueError, match="HAR windows lack raw source offsets"):
            load_windows_with_provenance(source, "train")
        reader.assert_not_called()


@pytest.mark.parametrize("value", [1.5, 2.0, True, np.bool_(True), "2", -1, 0])
def test_window_parameters_reject_fractional_or_noninteger_values_before_io(value):
    for name in ("window_size", "stride"):
        with pytest.raises(ValueError, match=name):
            load_windows_with_provenance("wisdm_v1", "train", **{name: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_sha256": "bad"},
        {"source": "har"},
        {"start_rows": [0.5]},
        {"start_rows": np.array([2**63], dtype=np.uint64)},
        {"stop_rows": [0]},
        {"segment_ids": [1]},
        {"segment_ids": [0, 0]},
        {"first_timestamps_ns": [1]},
        {"first_timestamps_ns": [3], "last_timestamps_ns": [2]},
    ],
)
def test_invalid_provenance_is_not_silently_coerced(kwargs):
    arguments: dict[str, Any] = {
        "source_sha256": "a" * 64,
        "segment_ids": [0],
        "start_rows": [0],
        "stop_rows": [2],
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        WindowProvenance(**arguments)


def test_enrollment_uses_raw_gap_and_full_interval_union_after_shuffled_subset():
    starts = np.r_[np.arange(0, 21, 2), np.arange(100, 121, 2), [200, 202]]
    provenance = metadata(starts, starts + 8)
    subjects = np.repeat([10, 20, 30], [11, 11, 2])
    order = np.random.default_rng(7).permutation(len(starts))
    shuffled = provenance.take(order)
    ids = subjects[order]
    enroll, probes, coverage = provenance_enrollment_split(ids, shuffled, 2, 3)
    np.testing.assert_array_equal(np.sort(order[enroll]), [0, 1, 11, 12])
    # Prefix union ends at 10; three valid raw samples require start >= 13.
    np.testing.assert_array_equal(np.sort(order[probes]), [7, 8, 9, 10, 18, 19, 20, 21])
    assert coverage["excluded_subject_ids"] == [30]
    assert coverage["n_purged"] == 10
    assert coverage["raw_offset_nonoverlap_verified"]
    assert not coverage["cross_session"]
    assert not support_conflicts(
        shuffled,
        np.repeat(enroll, len(probes)),
        np.tile(probes, len(enroll)),
        purge_samples=3,
    ).any()
    assert np.all(np.diff(enroll) > 0) and np.all(np.diff(probes) > 0)
    json.dumps(coverage, allow_nan=False)


def test_enrollment_checks_other_subject_support_and_excludes_no_probe_subjects():
    # Caller-supplied metadata may include conflicting identities. No cross-role
    # raw support is allowed, even when the claimed subject IDs differ.
    provenance = metadata([0, 3, 10, 4, 12, 20], [8, 11, 18, 12, 20, 28])
    enroll, probe, _ = provenance_enrollment_split([0, 0, 0, 1, 1, 1], provenance, 1)
    assert not support_conflicts(
        provenance, np.repeat(enroll, len(probe)), np.tile(probe, len(enroll))
    ).any()
    assert 2 not in probe  # [10,18) also overlaps subject 1's [4,12) prefix.
    empty = metadata([], [])
    e, p, coverage = provenance_enrollment_split(np.empty(0, dtype=np.int64), empty)
    assert not len(e) and not len(p) and coverage["n_input_subjects"] == 0


def test_touching_intervals_and_gaps_across_resets_do_not_infer_sessions():
    provenance = metadata([0, 4, 9], [4, 8, 13], segments=[0, 4, 9])
    np.testing.assert_array_equal(
        support_conflicts(provenance, [0, 1], [1, 2]), [False, False]
    )
    np.testing.assert_array_equal(
        support_conflicts(provenance, [0, 1], [1, 2], purge_samples=1), [True, False]
    )
    _, probes, coverage = provenance_enrollment_split([0, 0, 0], provenance, 1, 1)
    np.testing.assert_array_equal(probes, [2])
    assert not coverage["cross_session"]


def test_interval_pair_index_matches_exhaustive_ragged_intervals_and_labels():
    rng = np.random.default_rng(21)
    for _ in range(15):
        starts = rng.integers(0, 30, 17)
        stops = starts + rng.integers(1, 20, len(starts))
        ids = rng.integers(0, 4, len(starts))
        provenance = metadata(starts, stops)
        rows = rng.permutation(len(starts))[:13]
        for gap in (0, 3, 100):
            for exclude in (None, ids):
                index = _IntervalPairs(rows, provenance, gap, exclude_subjects=exclude)
                expected = {
                    tuple(sorted((int(i), int(j))))
                    for i, j in combinations(rows, 2)
                    if (stops[i] + gap <= starts[j] or stops[j] + gap <= starts[i])
                    and (exclude is None or ids[i] != ids[j])
                }
                actual = [index.pair_at(rank) for rank in range(index.total)]
                assert len(actual) == len(set(actual)) == len(expected)
                assert set(actual) == expected
                with pytest.raises(ValueError):
                    index.pair_at(index.total)


def test_balanced_reproducible_same_activity_pairs_have_no_shared_support():
    starts = np.tile(np.arange(0, 25, 3), 4) + np.repeat([0, 100, 200, 300], 9)
    ids = np.repeat([0, 1, 0, 1], 9)
    activities = np.repeat([0, 0, 1, 1], 9)
    provenance = metadata(starts, starts + 8)
    first = provenance_verification_pairs(
        ids, provenance, 101, 9, activity_ids=activities, purge_samples=2
    )
    pairs, labels, context = first
    repeated = provenance_verification_pairs(
        ids, provenance, 101, 9, activity_ids=activities, purge_samples=2
    )
    np.testing.assert_array_equal(pairs, repeated[0])
    np.testing.assert_array_equal(labels, repeated[1])
    assert context == repeated[2]
    np.testing.assert_array_equal(
        ids[pairs[:, 0]] == ids[pairs[:, 1]], labels.astype(bool)
    )
    assert np.all(activities[pairs[:, 0]] == activities[pairs[:, 1]])
    assert not support_conflicts(
        provenance, pairs[:, 0], pairs[:, 1], purge_samples=2
    ).any()
    assert (context["n_genuine_pairs"], context["n_impostor_pairs"]) == (50, 51)
    assert context["raw_offset_nonoverlap_verified"] and not context["cross_session"]
    assert context["n_unique_pairs"] + context["n_repeated_pairs"] == 101
    assert not np.array_equal(
        pairs,
        provenance_verification_pairs(
            ids, provenance, 101, 10, activity_ids=activities, purge_samples=2
        )[0],
    )
    json.dumps(context, allow_nan=False)


def test_only_one_genuine_pair_in_thousands_is_sampled_without_rejection():
    n = 5000
    starts = np.r_[0, np.ones(n - 2, dtype=np.int64), 198, 1000]
    stops = np.r_[2, np.full(n - 2, 200, dtype=np.int64), 200, 1010]
    ids = np.r_[np.zeros(n, dtype=np.int64), 1]
    provenance = metadata(starts, stops)
    pairs, labels, context = provenance_verification_pairs(ids, provenance, 1001, 5)
    np.testing.assert_array_equal(pairs[labels == 1], np.tile([0, n - 1], (500, 1)))
    assert context["per_activity"]["0"]["n_available_genuine_pairs"] == 1
    assert context["n_unique_genuine_pairs"] == 1
    assert context["n_repeated_pairs"] >= 499
    assert not support_conflicts(provenance, pairs[:, 0], pairs[:, 1]).any()


def test_pair_sampling_excludes_impostor_shared_support_and_ineligible_activity():
    provenance = metadata([0, 10, 0, 10, 100, 101], [5, 15, 5, 15, 110, 111])
    ids = np.array([0, 0, 1, 1, 0, 1])
    activities = np.array([0, 0, 0, 0, 1, 1])
    pairs, labels, context = provenance_verification_pairs(
        ids, provenance, 200, activity_ids=activities
    )
    assert context["eligible_activity_ids"] == [0]
    assert context["excluded_activity_ids"] == [1]
    assert context["per_activity"]["0"]["n_available_impostor_pairs"] == 2
    assert context["per_activity"]["0"]["n_excluded_raw_support_impostor_pairs"] == 2
    assert set(map(tuple, pairs[labels == 0])) == {(0, 3), (1, 2)}
    assert not support_conflicts(provenance, pairs[:, 0], pairs[:, 1]).any()


def test_repeated_subset_windows_never_form_false_disjoint_genuine_pairs():
    provenance = metadata([0, 10, 20, 30], [5, 15, 25, 35]).take([0, 0, 1, 2, 3])
    pairs, _, _ = provenance_verification_pairs([0, 0, 0, 1, 1], provenance, 100)
    assert not support_conflicts(provenance, pairs[:, 0], pairs[:, 1]).any()
    assert (0, 1) not in set(map(tuple, pairs))


def test_no_safe_genuine_pair_fails_clearly_without_partial_trials():
    provenance = metadata([0, 1, 100, 101], [10, 11, 110, 111])
    with pytest.raises(ValueError, match="nonoverlapping same-subject pair"):
        provenance_verification_pairs([0, 0, 1, 1], provenance, 10)


@pytest.mark.parametrize("value", [1.5, 2.0, True, np.bool_(False), "2", -1, 2**63])
def test_protocol_parameters_reject_fractional_boolean_and_overflow(value):
    provenance = metadata([0, 5, 20, 25], [4, 9, 24, 29])
    ids = [0, 0, 1, 1]
    for name in ("enrollment_windows", "purge_samples"):
        with pytest.raises(ValueError, match=name):
            provenance_enrollment_split(ids, provenance, **{name: value})
    for name in ("n_pairs", "seed", "purge_samples"):
        with pytest.raises(ValueError, match=name):
            provenance_verification_pairs(ids, provenance, **{name: value})


def test_large_raw_gap_does_not_wrap_int64_and_missing_provenance_is_rejected():
    limit = np.iinfo(np.int64).max
    provenance = metadata([0, 5, limit - 10, limit - 5], [4, 9, limit - 6, limit - 1])
    assert support_conflicts(provenance, [2], [3], purge_samples=100).all()
    index = _IntervalPairs(np.arange(4), provenance, 100)
    assert index.total == 4
    with pytest.raises(ValueError, match="raw-support-safe"):
        provenance_verification_pairs([0, 0, 1, 1], provenance, purge_samples=100)
    for function in (provenance_enrollment_split, provenance_verification_pairs):
        invalid: Any = None  # Intentionally test the runtime validation boundary.
        with pytest.raises(ValueError, match="cannot be guessed"):
            function([0, 0, 1, 1], invalid)
        with pytest.raises(ValueError, match="one entry"):
            function([0, 1], provenance)


def test_wrapper_rejects_misaligned_tensor_support_lengths():
    data: dict[str, Any] = {
        "X": torch.ones(1, 3, 4),
        "y_activity": torch.zeros(1, dtype=torch.int64),
        "y_subject": torch.zeros(1, dtype=torch.int64),
        "activity_names": ["Walking"],
        "subject_ids": [1],
        "sampling_rate": 20.0,
    }
    with pytest.raises(ValueError, match="sample dimension"):
        ProvenancedWindowData(**data, provenance=metadata([0], [3]))
    with pytest.raises(ValueError, match="equal row counts"):
        ProvenancedWindowData(**data, provenance=metadata([], []))
