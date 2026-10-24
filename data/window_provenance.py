"""Opt-in raw support metadata; the frozen adapter contract stays unchanged.

Only native WISDM v1 records provide the offsets required here. Coordinates are
ordinals among valid parsed records, never physical text lines or session IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from data.adapter import (
    WISDM_V1_ACTIVITY_NAMES,
    WindowData,
    _read_wisdm_v1,
    _window_wisdm,
    _wisdm_fingerprint,
    _wisdm_segments,
    _wisdm_subject_split,
)


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not minimum <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum} representable in int64"
        )
    return int(value)


def _integer_array(values, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            f"{name} must be a 1D nonnegative int64-compatible integer array"
        )
    result = values.astype(np.int64, copy=True)
    result.flags.writeable = False
    return result


def _take_indices(selection, size):
    if isinstance(selection, slice):
        return np.arange(size, dtype=np.int64)[selection]
    if isinstance(selection, torch.Tensor):
        selection = selection.detach().cpu().numpy()
    values = np.asarray(selection)
    if values.ndim == 1 and values.dtype.kind == "b":
        if len(values) != size:
            raise ValueError("boolean selection must have one entry per window")
        return np.flatnonzero(values)
    # An explicitly empty Python list is a useful, unambiguous empty subset.
    if isinstance(selection, (list, tuple)) and not len(selection):
        return np.empty(0, dtype=np.int64)
    rows = _integer_array(values, "selection")
    if np.any(rows >= size):
        raise ValueError("selection contains out-of-range window indices")
    return rows


@dataclass(frozen=True)
class WindowProvenance:
    """One source digest and one half-open valid-row interval per window.

    ``segment_ids`` are the first valid-row ordinal of each contiguous run.
    Timestamp endpoints, when present, are the first and LAST INCLUDED sample's
    integer nanoseconds. Only ``stop_rows`` is an exclusive endpoint.
    Arrays are copied and read-only. This describes alignment, not authenticity
    of arbitrary caller-created metadata; load from the wrapper for derivation.
    """

    source_sha256: str
    segment_ids: np.ndarray
    start_rows: np.ndarray
    stop_rows: np.ndarray
    first_timestamps_ns: np.ndarray | None = None
    last_timestamps_ns: np.ndarray | None = None
    source: str = "wisdm_v1"

    def __post_init__(self):
        if self.source != "wisdm_v1":
            raise ValueError("raw-sample provenance is supported only for wisdm_v1")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA256 hex digest")
        for name in ("segment_ids", "start_rows", "stop_rows"):
            object.__setattr__(self, name, _integer_array(getattr(self, name), name))
        if not (len(self.segment_ids) == len(self.start_rows) == len(self.stop_rows)):
            raise ValueError("provenance arrays must have one aligned entry per window")
        if np.any(self.stop_rows <= self.start_rows):
            raise ValueError("raw support must have start_rows < stop_rows (exclusive)")
        if np.any(self.segment_ids > self.start_rows):
            raise ValueError(
                "segment_ids must identify a run beginning at or before its window"
            )
        if (self.first_timestamps_ns is None) != (self.last_timestamps_ns is None):
            raise ValueError("provide both timestamp endpoint arrays or neither")
        if self.first_timestamps_ns is not None:
            for name in ("first_timestamps_ns", "last_timestamps_ns"):
                array = _integer_array(getattr(self, name), name)
                if len(array) != len(self) or np.any(array <= 0):
                    raise ValueError(
                        "timestamps must be positive and aligned with windows"
                    )
                object.__setattr__(self, name, array)
            if np.any(self.first_timestamps_ns > self.last_timestamps_ns):
                raise ValueError("a window must not cross a timestamp reset")

    def __len__(self):
        return len(self.start_rows)

    def take(self, selection) -> WindowProvenance:
        """Keep the exact selection order, including repeated or empty subsets."""
        rows = _take_indices(selection, len(self))
        return WindowProvenance(
            source_sha256=self.source_sha256,
            segment_ids=self.segment_ids[rows],
            start_rows=self.start_rows[rows],
            stop_rows=self.stop_rows[rows],
            first_timestamps_ns=None
            if self.first_timestamps_ns is None
            else self.first_timestamps_ns[rows],
            last_timestamps_ns=None
            if self.last_timestamps_ns is None
            else self.last_timestamps_ns[rows],
            source=self.source,
        )

    def summary(self) -> dict:
        """Small JSON-safe description; row arrays remain on the object."""
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "row_coordinate": "zero_based_valid_parsed_record_ordinal",
            "support": "[start_rows, stop_rows)",
            "segment_definition": "first valid-row ordinal of a contiguous preprocessing run",
            "n_windows": len(self),
            "n_segments": len(np.unique(self.segment_ids)),
            "timestamps_available": self.first_timestamps_ns is not None,
            "timestamp_endpoints": "first and last included sample; nanoseconds",
            "cross_session": False,
        }


@dataclass
class ProvenancedWindowData(WindowData):
    """A WindowData subclass accepted by existing consumers, with aligned take."""

    provenance: WindowProvenance

    def __post_init__(self):
        n = len(self.X)
        if len(self.provenance) != n or len(self.y_activity) != n:
            raise ValueError(
                "WindowData tensors and provenance must have equal row counts"
            )
        if self.y_subject is None or len(self.y_subject) != n:
            raise ValueError(
                "WISDM provenance requires one real subject label per window"
            )
        if self.X.ndim != 3 or np.any(
            self.provenance.stop_rows - self.provenance.start_rows != self.X.shape[2]
        ):
            raise ValueError(
                "raw support length must match each window's sample dimension"
            )

    def take(self, selection) -> ProvenancedWindowData:
        """Subset data and provenance together; retain the global identity mapping."""
        rows = _take_indices(selection, len(self.X))

        def tensor_take(tensor):
            indices = torch.tensor(rows.copy(), dtype=torch.int64, device=tensor.device)
            return tensor.index_select(0, indices)

        return ProvenancedWindowData(
            X=tensor_take(self.X),
            y_activity=tensor_take(self.y_activity),
            y_subject=tensor_take(self.y_subject),
            activity_names=list(self.activity_names),
            subject_ids=list(self.subject_ids),
            sampling_rate=self.sampling_rate,
            provenance=self.provenance.take(rows),
        )


def load_windows_with_provenance(
    source: str,
    split: str,
    window_size: int = 128,
    stride: int = 64,
    data_root: str = "datasets",
    *,
    include_timestamps: bool = True,
) -> ProvenancedWindowData:
    """Match adapter.load_windows order and values, with native WISDM offsets.

    Unsupported sources fail before any load: official HAR windows do not ship
    source offsets, and the legacy WISDM CSVs lack authentic subject identities.
    """
    if source != "wisdm_v1":
        raise ValueError(
            "raw-sample provenance requires source='wisdm_v1'; official HAR windows "
            "lack raw source offsets and legacy WISDM lacks real subject identities"
        )
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    size = _integer(window_size, "window_size", 1)
    step = _integer(stride, "stride", 1)
    if not isinstance(include_timestamps, (bool, np.bool_)):
        raise ValueError("include_timestamps must be boolean")  # noqa: TRY004 -- public validation contract
    root = Path(data_root) / "wisdm"
    path = root / "WISDM_ar_v1.1_raw.txt"
    if not path.is_file():
        path = root / "WISDM_ar_v1.1" / path.name
    if not path.is_file():
        raise FileNotFoundError(
            f"WISDM provenance requires native raw records at {path}"
        )
    full = _read_wisdm_v1(path)
    # Also catch a stale parser cache when bytes changed without a size/mtime
    # change. Never attach today's digest to windows from yesterday's frame.
    digest = _wisdm_fingerprint(path)
    if full.attrs.get("sha256") != digest:
        raise ValueError(
            "WISDM source changed since parsing; clear the reader cache and retry"
        )
    subject_ids = np.unique(full["subject"])
    train_ids, test_ids = _wisdm_subject_split(subject_ids)
    selected = train_ids if split == "train" else test_ids
    frame = full.loc[full["subject"].isin(selected)].reset_index(drop=True)
    data = _window_wisdm(frame, size, step, WISDM_V1_ACTIVITY_NAMES, subject_ids, 1e9)
    source_rows = frame["_source_row"].to_numpy(dtype=np.int64)
    times = frame["timestamp"].to_numpy(dtype=np.int64)
    starts, stops, segments, first_times, last_times = [], [], [], [], []
    for start, stop in _wisdm_segments(frame, 1e9):
        rows = np.arange(start, stop - size + 1, step, dtype=np.int64)
        if not len(rows):
            continue
        starts.append(source_rows[rows])
        stops.append(source_rows[rows + size - 1] + 1)
        segments.append(np.full(len(rows), source_rows[start], dtype=np.int64))
        if include_timestamps:
            first_times.append(times[rows])
            last_times.append(times[rows + size - 1])

    def concatenate(chunks):
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    provenance = WindowProvenance(
        source_sha256=digest,
        segment_ids=concatenate(segments),
        start_rows=concatenate(starts),
        stop_rows=concatenate(stops),
        first_timestamps_ns=concatenate(first_times) if include_timestamps else None,
        last_timestamps_ns=concatenate(last_times) if include_timestamps else None,
    )
    return ProvenancedWindowData(
        X=data.X,
        y_activity=data.y_activity,
        y_subject=data.y_subject,
        activity_names=data.activity_names,
        subject_ids=data.subject_ids,
        sampling_rate=data.sampling_rate,
        provenance=provenance,
    )
