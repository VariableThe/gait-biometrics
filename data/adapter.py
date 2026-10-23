"""Dataset adapter with the frozen shared WindowData contract.

``wisdm`` reads legacy activity-only CSVs. ``wisdm_v1`` reads the official
WISDM AR v1.1 raw records with genuine subject IDs; see docs/wisdm_identity.md.
Neither source silently reconstructs missing identities or normalizes inputs.

Every source returns:
  - X: float32 tensor of shape (N, C, T)
  - y_activity: int64 tensor of shape (N,), 0-indexed
  - y_subject: int64 tensor of shape (N,) 0-indexed, or None when the dataset
    has no subject ids
  - activity_names: list[str]
  - subject_ids: list[int]
  - sampling_rate: float
"""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class WindowData:
    X: torch.Tensor  # (N, C, T) float32
    y_activity: torch.Tensor  # (N,) int64, 0-indexed
    y_subject: (
        torch.Tensor | None
    )  # (N,) int64 0-indexed subject id; None when the dataset has none
    activity_names: list[str]
    subject_ids: list[int]
    sampling_rate: float


HAR_ROOT = os.path.join("datasets", "uci_har", "UCI HAR Dataset")
WISDM_ROOT = os.path.join("datasets", "wisdm")

HAR_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

WISDM_ACTIVITY_NAMES = [f"activity_{i}" for i in range(6)]

WISDM_V1_ACTIVITY_NAMES = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
]
WISDM_V1_URL = "https://www.cis.fordham.edu/wisdm/dataset.php"
WISDM_V1_RAW_SHA256 = "9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e"


def _wisdm_fingerprint(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_wisdm_v1(path):
    """Read a raw file once per process; invalidate when size or mtime changes.

    The returned cached frame is shared and must be treated as read-only.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _read_wisdm_v1_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=1)
def _read_wisdm_v1_cached(path, size, mtime_ns):
    """Read native subject,activity,timestamp_ns,x,y,z; records in source order.

    Semicolons can separate multiple records on one physical line. Invalid
    sensor records are counted and break continuity; identities are never guessed.
    Array buffers avoid retaining a million Python record objects in memory.
    """
    columns = {
        k: array("q" if k in ("subject", "Activity", "timestamp", "_segment") else "f")
        for k in (
            "subject",
            "Activity",
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z",
            "_segment",
        )
    }
    activities = {name: i for i, name in enumerate(WISDM_V1_ACTIVITY_NAMES)}
    max_float = np.finfo(np.float32).max
    invalid = 0
    segment = 0
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            for record in line.strip().split(";"):
                if not record.strip():
                    continue
                fields = [field.strip() for field in record.split(",")]
                if len(fields) != 6:
                    invalid += 1
                    segment += 1
                    continue
                try:
                    subject = int(fields[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Missing/invalid real WISDM subject ID at {path}:{line_no}"
                    ) from exc
                if not 1 <= subject <= 36:
                    raise ValueError(
                        f"Not a WISDM AR v1.1 subject ID at {path}:{line_no}: {subject}"
                    )
                if fields[1] not in activities:
                    raise ValueError(
                        f"Unknown WISDM AR v1.1 activity at {path}:{line_no}: {fields[1]!r}"
                    )
                try:
                    timestamp = int(fields[2])
                    xyz = tuple(float(value) for value in fields[3:])
                    valid = 0 < timestamp <= 9223372036854775807 and all(
                        math.isfinite(value) and abs(value) <= max_float
                        for value in xyz
                    )
                except ValueError:
                    valid = False
                if not valid:
                    invalid += 1
                    segment += 1
                    continue
                for key, value in zip(
                    columns, (subject, activities[fields[1]], timestamp, *xyz, segment)
                ):
                    columns[key].append(value)
    df = pd.DataFrame(
        {key: np.asarray(value) for key, value in columns.items()}, copy=False
    )
    df["_source_row"] = np.arange(len(df))
    if df.empty:
        raise ValueError(f"No valid subject-labelled WISDM AR v1.1 records in {path}")
    df.attrs.update(
        source_url=WISDM_V1_URL,
        sha256=_wisdm_fingerprint(path),
        invalid_records=invalid,
        valid_records=len(df),
        timestamp_unit="nanoseconds",
        schema="WISDM AR v1.1 raw",
        schema_version=1,
    )
    df.attrs["matches_official_sha256"] = df.attrs["sha256"] == WISDM_V1_RAW_SHA256
    if invalid:
        warnings.warn(
            f"WISDM AR v1.1: excluded {invalid} invalid records; each breaks window continuity",
            RuntimeWarning,
        )
    return df


def _wisdm_subject_split(subjects):
    """Fixed 80/20 split of real IDs, independent of record order and activity."""
    ids = np.unique(np.asarray(subjects, dtype=np.int64))
    if len(ids) < 2:
        raise ValueError(
            "Subject-disjoint WISDM train/test requires at least two real subjects"
        )
    shuffled = np.random.default_rng(42).permutation(ids)
    n_train = min(len(ids) - 1, max(1, int(0.8 * len(ids))))
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def _wisdm_segments(df, timestamp_scale=1.0):
    """Yield contiguous valid runs; never sort records or bridge boundaries.

    A reset, duplicate timestamp, or gap over one second ends a run. The gap
    threshold is a conservative preprocessing rule, not an inferred session ID.
    """
    if df.empty:
        return
    times = df["timestamp"].to_numpy()
    values = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
    labels = df["Activity"].to_numpy()
    valid = np.isfinite(times) & np.all(np.isfinite(values), axis=1) & pd.notna(labels)
    boundary = np.ones(len(df), dtype=bool)
    delta = np.diff(times)
    boundary[1:] = (
        (delta <= 0)
        | (delta > timestamp_scale)
        | (labels[1:] != labels[:-1])
        | ~valid[1:]
        | ~valid[:-1]
    )
    for key in ("subject", "session", "_segment"):
        if key in df:
            ids = df[key].to_numpy()
            valid &= pd.notna(ids)
            boundary[1:] |= (ids[1:] != ids[:-1]) | ~valid[1:] | ~valid[:-1]
    if "_source_row" in df:
        boundary[1:] |= np.diff(df["_source_row"].to_numpy()) != 1
    starts = np.flatnonzero(boundary)
    for start, stop in zip(starts, np.r_[starts[1:], len(df)]):
        if valid[start]:
            yield start, stop


def _load_har(split: str, data_root: str) -> WindowData:
    root = os.path.join(data_root, "uci_har", "UCI HAR Dataset")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    # HAR is already windowed: each row of the Inertial Signals files is one
    # 128-sample window. Stack the 9 channels -> (N, 9, 128).
    channels = []
    for ch in HAR_CHANNELS:
        path = os.path.join(root, split, "Inertial Signals", f"{ch}_{split}.txt")
        arr = np.loadtxt(path, dtype=np.float32)  # (N, 128)
        channels.append(arr)
    X = np.stack(channels, axis=1)  # (N, 9, 128)

    y_activity = (
        np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=np.int64) - 1
    )
    y_subject = (
        np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=np.int64)
        - 1
    )

    with open(os.path.join(root, "activity_labels.txt")) as f:
        activity_names = [line.split(None, 1)[1].strip() for line in f if line.strip()]

    subject_ids = sorted(int(s) for s in np.unique(y_subject))

    return WindowData(
        X=torch.from_numpy(X).float(),
        y_activity=torch.from_numpy(y_activity).long(),
        y_subject=torch.from_numpy(y_subject).long(),
        activity_names=activity_names,
        subject_ids=subject_ids,
        sampling_rate=50.0,
    )


def _load_wisdm(
    split: str, data_root: str, window_size: int, stride: int
) -> WindowData:
    path = os.path.join(data_root, "wisdm", f"{split}.csv")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    df = pd.read_csv(path)
    required = {"acc_x", "acc_y", "acc_z", "timestamp", "Activity"}
    if not required <= set(df):
        raise ValueError(f"WISDM CSV is missing columns: {sorted(required - set(df))}")
    labels = pd.to_numeric(df["Activity"], errors="raise").to_numpy()
    if not np.all(
        np.isfinite(labels)
        & (labels == np.floor(labels))
        & (labels >= 0)
        & (labels < 6)
    ):
        raise ValueError("Legacy WISDM activity labels must be integers 0..5")
    df["Activity"] = labels.astype(np.int64)
    return _window_wisdm(df, window_size, stride, WISDM_ACTIVITY_NAMES)


def _window_wisdm(
    df, window_size, stride, activity_names, subject_ids=None, timestamp_scale=1.0
):
    values = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
    chunks, activities, subjects = [], [], []
    for start, stop in _wisdm_segments(df, timestamp_scale):
        if stop - start < window_size:
            continue
        chunk = np.lib.stride_tricks.sliding_window_view(
            values[start:stop], window_size, axis=0
        )[::stride]
        chunks.append(chunk)
        activities.extend([int(df["Activity"].iloc[start])] * len(chunk))
        if subject_ids is not None:
            label = int(np.searchsorted(subject_ids, df["subject"].iloc[start]))
            subjects.extend([label] * len(chunk))
    X = (
        np.concatenate(chunks)
        if chunks
        else np.empty((0, 3, window_size), dtype=np.float32)
    )
    return WindowData(
        X=torch.from_numpy(X),
        y_activity=torch.tensor(activities, dtype=torch.int64),
        y_subject=torch.tensor(subjects, dtype=torch.int64)
        if subject_ids is not None
        else None,
        activity_names=list(activity_names),
        subject_ids=list(map(int, subject_ids)) if subject_ids is not None else [],
        sampling_rate=20.0,
    )


def _load_wisdm_v1(split, data_root, window_size, stride):
    root = Path(data_root) / "wisdm"
    path = root / "WISDM_ar_v1.1_raw.txt"
    if not path.is_file():
        path = root / "WISDM_ar_v1.1" / path.name
    if not path.is_file():
        raise FileNotFoundError(
            f"Real-subject WISDM source requires {root / 'WISDM_ar_v1.1_raw.txt'}; "
            "legacy train/test.csv have no recoverable subject identities"
        )
    df = _read_wisdm_v1(path)
    subject_ids = np.unique(df["subject"])
    train_ids, test_ids = _wisdm_subject_split(subject_ids)
    selected = train_ids if split == "train" else test_ids
    # Preserve source-row gaps as explicit boundaries after selecting subjects.
    # Every removed run also changes subject; records are never reordered.
    df = df.loc[df["subject"].isin(selected)].reset_index(drop=True)
    return _window_wisdm(
        df, window_size, stride, WISDM_V1_ACTIVITY_NAMES, subject_ids, 1e9
    )


def load_windows(
    source: str,
    split: str,
    window_size: int = 128,
    stride: int = 64,
    data_root: str = "datasets",
) -> WindowData:
    """Load HAR, legacy activity-only WISDM, or real-subject ``wisdm_v1``.

    For wisdm_v1, y_subject indexes the global sorted source IDs in subject_ids,
    identically in both source splits. Walking has activity label 0.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if source == "har":
        return _load_har(split, data_root)
    if source == "wisdm":
        return _load_wisdm(split, data_root, window_size, stride)
    if source == "wisdm_v1":
        return _load_wisdm_v1(split, data_root, window_size, stride)
    raise ValueError(f"source must be 'har', 'wisdm', or 'wisdm_v1', got {source!r}")


def build_loader(
    data: WindowData, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    """Build a DataLoader, adding identity labels only when the source has them."""
    dataset = TensorDataset(data.X, data.y_activity)
    if data.y_subject is not None:
        dataset = TensorDataset(data.X, data.y_activity, data.y_subject)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
