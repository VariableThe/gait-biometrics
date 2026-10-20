"""Dataset adapter — FROZEN SHARED CONTRACT (STUB).

This module is an explicitly-labelled STUB. Another team member (Student A)
will replace the bodies of `load_windows` and `build_loader` with a full
implementation (resampling, denoising, orientation elimination, gait-cycle
extraction, augmentation, ...). The public interface and the semantics below
are frozen and MUST NOT change.

Any replacement MUST return `WindowData` with unchanged shape and semantics:
  - X: float32 tensor of shape (N, C, T)
  - y_activity: int64 tensor of shape (N,), 0-indexed
  - y_subject: int64 tensor of shape (N,) 0-indexed, or None when the dataset
    has no subject ids
  - activity_names: list[str]
  - subject_ids: list[int]
  - sampling_rate: float
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass


@dataclass
class WindowData:
    X: torch.Tensor                 # (N, C, T) float32
    y_activity: torch.Tensor        # (N,) int64, 0-indexed
    y_subject: torch.Tensor | None  # (N,) int64 0-indexed subject id; None when the dataset has none
    activity_names: list[str]
    subject_ids: list[int]
    sampling_rate: float


HAR_ROOT = os.path.join("datasets", "uci_har", "UCI HAR Dataset")
WISDM_ROOT = os.path.join("datasets", "wisdm")

HAR_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

WISDM_ACTIVITY_NAMES = [f"activity_{i}" for i in range(6)]


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
        np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=np.int64) - 1
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


def _load_wisdm(split: str, data_root: str, window_size: int, stride: int) -> WindowData:
    path = os.path.join(data_root, "wisdm", f"{split}.csv")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    df = pd.read_csv(path)
    # Expected columns: acc_x, acc_y, acc_z, timestamp, Activity
    n = len(df)
    X_list, y_list = [], []

    i = 0
    while i + window_size <= n:
        # A backwards jump in timestamp marks a session boundary; windows must
        # not straddle it.
        block = df.iloc[i : i + window_size]
        ts = block["timestamp"].to_numpy()
        if np.any(np.diff(ts) < 0):
            i += 1
            continue
        X_list.append(
            block[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32).T
        )  # (3, T)
        y_list.append(int(block["Activity"].iloc[0]))
        i += stride

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 3, window_size), np.float32)
    y_activity = np.asarray(y_list, dtype=np.int64)

    return WindowData(
        X=torch.from_numpy(X).float(),
        y_activity=torch.from_numpy(y_activity).long(),
        y_subject=None,  # WISDM ships no subject ids (verified)
        activity_names=list(WISDM_ACTIVITY_NAMES),
        subject_ids=[],
        sampling_rate=20.0,
    )


def load_windows(source: str, split: str, window_size: int = 128,
                 stride: int = 64, data_root: str = "datasets") -> WindowData:
    """Load windowed data for `source` ("har" or "wisdm") and `split`.

    STUB implementation — see module docstring. Student A will replace this.
    """
    if source == "har":
        return _load_har(split, data_root)
    if source == "wisdm":
        return _load_wisdm(split, data_root, window_size, stride)
    raise ValueError(f"source must be 'har' or 'wisdm', got {source!r}")


def build_loader(data: WindowData, batch_size: int, shuffle: bool,
                 num_workers: int = 0) -> DataLoader:
    """Build a DataLoader over a WindowData instance.

    STUB implementation — see module docstring. Student A will replace this.
    """
    dataset = TensorDataset(data.X, data.y_activity)
    if data.y_subject is not None:
        dataset = TensorDataset(data.X, data.y_activity, data.y_subject)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
