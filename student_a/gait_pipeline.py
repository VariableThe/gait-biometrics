"""Prepare WISDM and UCI-HAR inertial data for gait/activity models.

The implementation follows the supplied papers' reusable preprocessing ideas:
timestamp-aware resampling, optional FIR-like low-pass filtering, sliding
windows, training-only standardization, and subject-safe splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
import zipfile
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import interp1d
    from scipy.signal import filtfilt, firwin
except ImportError:
    interp1d = firwin = filtfilt = None

ROOT = Path(__file__).resolve().parent

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


def resample(values, timestamps, hz):
    """Resample one already-separated recording; never sort across resets."""
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be positive and finite")
    if not len(t):
        return np.empty((0, x.shape[1]), dtype=np.float32)
    if (
        not np.all(np.isfinite(t))
        or not np.all(np.isfinite(x))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Resampling requires a finite, strictly increasing single segment"
        )
    t = t - t[0]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / hz)
    return np.column_stack(
        [np.interp(grid, t, x[:, j]) for j in range(x.shape[1])]
    ).astype(np.float32)


def lowpass(x, hz, cutoff=8.0):
    if not 0 < cutoff < hz / 2:
        raise ValueError("cutoff must be between zero and the Nyquist frequency")
    if filtfilt is None or firwin is None:
        return x
    taps = firwin(31, cutoff, fs=hz)
    padlen = 3 * (len(taps) - 1)
    if len(x) <= padlen:
        return x
    return filtfilt(taps, [1.0], x, axis=0, padlen=padlen).astype(np.float32)


def windows(x, y, subject, size, step):
    """Pure-activity windows, restarting stride at every label boundary."""
    if size <= 0 or step <= 0:
        raise ValueError("window size and step must be positive")
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("Signal and activity lengths differ")
    starts = (
        np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1]
        if len(y)
        else np.array([], dtype=int)
    )
    out, labels = [], []
    for start, stop in zip(starts, np.r_[starts[1:], len(y)]):
        if stop - start < size:
            continue
        block = np.lib.stride_tricks.sliding_window_view(x[start:stop], size, axis=0)[
            ::step
        ].transpose(0, 2, 1)
        out.append(block)
        labels.extend([y[start]] * len(block))
    X = (
        np.concatenate(out)
        if out
        else np.empty((0, size, x.shape[1]), dtype=np.float32)
    )
    return (
        X.astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.full(len(labels), subject, dtype=np.int64),
    )


def save_npz(out, name, X, y, subjects, meta, stats=None):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", X=X, y=y, subject=subjects)
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if stats:
        np.savez(out / f"{name}_normalization.npz", **stats)


def _prepare_wisdm_split(df, args, timestamp_scale):
    chunks, labels, subjects = [], [], []
    timestamps = df.timestamp.to_numpy()
    values = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
    activities = df.Activity.to_numpy()
    identities = df.subject.to_numpy() if "subject" in df else None
    for start, stop in _wisdm_segments(df, timestamp_scale):
        duration = (timestamps[stop - 1] - timestamps[start]) / timestamp_scale
        if duration * args.hz + 1 + 1e-7 < args.window:
            continue
        # Subtract integer nanoseconds before converting to seconds.
        ts = timestamps[start:stop]
        seconds = (ts - ts[0]) / timestamp_scale
        x = resample(values[start:stop], seconds, args.hz)
        x = lowpass(x, args.hz, args.cutoff)
        subject = int(identities[start]) if identities is not None else -1
        w, y, sub = windows(
            x, np.full(len(x), int(activities[start])), subject, args.window, args.step
        )
        if len(w):
            chunks.append(w)
            labels.append(y)
            subjects.append(sub)
    if not chunks:
        return (
            np.empty((0, args.window, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return np.concatenate(chunks), np.concatenate(labels), np.concatenate(subjects)


def process_wisdm(args, out):
    """Keep X/y/subject NPZ keys; unknown legacy subjects are explicitly -1."""
    if args.window <= 0 or args.step <= 0:
        raise ValueError("window and step must be positive")
    raw_path = getattr(args, "wisdm_raw", None)
    allsets = []
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        df = _read_wisdm_v1(path)
        train_ids, test_ids = _wisdm_subject_split(df.subject)
        provenance = {
            **df.attrs,
            "source": "wisdm_v1",
            "source_path": str(path),
            "identity_available": True,
            "subject_values": "original WISDM user IDs (1..36)",
            "activity_names": WISDM_V1_ACTIVITY_NAMES,
            "split_protocol": "sorted subjects, numpy default_rng(42), 80% train, remainder test",
            "train_subject_ids": train_ids.tolist(),
            "test_subject_ids": test_ids.tolist(),
        }
        for split, ids in [("train", train_ids), ("test", test_ids)]:
            part = df.loc[df.subject.isin(ids)].reset_index(drop=True)
            allsets.append((split, *_prepare_wisdm_split(part, args, 1e9)))
    else:
        path = ROOT / "Dataset" / "WISDM_Dataset.zip"
        provenance = {
            "source": "legacy_wisdm_csv",
            "source_path": str(path),
            "sha256": _wisdm_fingerprint(path),
            "identity_available": False,
            "subject_values": "-1 means unknown; not usable for identity verification",
            "activity_names": [f"activity_{i}" for i in range(6)],
            "timestamp_unit": "seconds",
            "split_protocol": "provided CSV splits; subject overlap unknown",
        }
        with zipfile.ZipFile(path) as archive:
            for split in ("train", "test"):
                with archive.open(f"{split}.csv") as stream:
                    df = pd.read_csv(stream)
                required = {"timestamp", "Activity", "acc_x", "acc_y", "acc_z"}
                if not required <= set(df):
                    raise ValueError(
                        f"Missing WISDM columns: {sorted(required - set(df))}"
                    )
                labels = df.Activity.to_numpy()
                if not np.all(
                    np.isfinite(labels)
                    & (labels == np.floor(labels))
                    & (labels >= 0)
                    & (labels < 6)
                ):
                    raise ValueError(
                        "Legacy WISDM activity labels must be integers 0..5"
                    )
                # Source split strings are provenance, never person identities.
                df = df.drop(columns=["subject"], errors="ignore")
                allsets.append((split, *_prepare_wisdm_split(df, args, 1.0)))
    train = allsets[0][1]
    if not len(train):
        raise ValueError(
            "No training windows survive WISDM segment boundaries; reduce window length"
        )
    mu, sd = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
    sd[sd < 1e-6] = 1
    for split, w, y, subjects in allsets:
        metadata = {
            **vars(args),
            **provenance,
            "normalization": "train-only standard score",
            "boundary_policy": "subject/activity/session change, nonincreasing timestamp, invalid record, gap >1 second",
            "windows": len(w),
            "shape": list(w.shape),
            "subject_ids": np.unique(subjects).tolist(),
            "scipy_filter_available": filtfilt is not None,
        }
        save_npz(
            out,
            f"wisdm_{split}",
            ((w - mu) / sd).astype(np.float32),
            y,
            subjects,
            metadata,
            {"mean": mu, "std": sd},
        )


def read_uci(z):
    with zipfile.ZipFile(z) as outer:
        nested = outer.read("UCI HAR Dataset.zip")
    import io

    with zipfile.ZipFile(io.BytesIO(nested)) as d:

        def txt(name):
            return np.loadtxt(d.open(name))

        Xtr, Xte = (
            txt("UCI HAR Dataset/train/X_train.txt"),
            txt("UCI HAR Dataset/test/X_test.txt"),
        )
        ytr, yte = (
            txt("UCI HAR Dataset/train/y_train.txt").astype(int).ravel(),
            txt("UCI HAR Dataset/test/y_test.txt").astype(int).ravel(),
        )
        str_, ste = (
            txt("UCI HAR Dataset/train/subject_train.txt").astype(int).ravel(),
            txt("UCI HAR Dataset/test/subject_test.txt").astype(int).ravel(),
        )
    # Dataset already provides 128-sample windows x 561 engineered features.
    return (Xtr, ytr, str_), (Xte, yte, ste)


def process_uci(args, out):
    tr, te = read_uci(
        ROOT / "Dataset" / "human+activity+recognition+using+smartphones.zip"
    )
    for name, (X, y, s) in [("uci_har_train", tr), ("uci_har_test", te)]:
        save_npz(out, name, X.astype(np.float32), y, s, vars(args))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="processed")
    p.add_argument("--hz", type=float, default=50)
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--step", type=int, default=45)
    p.add_argument("--cutoff", type=float, default=8)
    p.add_argument(
        "--wisdm-raw",
        help="Official WISDM_ar_v1.1_raw.txt with genuine user IDs; otherwise legacy activity-only CSV zip",
    )
    a = p.parse_args()
    out = ROOT / a.out
    process_wisdm(a, out)
    process_uci(a, out)
    print(
        json.dumps(
            {"output": str(out), "window_samples": a.window, "step_samples": a.step},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
