"""Opt-in, NumPy-only reader for the Santos v7 processed MCU recordings.

The archive must be integrity-checked by the caller. Payload access is restricted
to the explicitly requested real subject IDs; no extraction or global hashing
is performed here. See ``docs/santos_dataset.md`` for the protocol and provenance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import stat
from numbers import Integral
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SANTOS_SUBJECT_IDS = (
    1,
    3,
    5,
    6,
    7,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
)
SANTOS_CSV_HEADER = ("frame", "acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z")
SANTOS_ACCEL_CHANNELS = ("acc x", "acc y", "acc z")
SANTOS_ARCHIVE_PREFIX = "A multi-sensor human gait dataset/"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_RECORDING = re.compile(
    r"user(?P<directory>[0-9]{2})/"
    r"capture_user(?P<subject>[0-9]{2})_day(?P<day>[12])_"
    r"(?P<trial>000[1-9]|0010)_imu_walk\.csv\Z"
)


def _subject_ids(subject_ids):
    try:
        values = list(subject_ids)
    except TypeError as exc:
        raise ValueError("subject_ids must contain real integer subject IDs") from exc
    if not values:
        raise ValueError("subject_ids must not be empty")
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in values
    ):
        raise ValueError("subject_ids must contain real integer subject IDs")
    values = [int(x) for x in values]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate requested subject IDs")
    unknown = sorted(set(values) - set(SANTOS_SUBJECT_IDS))
    if unknown:
        raise ValueError(f"Unknown authentic Santos subject IDs: {unknown}")
    return sorted(values)


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _safe_member(info):
    name = info.orig_filename
    path = name.removesuffix("/")
    if (
        not path
        or name != info.filename
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    mode = stat.S_IFMT(info.external_attr >> 16)
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Nonregular ZIP member: {name!r}")


def _selected_recordings(archive, subjects):
    """Validate central-directory metadata without opening any member payload."""
    prefix = SANTOS_ARCHIVE_PREFIX + "processed_data/"
    names = set()
    records = {}
    for info in archive.infolist():
        _safe_member(info)
        if info.filename in names:
            raise ValueError(f"Duplicate ZIP member: {info.filename!r}")
        names.add(info.filename)
        if info.is_dir():
            continue
        components = info.filename.split("/")
        if not any(part.lower() == "processed_data" for part in components):
            continue
        # Other sensors and raw files never enter the payload candidate set.
        if not info.filename.lower().endswith("_imu_walk.csv"):
            continue
        if not info.filename.startswith(prefix):
            raise ValueError(f"Unexpected processed MCU ZIP prefix: {info.filename!r}")
        match = _RECORDING.fullmatch(info.filename[len(prefix) :])
        if match is None:
            raise ValueError(f"Invalid processed MCU recording path: {info.filename!r}")
        subject = int(match["subject"])
        if match["directory"] != match["subject"]:
            raise ValueError(f"Directory/filename subject mismatch: {info.filename!r}")
        if subject not in SANTOS_SUBJECT_IDS:
            raise ValueError(f"Unknown authentic Santos subject in ZIP: {subject}")
        key = (subject, int(match["day"]), int(match["trial"]))
        if key in records:
            raise ValueError(f"Duplicate recording identity: {key}")
        records[key] = info
    missing = sorted(set(subjects) - {key[0] for key in records})
    if missing:
        raise ValueError(
            f"Missing requested subject IDs in processed MCU files: {missing}"
        )
    return [(key, records[key]) for key in sorted(records) if key[0] in subjects]


def _member_record(key, info, payload):
    subject, day, trial = key
    return {
        "subject": subject,
        "day": day,
        "trial": trial,
        "member": info.filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uncompressed_bytes": len(payload),
        "recording_key": [7, subject, day, trial, "MCU", "processed_data"],
    }


def _members_sha256(recordings):
    fields = (
        "subject",
        "day",
        "trial",
        "member",
        "sha256",
        "uncompressed_bytes",
        "recording_key",
    )
    manifest = [{key: record[key] for key in fields} for record in recordings]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_santos_members(archive_path, subject_ids):
    """Hash only selected original CSV members, without parsing their signals.

    This still reads the selected subjects' payloads: do not request test IDs
    until the parent has frozen the protocol and calibration threshold.
    """
    subjects = _subject_ids(subject_ids)
    with ZipFile(archive_path) as archive:
        recordings = [
            _member_record(key, info, archive.read(info))
            for key, info in _selected_recordings(archive, subjects)
        ]
    return {
        "dataset_doi": "10.6084/m9.figshare.14727231.v7",
        "subject_ids": subjects,
        "recordings": recordings,
        "members_sha256": _members_sha256(recordings),
    }


def _parse_recording(payload, record, window_size, stride):
    try:
        stream = io.StringIO(payload.decode("utf-8"), newline="")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unknown CSV encoding in {record['member']}") from exc
    reader = csv.reader(stream, strict=True)
    try:
        header = next(reader, None)
        if header != list(SANTOS_CSV_HEADER):
            raise ValueError(
                f"Unknown Santos CSV schema in {record['member']}: {header!r}; "
                f"expected {SANTOS_CSV_HEADER!r}"
            )
        row_count = 0
        invalid = []
        breaks = []
        segments = []
        segment = []
        previous_frame = None
        for offset, row in enumerate(reader):
            row_count += 1
            valid = len(row) == len(SANTOS_CSV_HEADER)
            frame = None
            xyz = None
            if valid:
                try:
                    token = row[0].strip()
                    if _INTEGER.fullmatch(token) is None:
                        raise ValueError("Noninteger frame")
                    frame = int(token)
                    sensor_values = tuple(float(value) for value in row[1:])
                    valid = _INT64_MIN <= frame <= _INT64_MAX and all(
                        math.isfinite(value) and abs(value) <= _FLOAT32_MAX
                        for value in sensor_values
                    )
                    # Gyroscope values affect row validity, never model input.
                    xyz = sensor_values[:3]
                except (ValueError, OverflowError):
                    valid = False
            if not valid:
                if segment:
                    segments.append(segment)
                    segment = []
                invalid.append(offset)
                previous_frame = None
                continue
            if previous_frame is not None and frame != previous_frame + 1:
                segments.append(segment)
                segment = []
                reason = (
                    "gap"
                    if frame > previous_frame + 1
                    else ("duplicate" if frame == previous_frame else "reset")
                )
                breaks.append(
                    {
                        "row": offset,
                        "previous_frame": previous_frame,
                        "frame": frame,
                        "reason": reason,
                    }
                )
            segment.append((offset, frame, xyz))
            previous_frame = frame
        if segment:
            segments.append(segment)
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV in {record['member']}: {exc}") from exc

    windows, frames, starts = [], [], []
    segment_coverage = []
    for segment in segments:
        count = len(segment)
        positions = range(0, count - window_size + 1, stride)
        n_windows = len(positions)
        for start in positions:
            rows = segment[start : start + window_size]
            windows.append(np.asarray([row[2] for row in rows], dtype=np.float32).T)
            frames.append([row[1] for row in rows])
            starts.append(rows[0][0])
        covered = (
            window_size + (n_windows - 1) * min(stride, window_size) if n_windows else 0
        )
        tail = count - ((n_windows - 1) * stride + window_size) if n_windows else count
        segment_coverage.append(
            {
                "start_row": segment[0][0],
                "stop_row": segment[-1][0] + 1,
                "frame_start": segment[0][1],
                "frame_end": segment[-1][1],
                "rows": count,
                "windows": n_windows,
                "covered_rows": covered,
                "tail_rows": tail,
                "stride_gap_rows": count - covered - tail,
            }
        )
    record.update(
        {
            "header": header,
            "source_rows": row_count,
            "valid_rows": row_count - len(invalid),
            "invalid_rows": len(invalid),
            "invalid_row_offsets": invalid,
            "continuity_breaks": breaks,
            "segments": segment_coverage,
            "windows": len(windows),
            "covered_rows": sum(s["covered_rows"] for s in segment_coverage),
            "tail_rows": sum(s["tail_rows"] for s in segment_coverage),
            "stride_gap_rows": sum(s["stride_gap_rows"] for s in segment_coverage),
        }
    )
    return windows, frames, starts


def load_santos_windows(archive_path, subject_ids, *, window_size=256, stride=256):
    """Load trial/day-bounded 3-axis acceleration with authentic subject IDs.

    Returns NumPy arrays plus JSON-compatible metadata. ``starts`` contains
    zero-based original data-row offsets (header excluded, bad rows included).
    ``frames`` contains all original MCU frame integers. Member payloads for
    unrequested subjects are never opened. No normalization or unit conversion
    is applied, and no complete-archive checksum is computed.
    """
    subjects = _subject_ids(subject_ids)
    window_size = _positive_integer(window_size, "window_size")
    stride = _positive_integer(stride, "stride")
    windows, frame_windows, offsets = [], [], []
    labels, days, trials, recording_indices = [], [], [], []
    recordings = []
    with ZipFile(archive_path) as archive:
        selected = _selected_recordings(archive, subjects)
        for index, (key, info) in enumerate(selected):
            # Reading by validated ZipInfo avoids duplicate-name ambiguity.
            payload = archive.read(info)
            record = _member_record(key, info, payload)
            values, frames, starts = _parse_recording(
                payload, record, window_size, stride
            )
            recordings.append(record)
            windows.extend(values)
            frame_windows.extend(frames)
            offsets.extend(starts)
            labels.extend([key[0]] * len(values))
            days.extend([key[1]] * len(values))
            trials.extend([key[2]] * len(values))
            recording_indices.extend([index] * len(values))
    present = {(r["subject"], r["day"], r["trial"]) for r in recordings}
    missing = [
        {"subject": s, "day": d, "trial": t}
        for s in subjects
        for d in (1, 2)
        for t in range(1, 11)
        if (s, d, t) not in present
    ]
    coverage = []
    for subject in subjects:
        day_coverage = {}
        for day in (1, 2):
            rows = [
                r for r in recordings if r["subject"] == subject and r["day"] == day
            ]
            day_coverage[str(day)] = {
                "present_trials": [r["trial"] for r in rows],
                "eligible_trials": [r["trial"] for r in rows if r["windows"]],
                "windows": sum(r["windows"] for r in rows),
            }
        coverage.append(
            {
                "subject": subject,
                "days": day_coverage,
                "eligible_both_days": all(
                    day_coverage[str(d)]["eligible_trials"] for d in (1, 2)
                ),
            }
        )
    metadata = {
        "dataset_doi": "10.6084/m9.figshare.14727231.v7",
        "article_id": 14727231,
        "version": 7,
        "file_id": 35882624,
        "archive_path": str(Path(archive_path)),
        "archive_integrity": "caller must verify official size/MD5 and record SHA256",
        "subject_ids": subjects,
        "sampling_rate_hz": 100,
        "channels": list(SANTOS_ACCEL_CHANNELS),
        "units": "m/s^2",
        "source_header": list(SANTOS_CSV_HEADER),
        "omitted_channels": ["gyro x", "gyro y", "gyro z"],
        "placement_side": "unknown; trials pool both leg placements",
        "window_size": window_size,
        "stride": stride,
        "start_convention": "zero-based CSV data-row offset; header excluded; bad rows counted",
        "frame_convention": "original int64 MCU frame; each window has increments of one",
        "recordings": recordings,
        "members_sha256": _members_sha256(recordings),
        "subject_coverage": coverage,
        "exclusions": {
            "missing_recordings": missing,
            "invalid_rows": sum(r["invalid_rows"] for r in recordings),
            "continuity_breaks": sum(len(r["continuity_breaks"]) for r in recordings),
            "tail_rows": sum(r["tail_rows"] for r in recordings),
            "stride_gap_rows": sum(r["stride_gap_rows"] for r in recordings),
            "zero_window_recordings": [
                i for i, r in enumerate(recordings) if not r["windows"]
            ],
        },
    }
    return {
        "X": np.stack(windows).astype(np.float32, copy=False)
        if windows
        else np.empty((0, 3, window_size), dtype=np.float32),
        "subjects": np.asarray(labels, dtype=np.int64),
        "days": np.asarray(days, dtype=np.int64),
        "trials": np.asarray(trials, dtype=np.int64),
        "starts": np.asarray(offsets, dtype=np.int64),
        "frames": np.asarray(frame_windows, dtype=np.int64).reshape(-1, window_size),
        "recording_indices": np.asarray(recording_indices, dtype=np.int64),
        "metadata": metadata,
    }
