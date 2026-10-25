"""Statically extracted pure source; provenance in source-origins.json."""
import csv, hashlib, io, json, math, re, stat
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
SANTOS_GYRO_CHANNELS = ("gyro x", "gyro y", "gyro z")
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

def _parse_recording(payload, record, window_size, stride, n_channels=3):
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
                    # All six values affect validity in either channel mode.
                    xyz = sensor_values[:n_channels]
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
