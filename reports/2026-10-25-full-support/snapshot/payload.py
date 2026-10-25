"""Future-only exact DEV payload access. This module performs no access on import."""

import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import common as c
import kernel as k
import numpy as np
import santos_schema as santos
from planning import DEV


def load(archive_path, metadata):
    """Call only after the worker has validated the complete fresh freeze.

    No archive-wide hash, testzip, extraction, fallback, downloads, or non-DEV
    member reads. Central-directory checks happen HERE, never in preparation.
    Existing official authenticity audit is carried, not rerun across CALIB/TEST.
    """
    archive_path = Path(archive_path)
    c.require(archive_path.is_file(), "Missing frozen archive: actual attempt consumed")
    c.require(
        metadata["official_authenticity"]["size_bytes"] == archive_path.stat().st_size,
        "Official archive byte-count provenance mismatch",
    )
    ledger, segments = [], [None] * len(metadata["eligible"])
    with ZipFile(archive_path) as archive:
        selected = santos._selected_recordings(archive, list(DEV))
        c.require(len(selected) <= 300, "oversized DEV recording directory")
        c.same(
            [list(key) for key, _ in selected],
            [[r["subject"], r["day"], r["trial"]] for r in metadata["records"]],
            "frozen exact member identities/order",
        )
        for recording_index, ((key, info), expected) in enumerate(
            zip(selected, metadata["records"], strict=True)
        ):
            c.require(
                info.filename == expected["member"]
                and info.file_size == expected["uncompressed_bytes"]
                and not info.flag_bits & 1,
                "DEV ZipInfo provenance mismatch",
            )
            units = [
                i
                for i, s in enumerate(metadata["eligible"])
                if s["recording_index"] == recording_index
            ]
            if not units:
                ledger.append(
                    {
                        "recording_index": recording_index,
                        "member": info.filename,
                        "payload_read": False,
                        "reason": "outside original historical trial graph; short fragments only",
                        "carried_metadata": expected,
                    }
                )
                continue
            payload = archive.read(info)  # validated ZipInfo, exact DEV allowlist only
            record = santos._member_record(key, info, payload)
            c.same(
                {f: record[f] for f in record},
                {f: expected[f] for f in record},
                "allowed member payload identity/hash",
            )
            _, _, _ = santos._parse_recording(payload, record, 256, 256, n_channels=6)
            c.same(record, expected, "Santos continuity/row/schema diagnostics")
            rows = list(
                csv.reader(
                    io.StringIO(payload.decode("utf-8"), newline=""), strict=True
                )
            )[1:]
            for unit in units:
                item = metadata["eligible"][unit]
                a, b = item["original"]["start_row"], item["original"]["stop_row"]
                frames = np.array([int(row[0]) for row in rows[a:b]], dtype=np.int64)
                values = np.array(
                    [[float(x) for x in row[1:]] for row in rows[a:b]], dtype=np.float32
                ).T
                c.require(
                    values.shape == (6, item["length"])
                    and np.isfinite(values).all()
                    and np.all(np.diff(frames) == 1),
                    "incomplete/noncontiguous required complete source",
                )
                c.require(
                    int(frames[0]) == item["original"]["frame_start"]
                    and int(frames[-1]) == item["original"]["frame_end"],
                    "required frame interval mismatch",
                )
                segments[unit] = {
                    "values": values,
                    "frames": frames,
                    "rows": np.arange(a, b, dtype=np.int64),
                }
            ledger.append(
                {
                    "recording_index": recording_index,
                    "member": info.filename,
                    "payload_read": True,
                    "sha256": record["sha256"],
                    "bytes": len(payload),
                    "eligible_units": units,
                }
            )
    c.require(
        all(item is not None for item in segments), "missing complete source segment"
    )
    return segments, ledger


def physical_registration(metadata):
    rows = []
    expanded = hashlib.sha256()
    key_count = 0
    for item in metadata["eligible"]:
        a, b = item["original"]["start_row"], item["original"]["stop_row"]
        rows.append(
            {
                "recording_key": item["recording_key"],
                "member": item["member"],
                "member_sha256": item["member_sha256"],
                "rows": [a, b],
                "frame_interval_inclusive": [
                    item["original"]["frame_start"],
                    item["original"]["frame_end"],
                ],
                "channels": ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"],
            }
        )
        for source_row in range(a, b):
            for channel in range(6):
                expanded.update(k.physical_key(item, source_row, channel) + b"\n")
                key_count += 1
    return {
        "version": "physical-row-normal-v1",
        "seed": 20261025,
        "segments": rows,
        "ordered_coordinate_ledger_sha256": c.fingerprint(rows),
        "ordered_expanded_physical_key_sha256": expanded.hexdigest(),
        "expanded_key_count": key_count,
        "expanded_key_order": "registered eligible segments, ascending physical source row, channel0..5; canonical keys separated by newline",
        "generated_outcomes_in_registration": False,
    }


def frozen_r_paths(root):
    """Only path construction during prepare. Root freeze later hashes payloads."""
    names = ["registration.json", "development.json", "results.json"]
    for fold in range(20):
        names.append(f"thresholds_{fold:02}.json")
        for role in [*[f"inner{i}" for i in range(4)], "refit"]:
            names.append(f"fold{fold:02}_canonical362_{role}_model.npz")
        names.extend(f"fold{fold:02}_canonical362_inner{i}.npz" for i in range(4))
        names.extend(
            f"fold{fold:02}_canonical362_refit_{condition}.npz"
            for condition in k.CONDITIONS
        )
    return [Path(root) / name for name in names]


def reference_freeze(root):
    result = {}
    for path in frozen_r_paths(root):
        c.require(
            path.is_file() and not path.is_symlink(),
            "Missing frozen R provenance: " + str(path),
        )
        result[path.name] = {"sha256": c.sha(path), "bytes": path.stat().st_size}
    return result


def reference_check(root, frozen):
    c.same(
        reference_freeze(root), frozen, "all frozen R states/thresholds/scores/index"
    )
