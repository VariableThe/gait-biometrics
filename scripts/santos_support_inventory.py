"""A registered, metadata-only Santos DEV continuity/support inventory.

Only this module, its tests/documentation, and explicit JSON inputs are read.
No historical runner, signal reader, feature extractor, model or scorer is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import signal
import struct
import sys
from pathlib import Path
from typing import Any

PROTOCOL = "santos_dev_continuity_support_inventory_v1"
DEV_IDS = (1, 3, 5, 7, 9, 11, 13, 14, 16, 17, 19, 24, 25, 27, 29)
HISTORICAL_ORDER = (7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17)
DEFINITIONS = ("historical256", "fixed768", "complete_segment")
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/santos_support_inventory.py",
    "tests/test_santos_support_inventory.py",
    "docs/santos_support_inventory.md",
)
KNOWN_RECONCILIATION = {
    "historical_windows": 588,
    "continuity_breaks": 28,
    "historical_tail_rows": 33810,
}
HEADER = ["frame", "acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
MEMBER_FIELDS = (
    "subject",
    "day",
    "trial",
    "member",
    "sha256",
    "uncompressed_bytes",
    "recording_key",
)
CLAIM_COLUMNS = ("labels", "claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_UNITS = 100_000
PATH_PATTERN = re.compile(
    r"A multi-sensor human gait dataset/processed_data/user(?P<directory>[0-9]{2})/"
    r"capture_user(?P<subject>[0-9]{2})_day(?P<day>[12])_"
    r"(?P<trial>000[1-9]|0010)_imu_walk\.csv\Z"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("STOP: " + message)


def integer(value: Any, name: str, low: int = 0, high: int = 2**63 - 1) -> int:
    require(
        type(value) is int and low <= value <= high, name + ": exact integer required"
    )
    return value


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_string(value: Any, name: str) -> str:
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        name + ": lowercase SHA-256 required",
    )
    return value


def _validate_json(value: Any) -> None:
    """Validate exact JSON types using auxiliary space bounded by nesting depth."""
    kind = type(value)
    if kind is dict:
        for key, item in value.items():
            require(type(key) is str, "JSON object keys must be strings")
            _validate_json(item)
    elif kind is list:
        for item in value:
            _validate_json(item)
    elif kind is float:
        require(math.isfinite(value), "nonfinite JSON number")
    else:
        require(
            kind in (str, int, bool, type(None)), "non-JSON value: " + kind.__name__
        )


def _json_equal(actual: Any, expected: Any) -> bool:
    """Compare validated JSON trees without allocating parallel signatures."""
    kind = type(actual)
    if kind is not type(expected):
        return False
    if kind is dict:
        return len(actual) == len(expected) and all(
            key in expected and _json_equal(item, expected[key])
            for key, item in actual.items()
        )
    if kind is list:
        return len(actual) == len(expected) and all(
            _json_equal(item, other)
            for item, other in zip(actual, expected, strict=True)
        )
    if kind is float:
        return actual.hex() == expected.hex()
    return actual == expected


def canonical(value: Any) -> bytes:
    _validate_json(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(
        _json_equal(value, json.loads(encoded)), "JSON type-preserving roundtrip failed"
    )
    return encoded


def fingerprint(value: Any) -> str:
    return sha256(canonical(value))


def same(actual: Any, expected: Any, name: str) -> None:
    _validate_json(actual)
    _validate_json(expected)
    require(_json_equal(actual, expected), name + " differs")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse_integer(token: str) -> int:
    require(token != "-0", "ambiguous signed integer zero")
    return int(token)


def _bad_constant(token: str) -> Any:
    raise ValueError("STOP: nonfinite JSON constant: " + token)


def decode_metadata(payload: bytes, compressed: bool) -> tuple[dict[str, Any], bytes]:
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded = stream.read(MAX_JSON_BYTES + 1)
    else:
        decoded = payload
    require(len(decoded) <= MAX_JSON_BYTES, "decoded JSON exceeds frozen byte budget")
    obj = json.loads(
        decoded.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_int=_parse_integer,
        parse_constant=_bad_constant,
    )
    require(type(obj) is dict, "metadata must be a JSON object")
    canonical(obj)
    return obj, decoded


def read_file(path: Path, *, metadata: bool = False) -> bytes:
    if metadata:
        require(
            path.name.endswith((".json", ".json.gz")),
            "input must be .json or .json.gz metadata",
        )
    require(
        path.is_file() and not path.is_symlink(),
        "regular nonsymlink file required: " + str(path),
    )
    require(path.stat().st_size <= MAX_FILE_BYTES, "file exceeds frozen byte budget")
    with path.open("rb") as stream:
        value = stream.read(MAX_FILE_BYTES + 1)
    require(len(value) <= MAX_FILE_BYTES, "file grew past frozen byte budget")
    return value


def write_new(path: Path, value: Any) -> None:
    payload = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": 1,
        "dev_ids": list(DEV_IDS),
        "definitions": {
            "historical256": {"size": 256, "stride": 256, "anchor": "segment start"},
            "fixed768": {
                "size": 768,
                "stride": 768,
                "anchor": "segment start",
                "historical_tiles_per_block": 3,
            },
            "complete_segment": {
                "unit": "entire nonempty valid continuity segment",
                "conceptual_only": True,
                "extraction_eligibility": "unassessed",
            },
        },
        "bands": ["<256", "256..767", ">=768"],
        "summaries": ["exact lengths", "minimum", "median", "maximum"],
        "flags": {
            "identity_support": "at least one day1 enrollment trial and one day2 probe trial with units",
            "some_identity_support": "any of the fixed 15 identities has both-day raw support",
            "all_identity_support": "all fixed 15 identities have both-day raw support",
            "complete_historical_trial_support": "every original historical eligible day1/day2 trial has units",
            "train_gate": "at least two units per identity, both days combined, for each frozen 9/12 TRAIN owner",
            "train_gate_interpretation": "conservative proposed WCCN count gate only; no covariance or extraction feasibility proof",
            "complete_segment_interpretation": "raw support only; one sample is not a trainable feature; conceptual unit counts may be below two despite ample samples",
            "claim_flags": "any/all ORIGINAL enrollment trial support plus original probe support; diagnostics, no partial-failure policy",
        },
        "reconciliation_reference": dict(KNOWN_RECONCILIATION),
        "graph_projection": "frozen registration.plan.folds and development.coverage; validate using source_coverage historical256-positive trials and the documented ordered crossfit projection",
        "budgets": {
            "cpu_seconds": 60,
            "address_space_bytes": 512 * 1024 * 1024,
            "cpus": 1,
            "cuda": False,
            "raw_member_reads": 0,
            "fits": 0,
            "input_file_bytes": MAX_FILE_BYTES,
            "decoded_json_bytes": MAX_JSON_BYTES,
            "maximum_support_units_per_definition": MAX_UNITS,
        },
        "no_expansion": "same all-15 cohort and 20 outer folds; source-present old-graph-absent trials stay separate; no protocol choice or promotion",
    }


def runtime() -> dict[str, Any]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "cache_tag": sys.implementation.cache_tag,
        "byteorder": sys.byteorder,
    }


def source_hashes() -> dict[str, str]:
    return {name: sha256(read_file(ROOT / name)) for name in SOURCES}


def enforce_limits() -> None:
    """Linux administrative ceilings; never increase an existing limit."""
    require(hasattr(os, "sched_setaffinity"), "CPU affinity enforcement unavailable")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for kind, ceiling in (
        (resource.RLIMIT_CPU, 60),
        (resource.RLIMIT_AS, 512 * 1024 * 1024),
    ):
        soft, hard = resource.getrlimit(kind)
        finite = [
            value for value in (soft, hard, ceiling) if value != resource.RLIM_INFINITY
        ]
        limit = min(
            value for value in (hard, ceiling) if value != resource.RLIM_INFINITY
        )
        warning = max(1, limit - 1) if kind == resource.RLIMIT_CPU else limit
        # Leave a CPU second for a failure diagnostic before the immutable hard stop.
        resource.setrlimit(kind, (min(*finite, warning), limit))


def _ids(values: Any, name: str, count: int) -> list[int]:
    require(
        type(values) is list and len(values) == count, name + ": wrong identity count"
    )
    for value in values:
        integer(value, name, 1, 30)
    require(
        len(set(values)) == count and set(values) <= set(DEV_IDS),
        name + ": invalid cohort",
    )
    return values


def validate_folds(registration: dict[str, Any]) -> list[dict[str, Any]]:
    old = registration["plan"]
    same(old["protocol"], "santos_dev_canonical_orientation_v1", "historical protocol")
    _ids(old["payload_allowlist"], "historical DEV allowlist", 15)
    same(old["payload_allowlist"], list(HISTORICAL_ORDER), "historical DEV order")
    same(
        old["window"],
        {"samples": 256, "stride": 256, "sampling_rate_hz": 100},
        "historical window",
    )
    folds = old["folds"]
    require(
        type(folds) is list and len(folds) == 20, "exactly 20 frozen folds required"
    )
    triples = []
    for index, fold in enumerate(folds):
        integer(fold["fold"], "fold index", 0, 19)
        same(fold["fold"], index, "fold order")
        triples.append(_ids(fold["validation_ids"], "validation IDs", 3))
    require(
        len({tuple(sorted(ids)) for ids in triples}) == 20,
        "distinct frozen validation triples required",
    )
    for partition in range(4):
        same(
            sorted(
                s for ids in triples[partition * 5 : partition * 5 + 5] for s in ids
            ),
            list(DEV_IDS),
            "partition all-15 binding",
        )
    for index, fold in enumerate(folds):
        validation = fold["validation_ids"]
        _ids(fold["refit_ids"], "refit TRAIN IDs", 12)
        same(
            fold["refit_ids"],
            [s for s in HISTORICAL_ORDER if s not in validation],
            "refit TRAIN order",
        )
        require(
            type(fold["inner"]) is list and len(fold["inner"]) == 4,
            "four frozen inner owners required",
        )
        partition, position = divmod(index, 5)
        for offset, owner in enumerate(fold["inner"], 1):
            held = triples[partition * 5 + (position + offset) % 5]
            _ids(owner["held_ids"], "held IDs", 3)
            _ids(owner["train_ids"], "inner TRAIN IDs", 9)
            same(owner["held_ids"], held, "inner held order")
            same(
                owner["train_ids"],
                [s for s in HISTORICAL_ORDER if s not in validation + held],
                "inner TRAIN order",
            )
    require(
        type(registration["sources"]) is dict and bool(registration["sources"]),
        "historical source hashes missing",
    )
    for name, digest in registration["sources"].items():
        require(type(name) is str and bool(name), "historical source path missing")
        digest_string(digest, "historical source hash")
    return folds


def recording_identity(record: dict[str, Any]) -> tuple[int, int, int]:
    subject = integer(record["subject"], "recording subject", 1, 30)
    day = integer(record["day"], "recording day", 1, 2)
    trial = integer(record["trial"], "recording trial", 1, 10)
    require(subject in DEV_IDS, "recording outside exact DEV cohort")
    match = PATH_PATTERN.fullmatch(record["member"])
    require(match is not None, "exact official processed MCU walking path required")
    assert match is not None
    require(
        match["directory"] == match["subject"], "directory/filename subject mismatch"
    )
    same(
        [int(match["subject"]), int(match["day"]), int(match["trial"])],
        [subject, day, trial],
        "path/metadata identity",
    )
    same(
        record["recording_key"],
        [7, subject, day, trial, "MCU", "processed_data"],
        "recording key",
    )
    digest_string(record["sha256"], "member hash")
    integer(record["uncompressed_bytes"], "member byte count", 1)
    same(record["header"], HEADER, "recording header")
    return subject, day, trial


def validate_recording(record: dict[str, Any]) -> None:
    for key in (
        "source_rows",
        "valid_rows",
        "invalid_rows",
        "windows",
        "covered_rows",
        "tail_rows",
        "stride_gap_rows",
    ):
        integer(record[key], key)
    source_rows = record["source_rows"]
    invalid = record["invalid_row_offsets"]
    require(type(invalid) is list, "invalid-row offsets missing")
    for row in invalid:
        integer(row, "invalid row", 0, source_rows - 1)
    same(sorted(set(invalid)), invalid, "ordered unique invalid offsets")
    same(record["invalid_rows"], len(invalid), "invalid-row count")
    same(record["valid_rows"], source_rows - len(invalid), "valid-row count")
    segments = record["segments"]
    require(type(segments) is list, "continuity segments missing")
    pieces = [(row, row + 1) for row in invalid]
    breaks = []
    previous = None
    for segment in segments:
        for field in (
            "start_row",
            "stop_row",
            "rows",
            "windows",
            "covered_rows",
            "tail_rows",
            "stride_gap_rows",
        ):
            integer(segment[field], "segment " + field)
        start, stop, length = segment["start_row"], segment["stop_row"], segment["rows"]
        require(
            0 <= start < stop <= source_rows and stop - start == length,
            "invalid half-open segment interval/length",
        )
        a = integer(segment["frame_start"], "frame start", -(2**63))
        b = integer(segment["frame_end"], "frame end", -(2**63))
        require(b - a == length - 1, "segment frame endpoints disagree with length")
        if previous is not None:
            require(previous["stop_row"] <= start, "unordered or overlapping segments")
            if previous["stop_row"] == start:
                last = previous["frame_end"]
                require(a != last + 1, "unexplained split of continuous frames")
                breaks.append(
                    {
                        "row": start,
                        "previous_frame": last,
                        "frame": a,
                        "reason": "gap"
                        if a > last + 1
                        else "duplicate"
                        if a == last
                        else "reset",
                    }
                )
        same(
            [
                segment["windows"],
                segment["covered_rows"],
                segment["tail_rows"],
                segment["stride_gap_rows"],
            ],
            [length // 256, length // 256 * 256, length % 256, 0],
            "historical segment counts",
        )
        pieces.append((start, stop))
        previous = segment
    cursor = 0
    for start, stop in sorted(pieces):
        require(
            start == cursor,
            "invalid rows and segments must partition every source row exactly",
        )
        cursor = stop
    same(cursor, source_rows, "complete source-row partition")
    # Strict numeric checking is necessary even where Python equates True and 1.
    require(
        type(record["continuity_breaks"]) is list, "continuity-break ledger missing"
    )
    for item in record["continuity_breaks"]:
        integer(item["row"], "break row", 0, source_rows - 1)
        integer(item["previous_frame"], "previous break frame", -(2**63))
        integer(item["frame"], "break frame", -(2**63))
    same(
        record["continuity_breaks"], breaks, "continuity-break reasons/endpoints/order"
    )
    for key in ("windows", "covered_rows", "tail_rows", "stride_gap_rows"):
        same(record[key], sum(s[key] for s in segments), "recording " + key)
    same(
        record["valid_rows"],
        sum(s["rows"] for s in segments),
        "segment valid-row total",
    )


def _historical_trials(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int], list[int]]:
    return {
        (subject, day): [
            r["trial"]
            for r in records
            if r["subject"] == subject and r["day"] == day and r["windows"] > 0
        ]
        for subject in DEV_IDS
        for day in (1, 2)
    }


def _graph_projection(
    ids: list[int], historical: dict[tuple[int, int], list[int]]
) -> dict[str, Any]:
    identities = sorted(ids)
    result: dict[str, Any] = {name: [] for name in CLAIM_COLUMNS}
    result["enrollment_trials"] = {str(s): historical[s, 1] for s in identities}
    result["probe_trials"] = {str(s): historical[s, 2] for s in identities}
    for subject in identities:
        require(
            bool(historical[subject, 1]) and bool(historical[subject, 2]),
            "historical graph lacks necessary both-day trial metadata",
        )
        for trial in historical[subject, 2]:
            for claim in identities:
                for name, value in zip(
                    CLAIM_COLUMNS,
                    (int(claim == subject), claim, subject, trial),
                    strict=True,
                ):
                    result[name].append(value)
    result["n_genuine"] = sum(result["labels"])
    result["n_impostor"] = len(result["labels"]) - result["n_genuine"]
    return result


def _pool_projection(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        key: [value for graph in graphs for value in graph[key]]
        for key in CLAIM_COLUMNS
    }
    result["n_genuine"] = sum(result["labels"])
    result["n_impostor"] = len(result["labels"]) - sum(result["labels"])
    result["directed_pairs"] = sorted(
        {
            f"{c}<-{p}"
            for c, p in zip(
                result["claim_subject_ids"], result["probe_subject_ids"], strict=True
            )
            if c != p
        }
    )
    return result


def validate_metadata(
    registration: dict[str, Any], development: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed; all validation and projection uses saved JSON, never signals."""
    canonical(registration)
    canonical(development)
    folds = validate_folds(registration)
    source = development["source_coverage"]
    same(source["subject_ids"], list(DEV_IDS), "exact sorted all-15 source cohort")
    for field, expected in (
        ("dataset_doi", "10.6084/m9.figshare.14727231.v7"),
        ("article_id", 14727231),
        ("version", 7),
        ("file_id", 35882624),
        ("sampling_rate_hz", 100),
        ("window_size", 256),
        ("stride", 256),
        ("source_header", HEADER),
        ("channels", HEADER[1:]),
        ("channel_names", HEADER[1:]),
        ("channel_units", ["m/s^2"] * 3 + ["degrees/sec"] * 3),
    ):
        same(source[field], expected, "source " + field)
    records = source["recordings"]
    require(
        type(records) is list and 0 < len(records) <= 300,
        "source recording ledger missing/oversized",
    )
    identities = []
    for record in records:
        identities.append(recording_identity(record))
        validate_recording(record)
    require(len(set(identities)) == len(identities), "duplicate recording identities")
    same(
        [list(key) for key in identities],
        [list(key) for key in sorted(identities)],
        "source recording order",
    )
    present = set(identities)
    missing = [
        {"subject": s, "day": d, "trial": t}
        for s in DEV_IDS
        for d in (1, 2)
        for t in range(1, 11)
        if (s, d, t) not in present
    ]
    exclusions = {
        "missing_recordings": missing,
        "invalid_rows": sum(r["invalid_rows"] for r in records),
        "continuity_breaks": sum(len(r["continuity_breaks"]) for r in records),
        "tail_rows": sum(r["tail_rows"] for r in records),
        "stride_gap_rows": 0,
        "zero_window_recordings": [
            i for i, r in enumerate(records) if r["windows"] == 0
        ],
    }
    same(source["exclusions"], exclusions, "explicit source exclusions/counts")
    historical = _historical_trials(records)
    coverage = []
    for subject in DEV_IDS:
        days = {
            str(day): {
                "present_trials": [
                    r["trial"]
                    for r in records
                    if r["subject"] == subject and r["day"] == day
                ],
                "eligible_trials": historical[subject, day],
                "windows": sum(
                    r["windows"]
                    for r in records
                    if r["subject"] == subject and r["day"] == day
                ),
            }
            for day in (1, 2)
        }
        coverage.append(
            {
                "subject": subject,
                "days": days,
                "eligible_both_days": all(bool(historical[subject, d]) for d in (1, 2)),
            }
        )
    same(source["subject_coverage"], coverage, "historical subject coverage")
    manifest = [{key: record[key] for key in MEMBER_FIELDS} for record in records]
    member_hash = fingerprint(manifest)
    same(source["members_sha256"], member_hash, "ordered source member manifest hash")
    data = development["data"]
    same(data["members_sha256"], member_hash, "historical data member hash")
    same(
        data["members"],
        [
            {
                key: record[key]
                for key in ("member", "subject", "day", "trial", "sha256")
            }
            for record in records
        ],
        "historical ordered member projection",
    )
    digest_string(data["data_sha256"], "opaque historical signal fingerprint")
    expected_coverage = {}
    for fold in folds:
        inner = [
            _graph_projection(owner["held_ids"], historical) for owner in fold["inner"]
        ]
        expected_coverage[str(fold["fold"])] = {
            "inner": inner,
            "pool": _pool_projection(inner),
            "validation": _graph_projection(fold["validation_ids"], historical),
            "cohort_inner": [
                _graph_projection(owner["train_ids"], historical)
                for owner in fold["inner"]
            ],
            "cohort_refit": _graph_projection(fold["refit_ids"], historical),
        }
    same(
        development["coverage"],
        expected_coverage,
        "complete ordered original graph counts/order",
    )
    reconciliation = {
        "historical_windows": sum(r["windows"] for r in records),
        "continuity_breaks": exclusions["continuity_breaks"],
        "historical_tail_rows": exclusions["tail_rows"],
    }
    same(
        reconciliation, KNOWN_RECONCILIATION, "frozen metadata reconciliation reference"
    )
    return {
        "folds": folds,
        "source": source,
        "historical_trials": historical,
        "reconciliation": reconciliation,
        "missing": missing,
    }


def segment_support(segment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Count supports within this segment only; no concatenation or extraction."""
    length = integer(segment["rows"], "segment length", 1)
    start = integer(segment["start_row"], "segment anchor")
    stop = integer(segment["stop_row"], "segment stop", start + 1)
    require(stop - start == length, "segment support interval differs from length")
    result = {}
    for name, size in (
        ("historical256", 256),
        ("fixed768", 768),
        ("complete_segment", length),
    ):
        count, tail = divmod(length, size)
        require(count <= MAX_UNITS, "support-unit budget exceeded")
        result[name] = {
            "unit_count": count,
            "unique_original_row_coverage": count * size,
            "unused_valid_rows": tail,
            "covered_intervals": [[start, start + count * size]] if count else [],
            "unused_intervals": [[stop - tail, stop]] if tail else [],
        }
    return result


def _totals(rows: list[dict[str, Any]], definition: str) -> dict[str, int]:
    return {
        key: sum(row["support"][definition][key] for row in rows)
        for key in ("unit_count", "unique_original_row_coverage", "unused_valid_rows")
    }


def length_summary(lengths: list[int]) -> dict[str, Any]:
    for length in lengths:
        integer(length, "segment length", 1)
    ordered = sorted(lengths)
    if not ordered:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    middle = len(ordered) // 2
    numerator = (
        ordered[middle] * 2
        if len(ordered) % 2
        else ordered[middle - 1] + ordered[middle]
    )
    divisor = 1 if numerator % 2 == 0 else 2
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": {
            "numerator": numerator // 2 if divisor == 1 else numerator,
            "denominator": divisor,
        },
        "maximum": ordered[-1],
    }


def _claim_support(
    graph: dict[str, Any],
    lookup: dict[tuple[int, int, int], dict[str, Any]],
    definition: str,
) -> dict[str, Any]:
    enrollment = graph["enrollment_trials"]
    trial_rows = [
        {
            "subject": int(subject),
            "day": day,
            "trial": trial,
            "has_raw_support": lookup[int(subject), day, trial]["support"][definition][
                "unit_count"
            ]
            > 0,
        }
        for day, mapping in ((1, enrollment), (2, graph["probe_trials"]))
        for subject in sorted(mapping, key=int)
        for trial in mapping[subject]
    ]
    any_flags, all_flags = [], []
    for claim, probe, trial in zip(
        graph["claim_subject_ids"],
        graph["probe_subject_ids"],
        graph["probe_trial_ids"],
        strict=True,
    ):
        support = [
            lookup[claim, 1, t]["support"][definition]["unit_count"] > 0
            for t in enrollment[str(claim)]
        ]
        probe_ok = lookup[probe, 2, trial]["support"][definition]["unit_count"] > 0
        any_flags.append(any(support) and probe_ok)
        all_flags.append(bool(support) and all(support) and probe_ok)
    return {
        "original_trials": trial_rows,
        "failed_original_trials": [
            row for row in trial_rows if not row["has_raw_support"]
        ],
        "any_original_enrollment_trial_and_probe_raw_support": any_flags,
        "all_original_enrollment_trials_and_probe_raw_support": all_flags,
        "original_n_genuine": graph["n_genuine"],
        "original_n_impostor": graph["n_impostor"],
    }


def _owner(
    ids: list[int], lookup: dict[tuple[int, int, int], dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "train_ids": ids,
        "interpretation": plan()["flags"]["train_gate_interpretation"],
        "definitions": {},
    }
    for definition in DEFINITIONS:
        people = []
        for subject in ids:
            day_counts = {
                str(day): sum(
                    lookup[subject, day, trial]["support"][definition]["unit_count"]
                    for trial in range(1, 11)
                )
                for day in (1, 2)
            }
            count = sum(day_counts.values())
            people.append(
                {
                    "subject": subject,
                    "day_unit_counts": day_counts,
                    "unit_count": count,
                    "support": _totals(
                        [row for row in lookup.values() if row["subject"] == subject],
                        definition,
                    ),
                    "historical_graph_unit_count": sum(
                        row["support"][definition]["unit_count"]
                        for row in lookup.values()
                        if row["subject"] == subject and row["in_original_trial_graph"]
                    ),
                    "zero_unit_trials_by_day": {
                        str(day): [
                            trial
                            for trial in range(1, 11)
                            if lookup[subject, day, trial]["zero_units"][definition]
                        ]
                        for day in (1, 2)
                    },
                    "missing_trials_by_day": {
                        str(day): [
                            trial
                            for trial in range(1, 11)
                            if lookup[subject, day, trial]["recording_index"] is None
                        ]
                        for day in (1, 2)
                    },
                    "proposed_min_two_units": count >= 2,
                }
            )
        result["definitions"][definition] = {
            "per_identity": people,
            "support": _totals(
                [row for row in lookup.values() if row["subject"] in ids], definition
            ),
            "all_min_two_units": all(row["proposed_min_two_units"] for row in people),
            "raw_support_only": definition == "complete_segment",
        }
    return result


def derive_inventory(
    registration: dict[str, Any], development: dict[str, Any]
) -> dict[str, Any]:
    """Public pure metadata API. CLI callers must register exact bytes first."""
    try:
        validated = validate_metadata(registration, development)
    except KeyError as exc:
        raise ValueError("STOP: missing necessary metadata fact: " + str(exc)) from exc
    except (TypeError, IndexError, AttributeError) as exc:
        raise ValueError("STOP: malformed metadata schema: " + str(exc)) from exc
    records = validated["source"]["recordings"]
    segments: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for recording_index, record in enumerate(records):
        break_index = {item["row"]: item for item in record["continuity_breaks"]}
        for index, original in enumerate(record["segments"]):
            start, stop, length = (
                original["start_row"],
                original["stop_row"],
                original["rows"],
            )
            start_reason = (
                break_index[start]["reason"]
                if start in break_index
                else "recording_start"
                if start == 0
                else "after_invalid_rows"
            )
            end_reason = (
                break_index[stop]["reason"]
                if stop in break_index
                else "recording_end"
                if stop == record["source_rows"]
                else "before_invalid_rows"
            )
            support = segment_support(original)
            first_window = len(windows)
            for tile in range(support["historical256"]["unit_count"]):
                row = start + tile * 256
                windows.append(
                    {
                        "window_index": len(windows),
                        "recording_index": recording_index,
                        "segment_index": index,
                        "row_key": [
                            record["subject"],
                            record["day"],
                            record["trial"],
                            row,
                        ],
                        "rows": [row, row + 256],
                        "frames": [
                            original["frame_start"] + tile * 256,
                            original["frame_start"] + tile * 256 + 255,
                        ],
                    }
                )
            support["fixed768"]["blocks"] = [
                {
                    "rows": [start + block * 768, start + (block + 1) * 768],
                    "historical_window_indices": list(
                        range(first_window + block * 3, first_window + block * 3 + 3)
                    ),
                }
                for block in range(support["fixed768"]["unit_count"])
            ]
            segments.append(
                {
                    "recording_index": recording_index,
                    "segment_index": index,
                    "subject": record["subject"],
                    "day": record["day"],
                    "trial": record["trial"],
                    "member": record["member"],
                    "member_sha256": record["sha256"],
                    "recording_key": record["recording_key"],
                    "original": original,
                    "start_reason": start_reason,
                    "end_reason": end_reason,
                    "length": length,
                    "band": "<256"
                    if length < 256
                    else "256..767"
                    if length < 768
                    else ">=768",
                    "length_eligibility": {
                        "at_least_256_rows": length >= 256,
                        "at_least_768_rows": length >= 768,
                        "extraction_or_trainability": "unassessed",
                    },
                    "support": support,
                }
            )
            require(
                len(windows) <= MAX_UNITS and len(segments) <= MAX_UNITS,
                "total support-unit budget exceeded",
            )
    clean = development.get("geometry", {}).get("clean", {})
    row_link = "not available in metadata"
    if "row_keys_sha256" in clean:
        same(
            clean["row_key_columns"],
            ["subject", "day", "trial", "start"],
            "historical row-key columns",
        )
        same(clean["rows"], len(windows), "historical ordered window count")
        # The historical NumPy native int64 array was written on little-endian Linux.
        digest = hashlib.sha256(canonical(["int64", [len(windows), 4]]))
        for window in windows:
            digest.update(struct.pack("<4q", *window["row_key"]))
        same(
            clean["row_keys_sha256"],
            digest.hexdigest(),
            "historical ordered int64 row-key hash",
        )
        row_link = "verified historical little-endian int64 row-key projection; no window signals read"
    historical = validated["historical_trials"]
    by_identity = {
        (r["subject"], r["day"], r["trial"]): (i, r) for i, r in enumerate(records)
    }
    trials: list[dict[str, Any]] = []
    for subject in DEV_IDS:
        for day in (1, 2):
            for trial in range(1, 11):
                entry = by_identity.get((subject, day, trial))
                index, record = entry if entry else (None, None)
                group = (
                    [s for s in segments if s["recording_index"] == index]
                    if record
                    else []
                )
                trials.append(
                    {
                        "subject": subject,
                        "day": day,
                        "trial": trial,
                        "recording_index": index,
                        "status": "source_present"
                        if record
                        else "missing_as_frozen_metadata",
                        "member": record["member"] if record else None,
                        "recording_key": record["recording_key"] if record else None,
                        "member_sha256": record["sha256"] if record else None,
                        "valid_rows": record["valid_rows"] if record else 0,
                        "source_rows": record["source_rows"] if record else None,
                        "invalid_row_offsets": record["invalid_row_offsets"]
                        if record
                        else [],
                        "continuity_breaks": record["continuity_breaks"]
                        if record
                        else [],
                        "segment_count": len(group),
                        "fragmented": len(group) > 1,
                        "unbroken": len(group) == 1
                        and record is not None
                        and record["invalid_rows"] == 0,
                        "in_original_trial_graph": trial in historical[subject, day],
                        "support": {name: _totals(group, name) for name in DEFINITIONS},
                        "zero_units": {
                            name: _totals(group, name)["unit_count"] == 0
                            for name in DEFINITIONS
                        },
                    }
                )
    lookup = {(r["subject"], r["day"], r["trial"]): r for r in trials}
    subject_days = [
        {
            "subject": subject,
            "day": day,
            "present_trials": [
                r["trial"]
                for r in trials
                if r["subject"] == subject
                and r["day"] == day
                and r["recording_index"] is not None
            ],
            "historical_trials": historical[subject, day],
            "missing_trials": [
                r["trial"]
                for r in trials
                if r["subject"] == subject
                and r["day"] == day
                and r["recording_index"] is None
            ],
            "fragmented_trials": [
                r["trial"]
                for r in trials
                if r["subject"] == subject and r["day"] == day and r["fragmented"]
            ],
            "unbroken_trials": [
                r["trial"]
                for r in trials
                if r["subject"] == subject and r["day"] == day and r["unbroken"]
            ],
            "support": {
                name: _totals(
                    [r for r in trials if r["subject"] == subject and r["day"] == day],
                    name,
                )
                for name in DEFINITIONS
            },
            "zero_unit_trials": {
                name: [
                    t
                    for t in range(1, 11)
                    if lookup[subject, day, t]["zero_units"][name]
                ]
                for name in DEFINITIONS
            },
        }
        for subject in DEV_IDS
        for day in (1, 2)
    ]
    flags: dict[str, Any] = {}
    for name in DEFINITIONS:
        people = []
        for subject in DEV_IDS:
            available = {
                str(day): [
                    t
                    for t in range(1, 11)
                    if lookup[subject, day, t]["support"][name]["unit_count"] > 0
                ]
                for day in (1, 2)
            }
            people.append(
                {
                    "subject": subject,
                    "supported_trials_by_day": available,
                    "both_day_raw_support": all(
                        bool(values) for values in available.values()
                    ),
                    "complete_historical_trial_support": all(
                        lookup[subject, day, t]["support"][name]["unit_count"] > 0
                        for day in (1, 2)
                        for t in historical[subject, day]
                    ),
                }
            )
        flags[name] = {
            "per_identity": people,
            "some_identity_support": any(p["both_day_raw_support"] for p in people),
            "all_identity_support": all(p["both_day_raw_support"] for p in people),
            "complete_historical_trial_support": all(
                p["complete_historical_trial_support"] for p in people
            ),
            "raw_support_only": name == "complete_segment",
        }
    fold_ledger: list[dict[str, Any]] = []
    for fold in validated["folds"]:
        graphs = development["coverage"][str(fold["fold"])]
        support_graphs: dict[str, Any] = {}
        for role in ("inner", "cohort_inner", "validation", "cohort_refit"):
            group = graphs[role] if type(graphs[role]) is list else [graphs[role]]
            projected = [
                {name: _claim_support(graph, lookup, name) for name in DEFINITIONS}
                for graph in group
            ]
            support_graphs[role] = (
                projected if type(graphs[role]) is list else projected[0]
            )
        support_graphs["pool"] = {
            name: {
                key: [
                    value
                    for item in support_graphs["inner"]
                    for value in item[name][key]
                ]
                for key in (
                    "any_original_enrollment_trial_and_probe_raw_support",
                    "all_original_enrollment_trials_and_probe_raw_support",
                )
            }
            for name in DEFINITIONS
        }
        fold_ledger.append(
            {
                "fold": fold["fold"],
                "validation_support": {
                    name: _totals(
                        [r for r in trials if r["subject"] in fold["validation_ids"]],
                        name,
                    )
                    for name in DEFINITIONS
                },
                "inner_train_owners": [
                    _owner(owner["train_ids"], lookup) for owner in fold["inner"]
                ],
                "refit_train_owner": _owner(fold["refit_ids"], lookup),
                "original_graph_support": support_graphs,
            }
        )
    lengths = [s["length"] for s in segments]
    return {
        "protocol": PROTOCOL,
        "dev_ids": list(DEV_IDS),
        "definitions": list(DEFINITIONS),
        "original_folds": validated["folds"],
        "original_graphs": development["coverage"],
        "original_recording_order": [
            {
                "recording_index": index,
                "recording_key": record["recording_key"],
                "member": record["member"],
                "historical_windows": record["windows"],
            }
            for index, record in enumerate(records)
        ],
        "original_windows": windows,
        "historical_row_key_link": row_link,
        "segments": segments,
        "trials": trials,
        "subject_days": subject_days,
        "folds": fold_ledger,
        "source_present_absent_original_graph": [
            r
            for r in trials
            if r["recording_index"] is not None and not r["in_original_trial_graph"]
        ],
        "subject19_day2_trial1": lookup[19, 2, 1],
        "flags": flags,
        "totals": {name: _totals(trials, name) for name in DEFINITIONS},
        "length_summary": length_summary(lengths),
        "length_bands": {
            band: length_summary([s["length"] for s in segments if s["band"] == band])
            for band in plan()["bands"]
        },
        "metadata_reconciliation": {
            "observed": validated["reconciliation"],
            "reference": dict(KNOWN_RECONCILIATION),
            "matches": True,
            "interpretation": "saved metadata reconciliation only; no raw verification",
        },
        "integrity_limits": {
            "historical_signal_fingerprint": development["data"]["data_sha256"],
            "signal_fingerprint_verified": False,
            "member_payload_hashes_verified": False,
            "member_manifest_link_verified": True,
            "raw_member_reads": 0,
        },
        "interpretation": plan()["no_expansion"],
    }


def _load_bound(
    registration: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    values, decoded_bytes = {}, {}
    for role in ("historical_registration", "development_metadata"):
        binding = registration["inputs"][role]
        snapshot = binding["snapshot"]
        require(
            snapshot in (role + ".input.json", role + ".input.json.gz"),
            "invalid metadata snapshot assignment",
        )
        payload = read_file(root / snapshot, metadata=True)
        same(len(payload), binding["bytes"], role + " exact byte count")
        same(sha256(payload), binding["sha256"], role + " exact input bytes")
        values[role], decoded_bytes[role] = decode_metadata(
            payload, snapshot.endswith(".gz")
        )
    development = values["development_metadata"]
    same(
        development["registration_sha256"],
        sha256(decoded_bytes["historical_registration"]),
        "historical registration uncompressed byte hash link",
    )
    return values["historical_registration"], development


def run(args: argparse.Namespace) -> dict[str, Any]:
    enforce_limits()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=False)
    write_new(
        root / "attempt.json",
        {
            "protocol": PROTOCOL,
            "consumed": True,
            "retry": "forbidden in this output root, including failed attempts",
        },
    )
    phase = "freeze exact inputs and source closure"
    frozen: dict[str, Any] = {
        "plan": plan(),
        "runtime": runtime(),
        "sources": {},
        "inputs": {},
    }
    try:
        frozen["sources"] = source_hashes()
        payloads = {}
        for role, path in (
            ("historical_registration", Path(args.historical_registration_json)),
            ("development_metadata", Path(args.development_metadata_json)),
        ):
            payload = read_file(path, metadata=True)
            snapshot = role + (
                ".input.json.gz" if path.name.endswith(".gz") else ".input.json"
            )
            frozen["inputs"][role] = {
                "original_path": str(path.absolute()),
                "snapshot": snapshot,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
            payloads[snapshot] = payload
        # No metadata decode, validation, graph reconstruction or support derivation precedes this write.
        write_new(root / "registration.json", frozen)
        for snapshot, payload in payloads.items():
            with (root / snapshot).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        phase = "validate saved metadata and derive registered support inventory"
        old_registration, development = _load_bound(frozen, root)
        inventory = derive_inventory(old_registration, development)
        result = {
            "registration_sha256": sha256(read_file(root / "registration.json")),
            "inputs_sha256": fingerprint(frozen["inputs"]),
            "inventory": inventory,
        }
        write_new(root / "result.json", result)
        write_new(
            root / "diagnostics.json",
            {
                "status": "complete",
                "phase": phase,
                "registration_sha256": result["registration_sha256"],
                "result_sha256": sha256(read_file(root / "result.json")),
                "raw_member_reads": 0,
            },
        )
        return {"status": "complete", "protocol": PROTOCOL, "out_root": str(root)}
    except Exception as exc:
        if not (root / "registration.json").exists():
            # An unreadable input cannot supply a hash. Retain the incomplete freeze,
            # label it explicitly, and never parse/derive or retry this attempt.
            frozen["preflight_incomplete"] = True
            write_new(root / "registration.json", frozen)
        write_new(
            root / "diagnostics.json",
            {
                "status": "failed_consumed",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_member_reads": 0,
            },
        )
        raise


def verify(args: argparse.Namespace) -> dict[str, Any]:
    enforce_limits()
    root = Path(args.out_root)
    frozen_payload = read_file(root / "registration.json")
    frozen, _ = decode_metadata(frozen_payload, False)
    # Refuse changed source/runtime/flags before reading or deriving historical metadata.
    same(frozen["sources"], source_hashes(), "source closure")
    same(frozen["runtime"], runtime(), "Python runtime")
    same(frozen["plan"], plan(), "registered definitions/flags/budgets")
    attempt, _ = decode_metadata(read_file(root / "attempt.json"), False)
    same(
        attempt,
        {
            "protocol": PROTOCOL,
            "consumed": True,
            "retry": "forbidden in this output root, including failed attempts",
        },
        "attempt reservation",
    )
    diagnostics, _ = decode_metadata(read_file(root / "diagnostics.json"), False)
    require(
        diagnostics["status"] == "complete", "failed attempt consumed; no replay/retry"
    )
    same(
        diagnostics["registration_sha256"],
        sha256(frozen_payload),
        "diagnostic registration binding",
    )
    payload = read_file(root / "result.json")
    same(diagnostics["result_sha256"], sha256(payload), "result byte hash")
    saved, _ = decode_metadata(payload, False)
    require(
        saved.keys() == {"registration_sha256", "inputs_sha256", "inventory"}
        and type(saved["inventory"]) is dict,
        "result schema requires exactly registration_sha256, inputs_sha256 and inventory object",
    )
    same(
        saved["registration_sha256"],
        sha256(frozen_payload),
        "result registration binding",
    )
    same(saved["inputs_sha256"], fingerprint(frozen["inputs"]), "result input bindings")
    old_registration, development = _load_bound(frozen, root)
    same(
        saved,
        {
            "registration_sha256": sha256(frozen_payload),
            "inputs_sha256": fingerprint(frozen["inputs"]),
            "inventory": derive_inventory(old_registration, development),
        },
        "deterministic result replay",
    )
    return {
        "status": "verified",
        "protocol": PROTOCOL,
        "raw_member_reads": 0,
        "fits": 0,
    }


def _cpu_expired(_signum: int, _frame: Any) -> None:
    raise TimeoutError("STOP: frozen administrative CPU budget exhausted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("run", "verify"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--historical-registration-json")
    parser.add_argument("--development-metadata-json")
    args = parser.parse_args(argv)
    if args.stage == "run" and (
        not args.historical_registration_json or not args.development_metadata_json
    ):
        parser.error(
            "run requires --historical-registration-json and --development-metadata-json"
        )
    if args.stage == "verify" and (
        args.historical_registration_json or args.development_metadata_json
    ):
        parser.error("verify reads saved metadata only; specify --out-root")
    signal.signal(signal.SIGXCPU, _cpu_expired)
    try:
        report = run(args) if args.stage == "run" else verify(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        MemoryError,
        RecursionError,
    ) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(canonical(report).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
