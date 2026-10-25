"""Verify the compact synthetic bundle without extraction, fitting or writes.

This checks published bytes, rows and the two disclosed fingerprint forms.
It explicitly does not replay the omitted full numeric arrays.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
REPORT_SHA = "bd890b2065009c961c7f8dc53624d85435e50ea44186d86465323673b45761da"
SAVED_JSON_SHA = "9061bf1d4df7f88a56ef9dd156e55625a4b44933867e87fd55bc5083cf890112"
IN_MEMORY_SHA = "51a085a519c36fb465a0b69812c61bed11cac5bca9995c31d1da15d8a7af801b"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def fingerprint(value):
    return digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(value):
    if isinstance(value, float):
        return {"float64_hex": value.hex()}
    if isinstance(value, dict):
        return {key: bits(val) for key, val in value.items()}
    if isinstance(value, list):
        return [bits(val) for val in value]
    return value


def main():
    inventory = json.loads((ROOT / "inventory.json").read_text())
    listed = {entry["path"] for entry in inventory["files"]}
    present = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "inventory.json" and "__pycache__" not in p.parts
    }
    require(listed == present, "Missing or extra published files")
    for entry in inventory["files"]:
        payload = (ROOT / entry["path"]).read_bytes()
        require(
            len(payload) == entry["size_bytes"] and digest(payload) == entry["sha256"],
            f"Published bytes changed: {entry['path']}",
        )
    index = json.loads((RAW / "artifact_index.json").read_text())
    for path in RAW.iterdir():
        original_name = path.name.removesuffix(".gz")
        if original_name in index["files"]:
            payload = (
                gzip.decompress(path.read_bytes())
                if path.suffix == ".gz"
                else path.read_bytes()
            )
            expected = index["files"][original_name]
            require(
                len(payload) == expected["bytes"]
                and digest(payload) == expected["sha256"],
                f"Original bytes changed: {original_name}",
            )
    report_bytes = (RAW / "report.json").read_bytes()
    require(digest(report_bytes) == REPORT_SHA, "Frozen report hash mismatch")
    report = json.loads(report_bytes)
    require(report["status"] == "complete", "Incomplete measurement")
    for name, expected in report["preregistration_file_sha256"].items():
        path = RAW / name
        payload = (
            path.read_bytes()
            if path.exists()
            else gzip.decompress((RAW / (name + ".gz")).read_bytes())
        )
        require(digest(payload) == expected, f"Preregistration changed: {name}")
    scientific = {key: report[key] for key in report["scientific_fingerprint_fields"]}
    require(
        fingerprint(scientific) == SAVED_JSON_SHA, "Saved-JSON fingerprint mismatch"
    )
    recovered = deepcopy(scientific)
    for row in recovered["diagnostic_summary"]:
        histogram = row["nearmax_set_size_counts"]
        require(
            all(str(int(key)) == key for key in histogram), "Noncanonical histogram key"
        )
        row["nearmax_set_size_counts"] = {
            int(key): value for key, value in histogram.items()
        }
    require(
        fingerprint(recovered)
        == IN_MEMORY_SHA
        == report["scientific_fingerprint_sha256"],
        "Original integer-key fingerprint not recovered",
    )
    require(SAVED_JSON_SHA != IN_MEMORY_SHA, "Expected disclosed serialization defect")
    records_bytes = gzip.decompress((RAW / "records.jsonl.gz").read_bytes())
    diagnostics_bytes = gzip.decompress((RAW / "diagnostics.jsonl.gz").read_bytes())
    for name, payload in (
        ("records.jsonl", records_bytes),
        ("diagnostics.jsonl", diagnostics_bytes),
    ):
        require(
            digest(payload) == report["artifact_sha256"][name],
            f"Measured rows changed: {name}",
        )
    records = sorted(
        (json.loads(line) for line in records_bytes.splitlines()),
        key=lambda r: r["row_id"],
    )
    diagnostics = [json.loads(line) for line in diagnostics_bytes.splitlines()]
    require(
        [r["row_id"] for r in records] == list(range(5120)),
        "Record row population changed",
    )
    require(
        [r["fixture_index"] for r in diagnostics] == list(range(1280)),
        "Diagnostic row population changed",
    )
    require(
        len({(r["fixture_key"], r["length"], r["rule"]) for r in records}) == 5120,
        "Duplicate fixture assignments",
    )
    cache = json.loads(gzip.decompress((RAW / "cached_reference.json.gz").read_bytes()))
    baseline = [
        r
        for r in records
        if r["length"] == 256 and r["rule"] == report["manifest"]["rules_in_order"][0]
    ]
    projected = [
        {key: r[key] for key in old}
        for r, old in zip(baseline, cache["records"], strict=True)
    ]
    require(
        bits(projected) == bits(cache["records"]),
        "Cached short-window baseline changed",
    )
    groups = []
    for group in report["summary"]["by_length_rule"]:
        subset = [
            r
            for r in records
            if r["length"] == group["length"] and r["rule"] == group["rule"]
        ]
        require(len(subset) == group["records"] == 1280, "Group denominator changed")
        finite = sum(r["finite_features"] for r in subset)
        failures = dict(Counter(r["failure_reason"] or "none" for r in subset))
        aliases = sum(
            r["finite_features"] and r["selected_pair_alias"] == "half_stride"
            for r in subset
        )
        require(
            finite == group["finite"]
            and failures == group["failures"]
            and aliases == group["finite_half_alias"],
            "Group outcome mismatch",
        )
        require(
            all(
                not r["quality_applied"]
                and not r["quality_trained"]
                and r["quality_cutoff"] is None
                for r in subset
            ),
            "Unexpected quality gate",
        )
        groups.append(
            {
                "length": group["length"],
                "rule": group["rule"],
                "records": len(subset),
                "finite": finite,
                "failures": failures,
                "finite_half_alias": aliases,
            }
        )
    print(
        json.dumps(
            {
                "status": "pass_with_preserved_serialization_erratum",
                "records": len(records),
                "diagnostics": len(diagnostics),
                "cached_baseline_exact": True,
                "saved_json_fingerprint": SAVED_JSON_SHA,
                "original_in_memory_fingerprint": IN_MEMORY_SHA,
                "full_numeric_array_replay": False,
                "groups": groups,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
