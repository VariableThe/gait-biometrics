"""Check the publication, restore one fresh attempt, and replay saved metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

ATTEMPT_FILES = {
    "attempt.json",
    "registration.json",
    "diagnostics.json",
    "result.json",
    "historical_registration.input.json.gz",
    "development_metadata.input.json.gz",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_member(root: Path, name: str) -> bytes:
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(p in {".", ".."} for p in relative.parts):
        raise ValueError("Invalid bundle member")
    path = root.joinpath(*relative.parts)
    if path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink():
        return path.read_bytes()
    raise ValueError("Bundle member is not a contained regular file: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    inventory = json.loads((bundle / "inventory.json").read_bytes())
    if (
        set(inventory) != {"schema_version", "files"}
        or inventory["schema_version"] != 1
    ):
        raise ValueError("Unknown publication inventory schema")
    expected = {}
    for item in inventory["files"]:
        if set(item) != {"path", "bytes", "sha256"} or item["path"] in expected:
            raise ValueError("Invalid or duplicate publication entry")
        payload = read_member(bundle, item["path"])
        if len(payload) != item["bytes"] or digest(payload) != item["sha256"]:
            raise ValueError("Publication fingerprint mismatch: " + item["path"])
        expected[item["path"]] = item
    observed = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and path != bundle / "inventory.json"
    }
    if observed != set(expected):
        raise ValueError("Publication has missing or unregistered files")
    provenance = json.loads(read_member(bundle, "provenance.json"))
    bindings = provenance["restored_attempt"]
    if set(bindings) != ATTEMPT_FILES:
        raise ValueError("Unexpected attempt assignment")
    restored = {}
    for name, binding in bindings.items():
        if set(binding) != {"published_path", "encoding", "bytes", "sha256"}:
            raise ValueError("Unknown restored binding schema")
        if binding["published_path"] not in expected:
            raise ValueError("Attempt payload is not registered")
        payload = read_member(bundle, binding["published_path"])
        if binding["encoding"] == "gzip":
            payload = gzip.decompress(payload)
        elif binding["encoding"] != "identity":
            raise ValueError("Unknown storage encoding")
        if len(payload) != binding["bytes"] or digest(payload) != binding["sha256"]:
            raise ValueError("Restored fingerprint mismatch: " + name)
        restored[name] = payload
    # The restored root is an exclusive attempt; failures remain for inspection.
    target = args.out_root.absolute()
    target.mkdir(parents=True, exist_ok=False)
    for name, payload in restored.items():
        with (target / name).open("xb") as stream:
            stream.write(payload)
    access_report = target.with_name(target.name + "-access.json")
    command = [
        sys.executable,
        str(bundle / "guarded_inventory.py"),
        "--stage",
        "verify",
        "--repo-root",
        str(bundle.parents[1]),
        "--out-root",
        str(target),
        "--access-report",
        str(access_report),
    ]
    subprocess.run(command, check=True)
    access = json.loads(access_report.read_bytes())
    if access["exit_code"] != 0 or access["denied_accesses"]:
        raise ValueError("Saved-metadata access guard failed")
    print(
        json.dumps(
            {
                "publication_files": len(expected),
                "restored_files": len(restored),
                "saved_metadata_replay": "verified",
                "raw_member_reads": 0,
                "fits": 0,
                "access_report": str(access_report),
                "scope": "same-source/runtime metadata reconstruction; no signal or model validation",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
