#!/usr/bin/env python3
"""Extract the WISDM and UCI HAR datasets from CI-2026.zip into datasets/.

Reads (never modifies) the pre-existing CI-2026.zip and produces:

    datasets/wisdm/train.csv
    datasets/wisdm/test.csv
    datasets/uci_har/UCI HAR Dataset/   (README.txt, features.txt,
        activity_labels.txt, train/, test/, each split's Inertial Signals/)

The HAR archive is nested: human+activity+recognition+using+smartphones.zip
contains UCI HAR Dataset.zip, which is extracted one further level.

Idempotent: existing target files are left untouched and re-extraction is
skipped. Safe to run multiple times.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(PROJECT_ROOT, "CI-2026.zip")
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

WISDM_MEMBER = "CI-2026/Dataset/WISDM_Dataset.zip"
HAR_OUTER_MEMBER = "CI-2026/Dataset/human+activity+recognition+using+smartphones.zip"

HAR_TOP_DIR = "UCI HAR Dataset"


def _extract_zipfile_to(zip_bytes: bytes, dest: str, skip_prefixes=()) -> list[str]:
    """Extract a zip archive given as bytes into dest, skipping junk paths."""
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith(skip_prefixes) or name.endswith("/"):
                continue
            if os.path.basename(name).startswith("._") or name.endswith(".DS_Store"):
                continue
            target = os.path.join(dest, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(target)
    return written


def setup_wisdm() -> bool:
    """Extract WISDM_Dataset.zip -> datasets/wisdm/{train,test}.csv."""
    dest = os.path.join(DATASETS_ROOT, "wisdm")
    targets = [os.path.join(dest, "train.csv"), os.path.join(dest, "test.csv")]
    if all(os.path.exists(p) for p in targets):
        print("[wisdm] already present, skipping")
        return False
    with zipfile.ZipFile(ARCHIVE) as outer:
        inner_bytes = outer.read(WISDM_MEMBER)
    os.makedirs(dest, exist_ok=True)
    written = _extract_zipfile_to(inner_bytes, dest)
    print(f"[wisdm] extracted {len(written)} files -> {dest}")
    return True


def setup_har() -> bool:
    """Extract the nested HAR archive -> datasets/uci_har/UCI HAR Dataset/."""
    dest = os.path.join(DATASETS_ROOT, "uci_har")
    marker = os.path.join(dest, HAR_TOP_DIR, "README.txt")
    if os.path.exists(marker):
        print("[har] already present, skipping")
        return False
    with zipfile.ZipFile(ARCHIVE) as outer:
        outer_bytes = outer.read(HAR_OUTER_MEMBER)
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as mid:
        inner_bytes = mid.read(f"{HAR_TOP_DIR}.zip")
    written = _extract_zipfile_to(
        inner_bytes, dest, skip_prefixes=("__MACOSX/",)
    )
    print(f"[har] extracted {len(written)} files -> {dest}")
    return True


def _count_rows(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def print_layout() -> None:
    print("\nFinal layout:")
    for dirpath, dirnames, filenames in os.walk(DATASETS_ROOT):
        dirnames.sort()
        rel = os.path.relpath(dirpath, PROJECT_ROOT)
        print(f"{rel}/")
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            size = os.path.getsize(full)
            rows = _count_rows(full)
            print(f"  {fn}  ({rows} rows, {size} bytes)")


def main() -> int:
    if not os.path.exists(ARCHIVE):
        print(f"ERROR: archive not found: {ARCHIVE}", file=sys.stderr)
        return 1
    setup_wisdm()
    setup_har()
    print_layout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
