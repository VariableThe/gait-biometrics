"""Download and verify the version-pinned Santos cross-day archive (1.52 GB).

No extraction or signal parsing occurs. Reusing a verified archive is offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.setup_datasets import _open_no_follow, _safe_member_target
from scripts.setup_public_datasets import download

URL = "https://ndownloader.figshare.com/files/35882624"
SIZE = 1_523_321_221
MD5 = "35adf0860202f2aefa37560dc0748ebd"
SHA256 = "ed944fa88a98ce1058327ff4b3c8f919fbf58626fc7e3c5256360ee080a3a304"


def prepare_santos(data_root: Path, *, verify_only: bool = False) -> dict:
    root = Path(data_root)
    target = Path(_safe_member_target(str(root), "santos/santos-v7.zip"))
    _safe_member_target(str(root), "santos/santos-v7.zip.part")
    sidecar = _safe_member_target(str(root), "santos/source_provenance.json")
    if verify_only and not target.is_file():
        raise FileNotFoundError(
            "verified Santos archive is missing; run without --verify-only"
        )
    download(URL, target, SHA256)
    if target.stat().st_size != SIZE:
        raise ValueError("Santos v7 archive byte count differs from official manifest")
    with target.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != MD5:
        raise ValueError("Santos v7 archive MD5 differs from official manifest")
    result = {
        "dataset": "Santos multi-sensor gait",
        "version": 7,
        "doi": "10.6084/m9.figshare.14727231.v7",
        "url": URL,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "authors": "Geise Santos, Marcelo M. Wanderley, Tiago Tavares, Anderson Rocha",
        "archive": str(target),
        "size_bytes": SIZE,
        "md5": md5,
        "sha256": SHA256,
        "sha256_basis": "locally measured official v7 bytes, also matched publisher MD5 and size",
        "extracted": False,
        "signal_values_parsed": False,
    }
    with _open_no_follow(sidecar) as stream:
        stream.write((json.dumps(result, indent=2) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_santos(args.data_root, verify_only=args.verify_only), indent=2
        )
    )


if __name__ == "__main__":
    main()
