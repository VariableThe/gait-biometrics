"""Fetch the original public datasets without OneDrive credentials or Git LFS.

Run with ``uv run python -m scripts.setup_public_datasets --dataset all``.
Only selected ordinary files are extracted; source archives remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from scripts.setup_datasets import _extract_zipfile_to

WISDM_URL = (
    "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
)
WISDM_SHA256 = "e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7"
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, target: Path, expected_sha256: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as src,
                temporary.open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            if expected_sha256 and sha256(temporary) != expected_sha256:
                raise ValueError(
                    f"checksum mismatch for {url}; original source may have changed"
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if expected_sha256 and sha256(target) != expected_sha256:
        raise ValueError(f"checksum mismatch for {target}; remove it and retry")
    return target


def prepare_wisdm(root: Path, archive: Path) -> dict:
    if sha256(archive) != WISDM_SHA256:
        raise ValueError("WISDM v1.1 archive checksum mismatch")
    dest = root / "wisdm"
    dest.mkdir(parents=True, exist_ok=True)
    names = ["readme.txt", "WISDM_ar_v1.1_raw.txt", "WISDM_ar_v1.1_raw_about.txt"]
    with tarfile.open(archive, "r:gz") as stream:
        for name in names:
            member = stream.getmember(f"WISDM_ar_v1.1/{name}")
            if not member.isfile():
                raise ValueError(f"expected an ordinary dataset file: {name}")
            with stream.extractfile(member) as src, (dest / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return {
        "source": "wisdm_v1",
        "url": WISDM_URL,
        "archive_sha256": sha256(archive),
        "raw_sha256": sha256(dest / "WISDM_ar_v1.1_raw.txt"),
        "identity": "original integer subject IDs; no identity reconstruction",
        "sampling_rate": 20.0,
        "channels": ["acc_x", "acc_y", "acc_z"],
    }


def prepare_har(root: Path, archive: Path) -> dict:
    with zipfile.ZipFile(archive) as outer:
        if "UCI HAR Dataset.zip" in outer.namelist():
            contents = outer.read("UCI HAR Dataset.zip")
        else:
            contents = archive.read_bytes()
    _extract_zipfile_to(contents, str(root / "uci_har"), skip_prefixes=("__MACOSX/",))
    return {
        "source": "har",
        "url": HAR_URL,
        "archive_sha256": sha256(archive),
        "representation": "official 128-sample inertial windows, official subject split",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "har", "wisdm_v1"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".dataset-cache"))
    parser.add_argument(
        "--har-archive", type=Path, help="reuse the original bundled HAR zip"
    )
    parser.add_argument(
        "--wisdm-archive", type=Path, help="reuse the checksum-verified official tar.gz"
    )
    args = parser.parse_args()
    manifests = []
    if args.dataset in ("all", "wisdm_v1"):
        archive = args.wisdm_archive or download(
            WISDM_URL, args.cache_dir / "WISDM_ar_v1.1.tar.gz", WISDM_SHA256
        )
        manifests.append(prepare_wisdm(args.data_root, archive))
    if args.dataset in ("all", "har"):
        archive = args.har_archive or download(HAR_URL, args.cache_dir / "UCI_HAR.zip")
        manifests.append(prepare_har(args.data_root, archive))
    args.data_root.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        (args.data_root / f"{manifest['source']}_provenance.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
