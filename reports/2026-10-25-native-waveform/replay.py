"""Restore lossless published arrays and run a relocated numeric replay."""

import argparse
import gzip
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def contained(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe relative path")
    require(path.as_posix() == relative and path.parts, "Noncanonical relative path")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--full", action="store_true", help="Recompute all saved correlations"
    )
    args = parser.parse_args()
    output = args.out_root.absolute()
    require(not output.exists(), "Output directory already exists; choose a fresh path")
    inventory = json.loads((BASE / "inventory.json").read_bytes())
    expected = inventory["files"]
    actual = {
        p.relative_to(BASE).as_posix()
        for p in BASE.rglob("*")
        if p.is_file() and p != BASE / "inventory.json"
    }
    require(actual == set(expected), "Publication file inventory mismatch")
    for relative, item in expected.items():
        path = BASE / contained(relative)
        require(not path.is_symlink(), "Symlink in publication")
        require(
            path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
            "Publication digest mismatch: " + relative,
        )
    provenance = json.loads((BASE / "provenance.json").read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    # Every output remains after any failure. No retry or original study execution occurs.
    for relative, item in provenance["restored_files"].items():
        target = output / contained(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = BASE / contained(item["published_path"])
        with source.open("rb") as original:
            reader = gzip.GzipFile(fileobj=original) if item["gzip"] else original
            with target.open("xb") as handle:
                remaining = item["bytes"]
                while block := reader.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(block)
                    require(remaining >= 0, "Inflated file exceeds registered size")
                    handle.write(block)
        require(
            target.stat().st_size == item["bytes"] and sha(target) == item["sha256"],
            "Restored digest mismatch: " + relative,
        )
    compressed = output / "evidence.npz.xz"
    with compressed.open("xb") as handle:
        for name in provenance["numeric_evidence"]["parts"]:
            with (BASE / contained(name)).open("rb") as part:
                shutil.copyfileobj(part, handle, 1024 * 1024)
    numeric = provenance["numeric_evidence"]
    require(
        sha(compressed) == numeric["xz_sha256"], "Combined numeric archive mismatch"
    )
    archive = output / "evidence.npz"
    with lzma.open(compressed, "rb") as source, archive.open("xb") as target:
        remaining = numeric["npz_bytes"]
        while block := source.read(min(1024 * 1024, remaining + 1)):
            remaining -= len(block)
            require(remaining >= 0, "Numeric evidence exceeds registered size")
            target.write(block)
    require(
        archive.stat().st_size == numeric["npz_bytes"]
        and sha(archive) == numeric["npz_sha256"],
        "Restored NPZ mismatch",
    )
    arrays = output / "arrays"
    arrays.mkdir()
    report = json.loads((output / "saved/report.json").read_bytes())
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(
            len(names) == len(set(names))
            and set(names) == set(report["bindings"]["arrays"]),
            "NPZ member inventory mismatch",
        )
        for name in names:
            require(
                Path(name).name == name and name.endswith(".npy"), "Unsafe NPZ member"
            )
            target = arrays / name
            with source.open(name) as original, target.open("xb") as handle:
                shutil.copyfileobj(original, handle, 1024 * 1024)
            require(
                sha(target) == report["bindings"]["arrays"][name],
                "NPY bytes mismatch: " + name,
            )
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    command = [
        sys.executable,
        "-B",
        str(BASE / "numeric_replay.py"),
        "--saved-root",
        str(output),
    ]
    if args.full:
        command.append("--full")
    log = output / "numeric-replay.log"
    with log.open("x") as handle:
        result = subprocess.run(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(log.read_text(), end="")
    require(
        result.returncode == 0,
        "Numeric replay failed; outputs retained: " + str(output),
    )


if __name__ == "__main__":
    main()
