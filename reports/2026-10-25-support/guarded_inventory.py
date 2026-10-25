"""Run or verify the reviewed inventory behind explicit import/read/write guards."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import resource
import time
from pathlib import Path
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "verify"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--access-report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--historical-registration-json", type=Path)
    parser.add_argument("--development-metadata-json", type=Path)
    args = parser.parse_args()
    if args.stage == "verify" and (
        args.historical_registration_json or args.development_metadata_json
    ):
        parser.error("verify accepts saved snapshots only")
    repo = (
        args.repo_root or (Path(__file__).resolve().parents[1] / "gait-biometrics")
    ).resolve()
    source = repo / "scripts/santos_support_inventory.py"
    out = args.out_root.resolve()
    allowed_sources = {
        repo / name
        for name in (
            "scripts/santos_support_inventory.py",
            "tests/test_santos_support_inventory.py",
            "docs/santos_support_inventory.md",
        )
    }
    inputs = {
        "historical-registration-json": (
            args.historical_registration_json
            or (repo / "reports/2026-10-25-canonical/evidence/registration.json.gz")
        ).resolve(),
        "development-metadata-json": (
            args.development_metadata_json
            or (repo / "reports/2026-10-25-canonical/evidence/development.json.gz")
        ).resolve(),
    }
    allowed_inputs = set(inputs.values()) if args.stage == "run" else set()
    original_import = builtins.__import__
    original_open = Path.open
    accesses = []
    imports = set()
    denied = []

    def guarded_import(name, *pos, **kw):
        if name.split(".")[0] in {
            "numpy",
            "torch",
            "data",
            "models",
            "engine",
            "zipfile",
            "tarfile",
        }:
            denied.append({"import": name})
            raise ImportError("Forbidden model/dataset/archive import: " + name)
        imports.add(name)
        return original_import(name, *pos, **kw)

    def guarded_open(path, *pos, **kw):
        resolved = path.resolve()
        mode = pos[0] if pos else kw.get("mode", "r")
        writing = any(c in mode for c in "wax+")
        inside = resolved.parent == out
        allowed = (
            inside
            if writing
            else inside or resolved in allowed_sources or resolved in allowed_inputs
        )
        if not allowed:
            denied.append({"path": str(resolved), "mode": mode})
            raise PermissionError("Unregistered file access: " + str(resolved))
        accesses.append({"path": str(resolved), "mode": mode})
        return original_open(path, *pos, **kw)

    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("reviewed_support_inventory", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Source loading occurs before guarded job I/O; no historical payload is loaded.
    with mock.patch("builtins.__import__", side_effect=guarded_import):
        spec.loader.exec_module(module)
    argv = ["--stage", args.stage, "--out-root", str(out)]
    if args.stage == "run":
        for key, value in inputs.items():
            argv += ["--" + key, str(value)]
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    error = None
    code = 1
    try:
        with (
            mock.patch("builtins.__import__", side_effect=guarded_import),
            mock.patch.object(Path, "open", guarded_open),
            mock.patch(
                "builtins.open",
                side_effect=PermissionError("Unregistered builtins.open"),
            ),
        ):
            code = module.main(argv)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result = {
            "schema_version": 1,
            "stage": args.stage,
            "wrapper_sha256": wrapper_hash,
            "exit_code": code,
            "exception": error,
            "allowed_sources": sorted(str(p) for p in allowed_sources),
            "allowed_original_inputs": sorted(str(p) for p in allowed_inputs),
            "output_root": str(out),
            "file_accesses": accesses,
            "imports_seen": sorted(imports),
            "denied_accesses": denied,
            "raw_member_reads": 0,
            "process_cpu_seconds_for_job": time.process_time() - started_cpu,
            "wall_seconds_for_job": time.perf_counter() - started_wall,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "Python-level import/Path.open guard for this reviewed stdlib job; not an OS sandbox or raw-data authenticity audit",
        }
        with args.access_report.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
