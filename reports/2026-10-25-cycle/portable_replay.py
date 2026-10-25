"""Replay the preserved synthetic prototype in a new directory, never in place.

Only its hardcoded output-root assignment changes. This is a new source hash,
not a replay under the original hash. Numerical equality is checked against
the preserved report; the runtime and timestamps are deliberately separate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_ASSIGNMENT = (
    'ROOT = Path("/home/sherlock/Code/github/VariableThe/artifacts/'
    'cycle-feasibility-r1")'
)
ORIGINAL_SHA256 = "0cc33e195edf4d1a13591dd8b2d4abe9380bd7526509c9e714968b60abc8b85d"
ORIGINAL_REPORT_SHA256 = (
    "0ec0068ff1a2d08b8aa3b382d5f175dca37bcfced8a828d2631070c352a8b5a9"
)
COMPARISON_KEYS = (
    "status",
    "manifest_sha256",
    "manifest",
    "assumptions",
    "sources",
    "records",
    "analytical_evidence",
    "reference_checks",
    "controls",
    "roc_examples",
    "summary",
    "recommendation",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    # Exclusive creation also rejects existing empty directories and symlinks.
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    frozen_bytes = gzip.decompress((bundle / "raw/report.json.gz").read_bytes())
    if hashlib.sha256(frozen_bytes).hexdigest() != ORIGINAL_REPORT_SHA256:
        raise SystemExit("Preserved report SHA256 mismatch; no replay started")
    raw = (bundle / "raw/prototype.py").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise SystemExit("Preserved prototype SHA256 mismatch; no replay started")
    source = raw.decode("utf-8")
    if source.count(ROOT_ASSIGNMENT) != 1:
        raise SystemExit("Expected exactly one historical output-root assignment")
    relocated = source.replace(
        ROOT_ASSIGNMENT, "ROOT = Path(__file__).resolve().parent"
    )
    payload = relocated.encode("utf-8")
    (out / "prototype.py").write_bytes(payload)
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    provenance = {
        "original_source_sha256": ORIGINAL_SHA256,
        "relocated_source_sha256": hashlib.sha256(payload).hexdigest(),
        "only_change": "ROOT = Path(__file__).resolve().parent",
        "python_executable": sys.executable,
        "output_root": str(out),
        "comparison_keys": list(COMPARISON_KEYS),
        "excluded_provenance_keys": [
            "source_sha256",
            "freeze_utc",
            "measurement_start_utc",
            "runtime",
        ],
        "scope": "Same-runtime synthetic reproducibility check, not gait performance",
    }
    (out / "relocation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for mode in ("--freeze-only", "--run"):
        with (out / ("freeze.log" if mode == "--freeze-only" else "run.log")).open(
            "w"
        ) as log:
            subprocess.run(
                [sys.executable, "-B", str(out / "prototype.py"), mode],
                check=True,
                env=environment,
                cwd=out,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    frozen = json.loads(frozen_bytes)
    replay = json.loads((out / "report.json").read_text())
    differences = [key for key in COMPARISON_KEYS if replay[key] != frozen[key]]
    comparison = {
        "schema_version": 1,
        "all_scientific_fields_exact": not differences,
        "different_top_level_fields": differences,
        "comparison_keys": list(COMPARISON_KEYS),
        "original_source_sha256": ORIGINAL_SHA256,
        "relocated_source_sha256": provenance["relocated_source_sha256"],
        "original_runtime": frozen["runtime"],
        "replay_runtime": replay["runtime"],
        "limitation": "No cross-platform bitwise promise; near ties can depend on the numeric runtime",
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_scientific_fields_exact": not differences,
                "different_top_level_fields": differences,
                "output_root": str(out),
            }
        )
    )
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
