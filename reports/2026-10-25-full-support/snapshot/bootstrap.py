"""Preparation-only copying of public metadata and pure source; never import a study."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PROJECT = REPO / "gait-biometrics"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def main():
    records = []
    copies = (
        "models/gait_canonical.py",
        "models/gait_features.py",
        "models/gait_invariants.py",
        "models/gait_metric.py",
        "engine/template_scoring.py",
        "engine/secure_enrollment.py",
        "engine/robustness_selection.py",
    )
    for package in ("models", "engine"):
        put(
            ROOT / "vendor" / package / "__init__.py",
            b'"""Owned pure-source snapshot, without study entrypoints."""\n',
        )
    for name in copies:
        raw = (PROJECT / name).read_bytes()
        put(ROOT / "vendor" / name, raw)
        records.append(
            {
                "origin": str(PROJECT / name),
                "sha256": sha(raw),
                "destination": "vendor/" + name,
                "method": "byte-identical copy",
            }
        )
    # Small pure subsets are selected statically; no source is evaluated here.
    selections = (
        (
            "models/session_robust.py",
            "vendor/models/lag_spectrum.py",
            ("_real_array", "extract_lagged_covariance_features"),
            "from numbers import Integral, Real\nimport numpy as np\n_FLOOR = 1e-8\n",
        ),
        (
            "engine/calibration_selection.py",
            "vendor/engine/calibration_selection.py",
            (
                "_mapping",
                "_required",
                "_integer",
                "_counts",
                "_subject",
                "_names",
                "_Row",
                "_validate_metrics",
            ),
            "from collections.abc import Mapping\nfrom dataclasses import dataclass\nfrom fractions import Fraction\nfrom numbers import Integral\n",
        ),
        (
            "engine/tail_calibration.py",
            "vendor/engine/empirical.py",
            (
                "_probability",
                "_finite_scores",
                "_next_threshold",
                "empirical_threshold",
            ),
            "from numbers import Real\nimport numpy as np\n",
        ),
        (
            "scripts/mentor_backend_study.py",
            "vendor/engine/historical_metrics.py",
            ("roc", "eer", "fixed_metrics", "report"),
            "import numpy as np\nfrom common import fingerprint as object_hash\n",
        ),
        (
            "data/santos.py",
            "vendor/santos_schema.py",
            (
                "_safe_member",
                "_selected_recordings",
                "_member_record",
                "_members_sha256",
                "_parse_recording",
            ),
            "import csv, hashlib, io, json, math, re, stat\nimport numpy as np\n",
        ),
    )
    for origin, destination, names, header in selections:
        raw = (PROJECT / origin).read_bytes()
        text = raw.decode()
        tree = ast.parse(text)
        if origin == "data/santos.py":
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    header += ast.get_source_segment(text, node) + "\n"
        nodes = []
        for name in names:
            node = next(
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
            )
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            nodes.append("\n".join(text.splitlines()[start - 1 : node.end_lineno]))
        payload = (
            '"""Statically extracted pure source; provenance in source-origins.json."""\n'
            + header
            + "\n\n"
            + "\n\n".join(nodes)
            + "\n"
        ).encode()
        put(ROOT / destination, payload)
        records.append(
            {
                "origin": str(PROJECT / origin),
                "sha256": sha(raw),
                "destination": destination,
                "method": "AST source selection",
                "names": list(names),
            }
        )
    public = PROJECT / "reports/2026-10-25-support/evidence"
    inputs = (
        (
            public / "result.json.gz",
            "support-result.input.json.gz",
            "80be0bc4e54ff1244a2ff8cf909dccf5ca7c3c2ee5189f58f64e9991e32aa25b",
        ),
        (
            public / "development_metadata.input.json.gz",
            "development-metadata.input.json.gz",
            None,
        ),
        (
            REPO / "artifacts/canonical-orientation-v1/registration.json",
            "historical-registration.input.json",
            None,
        ),
        (
            PROJECT / "datasets/santos/source_provenance.json",
            "official-authenticity.input.json",
            None,
        ),
        (
            REPO / "artifacts/complete-support-root-metadata-analysis-20261025.json",
            "secondary-analysis.input.json",
            None,
        ),
    )
    for path, destination, expected in inputs:
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
        json.loads(decoded)
        if expected and sha(decoded) != expected:
            raise RuntimeError("Completed public support metadata digest mismatch")
        put(ROOT / "inputs" / destination, raw)
        records.append(
            {
                "origin": str(path),
                "sha256": sha(raw),
                "decoded_sha256": sha(decoded),
                "destination": "inputs/" + destination,
                "method": "public metadata, no signal/feature/model/score payload",
            }
        )
    # Read-only documentation/source authorities are also bound, without importing them.
    for name in (
        "docs/robustness_selection.md",
        "docs/gait_canonical.md",
        "docs/santos_dataset.md",
        "data/imu_perturbations.py",
        "scripts/canonical_orientation_study.py",
    ):
        raw = (PROJECT / name).read_bytes()
        records.append(
            {
                "origin": str(PROJECT / name),
                "sha256": sha(raw),
                "destination": None,
                "method": "source/doc authority only",
            }
        )
    guard_origin = REPO / "artifacts/joint-repeat-feasibility-r1/study.py"
    raw = guard_origin.read_bytes()
    records.append(
        {
            "origin": str(guard_origin),
            "sha256": sha(raw),
            "destination": "guard.py",
            "method": "R3 PID/startticks handoff design adapted to 1800CPU/3600wall and explicit output root; never imported/executed",
        }
    )
    put(
        ROOT / "source-origins.json",
        (json.dumps(records, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "copied_pure_modules": len(copies) + len(selections),
                "public_inputs": len(inputs),
                "payload_reads": 0,
                "earlier_entrypoint_imports": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
