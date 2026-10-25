"""Recompute corrected synthetic ROC examples; preserve all frozen r1 files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.gated_roc import gated_roc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    raw = gzip.decompress((bundle / "raw/report.json.gz").read_bytes())
    frozen = json.loads(raw)
    examples = {}
    for name, example in frozen["roc_examples"].items():
        claims = example["claims"]
        if isinstance(claims, dict):
            g, i = claims["genuine"], claims["impostor"]
            labels = [1] * len(g["scores"]) + [0] * len(i["scores"])
            scores = g["scores"] + i["scores"]
            valid = g["validity_mask"] + i["validity_mask"]
        else:
            labels = [int(c["genuine"]) for c in claims]
            scores = [c["score"] for c in claims]
            valid = [c["valid"] for c in claims]
        examples[name] = {
            "labels": labels,
            "scores": scores,
            "valid": valid,
            "result": gated_roc(labels, scores, valid),
        }
    scores = [0.9, 0.6, None, 0.8, None, None]
    labels = [1, 1, 1, 0, 0, 0]
    valid = [s is not None for s in scores]
    examples["terminal_equality_audit_regression"] = {
        "labels": labels,
        "scores": scores,
        "valid": valid,
        "result": gated_roc(labels, scores, valid),
    }
    result = {
        "schema_version": 1,
        "scope": "Synthetic count arithmetic only; no biometric performance measurement",
        "frozen_raw_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in (
                "engine/gated_roc.py",
                "tests/test_gated_roc.py",
                "docs/gated_roc.md",
            )
        },
        "examples": examples,
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"examples": len(examples), "output": str(args.out)}))


if __name__ == "__main__":
    main()
