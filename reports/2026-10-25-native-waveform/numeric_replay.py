"""Relocated saved-only N446 replay; registered gate failure remains STOP."""

import argparse
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument(
        "--full", action="store_true", help="Both modes replay all correlations"
    )
    args = parser.parse_args()
    root = args.saved_root.resolve(strict=True)
    sys.path.insert(0, str(root / "source"))
    import common as c
    import evidence
    import numpy as np
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    c.require(versions == c.EXPECTED_VERSIONS, "Registered package versions required")
    saved = root / "saved"
    source_hashes = c.decode_json((saved / "sources.json").read_bytes())["files"]
    for name, expected in source_hashes.items():
        c.require(
            c.sha(root / "source" / name) == expected, "Frozen source changed: " + name
        )
    report = c.decode_json((saved / "report.json").read_bytes())
    for name, expected in report["bindings"]["files"].items():
        path = root / name if name == "evidence.npz" else saved / name
        c.require(c.sha(path) == expected, "Saved binding changed: " + name)
    for name, expected in report["bindings"]["arrays"].items():
        c.require(
            c.sha(root / "arrays" / name) == expected, "Saved array changed: " + name
        )
    tests = c.decode_json((saved / "selftests.json").read_bytes())
    c.require(
        tests["result"]["status"] == "pass"
        and tests["source_files_sha256"] == source_hashes,
        "Fixed checks failed or stale",
    )
    rows = [
        c.decode_json(line)
        for line in (saved / "records.jsonl").read_text().splitlines()
    ]
    keys = c.decode_json((saved / "keys.json").read_bytes())
    arrays = {
        p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
        for p in sorted((root / "arrays").glob("*.npy"))
    }
    audit, gates = evidence.replay(rows, arrays, keys, True)
    c.require(gates is not None, "Full gates required")
    c.require(
        c.typed(gates) == c.typed(c.decode_json((saved / "gates.json").read_bytes())),
        "Gate replay mismatch",
    )
    c.require(
        c.typed(audit)
        == c.typed(c.decode_json((saved / "saved_replay.json").read_bytes())),
        "Saved audit mismatch",
    )
    expected_status = (
        "COMPLETE_PASS" if gates["all_pass"] else "COMPLETE_GATE_FAIL_STOP"
    )
    c.require(
        report["status"] == expected_status
        and report["eligibility"] == gates["next_step"]
        and report["DEV_execution_authorized"] is False,
        "Interpretation changed",
    )
    c.require(
        report["gate_fingerprint"] == c.fingerprint(gates), "Gate fingerprint changed"
    )
    result = {
        "scope": "relocated_saved_numeric_replay",
        "audit": audit,
        "all_gates_pass": gates["all_pass"],
        "next_step": gates["next_step"],
        "DEV_execution_authorized": False,
        "original_runtime_closure_verified": False,
        "original_preparation_history_restored": False,
        "no_generator_filter_or_fit": True,
        "versions": versions,
        "timing": c.elapsed(),
    }
    with (root / "numeric-verification.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
