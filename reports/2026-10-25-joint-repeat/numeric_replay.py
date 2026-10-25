"""Relocated saved-only numeric check; not strict original runtime verification."""

import argparse
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = args.saved_root.resolve(strict=True)
    sys.path.insert(0, str(root / "source"))
    import common as c
    import numpy as np
    import replay
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    c.require(versions == c.EXPECTED_VERSIONS, "Registered package versions required")
    saved = root / "saved"
    report = c.decode_json((saved / "report.json").read_bytes())
    for name, expected in report["source_files_sha256"].items():
        c.require(
            c.sha(root / "source" / name) == expected, "Frozen source changed: " + name
        )
    for name, expected in report["binding_files_sha256"].items():
        c.require(c.sha(saved / name) == expected, "Saved binding changed: " + name)
    c.require(
        c.sha(saved / "records.jsonl") == report["records_sha256"], "Records changed"
    )
    c.require(
        c.sha(saved / "diagnostics.jsonl") == report["diagnostics_sha256"],
        "Diagnostics changed",
    )
    rows = [
        c.decode_json(line)
        for line in (saved / "records.jsonl").read_text().splitlines()
    ]
    diagnostics = [
        c.decode_json(line)
        for line in (saved / "diagnostics.jsonl").read_text().splitlines()
    ]
    inventory = c.decode_json((saved / "keys.json").read_bytes())
    arrays = {
        p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
        for p in sorted((root / "arrays").glob("*.npy"))
    }
    audit = replay.verify(
        rows,
        arrays,
        inventory,
        diagnostics,
        schema=c.decode_json((saved / "array_schema.json").read_bytes()),
        full=args.full,
    )
    summary = replay.summary(rows, arrays, inventory)
    c.require(
        c.typed(summary)
        == c.typed(c.decode_json((saved / "summary.json").read_bytes())),
        "Summary replay mismatch",
    )
    result = {
        "scope": "relocated_saved_numeric_replay",
        "original_runtime_closure_verified": False,
        "original_regression_tree_restored": False,
        "no_generator_filter_or_fit": True,
        "versions": versions,
        "audit": audit,
        "timing": c.elapsed(),
    }
    with (root / "numeric-verification.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
