"""Compare old/new WISDM CSV windowing on identical homogeneous sensor rows.

Real CSV output counts change when unsafe boundary windows are removed. This
controlled fixture isolates speed while requiring bit-identical output tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.adapter import load_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref", required=True, help="original local git commit/ref"
    )
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/preprocessing_benchmark.json")
    )
    args = parser.parse_args()
    if args.rows < 128 or args.repeats < 1:
        parser.error("rows must be >=128 and repeats >=1")
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:data/adapter.py"], text=True
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "kind": "synthetic_homogeneous_csv_exact_output_comparison",
        "rows": args.rows,
        "window_size": 128,
        "stride": 64,
        "baseline_ref": args.baseline_ref,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": str(torch.__version__),
    }
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        root = Path(temporary)
        baseline_module = root / "_original_adapter.py"
        baseline_module.write_text(source)
        spec = importlib.util.spec_from_file_location(
            "_benchmark_original_adapter", baseline_module
        )
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        (root / "wisdm").mkdir()
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            rng.normal(size=(args.rows, 3)).astype(np.float32),
            columns=["acc_x", "acc_y", "acc_z"],
        )
        frame["timestamp"] = np.arange(args.rows) / 20
        frame["Activity"] = 0
        frame.to_csv(root / "wisdm/test.csv", index=False)
        reference = None
        for name, loader in (
            ("baseline", original.load_windows),
            ("candidate", load_windows),
        ):
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                data = loader("wisdm", "test", 128, 64, str(root))
                timings.append(time.perf_counter() - start)
            if reference is None:
                reference = data
            else:
                if not torch.equal(reference.X, data.X) or not torch.equal(
                    reference.y_activity, data.y_activity
                ):
                    raise AssertionError(
                        "baseline/candidate outputs differ on the homogeneous fixture"
                    )
            results[name] = {
                "seconds": timings,
                "median_seconds": statistics.median(timings),
                "windows": len(data.X),
            }
    results["outputs_identical"] = True
    results["speedup"] = (
        results["baseline"]["median_seconds"] / results["candidate"]["median_seconds"]
    )
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
