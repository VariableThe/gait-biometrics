"""Smoke test for the core classification stack.

Trains all three classification configs for 2 epochs on a 2000-window
stratified subset into runs/smoke_classification/<model>, asserts exit code 0
and that best.pt + metrics.json exist for each, and prints a summary table.

Run from the repository root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = ["cnn_baseline", "cnn_lstm", "multitask"]


def main() -> None:
    rows = []
    failed = False

    for name in CONFIGS:
        out_dir = os.path.join(ROOT, "runs", "smoke_classification", name)
        cmd = [
            sys.executable, "-m", "engine.train",
            "--config", os.path.join(ROOT, "configs", f"{name}.yaml"),
            "--epochs", "2",
            "--subset", "2000",
            "--out-dir", out_dir,
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)

        best_pt = os.path.join(out_dir, "best.pt")
        metrics_path = os.path.join(out_dir, "metrics.json")
        ok = proc.returncode == 0 and os.path.isfile(best_pt) and os.path.isfile(metrics_path)
        if not ok:
            failed = True

        row = {"model": name, "ok": ok, "best_epoch": "-", "final_train_loss": "-",
               "final_train_acc": "-", "n_params": "-"}
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            row.update(
                best_epoch=m["best_epoch"],
                final_train_loss=f"{m['final_train_loss']:.4f}",
                final_train_acc=f"{m['final_train_acc']:.4f}",
                n_params=m["n_params"],
            )
        rows.append(row)

    print("\n=== smoke test summary ===")
    print(f"{'model':<15} {'ok':<5} {'best_epoch':<11} {'final_loss':<12} {'final_acc':<11} {'n_params':<10}")
    for r in rows:
        print(f"{r['model']:<15} {str(r['ok']):<5} {str(r['best_epoch']):<11} "
              f"{str(r['final_train_loss']):<12} {str(r['final_train_acc']):<11} {str(r['n_params']):<10}")

    if failed:
        raise SystemExit("SMOKE TEST FAILED: at least one config did not produce best.pt + metrics.json")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
