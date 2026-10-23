"""Run full training and held-out evaluation immediately from checked configs.

Example: uv run python -m scripts.run_experiments --prepare --device auto
Runs sequentially by default to bound RAM/VRAM on a 4 GB laptop GPU. Independent
invocations with different --out-root paths can run on different machines.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            Path("configs/verification_har.yaml"),
            Path("configs/wisdm_verification.yaml"),
        ],
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/full_experiments"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--epochs", type=int, help="override each config's epoch count")
    parser.add_argument(
        "--prepare", action="store_true", help="fetch original public WISDM/HAR first"
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    prepared = []
    stems = set()
    for config_path in args.configs:
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            parser.error(f"{config_path}: expected a configuration mapping")
        if config_path.stem in stems:
            parser.error(
                "config filenames must have distinct stems for separate output directories"
            )
        stems.add(config_path.stem)
        if "loss_type" in config and int(config.get("val_subjects", 0)) < 2:
            parser.error(
                f"{config_path}: calibrated verification requires val_subjects >= 2; use configs/verification_har.yaml or configs/wisdm_verification.yaml"
            )
        folder = args.out_root / config_path.stem
        if (folder / "best.pt").exists():
            parser.error(
                f"{folder}/best.pt already exists; choose a fresh --out-root to preserve results"
            )
        prepared.append((config_path, config, folder))
    args.out_root.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def run(command):
        print(" ".join(str(part) for part in command), flush=True)
        subprocess.run([str(part) for part in command], check=True, env=environment)

    if args.prepare:
        run(
            [
                sys.executable,
                "-m",
                "scripts.setup_public_datasets",
                "--data-root",
                args.data_root,
            ]
        )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    summary = {
        "started_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_commit": commit,
        "experiments": [],
    }
    for config_path, config, folder in prepared:
        folder.mkdir(parents=True, exist_ok=True)
        config.update(
            data_root=str(args.data_root.resolve()),
            device=args.device,
            out_dir=str(folder.resolve()),
            subset=None,
        )
        if args.epochs is not None:
            config["epochs"] = args.epochs
        effective = folder / "effective_config.yaml"
        effective.write_text(yaml.safe_dump(config, sort_keys=False))
        verification = "loss_type" in config
        module = "engine.train_siamese" if verification else "engine.train"
        commands = [[sys.executable, "-m", module, "--config", effective]]
        evaluate = [
            sys.executable,
            "-m",
            "engine.evaluate",
            "--config",
            effective,
            "--checkpoint",
            folder / "best.pt",
            "--task",
            "verification" if verification else "activity",
            "--out-dir",
            folder / "evaluation",
        ]
        if verification:
            evaluate.extend(["--calibrate", "--calibration-target-far", "0.01"])
        commands.append(evaluate)
        started = time.monotonic()
        for command in commands:
            run(command)
        summary["experiments"].append(
            {
                "config": str(config_path),
                "effective_config": str(effective),
                "elapsed_seconds": time.monotonic() - started,
                "commands": [[str(part) for part in command] for command in commands],
            }
        )
        (args.out_root / "experiment_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
