"""Bounded DEV-only raw-window embedding timings for a locked multisensor study.

Imports and data/model loading are deferred until after argument parsing, so
--help requires only the standard library. This runner never fits a model.
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

BATCH_SIZES = (1, 32)
WARMUP_BATCHES = 5
THREADS = 2


def repeat_count(value):
    number = int(value)
    if not 5 <= number <= 1000:
        raise argparse.ArgumentTypeError("repeats must be in [5, 1000]")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive-path", type=Path, required=True)
    result.add_argument("--study-root", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True, help="new JSON result file")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--repeats", type=repeat_count, default=30)
    return result


def verified_lock(study, args):
    """Check all frozen bytes and selected references before loading DEV values."""
    root, registration = study.check_registration(
        SimpleNamespace(out_root=args.study_root, archive_path=args.archive_path)
    )
    previous = study.previous
    value = previous._read(root / "lock.json")
    self_hash = value.pop("self_hash")
    previous._same(previous._object_hash(value), self_hash, "study lock")
    registration_hash = previous.digest(root / "registration.json")
    previous._same(
        value["registration_sha256"], registration_hash, "locked registration"
    )
    previous._same(
        value["artifact_hashes"].get("registration.json"),
        registration_hash,
        "registration artifact",
    )
    for path, digest in value["artifact_hashes"].items():
        previous._same(previous.digest(root / path), digest, "locked artifact")
    previous._same(set(value["selected"]), {"baseline", "winner"}, "selected roles")
    for role, fitted in value["selected"].items():
        row = fitted["row"]
        expected = study.canonical_candidate(row["name"])
        previous._same(
            {key: row[key] for key in expected}, expected, "locked method identity"
        )
        previous._same(
            row["name"],
            study.plan()["baseline"] if role == "baseline" else value["winner"],
            "locked method role",
        )
        kind = "descriptor" if row["family"] == "descriptors" else "neural"
        previous._same(fitted["kind"], kind, "fitted method kind")
        paths = fitted["paths"]
        if len(paths) != (1 if kind == "descriptor" else 2) or len(set(paths)) != len(
            paths
        ):
            raise ValueError("expected one fitted metric or two distinct seed models")
        if any(path not in value["artifact_hashes"] for path in paths):
            raise ValueError("selected fitted artifact is not covered by lock hashes")
    return root, registration, value, self_hash


def resident_rows(study, dev):
    """Select 32 real rows, round-robin over all 15 original DEV identities."""
    import numpy as np

    ids = study.previous.DEV_IDS
    if len(ids) != 15 or set(dev["subjects"].tolist()) != set(ids):
        raise ValueError("benchmark requires exactly the original 15 DEV subjects")
    per_subject = [np.flatnonzero(dev["subjects"] == subject) for subject in ids]
    indices = [
        int(rows[offset])
        for offset in range(3)
        for rows in per_subject
        if len(rows) > offset
    ][: max(BATCH_SIZES)]
    if len(indices) != max(BATCH_SIZES):
        raise ValueError("not enough authentic DEV rows for batch 32")
    x = np.ascontiguousarray(dev["X"][indices], dtype=np.float32)
    if x.shape != (32, 6, 256) or not np.isfinite(x).all():
        raise ValueError("expected finite authentic six-axis DEV windows")
    return x, indices


def validate_embeddings(output, batch_size):
    import numpy as np

    if (
        output.ndim != 2
        or output.shape[0] != batch_size
        or output.shape[1] == 0
        or not np.isfinite(output).all()
        or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("inference returned invalid unit embeddings")


def descriptor_runner(study, fitted, root):
    """Restore one fixed metric; each prepared call includes fresh descriptors."""
    import numpy as np

    from models.gait_metric import WithinClassMetric

    with np.load(root / fitted["paths"][0], allow_pickle=False) as saved:
        metric = WithinClassMetric.from_state_dict(dict(saved))
    if set(metric.training_subject_ids_) != set(study.previous.DEV_IDS):
        raise ValueError("final metric was not fitted exclusively on all DEV users")
    spec = fitted["row"]["descriptor"]

    def prepare(batch):
        def run():
            return metric.transform(study.descriptor_features({"X": batch}, spec))

        return run, lambda: None, lambda z: validate_embeddings(z, len(batch))

    return prepare, {
        "actual_device": "cpu",
        "backend": "numpy_cpu",
        "members": 1,
        "precision": "native NumPy descriptors; float64 WCCN; float32 output",
        "input_residence": "host RAM, contiguous float32 raw six-axis windows",
        "output_residence": "host RAM",
        "included": [
            "channel selection",
            "descriptor extraction",
            "fixed WCCN transform",
            "L2 normalization",
        ],
        "host_to_device_transfer_timed": False,
        "device_to_host_transfer_timed": False,
        "transfer_policy": "CPU only, including when requested_device is cuda",
    }


def neural_runner(study, fitted, root, device, dev):
    """Restore both fixed encoders once; prepare raw tensors outside timing."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("neural CUDA benchmark requested but CUDA is unavailable")
    torch.set_num_threads(THREADS)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family = fitted["row"]["family"]
    channels, _ = study.family_config(family)
    count = 3 if channels == "accel" else 6
    models = [
        study.load_model(
            root / path,
            family,
            study.previous.DEV_IDS,
            seed,
            device,
            training_data=dev,
        )[0]
        for seed, path in zip((42, 43), fitted["paths"], strict=True)
    ]
    target = torch.device(device)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize(target)

    def prepare(batch):
        # Full raw windows are resident; selection and all encoder preprocessing
        # remain in the timed call. Neither H2D nor D2H is timed.
        resident = torch.as_tensor(batch, dtype=torch.float32, device=target)
        synchronize()

        @torch.inference_mode()
        def run():
            x = resident[:, :count]
            return torch.cat([model(x) for model in models], dim=1) / math.sqrt(2)

        def validate(z):
            validate_embeddings(z.detach().float().cpu().numpy(), len(batch))

        return run, synchronize, validate

    return prepare, {
        "actual_device": device,
        "backend": "torch",
        "members": len(models),
        "precision": "float32 inference, no autocast, TF32 disabled",
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(target)
        if device == "cuda"
        else platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "input_residence": f"{device} resident float32 raw six-axis tensor",
        "output_residence": device,
        "included": [
            "channel selection",
            "both seed encoder forwards with fixed normalization",
            "concatenation",
            "division by sqrt(2)",
        ],
        "host_to_device_transfer_timed": False,
        "device_to_host_transfer_timed": False,
        "transfer_policy": "input conversion/H2D before timing; validation/D2H after timing",
        "cuda_synchronization": "before and after every warm timed batch"
        if device == "cuda"
        else "not applicable",
    }


def summarize(samples, batch_size):
    """Samples are seconds per complete batch, with no per-window conversion."""
    import numpy as np

    if not samples or any(not math.isfinite(s) or s <= 0 for s in samples):
        raise ValueError("timing samples must be finite positive batch seconds")
    return {
        "batch_latency_ms_median": statistics.median(samples) * 1000,
        "batch_latency_ms_p95": float(np.percentile(samples, 95)) * 1000,
        "batch_latency_ms_samples": [sample * 1000 for sample in samples],
        "windows_per_second": batch_size * len(samples) / sum(samples),
        "timed_windows": batch_size * len(samples),
        "timed_seconds": sum(samples),
    }


def benchmark_batch(prepare, batch, repeats):
    run, synchronize, validate = prepare(batch)
    for _ in range(WARMUP_BATCHES):
        output = run()
    synchronize()
    validate(output)
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        output = run()
        synchronize()
        samples.append(time.perf_counter() - started)
    validate(output)
    return {
        "batch_size": len(batch),
        "input_shape": list(batch.shape),
        "embedding_shape": list(output.shape),
        **summarize(samples, len(batch)),
    }


def benchmark(args):
    # Do not import the study (or its scientific dependencies) for --help.
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    from scripts import multisensor_study as study

    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    root, registration, lock, lock_hash = verified_lock(study, args)
    dev = study.development(args, registration)
    inputs, indices = resident_rows(study, dev)
    results, runners = {}, {}
    with threadpool_limits(limits=THREADS):
        for role in ("baseline", "winner"):
            fitted = lock["selected"][role]
            key = (fitted["row"]["name"], tuple(fitted["paths"]))
            if key not in runners:
                runners[key] = (
                    descriptor_runner(study, fitted, root)
                    if fitted["kind"] == "descriptor"
                    else neural_runner(study, fitted, root, args.device, dev)
                )
            prepare, metadata = runners[key]
            batches = [
                benchmark_batch(prepare, inputs[:size], args.repeats)
                for size in BATCH_SIZES
            ]
            results[role] = {
                "name": fitted["row"]["name"],
                "template": fitted["row"]["template"],
                "template_scoring_timed": False,
                "requested_device": args.device,
                "artifact_hashes": {
                    p: lock["artifact_hashes"][p] for p in fitted["paths"]
                },
                **metadata,
                "batches": batches,
            }
        pools = threadpool_info()
    return {
        "kind": "locked_multisensor_dev_resident_embedding_benchmark",
        "created_utc": datetime.now(UTC).isoformat(),
        "study_root": str(root),
        "archive_path": str(args.archive_path),
        "lock_self_hash": lock_hash,
        "registration_sha256": lock["registration_sha256"],
        "benchmark_script_sha256": study.previous.digest(Path(__file__)),
        "registered_source_hashes": registration["sources"],
        "prior_test_inspection": True,
        "development_subject_ids": list(study.previous.DEV_IDS),
        "development_fingerprint": registration["development_fingerprint"],
        "available_dev_windows": len(dev["X"]),
        "resident_row_indices": indices,
        "resident_row_subjects": dev["subjects"][indices].tolist(),
        "selection": "first available rows round-robin in registered DEV ID order; no replacement; batch 1 is prefix of batch 32",
        "requested_device": args.device,
        "repeats": args.repeats,
        "warmup_batches_per_case": WARMUP_BATCHES,
        "thread_limit": THREADS,
        "threadpools": pools,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timer": "time.perf_counter; synchronized batch wall time on CUDA",
        "percentile": "NumPy percentile 95, default linear interpolation over batch samples",
        "throughput": "batch_size * repeats / sum(batch_seconds); windows/second",
        "excluded": [
            "archive auditing/loading",
            "DEV fingerprinting",
            "model loading",
            "fitting/training",
            "input selection/conversion",
            "H2D and D2H",
            "output validation",
            "trial/template scoring",
            "calibration",
            "CALIB/TEST numerical loading",
        ],
        "results": results,
    }


def main():
    args = parser().parse_args()
    result = benchmark(args)
    from scripts import multisensor_study as study

    args.out.parent.mkdir(parents=True, exist_ok=True)
    study.write(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
