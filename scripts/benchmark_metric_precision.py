"""Resident synthetic TRAIN microbenchmark: FP32 versus CUDA BF16 encoder AMP.

Measures forward, trainer losses, backward, clipping, and AdamW updates. Does
not load study data or assess biometric quality. Torch imports are deferred so
--help and argument checks need no GPU or torch workload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_list(value):
    try:
        numbers = [positive_int(part) for part in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "use distinct positive integers, e.g. 32,96"
        ) from exc
    if len(set(numbers)) != len(numbers):
        raise argparse.ArgumentTypeError("values must be distinct")
    return numbers


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", required=True, type=Path, help="JSON results file")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "both"),
        default=None,
        help="default: both on CUDA; float32 on CPU; CPU rejects BF16/both",
    )
    result.add_argument("--channels", type=positive_list, default=[3, 9])
    result.add_argument("--widths", type=positive_list, default=[32, 96])
    result.add_argument("--batch-sizes", type=positive_list, default=[32])
    result.add_argument(
        "--architecture", choices=("multiscale", "temporal"), default="multiscale"
    )
    result.add_argument("--embed-dim", type=positive_int, default=128)
    result.add_argument("--window-size", type=positive_int, default=128)
    result.add_argument("--identities-per-batch", type=positive_int, default=8)
    result.add_argument("--n-classes", type=positive_int, default=8)
    result.add_argument("--steps", type=positive_int, default=20)
    result.add_argument("--warmup", type=positive_int, default=5)
    result.add_argument("--repeats", type=positive_int, default=3)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--torch-threads", type=positive_int, default=2)
    return result


def validate_args(args):
    """Resolve precision and P/K without importing torch or querying a GPU."""
    precision = args.precision or ("both" if args.device == "cuda" else "float32")
    if args.device == "cpu" and precision != "float32":
        raise ValueError("CPU requires strict float32; bfloat16/both requires CUDA")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if min(args.widths) < 4:
        raise ValueError("encoder widths must be >= 4")
    if args.identities_per_batch < 2 or args.n_classes < args.identities_per_batch:
        raise ValueError("require n_classes >= identities_per_batch >= 2")
    if any(
        b % args.identities_per_batch or b // args.identities_per_batch < 2
        for b in args.batch_sizes
    ):
        raise ValueError("every batch size must equal P*K with K >= 2")
    args.precisions = ["float32", "bfloat16"] if precision == "both" else [precision]
    return args


def summarize(samples):
    """Summarize repeat-level milliseconds/step, not individual step latencies."""
    return {
        "median_ms_per_step": statistics.median(samples),
        "mean_ms_per_step": statistics.mean(samples),
        "stdev_ms_per_step": statistics.stdev(samples) if len(samples) > 1 else None,
        "min_ms_per_step": min(samples),
        "max_ms_per_step": max(samples),
    }


def state_sha256(modules):
    import torch

    digest = hashlib.sha256()
    for prefix, module in modules:
        if module is None:
            continue
        for name, tensor in sorted(module.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(
                json.dumps(
                    [prefix, name, str(tensor.dtype), list(tensor.shape)]
                ).encode()
            )
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_repeat(args, channels, width, batch_size, precision, device):
    import torch

    from engine.train_metric import (
        _build_classifier,
        _resolved_config,
        _training_losses,
    )
    from models.common import seed_everything
    from models.gait_encoders import build_gait_encoder

    # Every repeat and precision begins from the same CPU model/input state and
    # dropout RNG seed. Dedicated generators keep synthetic rows independent of
    # parameter initialization and normalization data distinct from batch data.
    seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    config = _resolved_config(
        {
            "source": "synthetic_train_only",
            "seed": args.seed,
            "precision": precision,
            "device": args.device,
            "objective": "hybrid",
            "classifier_type": "arcface",
            "window_size": args.window_size,
            "identities_per_batch": args.identities_per_batch,
            "windows_per_identity": batch_size // args.identities_per_batch,
            "torch_threads": args.torch_threads,
            "model_args": {
                "architecture": args.architecture,
                "width": width,
                "embed_dim": args.embed_dim,
                "normalization": True,
            },
        }
    )
    train_x = torch.randn(
        32,
        channels,
        args.window_size,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    x = torch.randn(
        batch_size,
        channels,
        args.window_size,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    encoder = build_gait_encoder(channels, config["model_args"])
    encoder.train()
    # Explicit, small, synthetic TRAIN-only CPU fit before any model GPU move.
    encoder.fit_normalization(train_x)
    del train_x
    classifier = _build_classifier(config, args.n_classes)
    initial_state_sha256 = state_sha256(
        (("encoder", encoder), ("classifier", classifier))
    )
    encoder.to(device).train()
    classifier.to(device).train()
    x = x.to(device)
    labels = (
        torch.arange(args.identities_per_batch)
        .repeat_interleave(batch_size // args.identities_per_batch)
        .to(device)
    )
    # Each synthetic raw row has a distinct ID, including same-identity rows.
    sample_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    parameters = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    observed_dtype = None
    observed_compute_dtype = None
    compute_layer_name, compute_layer = next(
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear))
    )

    def observe_compute_dtype(module, inputs, output):
        nonlocal observed_compute_dtype
        observed_compute_dtype = output.dtype

    def step(check=False):
        nonlocal observed_dtype
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            z = encoder(x)
        if z.shape != (batch_size, args.embed_dim):
            raise ValueError("encoder output must have shape (batch, embed_dim)")
        losses = _training_losses(z, labels, sample_ids, classifier, labels, config)
        loss = losses[-1]
        # Preserve the trainer's existing per-step finite-loss check and sync.
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite metric training loss")
        if check:
            observed_dtype = str(z.dtype)
            if not torch.isfinite(z).all() or not all(
                torch.isfinite(v).all() for v in losses
            ):
                raise FloatingPointError("nonfinite warmup forward/loss")
            expected_dtype = (
                torch.bfloat16 if precision == "bfloat16" else torch.float32
            )
            # Final normalization can promote embeddings to FP32 under AMP.
            # Verify reduced precision on an eligible encoder operation instead.
            if observed_compute_dtype != expected_dtype:
                raise RuntimeError(
                    f"{precision} mode produced {observed_compute_dtype} at "
                    f"{compute_layer_name}, expected {expected_dtype}"
                )
        loss.backward()
        if check:
            if any(p.grad is None for p in parameters):
                raise RuntimeError("missing warmup parameter gradient")
            if not all(torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("nonfinite warmup gradients")
        torch.nn.utils.clip_grad_norm_(
            parameters,
            float(config["gradient_clip"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        return loss.detach()

    # The observer is removed before remaining warmup and all timed work.
    observer = compute_layer.register_forward_hook(observe_compute_dtype)
    try:
        step(check=True)
    finally:
        observer.remove()
    for _ in range(args.warmup - 1):
        step()
    optimizer.zero_grad(set_to_none=True)
    memory = {
        "baseline_allocated_bytes": None,
        "baseline_reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory.update(
            baseline_allocated_bytes=torch.cuda.memory_allocated(device),
            baseline_reserved_bytes=torch.cuda.memory_reserved(device),
        )
    started = time.perf_counter()
    for _ in range(args.steps):
        final_loss = step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    if device.type == "cuda":
        memory.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return {
        "elapsed_seconds": elapsed_seconds,
        "ms_per_step": elapsed_seconds * 1000 / args.steps,
        "windows_per_second": batch_size * args.steps / elapsed_seconds,
        "timed_optimizer_updates": args.steps,
        "warmup_optimizer_updates": args.warmup,
        "warmup_finite_forward_loss_gradients": True,
        "observed_embedding_dtype": observed_dtype,
        "observed_compute_layer": compute_layer_name,
        "observed_compute_dtype": str(observed_compute_dtype),
        "final_loss": float(final_loss.item()),
        "initial_state_sha256": initial_state_sha256,
        "memory": memory,
    }, config


def run(args):
    import torch

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback")
        if "bfloat16" in args.precisions:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError(
                        "CUDA BF16 requires native BF16 support; no precision fallback"
                    )
    # Strict FP32 baseline, including CPU under any outer autocast context.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    hardware = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cuda_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "torch_threads": args.torch_threads,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    cases = []
    for channels, width, batch_size in itertools.product(
        args.channels, args.widths, args.batch_sizes
    ):
        paired_state = None
        for precision in args.precisions:
            repeats = []
            for repeat in range(args.repeats):
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                result, config = run_repeat(
                    args, channels, width, batch_size, precision, device
                )
                state = result["initial_state_sha256"]
                if paired_state is not None and state != paired_state:
                    raise RuntimeError(
                        "initial model/classifier state differs across precision/repeats"
                    )
                paired_state = state
                repeats.append({"repeat": repeat + 1, **result})
            summary = summarize([r["ms_per_step"] for r in repeats])
            summary["windows_per_second_at_median"] = (
                batch_size * 1000 / summary["median_ms_per_step"]
            )
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                values = [r["memory"][key] for r in repeats]
                summary[key] = max(values) if device.type == "cuda" else None
            cases.append(
                {
                    "channels": channels,
                    "width": width,
                    "batch_size": batch_size,
                    "precision": precision,
                    "config": config,
                    "repeats": repeats,
                    "summary": summary,
                }
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "resident_synthetic_metric_training_precision",
        "hardware": hardware,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol": {
            "timing": "synchronized perf_counter wall time over the entire step block",
            "includes": [
                "zero_grad",
                "encoder_forward",
                "trainer_metric_and_classifier_losses",
                "trainer_finite_loss_check",
                "backward",
                "trainer_gradient_clip",
                "AdamW_step",
            ],
            "excludes": [
                "CPU_input_generation",
                "normalization_fit",
                "model_initialization",
                "CPU_to_device_transfer",
                "dataloading",
                "augmentation",
                "scheduler",
                "trainer_logging",
                "validation",
                "checkpointing",
            ],
            "input": "one resident FP32 synthetic P*K batch reused each step; distinct raw row IDs",
            "normalization": "32 separate synthetic TRAIN-only CPU rows; fit before model.to(device)",
            "precision": "FP32 parameters/input/AdamW/losses; CUDA BF16 autocast only around encoder",
            "repeat_reset": "fixed seed and fresh encoder/classifier/optimizer/input each repeat and precision",
            "memory": "CUDA process allocator bytes; peaks reset after warmup with AdamW state resident, gradients cleared; reserved includes warmup cache; CPU null",
            "constraints": [
                "synthetic throughput only; no quality or convergence evidence",
                "resident batch reuse excludes input pipeline and does not estimate full epoch time",
                "wall time includes trainer-path host synchronizations and Python dispatch",
                "allocator counters exclude other processes and CUDA driver/library allocations",
                "run only on idle hardware; record contention and thermal conditions externally",
                "same seed/state does not imply identical FP32/BF16 training trajectories",
            ],
        },
        "cases": cases,
    }


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        cli.error(str(exc))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(result['cases'])} cases to {args.out}")


if __name__ == "__main__":
    main()
