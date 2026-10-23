"""Checkpoint-only inference timing on identical resident synthetic windows.

Restores the frozen raw baseline, the locked neural ensemble, and its first
member. Never loads datasets, trains, calibrates, or computes biometric scores.
Torch is imported only after argument and selection-lock validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be in [{low}, {high}]")
        return number

    return parse


def batch_sizes(value):
    try:
        sizes = [bounded_int(1, 512)(part) for part in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated batch sizes in [1, 512], e.g. 1,128"
        ) from exc
    if len(sizes) > 8 or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("use at most eight distinct batch sizes")
    return sizes


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--lock",
        required=True,
        type=Path,
        help="source folder or its selection_lock.json",
    )
    result.add_argument("--baseline-checkpoint", required=True, type=Path)
    result.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    result.add_argument("--batch-sizes", type=batch_sizes, default=[1, 128])
    result.add_argument("--repetitions", type=bounded_int(1, 100), default=30)
    result.add_argument("--warmup", type=bounded_int(0, 20), default=5)
    result.add_argument("--torch-threads", type=bounded_int(1, 32), default=2)
    result.add_argument("--seed", type=bounded_int(0, 2**32 - 1), default=42)
    result.add_argument(
        "--pack-multiscale",
        action="store_true",
        help="also benchmark inference-only packed copies of multiscale encoders",
    )
    result.add_argument("--out", required=True, type=Path)
    return result


def artifact_path(folder, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("locked artifact paths must be relative strings")
    path = (folder / name).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"locked artifact escapes source folder: {name}")
    return path


def read_lock(path, baseline):
    path = path.resolve()
    if path.is_dir():
        path /= "selection_lock.json"
    lock = json.loads(path.read_text())
    folder = path.parent
    if lock.get("source") not in {"har", "wisdm_v1"}:
        raise ValueError("lock source must be har (9 channels) or wisdm_v1 (3)")
    selected = lock["selection"]
    if selected["method"] == "features":
        raise ValueError(
            "selection is a feature method; this neural encoder benchmark cannot "
            "represent it. Do not substitute a neural candidate for the winner."
        )
    method = selected["method"]
    if (
        not isinstance(method, str)
        or Path(method).name != method
        or method in {".", ".."}
    ):
        raise ValueError("selected neural method must be a source-folder child")
    if selected["postprocess"] not in {"none", "wccn_auto"}:
        raise ValueError("unsupported locked neural postprocess")
    seeds = lock["seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 8
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("lock must contain one to eight distinct nonnegative seeds")
    if lock.get("test_loaded_before_lock") is not False:
        raise ValueError("lock must declare test_loaded_before_lock=false")
    if sha256(folder / "validation_registry.json") != lock["registry_sha256"]:
        raise ValueError("validation registry changed after locking")
    hashes = lock["artifact_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("lock requires artifact_sha256")
    for name, expected in hashes.items():
        if sha256(artifact_path(folder, name)) != expected:
            raise ValueError(f"selected artifact changed after locking: {name}")
    required = [f"{method}/seed_{seed}/best.pt" for seed in seeds]
    if selected["postprocess"] == "wccn_auto":
        required += [f"{method}/metric_{seed}.npz" for seed in seeds]
    if any(
        name not in hashes or name not in selected["artifacts"] for name in required
    ):
        raise ValueError("every selected checkpoint/postprocessor must be hash-locked")
    if selected["validation_cache"] not in hashes:
        raise ValueError("selected validation cache is not hash-locked")
    baseline_hash = sha256(baseline)
    # Earlier research-runner locks did not contain a baseline digest. These
    # remain usable; the supplied baseline is always fingerprinted in the output.
    if "baseline_sha256" in lock and baseline_hash != lock["baseline_sha256"]:
        raise ValueError("baseline checkpoint differs from selection lock")
    return path, lock, baseline_hash


def check_protocol(protocol, lock, label):
    channels = 9 if lock["source"] == "har" else 3
    if (
        protocol.get("source") != lock["source"]
        or protocol.get("n_channels") != channels
        or protocol.get("window_size") != 128
    ):
        raise ValueError(f"{label}: source/channels/window_size disagree with lock")
    for key in ("train_subject_ids", "val_subject_ids"):
        if set(protocol[key]) != set(lock[key]):
            raise ValueError(f"{label}: {key} disagree with lock")


def model_sizes(model):
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    parameter_bytes = sum(t.numel() * t.element_size() for _, t in parameters)
    buffer_bytes = sum(t.numel() * t.element_size() for _, t in buffers)
    unused = sum(t.numel() for name, t in parameters if "dae_decoder." in name)
    total = sum(t.numel() for _, t in parameters)
    return {
        "parameters": total,
        "parameters_unused_dae_decoder": unused,
        "forward_parameters": total - unused,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "tensor_state_bytes": parameter_bytes + buffer_bytes,
        "buffers": [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size(),
            }
            for name, value in buffers
        ],
    }


def restore_models(
    torch, lock_path, lock, baseline_path, device, pack_multiscale=False
):
    import numpy as np

    from engine.train_metric import load_metric_encoder
    from models.gait_metric import WithinClassMetric
    from models.siamese import EmbeddingNet

    class Member(torch.nn.Module):
        def __init__(self, encoder, metric=None):
            super().__init__()
            self.encoder = encoder
            self.with_metric = metric is not None
            if metric is not None:
                for name in ("mean", "scale", "projection"):
                    self.register_buffer(
                        name, torch.from_numpy(getattr(metric, name + "_").copy())
                    )

        def forward(self, inputs):
            result = self.encoder(inputs)
            if self.with_metric:
                # Match WithinClassMetric.transform's float64 arithmetic and
                # float32 output, keeping the frozen transform on-device.
                result = (
                    (result.to(torch.float64) - self.mean) / self.scale
                ) @ self.projection
                result = torch.nn.functional.normalize(result, dim=1, eps=1e-12).float()
            return result

    class Ensemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, inputs):
            return torch.cat(
                [member(inputs) for member in self.members], dim=1
            ) / math.sqrt(len(self.members))

    payload = torch.load(baseline_path, map_location="cpu", weights_only=True)
    protocol = payload["metrics"]["protocol"]
    check_protocol(protocol, lock, "baseline")
    if protocol.get("normalization", "none") != "none":
        raise ValueError("old baseline must use raw inputs without normalization")
    state = payload["model_state_dict"]
    if state and all(name.startswith("embedding.") for name in state):
        state = {name[len("embedding.") :]: value for name, value in state.items()}
    channels = 9 if lock["source"] == "har" else 3
    baseline = EmbeddingNet(channels, embed_dim=128, dropout=0.3)
    baseline.load_state_dict(state, strict=True)
    del state, payload
    members, details = [], []
    folder = lock_path.parent
    selected = lock["selection"]
    for seed in lock["seeds"]:
        name = f"{selected['method']}/seed_{seed}/best.pt"
        path = artifact_path(folder, name)
        encoder, protocol, config = load_metric_encoder(str(path), device="cpu")
        check_protocol(protocol, lock, name)
        if config.get("seed") != seed:
            raise ValueError(f"{name}: checkpoint seed differs from lock order")
        metric = None
        if selected["postprocess"] == "wccn_auto":
            metric_path = artifact_path(
                folder, f"{selected['method']}/metric_{seed}.npz"
            )
            with np.load(metric_path, allow_pickle=False) as archive:
                metric = WithinClassMetric.from_state_dict(dict(archive))
            if set(metric.training_subject_ids_) != set(lock["train_subject_ids"]):
                raise ValueError("WCCN training subjects differ from selection lock")
            if len(metric.mean_) != int(config["model_args"]["embed_dim"]):
                raise ValueError("WCCN dimension differs from selected encoder")
        member = Member(encoder, metric)
        members.append(member)
        details.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "checkpoint_sha256": lock["artifact_sha256"][name],
                "checkpoint_file_bytes": path.stat().st_size,
                "model_args": config["model_args"],
                **model_sizes(member),
            }
        )
    ensemble = Ensemble(members)
    for model in (baseline, ensemble):
        model.requires_grad_(False).to(device).eval()
    models = {
        "old_baseline_raw": baseline,
        "selected_ensemble": ensemble,
        "selected_single": members[0],
    }
    if pack_multiscale:
        import copy

        from models.gait_inference import pack_multiscale_branches

        packed_members = []
        for member in members:
            packed_encoder = pack_multiscale_branches(member.encoder)
            # Preserve the member's WCCN state, reusing the already copied
            # encoder rather than deep-copying that encoder a second time.
            packed_members.append(
                copy.deepcopy(member, {id(member.encoder): packed_encoder})
            )
        packed_ensemble = Ensemble(packed_members).requires_grad_(False).eval()
        models["selected_ensemble_packed"] = packed_ensemble
        models["selected_single_packed"] = packed_members[0]
    return models, details


def check_packed_output(torch, original, packed, inputs, device):
    """Compare complete inference outputs outside all timed intervals."""
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        expected = original(inputs)
        actual = packed(inputs)
        if (
            expected.ndim != 2
            or expected.shape[0] != len(inputs)
            or actual.shape != expected.shape
            or not torch.isfinite(expected).all().item()
            or not torch.isfinite(actual).all().item()
        ):
            raise ValueError(
                "original/packed outputs must have matching finite batch shapes"
            )
        max_abs_diff = (expected - actual).abs().max().item()
        if not math.isfinite(max_abs_diff) or max_abs_diff > 1e-5:
            raise ValueError(
                f"packed output mismatch for input {tuple(inputs.shape)}: "
                f"max_abs_diff={max_abs_diff}, tolerance=1e-5"
            )
    return max_abs_diff


def percentile95(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def measure(torch, model, inputs, device, warmup, repetitions):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        # Validate once, outside timing; then discard outputs each iteration.
        output = model(inputs)
        if (
            output.ndim != 2
            or output.shape[0] != len(inputs)
            or not torch.isfinite(output).all().item()
        ):
            raise ValueError(
                "encoder output must be a finite (batch, dimensions) tensor"
            )
        output_shape = list(output.shape)
        del output
        for _ in range(warmup):
            model(inputs)
        synchronize()
        memory = None
        if device.type == "cuda":
            # Inputs, models, validation output and warmup allocations precede
            # the reset. Delta is incremental allocated memory, not reserved VRAM.
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
        timings = []
        for _ in range(repetitions):
            synchronize()
            started = time.perf_counter()
            output = model(inputs)
            synchronize()
            timings.append(time.perf_counter() - started)
            del output
        if device.type == "cuda":
            memory = {
                "allocated_before_bytes": allocated,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_delta_bytes": torch.cuda.max_memory_allocated(device) - allocated,
            }
    median = statistics.median(timings)
    return {
        "batch_size": len(inputs),
        "output_shape": output_shape,
        "repetitions": repetitions,
        "batch_latency_ms_median": median * 1000,
        "batch_latency_ms_p95": percentile95(timings) * 1000,
        "windows_per_second": len(inputs) * repetitions / sum(timings),
        "windows_per_second_at_median": len(inputs) / median,
        "batch_latency_ms_samples": [value * 1000 for value in timings],
        "cuda_memory": memory,
    }


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        lock_path, lock, baseline_hash = read_lock(args.lock, args.baseline_checkpoint)
        protected = {
            lock_path,
            lock_path.parent / "validation_registry.json",
            args.baseline_checkpoint.resolve(),
        }
        protected.update(
            artifact_path(lock_path.parent, name) for name in lock["artifact_sha256"]
        )
        if args.out.resolve() in protected:
            raise ValueError("--out must not overwrite any study input")
        # Support both direct-script and python -m execution from any directory.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import torch

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        models, members = restore_models(
            torch,
            lock_path,
            lock,
            args.baseline_checkpoint,
            device,
            pack_multiscale=args.pack_multiscale,
        )
        channels = 9 if lock["source"] == "har" else 3
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inputs_cpu = torch.randn(
            max(args.batch_sizes),
            channels,
            128,
            generator=generator,
            dtype=torch.float32,
        )
        if not torch.isfinite(inputs_cpu).all().item():
            raise ValueError("synthetic inputs must be finite")
        input_hash = hashlib.sha256(inputs_cpu.numpy().tobytes()).hexdigest()
        inputs = inputs_cpu.to(device)
        del inputs_cpu
        packed_checks = {}
        if args.pack_multiscale:
            for name in ("selected_ensemble", "selected_single"):
                packed_name = name + "_packed"
                packed_checks[packed_name] = {
                    size: check_packed_output(
                        torch, models[name], models[packed_name], inputs[:size], device
                    )
                    for size in args.batch_sizes
                }
        results = {}
        for name, model in models.items():
            results[name] = {**model_sizes(model), "batches": []}
            for size in args.batch_sizes:
                print(
                    f"Measuring {name} batch={size} on {device}",
                    file=sys.stderr,
                    flush=True,
                )
                results[name]["batches"].append(
                    measure(
                        torch,
                        model,
                        inputs[:size],
                        device,
                        args.warmup,
                        args.repetitions,
                    )
                )
                if name in packed_checks:
                    results[name]["batches"][-1].update(
                        max_abs_diff=packed_checks[name][size],
                        reference_case=name.removesuffix("_packed"),
                    )
        report = {
            "schema_version": 1,
            "kind": "checkpoint_synthetic_inference",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": lock["source"],
            "selection": lock["selection"]["name"],
            "postprocess": lock["selection"]["postprocess"],
            "selection_lock": str(lock_path),
            "selection_lock_sha256": sha256(lock_path),
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            "baseline_sha256": baseline_hash,
            "baseline_hash_in_lock": "baseline_sha256" in lock,
            "baseline_checkpoint_file_bytes": args.baseline_checkpoint.stat().st_size,
            "ensemble_members": len(members),
            "single_member_seed": lock["seeds"][0],
            "members": members,
            "hardware": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
            },
            "protocol": {
                "input_shape": list(inputs.shape),
                "input_dtype": str(inputs.dtype),
                "input_seed": args.seed,
                "input_sha256": input_hash,
                "batch_sizes": args.batch_sizes,
                "warmup_batches": args.warmup,
                "repetitions": args.repetitions,
                "inference_mode": True,
                "tf32": False,
                "autocast": False,
                "compiled": False,
                "pack_multiscale": args.pack_multiscale,
                "timer": "perf_counter; CUDA synchronized before/after each forward",
                "transfer_included": False,
                "included": "raw baseline; complete encoders; selected WCCN; ensemble concatenation/scaling",
                "excluded": "loading, input generation, transfers, calibration, scoring, training",
                "throughput_unit": "input windows/second; one ensemble input counts once",
                "throughput_formula": "batch_size * repetitions / sum(batch_seconds)",
                "p95_method": "linear interpolation at (n-1)*0.95",
                "memory_scope": "incremental CUDA allocated peak after warmup, all models co-resident",
            },
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".partial")
        if temporary.resolve() in protected:
            raise ValueError("temporary output path would overwrite a study input")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out)
        print(f"Wrote {args.out}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        arguments.exit(2, f"benchmark error: {exc}\n")


if __name__ == "__main__":
    main()
