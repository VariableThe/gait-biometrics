"""Measure identical embedding workloads on CPU and CUDA, with warm-up.

These are synthetic execution timings, never biometric accuracy measurements.
CUDA synchronization includes each bounded batch's host-to-device transfer.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from models.siamese import EmbeddingNet, triplet_loss


def benchmark(
    device: str, batch_size: int, channels: int, steps: int, repeats: int
) -> dict:
    torch.manual_seed(42)
    model = EmbeddingNet(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batches = [torch.randn(batch_size, channels, 128) for _ in range(3)]
    if device == "cuda":
        batches = [x.pin_memory() for x in batches]

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def step(training: bool):
        inputs = [
            x.to(device, non_blocking=True)
            for x in (batches if training else batches[:1])
        ]
        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = [model(x) for x in inputs]
            triplet_loss(*outputs).backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                model(inputs[0])

    result = {}
    for training in (False, True):
        model.train(training)
        for _ in range(5):
            step(training)
        synchronize()
        timings = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                step(training)
            synchronize()
            timings.append((time.perf_counter() - start) / steps)
        result["training" if training else "inference"] = {
            "seconds_per_batch": timings,
            "median_seconds": statistics.median(timings),
            "items_per_second": batch_size / statistics.median(timings),
            "item_unit": "triplets" if training else "windows",
            "batch_semantics": "one anchor batch"
            if not training
            else "anchor+positive+negative batches, three forwards, Adam step",
        }
    result["device_name"] = (
        torch.cuda.get_device_name() if device == "cuda" else platform.processor()
    )
    if device == "cuda":
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=9)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all")
    parser.add_argument("--out", type=Path, default=Path("runs/runtime_benchmark.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.channels, args.steps, args.repeats, args.threads) <= 0:
        parser.error("all sizes/counts must be positive")
    torch.set_num_threads(args.threads)
    devices = (
        ["cpu", "cuda"]
        if args.device == "all" and torch.cuda.is_available()
        else ["cpu"]
        if args.device == "all"
        else [args.device]
    )
    results = {
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "channels": args.channels,
        "window_size": 128,
        "steps": args.steps,
        "repeats": args.repeats,
        "torch_threads": args.threads,
        "warmup_batches": 5,
        "kind": "synthetic_execution_benchmark",
    }
    for device in devices:
        results[device] = benchmark(
            device, args.batch_size, args.channels, args.steps, args.repeats
        )
    if "cpu" in results and "cuda" in results:
        results["cuda_speedup_over_cpu"] = {
            mode: results["cpu"][mode]["median_seconds"]
            / results["cuda"][mode]["median_seconds"]
            for mode in ("training", "inference")
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
