"""CPU timings include descriptor extraction and the frozen covariance metric."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from models.gait_features import extract_gait_features
from models.gait_metric import WithinClassMetric
from scripts.research_verification import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error("repetitions must be in [1,100]")
    if args.out.resolve() == args.metric.resolve():
        parser.error("output must not overwrite the frozen metric")
    with np.load(args.metric, allow_pickle=False) as state:
        metric = WithinClassMetric.from_state_dict(state)
        tensor_bytes = sum(state[name].nbytes for name in state.files)
    inputs = np.random.default_rng(42).normal(size=(128, 9, 128)).astype(np.float32)
    cases = []
    with threadpool_limits(limits=2):
        for size in (1, 128):
            x = inputs[:size]
            for _ in range(5):
                output = metric.transform(extract_gait_features(x, 50))
            if not np.isfinite(output).all() or not np.allclose(
                np.linalg.norm(output, axis=1), 1, atol=1e-5
            ):
                raise ValueError("feature inference returned invalid unit embeddings")
            samples = []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                output = metric.transform(extract_gait_features(x, 50))
                samples.append(time.perf_counter() - started)
            cases.append(
                {
                    "batch_size": size,
                    "batch_latency_ms_median": statistics.median(samples) * 1000,
                    "batch_latency_ms_p95": float(np.percentile(samples, 95)) * 1000,
                    "batch_latency_ms_samples": [value * 1000 for value in samples],
                    "windows_per_second": size * args.repetitions / sum(samples),
                }
            )
    write(
        args.out,
        {
            "kind": "frozen_har_feature_pipeline",
            "metric_sha256": digest(args.metric),
            "tensor_state_bytes": tensor_bytes,
            "hardware": platform.platform(),
            "numpy_version": np.__version__,
            "threads": 2,
            "input_shape": list(inputs.shape),
            "seed": 42,
            "warmup_batches": 5,
            "repetitions": args.repetitions,
            "included": "26 descriptors per physical/magnitude channel, train-only scale, dense covariance projection, L2 normalization",
            "excluded": "loading, input generation, transfers, scoring, calibration",
            "batches": cases,
        },
    )
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
