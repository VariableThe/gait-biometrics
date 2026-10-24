"""Bounded DEV-only CPU cost of frozen lagcov embeddings and trial scoring."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
from scripts import secure_operating_study as study
from threadpoolctl import threadpool_limits


def measure(function, repeats):
    for _ in range(5):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {
        "seconds": samples,
        "median_ms": float(np.median(samples) * 1000),
        "p95_ms": float(np.percentile(samples, 95) * 1000),
    }


def benchmark(args):
    if not 5 <= args.repeats <= 1000:
        raise ValueError("repeats must be 5..1000")
    if args.out.exists():
        raise ValueError("output already exists")
    prior = study.prior
    root, registration = study._registration(args)
    lock = prior.read(root / "lock.json")
    checksum = lock.pop("self_hash")
    prior.same(prior.object_hash(lock), checksum, "frozen operating lock")
    prior.same(
        lock["registration_sha256"],
        prior.digest(root / "registration.json"),
        "frozen registration",
    )
    prior.same(
        lock["development_sha256"],
        prior.digest(root / "development.json"),
        "frozen development",
    )
    selection = prior.read(root / "development.json")["selection"]
    prior.same(selection, lock["selection"], "frozen selection")
    candidates = list(dict.fromkeys((selection["winner"], study.CONTROL)))
    prior.same(sorted(lock["models"]), sorted(candidates), "frozen models")
    data = prior.load_data(args.archive_path, prior.DEV_IDS)
    prior.same(prior._manifest(data), registration["development"], "registered DEV")
    rows_by_subject = [
        np.flatnonzero(data["subjects"] == subject).tolist()
        for subject in prior.DEV_IDS
    ]
    rows = [
        row[level]
        for level in range(max(map(len, rows_by_subject)))
        for row in rows_by_subject
        if len(row) > level
    ][:32]
    if len(rows) != 32:
        raise ValueError("at least 32 original DEV windows required")
    result = {
        "kind": "warm DEV-only frozen CPU execution benchmark",
        "lock_hash": checksum,
        "script_sha256": prior.digest(Path(__file__)),
        "sources": study.hashes(),
        "machine": platform.node(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "warmup": 5,
        "threads": 2,
        "dev_indices": rows,
        "actual_device": "cpu",
        "fits": 0,
        "calibration_or_test_payloads_loaded": False,
        "results": {},
    }
    with threadpool_limits(limits=2):
        values = study._features(data)
        for candidate in candidates:
            metric, cohort = prior.load_model(
                prior._verify_artifact(root, lock["models"][candidate]),
                "control",
                prior.DEV_IDS,
            )
            prior.verify_cohort(metric, cohort, values, data, prior.DEV_IDS)
            batches = []
            for size in (1, 32):
                batch = np.ascontiguousarray(data["X"][rows[:size]])

                def embed(batch=batch, metric=metric):
                    return metric.transform(study._features({"X": batch}))

                output = embed()
                if output.shape != (size, 362) or not np.isfinite(output).all():
                    raise ValueError("invalid frozen embedding")
                batches.append({"batch_size": size, **measure(embed, args.repeats)})
            method = candidate.split("/")[1]

            def score(method=method, metric=metric):
                return study._score(metric, values, data, prior.DEV_IDS, method)

            scored = score()
            result["results"][candidate] = {
                "raw_window_to_embedding": batches,
                "all_dev_trial_scoring": {
                    "windows": len(values),
                    "claims": len(scored["labels"]),
                    **measure(score, args.repeats),
                },
            }
    result["timing_scope"] = {
        "embedding": "raw resident RAM windows through exact registered descriptor extraction, WCCN and normalization; includes unused legacy descriptor calculation performed by current wrapper",
        "trial_scoring": "precomputed DEV descriptors through frozen metric, trial pooling, enrollment templates and all directed claims; no raw feature extraction",
        "excluded": "cold start, archive audit/loading, model restore, sensor acquisition, I/O, phone energy; not end-to-end authentication",
    }
    prior.write_new(args.out, result)
    return {
        candidate: {
            "embedding_median_ms": [
                batch["median_ms"] for batch in timings["raw_window_to_embedding"]
            ],
            "dev_scoring_median_ms": timings["all_dev_trial_scoring"]["median_ms"],
        }
        for candidate, timings in result["results"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    print(study.prior.canonical(benchmark(args)).decode())


if __name__ == "__main__":
    main()
