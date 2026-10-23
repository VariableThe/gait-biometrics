# Measured integration results — 23 October 2026

The authentic WISDM path now supports identity verification. These models remain research baselines: fixed-threshold errors and recording dependence prevent a deployment-ready claim. All rows below are actual dataset-backed runs or explicitly labeled synthetic timing experiments. No test metric selected a model, seed or threshold.

## Frozen-code full verification runs

Source snapshot: `cfbb8c647268613d5987767e7e0e04ad35dafd32` in the merged repository. The Student B code is mirrored; measured runs used the merged code on Darkstar (RTX 5070 Ti 16 GB), Python 3.14.8, torch 2.14.1+cu130, seed 42, 20 epochs, no window subset, 4 held-out validation subjects, walking label 0, native-rate windows T128/stride64, 10,000 balanced same-activity comparisons with replacement. `best.pt` is selected on validation EER. A threshold with validation FAR<=1% is frozen before test scoring; the table shows its actual test rates, not a promised test FAR.

| Source/task | Test windows | EER | ROC AUC | Fixed test FAR | Fixed test FRR | Training + evaluation wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| har walking | 496 | 0.1764 | 0.9252 | 0.0358 | 0.3728 | 23.18 s |
| wisdm_v1 walking | 345 | 0.2010 | 0.8686 | 0.1520 | 0.2468 | 25.48 s |

HAR optimizer training uses 955 walking windows from 17 subjects; validation uses 271 windows from 4 different subjects; test uses 496 windows from the 9 official test subjects. WISDM optimizer training uses 725 walking windows from 23 subjects; validation 202 windows from 4 other subjects; test 345 windows from 8 subjects. Subject labels in checkpoint metadata use the adapter's encoded global identity domain; original WISDM IDs are documented in the source provenance.

The canonical HAR/WISDM evaluations sampled 9,072 / 7,665 unique unordered pairs respectively. These are correlated, not independent trials. Genuine pairs may share samples because the frozen adapter contract has no recording offsets. Session separation and device/day shift remain unverified. Validation subjects also select checkpoints, so calibration is not an independent third cohort. An online walking detector, enrollment/probe sessions and continuous takeover-delay evaluation are not implemented.

An earlier pre-commit integration run produced HAR EER 0.2044/AUC 0.8987 and WISDM EER 0.2170/AUC 0.8809. It is retained separately as preliminary evidence, rather than averaged with the frozen-code run. Seed setting does not guarantee bitwise-identical CUDA training. Historical HAR EER 0.4162/AUC 0.6244 used all activities and an older pairing protocol; it is not a controlled quality baseline for these walking-only results.

## All five activity tracks

Fresh 20-epoch full-data runs on the official HAR test split (2,947 windows), device auto/CUDA, checkpoint selected on subject-disjoint validation activity loss. Existing model configs omit `val_subjects`, so this sweep uses the documented ceil(20% of 21)=5 validation subjects. This is a different cohort from `configs/validation_har.yaml` (4).

| Config | Test accuracy | Selected epoch | Training + evaluation wall time |
| --- | ---: | ---: | ---: |
| cnn_baseline | 0.821513 | 2 | 37.19 s |
| multitask | 0.905327 | 5 | 27.87 s |
| cnn_lstm | 0.893112 | 1 | 22.81 s |
| se_attention | 0.896166 | 1 | 24.95 s |
| vit | 0.872752 | 10 | 48.99 s |

The separate full 20-epoch Wiker GTX 1650 Ti run using `validation_har.yaml` (4 validation subjects) reached test accuracy **0.936885**, epoch 18 selected on validation loss. It is included with its complete training and evaluation JSON. Its total wall time was 414.07 s under unrelated host memory pressure; first epoch 378.05 s and median epoch 1.50 s. This total is not a clean speed benchmark. The optional old-trainer CPU run was stopped after 12 epochs to release local memory; no completed baseline result is claimed for it. All five track accuracies above are included, including lower scores than historical training-loss-selected runs; validation correctness is not an accuracy guarantee.

## Controlled performance measurements

- Identical 100,000-row homogeneous CSV, T128/stride64, 1,561 windows: old median **0.800901 s**, new **0.050937 s**, **15.72x** faster; `torch.equal` confirms bit-identical sensor tensors and labels. Three repeats; first run is included. Baseline is merged commit `0f0eeb3`.
- Real legacy WISDM test CSV: old median 6.121842 s / 2,958 windows; new median 0.157856 s / 611 windows. Different retention follows stricter activity/reset/gap boundaries, so this is a correctness-plus-runtime observation, not an equal-output speed comparison. The legacy numeric activity-name encoding is not established by the source CSV.
- Wiker identical synthetic embedding workload (GTX 1650 Ti 4 GB versus the same host CPU): **10.98x training** and **12.59x inference**. Darkstar RTX 5070 Ti versus its CPU: **9.83x / 15.98x**. These are hardware-path gains enabled by CUDA support, not architecture improvements.
- Runtime harness: batch64, 9 channels, T128, 2 CPU threads, 5 warm-up batches, 5 repeats of 10 steps; synchronized CUDA timings include input transfer. Inference transfers/processes one anchor batch; training transfers/processes anchor/positive/negative, three separate forwards plus Adam backward/step. Training throughput units are triplets/sec, inference units windows/sec. Raw samples and GPU peak allocations are in JSON; no phone energy/latency claim follows.
- Official WISDM raw parser: cold train load 10.586 s, warm test 0.107 s, repeated train 0.297 s on Wiker before final parser formatting. This compares process-local cache reuse with cold parsing; cache keys include resolved file path, size and mtime_ns and store no pickle. Raw windows are 2,155 train/718 test across all activities; 927/345 walking.
- Standalone Student A real/legacy full preprocessing completed in 23.67 / 13.74 s under concurrent load, with full metadata. These are absolute workload times without a before/after speed claim.

## Reproduce immediately

```bash
uv venv --python 3.14
uv pip install --python .venv/bin/python --compile-bytecode -r requirements-dev.txt
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments --prepare --device auto
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments --configs configs/cnn_baseline.yaml configs/multitask.yaml configs/cnn_lstm.yaml configs/se_attention.yaml configs/vit.yaml --epochs 20 --out-root runs/all_activity_tracks
uv run --no-project --python .venv/bin/python python -m scripts.benchmark_runtime --device all
uv run --no-project --python .venv/bin/python python -m scripts.benchmark_preprocessing --baseline-ref 0f0eeb3
```

For Student B's separate history use baseline ref `e2291d3` in the last command. The runner downloads public original datasets without OneDrive credentials/LFS, preserves existing checkpoints by requiring a fresh output root, overrides smoke subsets to full data, and records effective configs and subprocess commands. CUDA is selected when usable; CPU remains supported. Raw data and checkpoints stay local; only compact evidence/config JSON is checked in.

## Validation and review

Complete suites were run on Wiker and Darkstar; final counts and logs appear in `validation.json`. Dependency audit: no known vulnerabilities in the installed environment. Meaningful regressions cover real identity preservation, unknown legacy identities, segment discontinuities, split/cache stability, train-only normalization, validation-selected checkpoints, unseen multitask identities, chronological purge, bounded batches, safe checkpoint loading, tied ROC thresholds, exact 1% boundaries, validation-only calibration and checkpoint/config compatibility. Synthetic test scores never count as biometric results.

Independent review found no critical protocol defect in the shipped walking configurations. Six P2 findings were fixed: setup rejects destination symlinks, both public archives have trusted checksums and HAR layout validation, ONNX explicitly selects the supported exporter without an undeclared onnxscript dependency; omitted windowing values inherit checkpoint metadata, inference benchmarks no longer transfer unused triplet inputs, and invalid calibration plans fail before downloads/training. Final tests include those regressions. [Protocol limits](../../docs/verification_protocol.md), [training protocol](../../docs/training_protocol.md), [WISDM source details](../../docs/wisdm_identity.md), and [research fidelity audit](../../reading_notes/research_audit.md) explain the evidence scope.

## Compression diagnostic

A real export from the local validation-selected DeepConvLSTM checkpoint completed with structured prune=0.3 and a separate dynamic-quantization measurement. On a seeded 128-window HAR test diagnostic, accuracy fell from 0.953125 to 0.6640625 after pruning (−0.2890625); parameters fell 350,406→172,467 and estimated FLOPs 7,045,888→3,543,616. The FP32 pruned ONNX artifact is 694,388 bytes; the separate dynamically-quantized eager copy had estimated serialized size 312,168 bytes. No compressed-model latency gain or acceptable accuracy retention is claimed, and no hyperparameter was chosen from this test subset. Fine-tuning with validation-selected compression is future work. The exported FP32 artifact is explicitly not mislabeled as quantized. An ONNX checker plus PyTorch/ONNX Runtime numerical comparison at batch sizes 1 and 2 verifies graph validity and dynamic-batch inference on a tiny DeepConvLSTM; it is not a biometric-quality test.
