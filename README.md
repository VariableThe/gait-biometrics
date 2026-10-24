# Gait-based continuous authentication

Minor specialization project, MAHE. Mentor: **Manjunath Hegde**. Student A owns preprocessing, Student B models, and Student C evaluation.

The pipeline now loads authentic WISDM subject IDs, keeps windows within recording/activity boundaries, selects checkpoints on held-out validation subjects, and runs bounded batches on CPU or CUDA. **It remains a research baseline, not a usable passive authenticator.**

The [24 October hillclimb report](reports/2026-10-24-hillclimb/README.md) adds six-axis sensor fusion, rotation invariants, lagged covariance and independently calibrated enrollment-score blending. On five reused Santos cross-day test users, the latest selected method has descriptive EER 0%, with **3/200 false accepts (1.5%) and 0/50 genuine rejections** at its frozen calibration threshold. The matched lagcov control has 7/200 false accepts and 0/50 rejections. This is exploratory evidence: the threshold still misses the intended 1% test FAR, and these controlled leg recordings do not establish phone deployment. The report retains all candidate results, failed audit compute, source/artifact hashes, per-person errors, and runtime costs.

## Run immediately

Python 3.14.8 and the existing pinned torch 2.14.1 dependency set were validated on Linux x86_64. Create an isolated environment with uv:

```bash
uv venv --python 3.14
uv pip install --python .venv/bin/python --compile-bytecode -r requirements-dev.txt
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments --prepare --device auto
```

This downloads the original public HAR and WISDM AR v1.1 sources over HTTPS, verifies both archive checksums, and runs **20 full epochs each** for walking-only HAR and WISDM verification followed by held-out test evaluation. No OneDrive login or Git LFS is required. Effective configs, checkpoints, metrics and commands go under `runs/full_experiments/`. Choose a fresh `--out-root` for another run; existing checkpoints are preserved. Legacy `configs/siamese.yaml` is a smoke configuration without calibration subjects and is rejected by this full calibrated runner; use the two verification configs instead.

CUDA is chosen when available. CPU is supported. Use `--device cpu` explicitly when needed; batches stay bounded and the dataset remains on host memory. Run independent jobs on different machines or output roots. RAM pressure can dominate runtime, so the default runner executes sequentially on a single device.

```bash
# All five classification tracks: full data, validation-selected checkpoints.
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments \
  --configs configs/cnn_baseline.yaml configs/multitask.yaml configs/cnn_lstm.yaml configs/se_attention.yaml configs/vit.yaml \
  --epochs 20 --out-root runs/all_activity_tracks

# A defined four-validation-subject DeepConvLSTM activity experiment.
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments \
  --configs configs/validation_har.yaml --out-root runs/har_activity

uv run --no-project --python .venv/bin/python python -m pytest tests -q
```

## Measured results

The [controlled scaling study](reports/2026-10-23-scaling/README.md) adds 48 GPU
runs with three subject folds, two seeds, fixed final epochs and independent
calibration. Wider encoders help WISDM but slightly worsen HAR; descriptor
methods win the development comparison on both. These fold scores are not
new test improvements. BF16 was slower in every matched GPU shape, so FP32
remains the default.

The [first genuine cross-day study](reports/2026-10-23-cross-day/README.md) uses
a separate public Santos dataset and a frozen 15/5/5 development/calibration/test
population split. Day1 enrollment against day2 trial probes gives **7.0% test
EER**, AUC **0.9888**, and fixed-threshold FAR/FRR **0.5% / 22%**. Only five test
users contributed 50 genuine and 200 impostor claims; one user had 8/10 genuine
claims rejected. This is a small controlled leg-sensor result, not a phone or
deployment claim. [The stage guide](docs/cross_day_verification.md) records the
single test attempt, calibration lock, data/source checks and reproduction.

The [research-extension report](reports/2026-10-23-quality/README.md) adds balanced batch-hard/SupCon training, multiscale and temporal encoders, train-only normalization, coherent augmentation, statistical baselines, covariance normalization and enrollment scoring. Thirty-six final 40-epoch neural runs compare three seeds per method with fixed subject cohorts.

| Source/method | Baseline EER | New EER | Evidence |
| --- | ---: | ---: | --- |
| WISDM, three-seed multiscale batch-hard | 0.2010 | **0.0958** | Validation-locked primary study |
| HAR, subject-CV descriptors + WCCN | 0.1764 | **0.1044** | Exploratory follow-up; test previously inspected |

WISDM fixed test FAR/FRR improved from 0.1520/0.2468 to 0.0826/0.1072. Secondary HAR rates improved from 0.0358/0.3728 to 0.0054/0.3046. **The primary HAR neural winner worsened to EER 0.2146**; that negative result is retained. Its failure motivated separate subject-fold selection of descriptors. The secondary result needs a fresh cohort for confirmation. The new neural ensemble is smaller but slower than the old encoder; complete latency and parameter measurements are reported without claiming an architecture speedup.

Use the [research reproduction guide](docs/research_experiments.md) for the full train→lock→evaluate workflow and separate one-seed smoke configs. New checkpoint format `gait_metric_v1` uses its explicit loader/runner; the existing evaluator continues to handle legacy checkpoints. Activity-classification results below are unchanged.

The [complete 23 October evidence report](reports/2026-10-23/README.md) contains effective configs, every track's results, raw timing samples, calibration counts, test provenance and limitations. Frozen-code full 20-epoch walking results:

| Dataset | Test windows | EER | ROC AUC | Test FAR / FRR at validation-calibrated threshold |
| --- | ---: | ---: | ---: | ---: |
| UCI HAR, walking | 496 | 0.1764 | 0.9252 | 0.0358 / 0.3728 |
| WISDM AR v1.1, walking | 345 | 0.2010 | 0.8686 | 0.1520 / 0.2468 |

These measure **held-out-subject, within-recording discrimination**. EER is descriptive; the fixed threshold is chosen from validation scores, never test scores. The validation FAR target of 1% does not guarantee 1% test FAR. Genuine windows can overlap, comparisons repeat, and session/day separation is unverified. New enrollment scoring uses ordered indices within existing recordings; independent sessions, an online walking gate, temporal decisions and takeover-delay evaluation remain necessary.

A separate 20-epoch local GTX 1650 Ti DeepConvLSTM run with four validation subjects reached HAR test activity accuracy **0.936885**. The five-track sweep used five validation subjects and produced lower accuracies for several tracks; all are reported. Historical HAR EER 0.4162 used all activities and an older pairing protocol, so it is not a controlled comparison with the walking results above.

Controlled performance measurements:

- **15.72x preprocessing speedup**, bit-identical outputs on the same 100,000-row homogeneous CSV fixture (1,561 windows).
- **10.98x training / 12.59x inference** with Wiker's GTX 1650 Ti versus its CPU; **9.83x / 15.98x** on Darkstar's RTX 5070 Ti versus its CPU, on the same synthetic embedding workload. These are hardware-path execution gains, not biometric-quality improvements.
- Real CSV boundaries now discard unsafe windows. Their changed counts are documented separately from the equal-output benchmark.

```bash
uv run --no-project --python .venv/bin/python python -m scripts.benchmark_runtime --device all
uv run --no-project --python .venv/bin/python python -m scripts.benchmark_preprocessing --baseline-ref 0f0eeb3
```

Use `0f0eeb3` from this repository's evidence report. Timings include transfer and synchronized GPU work after warm-up; training throughput is triplets/sec and inference throughput is windows/sec. No phone battery or deployment latency measurement is implied.

## Data contract and provenance

`data/adapter.py` preserves the shared `load_windows` / `build_loader` signatures and `WindowData` fields. Models consume `(N,C,T)` float32 tensors; activity and optional identity labels are int64.

| Source | Representation | Identities | Activity encoding |
| --- | --- | --- | --- |
| `har` | Official 50 Hz, 9-channel, fixed 128-sample windows | Original official subject-disjoint train/test split | Original 1–6 converted to 0–5 |
| `wisdm_v1` | Official AR v1.1, native 20 Hz, 3 accelerometer channels | Authentic IDs 1–36 with stable global encoding and project-defined 80/20 subject split | Walking=0, Jogging=1, Upstairs=2, Downstairs=3, Sitting=4, Standing=5 |
| `wisdm` | Existing derived train/test CSVs, 3 accelerometer channels | Unknown: `y_subject=None` | Integer range 0–5 checked; semantic names remain unverified |

`wisdm_v1` is a different explicit source from the anonymous CSV. No identity is inferred from timestamps, row order or activity. The adapter deliberately rejects UCI 507's different 18-activity/1600–1650 schema rather than mislabeling it as v1.1. No gyroscope is invented for accelerometer-only WISDM.

Default activity training requires real identity labels for subject-disjoint validation. Legacy anonymous CSV activity training requires an explicit `validation_strategy: chronological`; it purges potentially overlapping boundary windows and makes no subject-independent claim. Multitask training maps optimizer-training identities only and never scores unseen validation identities as known classes.

## Implementation and research

- `data/`: stable model-facing adapter, raw identity parser, segmented/vectorized windowing and bounded process cache.
- `engine/train.py`: validation-selected activity training, multitask label handling, CPU/CUDA batches.
- `engine/train_siamese.py`: triplet/contrastive embedding training, cached identity groups, bounded index gathers, validation EER selection.
- `engine/evaluate.py`: tied ROC handling, empirical independent operating points, validation-calibrated fixed decisions, pairing/dependence metadata and leakage guards.
- `models/`: existing CNN, DeepConvLSTM, multitask, SE, transformer, Siamese and compression tracks. Architecture labels are adaptations, not verified paper reproductions.
- `scripts/`: public dataset preparation, full experiments, performance benchmarks and ONNX export.

Read [WISDM identities](docs/wisdm_identity.md), [training protocol](docs/training_protocol.md), [evaluation protocol](docs/verification_protocol.md), and the [audit of 12 supplied papers plus additional primary studies](reading_notes/research_audit.md). The audit distinguishes IMU methods from visual joint/silhouette methods and links each reusable lesson to its source. Earlier `UNVERIFIED` reading notes remain historical; they are not proof of implementation fidelity.

The [official Fordham WISDM source](https://www.cis.fordham.edu/wisdm/dataset.php) and [official UCI HAR source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) define the data. Original WISDM `readme.txt` and schema notes are retained by the downloader.

Student A's corrected standalone pipeline is also mirrored in `student_a/gait_pipeline.py`; its NPZ export contract is separate from the raw-window adapter.
