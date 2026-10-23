# Reproducing the locked research verification study

Run from `gait-biometrics/`. This opt-in study compares research candidates
with the existing neural baseline on the same walking-only, subject-held-out
HAR/WISDM cohorts. It preserves the existing training and evaluation paths.
Implementation details are in [encoder methods](gait_encoder_methods.md),
[metric learning](metric_learning_methods.md), and
[enrollment methods](enrollment_methods.md).

## Environment and official data

Reuse a compatible existing uv environment. For a new local environment, use
the existing pinned requirements; no additional research dependencies are needed:

```bash
uv venv --python 3.14
uv pip install --python .venv/bin/python --compile-bytecode -r requirements.txt
```

The baseline command below prepares both original public datasets over HTTPS,
checks pinned archive SHA-256 digests, and retains provenance in
`datasets/har_provenance.json` and `datasets/wisdm_v1_provenance.json`.
HAR uses the official UCI inertial prewindows and subject train/test split;
WISDM AR v1.1 uses Fordham's original raw records and authentic subject IDs,
with the adapter's deterministic subject-held-out split. No identity
reconstruction, OneDrive credentials, or Git LFS is needed. The source layouts
are `datasets/uci_har/UCI HAR Dataset/` and
`datasets/wisdm/WISDM_ar_v1.1_raw.txt`.

All stages use walking label `[0]`, 128 samples, stride 64, and four validation
subjects selected after activity filtering with split seed 42. HAR has nine
channels at 50 Hz; WISDM has three at 20 Hz. Model seeds vary while validation
subjects and pair seed 42 remain fixed. Channel/statistical normalization,
feature scaling, and covariance metrics fit optimizer-training subjects only.

## Run the baseline, then train, lock, and evaluate

First create the full baseline using the runner's default
`configs/verification_har.yaml` and `configs/wisdm_verification.yaml`. Both
configs specify 20 epochs; the runner disables subsampling. `--prepare` also
runs training and held-out evaluation, so this is not a download-only command.

```bash
uv run --no-project --python .venv/bin/python python -m scripts.run_experiments \
  --prepare --device auto --out-root runs/confirmed_verification
```

Then run the research stages in order, retaining the same roots and datasets:

```bash
uv run --no-project --python .venv/bin/python python -m scripts.research_verification \
  --stage train --device auto --out-root runs/research_verification

uv run --no-project --python .venv/bin/python python -m scripts.research_verification \
  --stage lock --device auto --out-root runs/research_verification

uv run --no-project --python .venv/bin/python python -m scripts.research_verification \
  --stage evaluate --device auto --out-root runs/research_verification
```

Research defaults are both `--sources har wisdm_v1`, `--epochs 40`,
`--seeds 42 43 44`, `--data-root datasets`, and
`--baseline-root runs/confirmed_verification`. Each neural run completes all
40 epochs and retains the earliest checkpoint attaining minimum validation
pair EER. Auto device selects CUDA when available, otherwise CPU;
`--device cpu` is supported. Sources, methods, and seeds run sequentially.
If changing roots, supply the corresponding `--data-root` and
`--baseline-root` consistently at every research stage.

The six neural methods per source are:

- `legacy_hard`: historical backbone with batch-hard training.
- `multiscale_hard`: multiscale backbone with batch-hard training.
- `multiscale_supcon`: multiscale backbone with supervised contrastive training.
- `multiscale_hybrid`: multiscale backbone with the hybrid objective.
- `temporal_hybrid`: temporal-bin backbone with the hybrid objective.
- `multiscale_stats_hybrid`: multiscale backbone with statistical fusion and
  the hybrid objective.

Each uses three seeds: **36 neural training runs across two sources**.
The registered plan uses 128-dimensional embeddings, width 32, dropout 0.1,
training-only normalization, balanced batches of eight identities and four
windows each, auxiliary identity cross-entropy, and small jitter/scale/roll
augmentation. The effective configs are the authoritative settings.

Each neural method supplies two validation candidates: the uniform three-seed
ensemble without postprocessing, and the ensemble with separately fitted
training-only WCCN per seed. Concatenating the seed embeddings and dividing by
the square root of three implements the uniform ensemble. Three time/spectral
feature candidates use standardization alone, automatic-shrinkage WCCN, or
WCCN with shrinkage 0.2. These add **six feature candidate fits across both
sources**, separate from the neural runs and their covariance fits. The total
is **15 validation candidates per source**.

`train` loads development data only and writes a progressively populated
validation registry. `lock` requires the complete registered candidate set,
40 epochs, and seeds exactly `[42, 43, 44]`; it rejects reduced studies. It
selects minimum validation ensemble pair EER, breaking exact ties by feature
method first, no extra covariance stage, registered method order, then name.
It hashes the registry, selected checkpoints/preprocessing/cache, development
data, and baseline checkpoint before writing the selection lock.

`evaluate` verifies those hashes and cohorts before loading research test
data, then evaluates only the locked winner and frozen baseline. Pair and
enrollment thresholds are computed from validation scores separately and
applied unchanged to their corresponding test scores; the target validation
FAR is 0.01. The selection lock fixes the candidate/artifacts, while the
calibrated thresholds are recorded in the evaluation JSON. Test outcomes must
not drive a new candidate, seed, epoch, or threshold choice. An unfavorable
locked outcome must remain in the report. The earlier baseline reproduction
does evaluate test data; the research runner's development-only claim applies
to its own training and locking stages.

Use fresh output roots for a new study. The baseline runner refuses an existing
`best.pt`; research training refuses an already locked study. Before locking,
research training can reuse completed checkpoints with matching effective
configs and verified development provenance; it does not resume optimizer
state partway through a run. Keep all artifacts intact after locking.

## Separate smoke run

For an installation check, save this minimal YAML outside the study output,
for example as `/tmp/gait_metric_smoke.yaml`:

```yaml
source: har
data_root: datasets
device: cpu
epochs: 1
steps_per_epoch: 2
validation_pairs: 100
objective: hybrid
model_args:
  architecture: multiscale
  embed_dim: 128
```

```bash
uv run --no-project --python .venv/bin/python python -m engine.train_metric \
  --config /tmp/gait_metric_smoke.yaml --out-dir runs/metric_smoke
```

This exercises training and validation only on already prepared data. It is
not a full study and produces no test score. Keep its outputs separate; do not
use a reduced runner study to unlock test access or report smoke validation
as benchmark evidence.

## Evidence and checkpoint handling

Inspect JSON evidence rather than relying on terminal summaries:

- `runs/confirmed_verification/experiment_summary.json`: executed baseline
  commands and effective-config locations.
- `runs/confirmed_verification/{verification_har,wisdm_verification}/metrics.json`
  and `evaluation/eval_verification.json`: baseline training provenance and
  original calibrated pair evaluation, respectively. The evaluation path is
  inside each source/config directory.
- `runs/research_verification/{har,wisdm_v1}/validation_registry.json`:
  registered candidates, validation metrics, seed results, cohorts, data
  fingerprint, and artifact paths.
- `runs/research_verification/{har,wisdm_v1}/selection_lock.json`: selected
  candidate and frozen artifact hashes.
- `runs/research_verification/{har,wisdm_v1}/locked_test_results.json`:
  baseline/selected pair and enrollment results, separate calibrations,
  coverage, comparison counts, and the lock hash. Neural winners also include
  descriptive per-seed test results; these are not a seed-selection rule.
- `runs/research_verification/{source}/{method}/seed_{seed}/effective_config.json`,
  `metrics.json`, and `per_epoch.jsonl`: resolved experiment inputs, training
  protocol/hardware metadata, and epoch diagnostics. Feature/covariance states
  and validation embeddings are retained as `.npz` artifacts referenced by
  the registry.

Research checkpoints declare `checkpoint_format: gait_metric_v1`. Load them
with `engine.train_metric.load_metric_encoder`, which reconstructs the encoder
and validates normalization fingerprints, or use the explicit locked research
evaluator above. The legacy `engine.evaluate` checkpoint path cannot infer
this format. Existing `EmbeddingNet`, baseline training, and legacy checkpoint
evaluation remain unchanged.

Frozen trained weights are not committed. A fresh checkout must retrain the
baseline and research seeds; fixed seeds do not guarantee bitwise identical
weights or scores across hardware/software versions. Preserve the actual
weights, effective configs, JSON/JSONL reports, preprocessing states, and
provenance together when archiving a completed study.

## Interpretation limits

This is a same-cohort comparison of the existing 20-epoch baseline against
40-epoch research training and, for neural candidates, a three-seed ensemble.
Loss, sampling, augmentation, normalization, covariance processing, and
training budget also change. The entire difference cannot be attributed to
architecture; `legacy_hard` is a useful control, not a fully matched historical
training recipe. A feature candidate may win. No state-of-the-art or
cross-session authentication claim follows from this study.

The same four-subject validation cohort supplies checkpoint selection,
candidate selection, and threshold calibration. This reuse and the small
identity count limit generalization and calibration confidence. A validation
FAR target does not guarantee that FAR on test or deployment. Overlapping
windows and shared pairs/templates make comparison counts larger than the
number of independent subjects or trials.

Enrollment builds a normalized mean template from each eligible identity's
first five ordered window embeddings, purges its next two windows, and scores
the remaining probes. Coverage and exclusions must accompany its metrics.
The purge establishes disjoint window
indices and conditional separation under the adapters' current source order
and 128/64 windowing. Raw offsets/session IDs are unavailable:
`raw_offset_nonoverlap_verified` is false. It is not an audited raw-interval
nonoverlap guarantee or proof of a later session. Both baseline and selected
candidate receive this same enrollment evaluation. Compare enrollment EER to
enrollment EER, and pair EER to pair EER; comparing the two result families
directly does not establish improvement.

## Primary method sources

These are adaptations of specific ideas, not reproductions of published
systems or transfers of their benchmark results:

- [Hermans, Beyer, and Leibe, *In Defense of the Triplet Loss for Person
  Re-Identification* (2017)](https://arxiv.org/abs/1703.07737): balanced identity
  batches and batch-hard positive/negative mining.
- [Khosla et al., *Supervised Contrastive Learning*
  (2020)](https://arxiv.org/abs/2004.11362): same-identity positives in the
  supervised contrastive objective. Here windows supply a single encoded
  view; the hybrid loss is an experimental combination.
- [Fawaz et al., *InceptionTime: Finding AlexNet for Time Series
  Classification* (2019)](https://arxiv.org/abs/1909.04939): bottlenecks and
  parallel temporal receptive fields in the multiscale backbone. Kernels,
  normalization, pooling, and objective differ here.
- [Hu, Shen, and Sun, *Squeeze-and-Excitation Networks*
  (2017)](https://arxiv.org/abs/1709.01507v1): channel recalibration adapted to
  one-dimensional gait features.
- [Hatch, Kajarekar, and Stolcke, *Within-Class Covariance Normalization for
  SVM-Based Speaker Recognition* (Interspeech 2006)](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html):
  WCCN motivates suppressing training within-identity variation. The
  standardized, regularized gait covariance metric and cosine scoring here
  adapt a speaker-recognition method; they do not reproduce its SVM system.


## Secondary HAR subject-fold diagnostic

The primary HAR neural winner failed to generalize. A separate exploratory
study selects three fixed descriptor transformations using three disjoint
subject folds over the original 17 optimizer identities. The original four
validation identities are reserved for calibration. Test results acknowledge
that the HAR test cohort was already inspected; they are not a new blind test.

```bash
uv run --no-project --python .venv/bin/python python -m scripts.cross_validate_gait_features --stage train
uv run --no-project --python .venv/bin/python python -m scripts.cross_validate_gait_features --stage evaluate
uv run --no-project --python .venv/bin/python python -m scripts.benchmark_gait_features --metric runs/har_feature_cv/metric.npz --out runs/har_feature_cv/benchmark_cpu.json
```

Outputs include all subject folds, every candidate's fold scores, a frozen
transform, calibrated threshold, artifact hashes and explicitly exploratory
test metrics. Use a fresh `--out-root` for another study. The selected transform
is fitted on the 17 training identities only; no normalization or covariance
statistics are fitted on calibration or test windows. See the
[measured report](../reports/2026-10-23-quality/README.md) for both successes and
negative results.
