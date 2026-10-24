# Exploratory DEV descriptor ablation: HAR and WISDM v1

`scripts/invariant_transfer_study.py` compares three fixed descriptors within
each existing source's TRAIN split. Despite the runner's historical "transfer"
name, it does not train on one dataset and evaluate on another. It produces
three method/source rows per source, six with `--source all`. This document
contains the protocol, not measured performance results.

## Fixed protocol

Load only `data.adapter.load_windows(source, split="train", window_size=128,
stride=64)`, with source `har` or authentic-subject `wisdm_v1`. Keep only activity
`0` (walking). HAR retains its nine physical channels and 50 Hz sampling;
WISDM v1 retains its three acceleration channels and 20 Hz sampling. HAR's
existing 128-sample windows are used as supplied; WISDM is windowed by the
existing adapter, preserving source segment boundaries. No raw-axis scaling,
resampling, or sensor synthesis is added.

Before constructing folds, retain only identities with **at least two walking
windows**. Count walking windows only, excluding nonwalking activity from this
eligibility decision. This applies consistently to both sources, including
identities that might otherwise have become calibration subjects. No singleton
identity is fitted, transformed, or scored. Preserve full source arrays and the
global encoded-to-original subject mapping unchanged for provenance.

Call `engine.subject_protocol.make_subject_folds` on eligible walking TRAIN labels
with `calibration_subjects=4, n_folds=3, seed=2026`, matching the scaling study's
helper and settings. **Filtering may change the comparison cohorts despite the
same seed.** Preserve authentic encoded IDs, including gaps.
The four reserved calibration identities appear only in provenance and window
counts. Remove them before any descriptor extraction; never fit, transform,
score, calibrate, select, or refit on them. Every other identity validates once
and trains in the other two folds. Filtering must leave enough identities for
four calibration subjects and three verification folds; otherwise the helper
fails without silently changing settings.

The parent audit reported only one walking window for WISDM original user `7`
(encoded `6`). A synthetic `1–36` global-ID universe, with the adapter's fixed
80/20 source split and the helper's seed `2026`, places encoded `6` in DEV
validation fold `2` before filtering. In this synthetic reconstruction,
calibration IDs remain `[18, 25, 28, 30]` after excluding `6`, while development
and fold membership change. Actual loaded walking IDs determine the runner's
partition; these synthetic memberships are not an audit of prior result files.
The pre-fit audit records the full unfiltered reference partition and explicit
added/removed IDs for calibration, development, and each fold so comparison
cohort changes cannot be mistaken for a matched-cohort scaling comparison.

The three candidates, in fixed order, are:

1. `axis`: `models.gait_features.extract_gait_features`, including its existing
   axis and vector-magnitude statistics.
2. `invariant`: `models.gait_invariants.extract_invariant_features`, with default
   adjacent physical-vector triads.
3. `fusion`: column concatenation `[axis, invariant]`.

Each receives a fresh `WithinClassMetric("wccn", "auto")` per fold. Fit means,
scales, and regularized within-class covariance on that fold's training
identities only. Transform only that fold's held-out development identities.
Feature extraction is label-free and shared across folds; fusion reuses both
feature blocks. Expected dimensions are HAR `312 / 192 / 504` and WISDM v1
`104 / 64 / 168` for axis / invariant / fusion.

Evaluate with the existing `verification_metrics`, explicitly passing
`n_pairs=10000, seed=42, activity_ids=walking_labels`. These are same-activity
window-pair scores on held-out identities. Every candidate uses the same fold
rows and pair-generation settings. Retain all returned metrics and every fold.
Report EER and ROC AUC as equally weighted three-fold means and descriptive
sample SD (`ddof=1`) through `engine.subject_protocol.fold_summary`. Folds share
training identities, sampled pairs reuse windows, and overlapping source
windows are dependent. SD is not a standard error or population confidence
interval. Evaluation-derived EER thresholds and operating points returned by
the existing evaluator are descriptive; no deployment threshold is calibrated.

There is no candidate winner, selection lock, calibration stage, final refit,
or TEST stage. Previously inspected HAR/WISDM cohorts do not become fresh blind
evidence through this follow-up. HAR/WISDM window-pair scores must not be pooled
with, ranked against, or called replications of Santos cross-day trial/template
claim scores. No cross-day or independent fresh performance claim is supported.

## Reproduction and artifacts

Use the parent's existing environment; no package installation is required by
the runner. From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project python -m scripts.invariant_transfer_study \
  --data-root datasets \
  --out-dir runs/invariant_transfer_dev_2026_10_24 \
  --source all
```

`--data-root` and `--out-dir` are required `Path` arguments. Source may also be
`har` or `wisdm_v1`. The output directory must not already exist, even empty;
existing artifacts are never overwritten. A failed run leaves an incomplete
directory for inspection; rerun into a new directory. Execution uses one numeric
thread, CPU descriptor/metric operations, no neural training or device options.
Importing the runner or invoking `--help` does not import Torch; execution lazily
imports the existing adapter/evaluator, which depend on Torch for CPU tensor and
pair operations.

Before any descriptor extraction or metric fit, `pre_fit_protocol.json` records
the fixed settings, package versions, platform and thread context, code source
hashes, and each source's metadata. Fingerprints cover full loaded TRAIN arrays,
walking arrays, eligible walking arrays, and actual development-only arrays,
each with labels. Arrays
are serialized as canonical little-endian C-order float32 signals and int64 IDs,
with dtype and shape in the hash and floating signed zeros canonicalized. This
is serialization normalization, not a learned normalization of sensor data.
Fingerprints identify adapter outputs, not original archive bytes; the shared
WISDM parser necessarily reads the official raw file to implement its TRAIN
subject filter. The runner never requests or evaluates the TEST split.

Metadata includes exact encoded subject lists, mappings back to original source
IDs, per-subject window counts, retained original TRAIN row indices, and every
fold's training/validation IDs and counts. `walking_eligibility` contains retained
and excluded identity/window counts, original TRAIN row indices, and a reason
for every excluded identity. `subject_protocol_comparison` reconstructs the
unfiltered reference partition for provenance only; no reference cohort is fitted
or scored. Code hashing conservatively covers
the runner, scripts package initializer, and all Python sources under `data`,
`engine`, and `models`, including lazy/transitive evaluator dependencies.
The same source hashes must hold after evaluation or no result is published.

`results.json` references the SHA-256 of the exact pre-fit protocol bytes and
includes all candidate/fold metric records, fitted shrinkage, feature dimension,
dense metric numeric storage count (`D² + 2D`, excluding subject IDs), and
equal-fold summaries. It records feature extraction, metric fitting, held-out
metric transformation, and pair sampling/scoring/metric wall time separately.
Extraction is performed once per source/block and reused, so fusion's reported
extraction cost includes both shared blocks plus concatenation. These are single
wall observations without warmup or repeated measurement. Small pair operations,
imports, caches, and order effects can dominate; no speedup, production latency,
or throughput claim follows from these timings.

Synthetic checks require no dataset access or Torch import:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project python -m pytest -q tests/test_invariant_transfer_study.py
```

## Rotation-invariant context

The executable feature schema, physical units, constant-rotation proof, and
limitations remain documented in [multisensor_features.md](multisensor_features.md).
This study reuses that implementation; it introduces no new invariance claim.
A constant rotation of a vector triad preserves norms, temporal dot products,
covariance eigenvalues, and summed vector spectral power. A moving sensor,
changed attachment, different units, or time-varying rotation need not preserve
these descriptors. The axis and fusion candidates retain orientation-sensitive
features; the fusion representation is not rotation invariant as a whole.

[Gadaleta and Rossi, IDNet, Sections 3.2–3.3](https://arxiv.org/html/1606.03238)
uses acceleration magnitude for orientation-independent cycle extraction and
then a gravity/PCA reference frame. It motivates the problem but does not
validate this fixed-descriptor protocol or its verification performance.
[Kobayashi, Hasida and Otsu, ICASSP 2011, Section 2.1](https://staff.aist.go.jp/takumi.kobayashi/publication/2011/ICASSP2011.pdf)
derives invariant Fourier correlation features whose diagonal is xyz-summed
spectral power. The reused descriptor summarizes that diagonal and omits their
cross-frequency correlations and classifier. Mathematical invariance does not
establish improved subject verification or transfer performance.
