# Opt-in identity metric learning

These methods adapt image metric-learning ideas to inertial gait windows.
They are not exact paper reproductions: architectures, data, augmentations,
and optimization differ. Better authentication metrics remain an experimental
hypothesis. Existing training and evaluation paths are unchanged.

## Objectives and research basis

[Hermans, Beyer, and Leibe, *In Defense of the Triplet Loss for Person
Re-Identification* (2017)](https://arxiv.org/abs/1703.07737) motivates balanced
P×K batches and hardest-positive/nearest-negative mining within a batch.
For each valid anchor, `batch_hard_triplet_loss` uses Euclidean distances and
`max(0, d_positive - d_negative + margin)`, averaged across anchors. The
default margin is 0.2. With `softplus: true`, the loss instead uses
`softplus(d_positive - d_negative)` without a margin.

[Khosla et al., *Supervised Contrastive Learning*
(2020)](https://arxiv.org/abs/2004.11362) motivates the outside-log mean over
same-class positives (Equation 2). Here classes are real training identities;
different sensor windows supply positives. Normalized dot products are
divided by temperature (default 0.1). Each anchor's loss is the log-sum-exp
over allowed other windows minus its mean positive logit. This adaptation
uses one encoded view per window and no separate projection/pretraining stage.

`objective` accepts `batch_hard`, `supcon`, or `hybrid`. Hybrid explicitly uses
`0.5 * (batch_hard + supcon)`; that combination is an experimental choice.
An optional linear identity head adds `ce_weight * cross_entropy`, default
weight 0.5 and label smoothing 0.05. Only optimizer-training IDs are remapped
to consecutive head labels. Validation IDs never enter its vocabulary or loss.

Both objectives exclude self-positives and copies of the same original row.
Anchors without a distinct-row positive or a negative are skipped. If none
is valid, the differentiable zero has finite gradients. Direct `torch.cdist`
avoids a manually differentiated singular square root at coincident points.
SupCon supports zero embeddings without NaNs.

## Sampling and small cohorts

`IdentityBalancedBatchSampler` draws P distinct identities and K windows
each, with P≥2 and K≥2 (defaults 8×4). Shuffled identity cycles cover eligible
identities before restarting. Within each identity rows are unique when
possible; when fewer than K exist, all are used before replacement.

Singletons cannot provide distinct-row positives and are excluded from the
P×K pool. Their exact IDs/counts and the resulting negative-coverage limitation
are recorded in `protocol.sampler`. The classifier vocabulary still records
all training IDs, although excluded singletons supply no batches. Actual P
is capped to the number of eligible identities and recorded alongside
requested P. Fewer than two eligible identities raises an error.

Default steps are `ceil(N_train / (actual_P * K))`, including singleton rows
in N. An explicit shorter step budget may not cover all eligible identities;
the sampler reports whether its epoch budget guarantees identity coverage.

## Leakage controls and selection

The trainer loads only the original source `train` split. It checks finite
float32 `(N,C,T)` windows and label dimensions (HAR nine channels, `wisdm_v1`
three). It filters `activity_ids` first (default walking `[0]`), then calls
`engine.train._split_validation(data, val_subjects, split_seed)`. This exactly
matches the walking baseline: identities lacking walking windows do not enter
the validation lottery. No additional window subsampling occurs.

The default validation cohort has four subjects selected by `split_seed=42`.
Training seed `seed` (alias `seedmodel`) defaults to 42. The separate pair
seed `val_seed=42` stays fixed across training seeds 42, 43, and 44. Changing
any cohort, activity, or pair setting must be declared in comparisons.

`encoder.fit_normalization(X)` receives only optimizer-training windows,
after filtering and splitting, before device transfer. It is invoked even
with input normalization disabled so optional statistics-fusion moments are
still fitted on training data. Mean/std tensors and the fitted flag are
embedded checkpoint buffers. Inference never fits them. A SHA-256 fingerprint
records their exact tensor bytes, names, shapes, and dtypes.

Each epoch embeds unaugmented validation windows in bounded batches and calls
the existing `engine.evaluate.verification_metrics` with real identity and
activity labels, 10,000 pairs by default, and `val_seed`. Pairing is
same-activity. Strict validation-EER improvement saves the best checkpoint;
equal EER retains the earliest epoch. Training and checkpoint loading never
access test windows, labels, metrics, or thresholds.

These results describe held-out-subject **within-recording discrimination**.
Overlapping windows from one recording can be strongly related; this does
not establish independent-session enrollment/probe authentication. Evaluate
and report an independent enrollment protocol separately.

Select architectures/objectives, optional train-only covariance whitening,
and any uniform multi-seed ensemble using validation only. Freeze candidate
configurations, checkpoint paths, and calibrated thresholds before a locked
test evaluation. Test results must not drive further candidate selection.

## API and configuration

```python
from engine.train_metric import load_metric_encoder, train_metric

metrics = train_metric(config, out_dir="runs/metric_candidate")
encoder, protocol, resolved_config = load_metric_encoder(
    "runs/metric_candidate/best.pt", device="cpu"
)
# Eval-mode encoder consumes raw (N,C,T) windows and returns unit embeddings.
```

```yaml
source: har                    # or wisdm_v1
data_root: datasets
epochs: 40
seed: 42                       # vary 42/43/44 with fixed seeds below
split_seed: 42
val_seed: 42
val_subjects: 4
activity_ids: [0]
window_size: 128
stride: 64
model_args:
  architecture: multiscale     # legacy | multiscale | temporal
  embed_dim: 128               # trainer default is 64
  width: 32
  dropout: 0.1
  stats_fusion: false
  normalization: true
objective: hybrid             # batch_hard | supcon | hybrid
identities_per_batch: 8
windows_per_identity: 4
# steps_per_epoch: omitted -> ceil(N_train / (actual_P * K))
margin: 0.2
softplus: false
temperature: 0.1
ce_weight: 0.5                 # zero disables the classifier
label_smoothing: 0.05
lr: 0.001
weight_decay: 0.0001
gradient_clip: 1.0
augment: {}                   # identity by default
# Optional: {jitter_std: 0.01, scale_std: 0.05, roll_fraction: 0.05}
# Also accepts rotation_degrees and vector_groups.
validation_pairs: 10000
eval_batch_size: 256
device: auto
torch_threads: 2
```

AdamW uses epoch-wise cosine decay and gradient clipping at norm 1 by default.
Augmentation runs only on optimizer batches through
`data.gait_transforms.augment_gait`, within individual windows. The supported
circular-shift key is `roll_fraction`; `time_shift` is not valid.
`python -m engine.train_metric --config CONFIG.yaml --out-dir RUN_DIR` provides
the opt-in CLI; `--epochs` overrides duration. Use the shared uv environment.

## Checkpoints and diagnostics

`best.pt` contains `checkpoint_format: gait_metric_v1`, raw unprefixed encoder
`model_state_dict`, optional separate `classifier_state_dict`, epoch,
`train_seed`, `split_seed`, `val_seed`, JSON-compatible resolved `config`,
hardware metadata, and `metrics.protocol`. Protocol includes exact train/val
IDs, classifier mapping, source/dimensions, activity filter, sampler exclusions,
normalization fingerprint, objective, model args, configuration, and selection
rule. Optimizer resume is not implemented.

`load_metric_encoder(checkpoint, device='cpu')` uses
`torch.load(..., weights_only=True)`, constructs the declared encoder, strictly
loads its raw state, verifies normalization fingerprints, and returns
`(encoder.eval(), protocol, config)`. The classifier is not needed for
inference. The legacy evaluator intentionally cannot load this format;
use the explicit loader in the locked candidate evaluator.

`metrics.json` records the full report; `per_epoch.jsonl` flushes each epoch.
Logs contain total/hard/SupCon/CE loss, collapsed-pair fraction (distance
≤1e-4), gradient norm before clipping, learning rate, validation EER/AUC,
and elapsed time. Collapse statistics count distinct batch positions, so
repeated rows in undersized classes can increase that diagnostic. Hardware
metadata records device, CUDA/Torch versions, platform, and thread count.

## Focused checks

`tests/test_metric_learning.py` checks analytic losses, self/duplicate-positive
exclusion, finite zero/random gradients, deterministic P×K coverage, singleton
reporting/replacement, and tiny mocked training with real encoders. It checks
training-only fitted moments, walking-before-split behavior, weights-only
loading and fingerprint tampering, earliest validation ties, and fixed
validation identities/pair seeds across three model seeds. All run on tiny
synthetic windows with one Torch thread; no dataset, test split, or GPU is used.
