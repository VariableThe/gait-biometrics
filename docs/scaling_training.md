# Subject-disjoint metric training and scaling

`engine.train_metric.train_metric(config, out_dir)` keeps its existing defaults:
linear identity classification, FP32, and the earliest epoch attaining the lowest
held-out TRAIN-subject validation EER. Only the source's original `train` split is
loaded. Activity filtering (walking by default) happens before any subject split.

## Explicit cohorts

Supply both `train_subject_ids` and `val_subject_ids`, or omit both for the seeded
legacy split. Each list needs at least two unique nonnegative integer IDs. Boolean,
fractional and string IDs are rejected; lists are sorted without remapping real
identities. The cohorts must be disjoint, and every requested identity must have
rows after activity filtering. A missing ID, including one with only nonwalking
rows, is an error. `val_subjects` applies only to the legacy seeded split.

Available TRAIN identities outside both lists remain unused. Reserve independent
calibration subjects this way: do not place them in either trainer list. The
trainer does not calibrate a deployment threshold. Its validation cohort is for
monitoring or model selection, so a cohort passed here is not an independent
calibration cohort. Normalize inputs using only the explicit training windows.
The identity-balanced sampler's existing singleton exclusions remain visible in
protocol metadata; exact cohort availability does not imply sampler eligibility.

For example, with real source IDs chosen by a separate TRAIN-only fold protocol:

```yaml
source: har
train_subject_ids: [0, 1, 2, 3, 4, 5]
val_subject_ids: [6, 7]
epochs: 40
epoch_selection: final
objective: hybrid
classifier_type: arcface
angular_scale: 30.0
angular_margin: 0.3
precision: float32
model_args:
  architecture: multiscale
  width: 32
  embed_dim: 64
```

The IDs above illustrate syntax; actual IDs must come from the selected source's
post-filter TRAIN pool. Keep the original test cohort out of candidate, epoch,
width, loss, precision and hyperparameter selection, especially if its results
have already been inspected. A validation improvement need not generalize to new
subjects; report fold variation and negative results.

## Fixed training budget

`epoch_selection: validation_eer` is the backward-compatible default.
`epoch_selection: final` trains exactly `epochs` epochs and saves the last epoch,
even when an earlier epoch has lower validation EER. Validation embeddings and
verification diagnostics are still computed each epoch, but do not influence
optimization, scheduling, stopping or checkpoint choice. For the planned width
32/96 × linear/ArcFace comparison, predeclare 40 epochs and hybrid loss for every
neural run; compare the final-epoch validation scores across candidates.

The checkpoint filename remains `best.pt` in both modes. `selected_epoch` and
`selected_val_eer` describe the retained checkpoint. Historical `best_epoch` and
`best_val_eer` are aliases for those values; in final mode they do not describe
the observed validation minimum. `observed_min_val_eer` is a reporting diagnostic.
The protocol records `epoch_selection`, `selection_metric` (`val_eer` or
`final_epoch`), `validation_used_for_selection`, and the tie policy.

## Identity classifier adaptation

`classifier_type: arcface` uses
`models.angular_margin.AngularMarginClassifier(in_features, n_classes, scale, margin)`
instead of the default linear head. Sorted training IDs map to contiguous class
indices. Labels are passed only during training; validation uses the encoder
alone and never applies cross entropy to unseen subjects. `ce_weight: 0` omits
the head. Scale must be positive and margin must be in `[0, π/2)` radians.

[ArcFace](https://arxiv.org/abs/1801.07698v3) introduces a target-class angular
margin with normalized embeddings and classifier weights. This implementation
adapts that classification objective to inertial gait embeddings alongside the
existing metric loss and configurable label smoothing. Its monotonic fallback
and finite endpoint gradients are implementation choices, not a reproduction of
the face-recognition experiment or evidence of improved gait generalization.

## Optional BF16 benchmark

`precision: bfloat16` requires a CUDA device for which
`torch.cuda.is_bf16_supported()` returns true; unsupported requests fail before
loading data. The check runs on the explicitly requested GPU. FP32 remains the
default. The encoder forward uses BF16 autocast; embeddings are cast to FP32 for
distance losses, angular/linear classifier logits, cross entropy and reductions.
Parameters and optimizer state stay FP32. Backward runs outside autocast, followed
by gradient clipping with `error_if_nonfinite=True`; no GradScaler is used.
All validation embedding forwards run with autocast disabled in FP32.

These choices follow [PyTorch's autocast contract](https://docs.pytorch.org/docs/2.9/amp.html):
explicitly disable autocast and cast lower-precision outputs when a region must
execute in FP32. The BF16 capability query's default may include emulation; it is
a supported-execution gate, not a claim of native Tensor Core acceleration.
Measure representative wall time, throughput, memory, and final validation
behavior on the target device before adopting BF16. No compile path or speedup
claim is introduced here.

## Saved evidence and compatibility

`per_epoch.jsonl`, `metrics.json`, and the v1 checkpoint retain complete completed
epoch histories, exact training/validation/unused subject sets, model/classifier
configuration, precision and selection policy. The checkpoint still contains raw
encoder keys, the optional training classifier state, train-only normalization
buffers and their fingerprint. Its `epoch` is the retained epoch;
`completed_epochs` records the full run even when an earlier epoch was retained.
The existing safe `load_metric_encoder` restores old and new `gait_metric_v1`
checkpoints without constructing the classifier or refitting normalization.

`tests/test_metric_cohorts.py` covers these boundaries with tiny synthetic CPU
tensors. A real CUDA BF16 benchmark and remote integration checks belong to the
parent runner; CPU contract tests do not establish GPU speed or numerical parity.
