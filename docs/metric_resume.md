# Interrupted metric training

Metric training can opt in to restart at a committed epoch boundary. The total
epoch budget must be chosen before the first run; it is also the cosine
scheduler's horizon. Resume continues that same run and cannot extend it.

```bash
# Fresh run, with the complete intended budget in the YAML (or --epochs).
uv run --python ../.venv/bin/python python -m engine.train_metric \
  --config configs/metric_har.yaml --out-dir runs/metric-resumable --resumable

# After interruption: identical config, budget, source checkout and output path.
uv run --python ../.venv/bin/python python -m engine.train_metric \
  --config configs/metric_har.yaml --out-dir runs/metric-resumable --resume
```

The Python equivalents are `train_metric(config, out_dir, resumable=True)` and
`train_metric(config, out_dir, resume=True)`. Resume implies resumable output.
The default call still writes `best.pt`, `per_epoch.jsonl` and `metrics.json`
without `last.pt`. Existing `gait_metric_v1` inference checkpoints and
`load_metric_encoder` retain their format and behavior. Integer budgets and
sampling dimensions now reject fractions, booleans and strings instead of
silently truncating or coercing them.

## Saved state and recovery

An opt-in run writes `last.pt` before epoch one, then after each completed epoch,
including validation, selection and the scheduler step. Each write uses a
flushed temporary file in the output directory followed by atomic replacement.
Only one training process should write to an output directory at a time.

`last.pt` has format `gait_metric_resume_v1` and contains tensors and plain
numeric/JSON-compatible containers. It loads with `torch.load(...,
weights_only=True)`; NumPy arrays or executable custom objects are not pickled.
It records:

- Current encoder and optional classifier parameters and buffers; AdamW state;
  cosine scheduler state. FP32/BF16 training does not use a gradient scaler, so
  `scaler_state_dict` is explicitly `None`.
- CPU and, for CUDA training, every CUDA generator state; Python and NumPy RNG
  states, including the cached NumPy Gaussian. Dropout and augmentations resume
  from these states after model/optimizer reconstruction.
- The sampler seed, next epoch, configuration and next iterator's Python RNG
  state. The current sampler starts a local `Random(seed + epoch)` each epoch;
  there are no pending sampler draws at a saved boundary. Resume validates this
  state and reconstructs that exact next iterator.
- Completed epoch count, full epoch history, selected epoch/EER and an embedded
  copy of the selected inference checkpoint. Validation-EER ties retain the
  earliest epoch, including across interruptions. With `epoch_selection: final`,
  no best snapshot exists until the final intended epoch.
- Full resolved configuration and its SHA-256; SHA-256 hashes of Python source
  files under `engine/`, `models/`, and `data/`; fingerprints of the loaded
  source TRAIN tensors, optimizer TRAIN cohort, and validation cohort; complete
  protocol/cohort metadata and normalization fingerprint. Tensor hashes include
  roles, shapes, dtypes, row order, values, activity names, subject metadata and
  sampling rate. These hash the loaded representation, not raw archive bytes.
- Resolved output/data-root paths and runtime/device details, including Python,
  NumPy, PyTorch, CUDA, thread count and determinism/precision settings.

Resume checks configuration, source, data, cohort/protocol, runtime and directory
provenance before changing output artifacts. Different epochs, seeds, settings,
subject roles, data values/order/interpretation, source files or resolved paths
are refused. A copied `last.pt` cannot be resumed in a different directory.
The source match intentionally includes dependencies beyond the chosen encoder.
Restore the original environment/source for a run; start a new output directory
when changing an experiment.

`last.pt` is authoritative. If interruption leaves an extra or partial log row,
or an absent/stale/damaged `best.pt`, resume reconstructs those derived artifacts
from the committed snapshot. It restores the selected best weights separately
from the latest optimizer weights. It appends each remaining epoch once and
rebuilds final `metrics.json`. A fully completed run performs no further
optimization or validation. Work since the last committed boundary is repeated;
completed epochs are retained. Wall-clock timings include actual execution and
can differ between uninterrupted and restarted runs.

A fresh resumable run refuses existing training artifacts. A default call also
refuses to overwrite a directory containing `last.pt`. `best.pt` alone, a legacy
checkpoint renamed to `last.pt`, or a missing/corrupt resume snapshot cannot be
used for optimizer resume. There is no migration from historical inference-only
checkpoints: optimizer/RNG/scheduler state cannot be reconstructed from them.
Training continues to load only the source TRAIN split; this feature adds no
CALIB/TEST access or new study results.

## Verification and historical replay

The tests patch `load_windows` with generated TRAIN tensors and use one CPU
thread. They compare split versus uninterrupted training at the same total
budget, with dropout, augmentation, an optional classifier, AdamW and cosine
scheduling. Parameters, optimizer/scheduler/RNG states, history and epoch
selection match exactly. A fixed observational clock permits exact history
comparison without making runtime claims. Tests also exercise uncommitted epoch
rollback, initial/final boundaries, ties, log/best repair, safe loading, atomic
write failure, mismatch refusal and the unchanged default checkpoint contract.
CUDA RNG bookkeeping is tested with mocks; actual CUDA training equivalence has
not been measured. CUDA kernels may require deterministic settings for exact
numerical replay even on an otherwise matching runtime.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_metric_resume.py tests/test_metric_cohorts.py
```

Historical studies pin source-file hashes, including `engine/train_metric.py`.
These implementation changes legitimately change those hashes. Do not replace
the hashes in frozen registrations or reinterpret old results as new runs.
Replay a historical study from its original source commit in a separate
checkout, verify its registered hashes there, and use its original environment
and artifacts. The pre-change checkout for this implementation was
`4014d85e3c1962c3d27f5f425b2b6cc6e7fc6175`; older registrations may require an
earlier commit recorded by their own provenance. For example, after choosing the
registration's actual source commit:

```bash
git worktree add --detach ../gait-historical-replay <historical-source-commit>
```

This is a replay instruction, not an operation performed by the resume tests.
New studies should register the new implementation hashes before any fitting
or model selection.
