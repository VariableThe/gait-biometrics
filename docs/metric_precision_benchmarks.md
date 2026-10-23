# Metric training precision microbenchmark

`scripts/benchmark_metric_precision.py` measures a resident synthetic training
step using the existing metric trainer helpers and encoder factory. No measured
results are recorded here. Do not run it alongside the active frozen GPU study
or CPU test suite; the parent runner will measure it after those workloads finish.

The default CUDA grid compares FP32 with BF16 encoder autocast for channels 3/9,
widths 32/96, batch 32, multiscale architecture, embedding dimension 128, ArcFace,
and the hybrid batch-hard/SupCon objective. Each case has 5 updating warmup steps
and 3 independently initialized repeats of 20 timed updating steps. Default
P8/K4 uses eight classifier classes. `--batch-sizes 32,128` adds P8/K16; P remains
8 and K is derived explicitly from each batch size. `--n-classes` can represent a
larger training identity classifier while a batch still contains P identities.

Run from `gait-biometrics` with the existing shared environment. These commands
are for a later idle-hardware measurement, not validation during the study:

```bash
VIRTUAL_ENV="$(realpath ../.venv)" TMPDIR="$(realpath ../artifacts/tmp)" \
  uv run --no-project --active python scripts/benchmark_metric_precision.py \
  --device cuda --out artifacts/metric_precision_cuda.json

# Optional larger batch and classifier; use the same settings for both modes.
VIRTUAL_ENV="$(realpath ../.venv)" TMPDIR="$(realpath ../artifacts/tmp)" \
  uv run --no-project --active python scripts/benchmark_metric_precision.py \
  --device cuda --batch-sizes 32,128 --n-classes 24 \
  --out artifacts/metric_precision_cuda_batches.json

# CPU has only strict FP32; this command also performs timed training work.
VIRTUAL_ENV="$(realpath ../.venv)" TMPDIR="$(realpath ../artifacts/tmp)" \
  uv run --no-project --active python scripts/benchmark_metric_precision.py \
  --device cpu --out artifacts/metric_precision_cpu.json
```

`--out` is required and contains JSON, including resolved trainer configuration,
hardware/software metadata, raw repeat results, repeat summaries, and protocol
constraints. `--help` imports only the standard library. CPU defaults to
`--precision float32` and rejects explicit `both`/`bfloat16` before importing
torch. CUDA defaults to `both` and rejects an unavailable CUDA device or BF16
without native hardware support; it never silently substitutes another device
or precision. `--precision float32` permits a CUDA FP32-only run. TF32 is disabled
for both matmul and cuDNN; cuDNN autotuning is disabled and these choices are
recorded. No guarantee of bitwise determinism is made.

## Work and timing boundaries

The script calls `engine.train_metric._resolved_config`, `_build_classifier`,
and `_training_losses`, and `models.gait_encoders.build_gait_encoder`. Model
arguments and trainer loss/optimizer defaults come from the existing code. It
does not change the registered trainer, models, study, checkpoints, or data.

Before moving the encoder to the selected device, each fresh encoder explicitly
fits normalization in train mode on 32 separate, small synthetic TRAIN-only CPU
windows. Those windows use a dedicated seeded generator and are distinct from
the benchmark batch. There is no validation/test data or normalization leakage.

Each repeat recreates the encoder, ArcFace classifier, AdamW optimizer, and
synthetic batch with the same fixed seed. CPU state hashes must match across
precisions and repeats within each architecture/shape case. Parameters,
normalization, raw input, labels, row IDs, batch shape, loss settings, dropout,
gradient clipping, learning rate, and weight decay are held constant between
the paired modes. Every batch row has a distinct original row ID, so the
trainer's duplicate-row mask retains real same-identity positives. The same
resident synthetic batch is reused at every step; no sampler is timed.

Both modes keep parameters, raw inputs, losses, classifier calculations, and
AdamW state in FP32. BF16 autocast surrounds only encoder forward, matching the
trainer path. A hook on the first eligible convolution/linear layer verifies
its actual compute output dtype during the first untimed warmup, then is removed
before subsequent warmup and timing. Warmup also records the final embedding
dtype, which may be FP32 because final normalization promotes its result under
autocast. There is no gradient scaler.
Warmup performs real optimizer updates and initializes AdamW state before
measurement. The first untimed warmup checks finite embeddings, all loss
components, and every parameter gradient. Timed steps retain the trainer's
finite-loss check and `clip_grad_norm_(error_if_nonfinite=True)` without extra
per-parameter or embedding finite checks.

Timing uses `time.perf_counter()` around the entire step block, with CUDA device
synchronization immediately before and after it. Total wall time is stored in
seconds; milliseconds per step are `elapsed_seconds * 1000 / steps` and windows
per second are `batch_size * steps / elapsed_seconds`. This includes Python
dispatch and the existing trainer-path host synchronizations. Summary statistics
describe the independent repeat-level means, not a distribution of individual
step latencies. The median throughput is derived from median milliseconds per
step; no FP32/BF16 speed ratio is asserted by the script.

Included work: `zero_grad(set_to_none=True)`, encoder forward in train mode,
metric and classifier losses, trainer finite-loss guard, backward, gradient
clipping, and AdamW updates. Excluded work: synthetic CPU construction, model
initialization, normalization fitting, transfers, dataloading, augmentation,
scheduler, trainer scalar/collapse logging, validation, and checkpoints. Final
loss conversion to a Python number occurs after timing. This is a microbenchmark
of the training compute path, not full trainer/epoch wall time.

## Memory and interpretation

For CUDA, each repeat releases unused cache before setup, warms up, clears
gradients, synchronizes, and resets allocator peaks just before measurement.
Baseline and peak allocated/reserved values are reported in **bytes**. The
baseline includes the resident model, classifier, batch, and initialized AdamW
state. Reserved memory includes cached warmup workspace; peaks are absolute
process allocator values during the measured interval, not a subtraction from
baseline. Per-case summaries take the maximum peak across repeats. CPU memory
fields are `null` because this script does not measure CPU peak memory.

Allocator counters do not describe total GPU usage, other processes, or all
driver/cuDNN allocations. Idle hardware, thermal/power state, and competing
activity can change timings; record those external conditions with any result.
Synthetic resident throughput excludes input-pipeline costs and cannot predict
full-study duration. Same initial state and seed do not imply identical FP32 and
BF16 update trajectories. Finite checks and timings support capability and
microbenchmark observations only, not accuracy, verification quality,
convergence, or training equivalence claims. No optional embedding diagnostic
is run by this implementation.
