# Checkpoint-based gait encoder inference benchmark

`scripts/benchmark_gait_encoders.py` measures the frozen legacy encoder, the
validation-selected neural ensemble, and its first selected member on identical
synthetic windows. It does not train, load train/test windows, calibrate a
threshold, or calculate biometric accuracy. No runtime measurements have been
produced for this document; speed, efficiency, and energy improvements are not
claimed.

Run from the repository with its existing dependencies. Do not install another
environment to run the benchmark. Substitute the actual study source folder:

```bash
uv run python scripts/benchmark_gait_encoders.py \
  --lock runs/NEW_STUDY/har/selection_lock.json \
  --baseline-checkpoint runs/confirmed_verification/verification_har/best.pt \
  --device cpu --torch-threads 2 --batch-sizes 1,128 \
  --repetitions 30 --warmup 5 --out runs/NEW_STUDY/har/inference_cpu.json
```

For WISDM use the `wisdm_v1` source folder and frozen checkpoint
`runs/confirmed_verification/wisdm_verification/best.pt`. `--lock` also accepts
the source folder directly. For CUDA use `--device cuda` and a separate output
file, after training and other GPU workloads finish. `--device auto` selects
CUDA when available; the default is CPU. On a busy CPU or for an initial check,
use `--repetitions 3 --warmup 1`; retain the actual count when reporting results.
Batch sizes are bounded to 1–512, with at most eight distinct sizes; repetitions
are bounded to 1–100 and warmup batches to 0–20. The default CPU thread count is
two, with one interop thread. Do not benchmark while the host is swapping or
while a competing workload is active.

The lock's registry and every artifact hash are checked without opening source
datasets. Every selected checkpoint and optional postprocessor must be covered
by the lock. The baseline digest is checked against `baseline_sha256` when that
field exists; older locks without it are explicitly identified in the output,
which always records the supplied baseline's digest. Source, channel count,
128-sample window length, and development subject cohorts must agree across
checkpoints. A feature-method winner fails with a clear error: its cost cannot
be represented by silently substituting a neural candidate.

The three reported cases have distinct costs:

- `old_baseline_raw` strictly restores `EmbeddingNet` from the supplied frozen
  checkpoint, removing the Siamese `embedding.` prefix when present. Its raw
  input path has no new normalization. Its ordinary forward never calls the
  saved denoising decoder.
- `selected_ensemble` restores every seed in lock order through
  `engine.train_metric.load_metric_encoder`. Each complete encoder retains its
  frozen channel, magnitude, and statistics buffers and its actual forward
  processing. Every member runs on the same input. Their embeddings are
  concatenated and divided by the square root of the number of members.
- `selected_single` uses the first seed in lock order, with that member's
  selected postprocessing. This is a computation ablation, not a newly selected
  winner or an accuracy result for a deployed single-model system.

Optionally add `--pack-multiscale` to include `selected_ensemble_packed` and
`selected_single_packed` alongside all three ordinary cases. This requires a
multiscale selection. Each frozen, eval-mode member encoder is converted with
`models.gait_inference.pack_multiscale_branches`; its WCCN wrapper and the
ensemble concatenation/scaling are preserved. The original models and ordinary
checkpoint loading (`weights_only=True`) remain unchanged. Packed models are
inference-only runtime copies, not a resumable training checkpoint format.

Packing combines branch convolutions using zero-padded kernels. It increases
stored parameters and nominal FLOPs, including the zero-padded coefficients,
while reducing convolution dispatches. Whether that tradeoff improves latency
depends on the device, backend, and batch shape; no speed improvement is claimed.
Before any timing, the complete original and packed ensemble outputs and single
member outputs are compared for **every requested input batch shape**, with
autocast disabled. Nonfinite outputs, shape mismatches, or a maximum absolute
difference greater than `1e-5` fail the run. Each packed case's batch record
contains `max_abs_diff` and `reference_case`; these checks are outside timing.

For `wccn_auto`, each member applies its own saved centering, scaling, projection,
and L2 normalization before concatenation. The benchmark expresses this frozen
transform in Torch on the selected device, retaining the reference transform's
float64 arithmetic and float32 output. This includes its computation and buffer
storage but excludes the research runner's CPU/NumPy round trips. Unknown
postprocessing fails. A selection with `none` adds no WCCN transform.

A three-seed ensemble executes **three complete model forwards per input
window**, plus any three postprocessors and concatenation/scaling. It stores
all three members. A smaller individual encoder does not establish a faster or
smaller ensemble. Measured ensemble latency and aggregate storage must be used
for comparison; runtime is not assumed to be exactly three times the single
member. The report records the actual member count rather than assuming three.

One seeded, finite, contiguous float32 tensor is generated with shape
`(max_batch_size, C, 128)`, using native HAR `C=9` or WISDM `C=3`. Every case
uses the same prefix slices. Generation, hashing, model loading, and the single
input transfer happen before timing; the input remains on the selected device.
The input hash identifies the full generated tensor. To compare CPU and CUDA,
use the same seed, batch list, checkpoints, and Torch version. These random
inputs characterize execution rather than the source signal distribution.

Models are frozen and evaluated under `torch.inference_mode()`, without
autocast, TF32, or `torch.compile`. After an untimed finite-output check and the
requested warmup, each sample measures one complete batch forward using
`perf_counter`. CUDA synchronizes immediately before and after each forward;
these are host-observed completion latencies including dispatch and completion
wait, rather than CUDA-event kernel-only times. There are no transfers inside
the timed interval. Outputs are discarded between samples.

Each case reports median and interpolated p95 batch latency in milliseconds,
the raw samples, and windows per second as
`batch_size * repetitions / sum(batch_seconds)`. It also reports
`batch_size / median(batch_seconds)` separately. One ensemble input window
counts once even though every member processes it. Batch one describes
single-window latency; batch 128 describes a batched workload. A p95 from only a
few repetitions is descriptive and should not be treated as a reliable tail
latency estimate.

Parameter counts and tensor bytes are reported for each member and the whole
ensemble, and separately for every actual benchmark case. Packed case counts
include zero kernel padding; they are not copied from the original models.
`parameter_bytes` counts all restored parameters, including the
legacy decoder; `parameters_unused_dae_decoder` identifies parameters excluded
from ordinary inference. `forward_parameters` subtracts that decoder count.
`buffer_bytes` includes encoder normalization, running statistics, indices, and
selected WCCN state. `tensor_state_bytes` is parameter plus buffer bytes, not a
CUDA allocation estimate or serialized checkpoint size. Checkpoint file sizes
are separate because training checkpoints can contain optimizer state and
metadata. The single case shares its loaded member with the ensemble; it is
not an extra model copy. When packing is enabled, the packed single case likewise
shares its first member with the separate packed ensemble. Both original and
packed ensembles are resident during that run.

CUDA reports optional-to-interpret incremental peak allocated bytes: inputs and
all models are already resident, warmup is synchronized, and peak counters are
reset before sampling. The peak delta includes temporary activations and
outputs above that resident allocation. It excludes reserved allocator memory,
driver/library allocations, and model/input residency itself. All models are
co-resident during the run, so this is not a standalone deployment memory
measurement. CPU memory is not measured.

Publish actual JSON measurements with hardware, software, thread count,
checkpoint hashes, input shape, repetition count, and competing-workload context.
Compare cases separately for each source/device/batch, including the optional
packed cases when enabled. There are no
energy measurements; latency, parameter count, and memory are not energy
measurements. Biometric test scores belong to the separate locked study
evaluation and are not inputs to this benchmark.
