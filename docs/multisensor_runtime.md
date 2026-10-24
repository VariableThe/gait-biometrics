# Frozen multisensor embedding runtime

`scripts/benchmark_multisensor_methods.py` measures the registered baseline and
the winner in an existing `scripts.multisensor_study` **final `lock.json`**.
It restores fixed final artifacts and never fits, trains, selects a winner,
calibrates a threshold, or scores biometric trials. A completed final lock is
required; CV artifacts alone are insufficient. Runtime results must be generated
after the parent study completes that lock. No runtime measurements accompany
this implementation.

Run from `gait-biometrics` in the existing prepared environment, retaining the
archive argument spelling used at registration:

```bash
study_python='/home/sherlock/Code/github/VariableThe/.venv/bin/python'
study_archive='datasets/santos/santos-v7.zip'
study_root='artifacts/santos-multisensor-exploratory-v1'

uv run --no-project --python "$study_python" python -m scripts.benchmark_multisensor_methods \
  --archive-path "$study_archive" --study-root "$study_root" \
  --out reports/multisensor-runtime-cpu.json --device cpu --repeats 30

uv run --no-project --python "$study_python" python -m scripts.benchmark_multisensor_methods \
  --archive-path "$study_archive" --study-root "$study_root" \
  --out reports/multisensor-runtime-cuda-request.json --device cuda --repeats 30
```

`--archive-path`, `--study-root`, and `--out` are required paths; `--device`
accepts `cpu` (default) or `cuda`. Repeats default to 30 and must be between 5
and 1,000. Each role has batch sizes 1 and 32, five untimed warm-up calls per
batch size, and exactly `repeats` timed calls. Each call handles one batch;
there is no unbounded duration loop. The output is exclusively created and
an existing result is never overwritten. `--help` imports neither Torch nor
the study's scientific dependencies.

The runner first calls `check_registration`, checking the registered plan,
study source hashes, and official archive audit. It then checks the lock's
`self_hash`, registration hash, **every** locked artifact hash, selected method
identities, and final-artifact references before any DEV numerical load. Archive
auditing reads compressed archive bytes and member inventory; it does not parse
CALIB or TEST sensor values. `study.development` loads only the original fifteen
DEV users and verifies the registered data fingerprint:
`7,16,11,9,29,27,5,13,3,1,25,24,14,19,17`.

Thirty-two real windows are selected without replacement, taking the first
available row from each DEV user in that order, then the second, then as many
third rows as needed. Batch 1 uses the first selected row; batch 32 spans all
fifteen users. The same resident inputs are reused for every repeat and both
methods. The JSON records original DEV row indices and subjects. This is a
small, warm resident-input microbenchmark; it does not sample the entire DEV
population's latency distribution. CALIB and TEST numerical arrays are never
loaded. Historical TEST inspection is already known, as documented in the
[study protocol](multisensor_study.md); runtime timing creates no new independent
accuracy evidence.

Descriptor inference uses the existing
`descriptor_features({"X": batch}, descriptor_spec)` wrapper and the restored
`WithinClassMetric.transform`. Every timed call starts with raw `(B,6,256)`
float32 windows in host RAM and includes channel selection, the selected
axes/invariant/fusion features, fixed WCCN standardization/projection, and L2
normalization. It recomputes features on every call. The metric is restored once
outside timing, with its training identities checked against DEV. Feature
arithmetic follows the existing NumPy helpers; WCCN uses float64 internally and
returns float32 embeddings. **Descriptors always report `actual_device: cpu`
and `backend: numpy_cpu`, including under `--device cuda`.** Such a result is
CPU execution and provides no CUDA speedup measurement. If both selected
methods are descriptors, Torch is never imported and CUDA is never queried.

If the winner is neural, both seed-42 and seed-43 final encoders are restored
once using `load_model`, including training-record and DEV-input checks.
Normalization remains fixed. Raw float32 Torch input tensors are prepared on
the requested device before timing. Each inference-mode call slices the
registered channels, runs **both** encoders, concatenates their 64-dimensional
embeddings, and divides by `sqrt(2)`, yielding one 128-dimensional embedding
per window. The timer therefore includes the full ensemble cost. Both forwards
run sequentially on the same device. No AMP or TF32 is used. Torch outputs and
ensemble arithmetic remain float32 on that device; the study's host NumPy
assembly may have different scalar-promotion rounding. This benchmark measures
embedding execution and does not rerun accuracy evaluation.

For CUDA neural cases, `torch.cuda.synchronize` runs immediately before each
timer starts and before it stops, so batch wall time includes Python dispatch,
GPU computation, allocation, and completion synchronization. **H2D is excluded**:
the full raw batch is already resident on the GPU. **D2H is excluded**: embeddings
remain on the GPU throughout timing and are copied to CPU only for untimed
finite/unit-norm validation. A neural CUDA request fails if CUDA is unavailable;
it does not silently fall back. The baseline descriptor remains CPU in the
same result. The runner calls neither `final_embeddings` (which reloads fitted
artifacts each call) nor the host-loop `encode` helper. Identical baseline and
winner artifacts share one restored runner, with each role timed separately.

CPU numerical thread pools and neural Torch intra-op threads are limited to two.
The result records thread pools, platform/Python/NumPy versions, Torch/CUDA
versions and GPU name when applicable, selected artifact hashes, lock and
registration hashes, and this benchmark script's own hash. The new script and
this guide are outside the frozen study's `SOURCES` tuple; adding them does not
change the study source bytes or its registration. Keep hardware load and
power settings comparable when comparing reports from different runs.

`batch_latency_ms_median` and `batch_latency_ms_p95` summarize **whole-batch**
wall time in milliseconds. P95 uses NumPy's default linear interpolation on
the per-call samples; five or thirty samples provide only a coarse tail estimate.
Raw batch samples are saved. `windows_per_second` is
`batch_size * repeats / sum(batch_seconds)`, with seconds in the denominator;
for example, thirty batches of 32 taking 0.960 seconds yield 1,000 windows/s.
The two-member neural ensemble still processes 32 input windows per batch,
not 64. Throughput uses total measured time, not inverse median latency.

Archive checks, loading, fingerprinting, model restoration, input preparation,
transfers, validation, and report writing are outside the timer. These results
exclude cold starts, sensor acquisition, window construction, deployment transfer
cost, enrollment/trial/template aggregation, and authentication decisions. They
are raw-window-to-embedding execution timings, not end-to-end authentication
latencies or evidence of accuracy improvement.
