# Existing-data engineering follow-up

The audited training, setup, evaluation and continuous-decision gaps now have
implemented, tested paths. The conditional claimant-normalization experiment
**retains the raw scorer**: neither normalization passes every registered gate.
No new original Santos CALIB/TEST windows were loaded or scored, and the
published TEST operating point is unchanged.

## What changed

| Gap | Result |
| --- | --- |
| Fractional settings silently truncated; invalid sampling rates accepted | Strict preflight checks in activity, Siamese and metric training |
| Same-activity validation could fail after optimization | Pair feasibility checked before allocating the model/optimizer |
| Interrupted metric training lost optimizer/RNG state | Opt-in atomic epoch-boundary resume, strict configuration/data/source/runtime binding and best/log recovery |
| Partial HAR extraction treated as complete | Complete nonempty required-file checks and safe local-archive repair |
| Float32 cutoff rounding and invalid numeric artifacts | Float64 decisions, scalar/label/identity validation and finite, shape-checked metric state |
| Window pairs could share raw samples | Opt-in WISDM raw-support provenance, exact sample-gap exclusion and bounded pair sampling |
| No causal policy or takeover accounting | Fresh gated quorum, silence expiry, latched fallback, strict JSONL and right-censored takeover metrics |
| No source-bound walking gate | One fixed CPU logistic fit, exact sensor/unit contract and numeric JSON artifact |
| Gate/encoder/score-policy integration | Single-window adapter with fixed enrollment and explicit source/encoder/template calibration bindings |
| Repeated installations and manual checks | Python/backend/tool pins, frozen uv lock, CPU CI and dependency audits |

The guides in [`docs/open_problems.md`](../../docs/open_problems.md) link each
implementation and its contract. `WindowAuthenticator` requires an externally
frozen **single-window** calibration; it rejects a trial-score artifact. These
changes do not manufacture a compatible phone encoder, deployment threshold,
sensor attestation, trusted fallback verifier or device acquisition service.

## Normalization result: no promotion

The 26-source registration froze lagcov362, automatic WCCN, the equal
mean/two-medoid score blend, policy C and a 0.001 standard-deviation floor.
Twenty outer DEV splits share 100 fresh covariance fits. TRAIN-only cohorts
remain separate from scored identities. Source-derived target **and cohort**
trial inventories register before fitting; cancellation cannot remove a cohort
trial silently. The run contains 240 inner and 60 outer comparisons.

| Arm | Mean DEV FAR | Mean DEV FRR | Splits FAR >2% | People no worse than raw | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw | 0.8362% | 15.1264% | 3/20 | 15/15 | Retain |
| Z | 1.0862% | 14.9540% | 5/20 | 9/15 | Fails FAR, fold-tail and person gates |
| S | 1.0000% | 12.4540% | 5/20 | 12/15 | Fails fold-tail gate |

These are exact **equal-fold means**, not pooled claim counts. Raw FAR is
`97/11600` and FRR `329/2175`; S FRR is `2167/17400`. Selection uses integer
counts and exact rational comparisons. The apparently better S rejection rate
does not override the predeclared safety gate. Every arm is retained. One DEV
person remains completely rejected by raw and S; see all per-person results in
[`summary.json`](summary.json).

All 401 numeric artifact digests, 100 model/cohort assignments and all 300 score
comparisons replayed. Verification checks runtime versions, exact metadata,
cache-derived thresholds/counts and identical fresh/cache decisions at every
frozen threshold in addition to numerical tolerance. Independent review also
reproduced the selection from counts. No further normalization variant or grid
was selected from these results.

## Source-bound walking detector

Only repository WISDM TRAIN windows were materialized. Its 28 source TRAIN
identities were assigned **16 fit / 6 calibration / 6 held DEV validation** before
the single class-balanced logistic fit. Fixed magnitude/cadence features and
standardization use fit identities only. Calibration chooses balanced accuracy;
validation cannot change weights, features or threshold.

The native-unit contract is `wisdm_v1`, three total-acceleration axes, 20 Hz,
128 samples and `wisdm_native_10_per_g`. The supplied raw-about metadata defines
10 raw units as 1g (9.81 m/s²); values are not silently relabelled SI. Source,
unit, channel, window-length or sampling-rate mismatches fail.

| Held DEV validation | Count | Rate |
| --- | ---: | ---: |
| Walking accepted | 145/154 | 94.16% recall |
| Walking missed | 9/154 | 5.84% |
| Other activities rejected | 181/205 | 88.29% |
| Other activities passed | 24/205 | 11.71% |
| Balanced accuracy | 359 total windows | 91.22% |

The calibration-only threshold is `0.5827542706696185`. It is an activity gate
threshold, not an authentication threshold; logistic outputs are not asserted
to be calibrated probabilities. Per-person and per-activity errors, including
stairs and jogging false passes, remain in the full result.

Five warmed, one-thread CPU inference repetitions over 359 completed windows
measured median throughput **30,010 windows/sec**, or **0.0333 ms amortized per
window**. This is batched feature-plus-gate throughput, excluding imports,
loading, fitting and disk I/O. Each window still needs nominally **6.4 seconds**
of acquisition, with 6.35-second first-to-last sample span. It is not measured
single-call latency, full authentication latency or phone performance.

Strict cache-only verification passed in the locked Torch `2.14.1+cpu`
environment. The independent review's separate CUDA environment was correctly
refused by the version guard; its helper-level numeric checks passed. Do not
interpret that review as a matching-environment strict verification.

## Raw support and continuous behavior

An existing-data TRAIN integration check found identical tensors/activity and
identity labels between the original and provenance loaders: 2,155 windows,
including 927 walking windows. All 10,000 sampled walking verification pairs
had disjoint raw support. They contain 8,256 unique pairs and 1,744 repeats;
nonoverlap does not make them independent. A five-window enrollment prefix
with a 128-raw-sample purge retained 120 enrollment windows and 780 probes over
24 eligible identities, purging 18 unsafe windows. Coordinates are ordinals
among valid parsed source records, not raw lines or independent sessions.
See [`train-provenance.json`](train-provenance.json).

Continuous checks use synthetic timed events and an actual synthetically fitted
walking gate. They cover gate/quality abstention, fresh noncontradictory quorum,
nonoverlap, stale evidence, silence, malformed JSON, untrusted reset rejection,
latched fallback and source/encoder/template mismatches. Owner false-challenge
rates use explicitly observed owner duration. Undetected takeovers retain null
latency and their censor time. These establish software accounting, **not
measured real takeover latency or daylong challenge burden**.

The host must schedule clock ticks during silence, act on each current decision
and independently verify fallback before creating a new episode. The adapter's
declared support conservatively includes inference/delivery delay; it can reject
otherwise disjoint acquisition windows. Real phone timing and operating policy
still require appropriate observation.

## Actual CUDA recovery check

A separate generated-TRAIN check used the idle local GTX 1650 Ti with Torch
2.14.1+cu130. Four uninterrupted FP32 epochs (eight optimizer steps) matched an
epoch-two committed interruption followed by two resumed epochs exactly:
encoder, classifier, AdamW, cosine scheduler, CPU/CUDA RNG, sampler, history and
the entire selected `best.pt` payload. Live parameters/gradients/optimizer
moments were on CUDA, and actual CUDA generator state was saved/restored.

It enabled deterministic algorithms, cuDNN determinism, disabled benchmarking
and TF32, and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Dropout and jitter were active.
The trainer's observational clock was fixed to compare histories; raw elapsed
times are diagnostic and support no speed claim. This is one device, precision
and tiny synthetic encoder, not universal CUDA equivalence or biometric evidence.
The JSON, log and original one-off harness snapshot are included. The snapshot
expects the original workspace `artifacts/` beside `gait-biometrics/`; it is
evidence source, not an installed CLI.

## Reproduction and evidence

The study source snapshot is main commit
[`fa61257b0e016cfc7a47a31dc89d94ff7ca0979f`](https://github.com/VariableThe/gait-biometrics/commit/fa61257b0e016cfc7a47a31dc89d94ff7ca0979f).
The later single-window adapter is separate and does not change either study's
registered sources. The release also tightens type annotations after this
freeze. For strict replay, check out the registered `fa61257` source snapshot;
do not substitute current hashes in a published registration. These annotation
updates add no model fit or new biometric result. Source fingerprints, runtime versions, actual source
coverage and all failed/passing gates remain immutable.

`normalization/` publishes lossless gzip JSON registrations/results/validation
and all 300 score caches. Its inventory includes hashes for all 401 numeric
artifacts. Large feature/model/cohort states remain in the local 160 MB experiment
archive and are **not** included in Git; full numerical model replay needs that
archive or a separately registered reproduction. `walking-gate/` publishes every
artifact, including fixed features and numeric model. JSON is compressed without
changing bytes; for example:

```bash
gzip -cd reports/2026-10-24-engineering/walking-gate/model.json.gz > /tmp/walking-gate-model.json
```

Use fresh output roots and the commands in
[`claimant_normalization.md`](../../docs/claimant_normalization.md),
[`walking_gate.md`](../../docs/walking_gate.md) and
[`reproducible_environment.md`](../../docs/reproducible_environment.md).
No source download is required when the existing datasets are present.

The locked CPU integration logs record **913 main / 909 Student B tests**,
each with **1,270 passing subtests**. The three dependency audits are included.
The lock audit found no known vulnerabilities/adverse statuses. Installed-package
auditing skips torch's local `+cpu` suffix, so the corresponding pinned upstream
release was audited separately and passed. This is a dated audit, not a claim
that unknown vulnerabilities cannot exist. Eight existing ONNX/quantization
warnings remain visible in the test logs.

Student A's standalone two-commit identity/preprocessing contribution still
lacks upstream collaborator access. Its four standalone tests pass and the
changes already exist under the merged project's Student A bridge.
[`student-a-ready.patch`](student-a-ready.patch) is the prepared standalone
contribution; it can be applied on that repository's current main with `git am`.
No public fork was created while the earlier collaborator-access request remains
pending. This publication step needs access, not another dataset.

## Remaining evidence limits

No software can recover identity mappings erased from anonymous legacy CSVs or
derive HAR raw offsets absent from released windows. Previously inspected TEST
people cannot supply independent confirmation. Dependent claims from these
small cohorts cannot certify population FAR=1%. The walking detector is one
source-bound DEV experiment; it has not validated placement, transition, repeated
session or device shifts. Fresh repeated phone sessions and actual device
measurements remain necessary for those claims.
