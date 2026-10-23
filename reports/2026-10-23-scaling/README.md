# Subject generalization and scaling study — 23 October 2026

Forty-eight complete GPU runs tested a matched size/objective factorial. Larger
encoders improved WISDM development performance, slightly worsened HAR, and
lost to descriptor baselines on both sources. This is a development study,
not a new independent test result or a claim of usable authentication.

The small linear-head comparator has mean fold EER 18.203% on HAR and 18.560%
on WISDM. Development-selected descriptors reach **11.647%** and **12.633%**
respectively: relative development reductions 36.0% and 31.9%. Larger ArcFace
encoders reach 18.593% HAR and 16.250% WISDM; scaling helps only the latter.
All candidates and the considerable fold variation remain visible below.

## Registered comparison

Width 32/96 × linear/ArcFace identity heads, shared hybrid triplet/SupCon loss,
three subject folds × seeds 42/43 × exactly 40 final epochs per neural run.
Descriptors use the same folds and a single deterministic fit. Four users per
source are excluded from every optimizer and reporting fold for calibration
after selection. Normalization and covariance fit fold-training users only.
Source TEST is never loaded. Earlier studies inspected those test cohorts;
this work does not relabel them as fresh. Equal-fold/seed means choose the
candidate. SDs describe dependent fold variation and are not confidence intervals.

Scientific implementation: commit `40789b491d34c10fcd99a31f2b60c7b0d4549ade`;
per-module source hashes and complete pretraining configurations are retained.
The two source jobs ran in parallel; their train-stage wall times were
332.217 seconds HAR and 257.748 seconds WISDM. Summed run times are compute
records, not end-to-end study wall time. Both lock stages independently
revalidated all 24 real checkpoints and recomputed descriptor/neural scores.

## Complete outcomes

### UCI HAR

| Candidate | Mean fold EER | Fold SD (pp) | AUC | Encoder parameters |
| --- | ---: | ---: | ---: | ---: |
| linear_w32 | 18.203% | 7.61 | 0.8934 | 30,252 |
| linear_w96 | 18.697% | 6.37 | 0.8941 | 207,236 |
| arcface_w32 | 17.980% | 8.15 | 0.8995 | 30,252 |
| arcface_w96 | 18.593% | 8.75 | 0.8958 | 207,236 |
| features_standard | 11.647% | 4.15 | 0.9496 | N/A |
| features_wccn_auto | 12.027% | 6.74 | 0.9452 | N/A |
| features_wccn_0.2 | 11.713% | 7.07 | 0.9486 | N/A |

### WISDM AR v1.1

| Candidate | Mean fold EER | Fold SD (pp) | AUC | Encoder parameters |
| --- | ---: | ---: | ---: | ---: |
| linear_w32 | 18.560% | 9.86 | 0.8884 | 28,972 |
| linear_w96 | 16.330% | 8.89 | 0.9034 | 203,396 |
| arcface_w32 | 17.270% | 8.96 | 0.9062 | 28,972 |
| arcface_w96 | 16.250% | 7.98 | 0.9026 | 203,396 |
| features_standard | 14.700% | 7.53 | 0.9238 | N/A |
| features_wccn_auto | 12.700% | 8.82 | 0.9403 | N/A |
| features_wccn_0.2 | 12.633% | 8.64 | 0.9419 | N/A |

## Calibration and deployment limits

Selected methods are HAR `features_standard` and WISDM `features_wccn_0.2`.
Their separate calibration users determine attainable thresholds at empirical
FAR <=1%; see each `selection_lock.json` for actual FAR/FRR, thresholds and
sample/reuse counts. Calibration errors are not test errors. Reused pairs,
overlapping windows and only four calibration identities cannot establish
low population FAR, cross-day robustness or takeover detection time. The
neural finalization path uses a six-member fold/seed ensemble; its calibration
representation differs from individual fold predictions and is disclosed.

## Runtime and precision

| Channels / width | Wiker CPU FP32 ms/step | Darkstar CUDA FP32 ms/step | CUDA BF16 ms/step |
| --- | ---: | ---: | ---: |
| 3 / 32 | 50.424 | 9.793 | 11.793 |
| 3 / 96 | 108.187 | 9.467 | 11.701 |
| 9 / 32 | 37.555 | 9.881 | 14.560 |
| 9 / 96 | 92.631 | 9.591 | 11.881 |

These are resident synthetic ArcFace/hybrid training microbenchmarks, batch 32,
20 timed optimizer steps after five warmups, three independent repeats. They
include forward/backward, finite-loss checks, clipping and AdamW; exclude
loading, transfers, augmentation, validation, checkpointing and scalar logging.
CPU and GPU use different machines; Wiker had unrelated memory pressure, so
these figures are not an isolated same-machine device speedup. Raw repeats,
initialization hashes, actual convolution dtypes, allocator peaks and hardware
are attached. All eight CUDA cases passed finite forward/loss/gradient checks.
BF16 was **slower in all four matched CUDA cases** despite modest memory savings;
FP32 remains the default and all 48 study runs used FP32. BF16 is opt-in, never
silently substituted. These timings do not demonstrate equivalent convergence.

## Reproduction and evidence

Follow [the study guide](../../docs/subject_scaling_study.md). Source data come
from the checksum-verified public HAR/WISDM setup in the repository. Run train,
then lock in a fresh output root; complete budget and unchanged registrations
are required. The script has no source-test evaluation stage.

- Each source folder: original pretraining plan, complete candidate matrix,
  frozen choice and hashed calibration lock.
- `training_runs.jsonl`: all 48 configurations, exact cohorts, hardware,
  normalization provenance, checkpoint/metrics hashes and run metadata.
- `training_histories.jsonl`: all 1,920 epoch records, without duplicating them
  in every configuration document. Checkpoint weights remain in the workspace
  experiment artifacts and are reproducible; they are not committed as binaries.
- `validation.json` and full local test logs: 290+215 subtests main;
  286+215 Student B. The mirror is the same experiment, not a replication.
- `metric_precision_cpu.json` / `metric_precision_cuda.json`: raw microbenchmarks.

ArcFace is an adaptation from [Deng et al.](https://arxiv.org/abs/1801.07698),
not a reproduction of face-recognition results. Subject generalization and
fresh cross-day dataset options are documented in
[the protocol research](../../docs/subject_generalization.md).
