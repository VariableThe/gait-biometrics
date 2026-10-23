# Research extensions and measured efficacy — 23 October 2026

WISDM verification improved in the validation-locked neural study. HAR's neural validation winner failed to generalize; a separate subject-cross-validated feature method improved HAR in an explicitly exploratory follow-up. These are within-recording research results, with substantial remaining fixed-threshold errors.

## Matched source cohorts and results

The frozen baselines are the preceding 20-epoch walking models. All comparisons retain native-rate T128/stride64 windows, walking label 0, identical development/test subjects and the same 10,000 balanced same-activity comparisons, seed 42. Thresholds target FAR≤1% on validation and remain fixed on test. This target does **not** promise 1% test FAR.

| Dataset and method | Test EER | ROC AUC | Fixed test FAR | Fixed test FRR | Evidence status |
| --- | ---: | ---: | ---: | ---: | --- |
| HAR frozen baseline | 17.64% | 0.9252 | 3.58% | 37.28% | Prior baseline, re-evaluated |
| HAR multiscale hybrid ensemble | 21.46% | 0.8900 | 6.86% | 27.38% | Primary validation winner; negative result |
| HAR descriptors + WCCN, shrinkage 0.2 | **10.44%** | **0.9608** | **0.54%** | **30.46%** | Secondary subject-CV selection; exploratory test reuse |
| WISDM frozen baseline | 20.10% | 0.8686 | 15.20% | 24.68% | Prior baseline, re-evaluated |
| WISDM multiscale batch-hard ensemble | **9.58%** | **0.9622** | **8.26%** | **10.72%** | Primary validation-locked selection |

WISDM EER decreased 52.3% relative to its matched baseline. The secondary HAR reduction is 40.8%; its evidence is weaker because the test cohort had already been inspected. Neither comparison isolates architecture: the neural bundle changes normalization, sampling, objective, augmentation, 40 versus 20 epochs, and three-member versus single-model inference. No historical all-activity EER of 0.4162 is used as a controlled comparator.

HAR retains 955 optimizer-training windows/17 subjects, 271 validation windows/4 subjects, and 496 test windows/9 subjects. WISDM retains 725/23, 202/4 and 345/8 respectively. Singleton identities excluded from positive training batches are explicitly reported in checkpoint sampler metadata. Validation windows never fit normalization or the identity classifier.

## Primary study: all candidates, then locked test access

The locally recorded [plan](preregistered_plan.json) precedes new candidate test scoring. Six neural methods each trained for 40 full epochs with seeds 42, 43 and 44, fixed split seed 42, fixed pair seed 42: legacy batch-hard; multiscale batch-hard, supervised contrastive and hybrid; temporal hybrid; and multiscale hybrid with statistics fusion. Each has no postprocessing and train-only automatic-shrinkage WCCN options. Three descriptor baselines complete **15 validation candidates per dataset**. Uniform seed ensembles concatenate normalized embeddings; no seed is selected using test scores.

All **36 final neural runs completed**. Full epoch histories and effective configs are included beneath `har/` and `wisdm_v1/`. Summed per-run elapsed times were 287.19 seconds for HAR and 225.19 seconds for WISDM on Darkstar's RTX 5070 Ti; these are workload totals, not end-to-end study wall time. Both source sweeps ran concurrently. Selection chose HAR hybrid at validation EER 3.04% and WISDM batch-hard at 1.98%. The generalization gap is visible rather than hidden.

The registries include every candidate, unsuccessful temporal/statistics variants, per-seed validation metrics and fitted-transform paths. `selection_lock.json` freezes selected artifact hashes, development tensor hashes and the baseline checksum. The runner validates actual checkpoint configuration, seed, all 40 completed epochs and protocol consistency before reuse and independently before locking. Reduced smoke runs cannot unlock test evaluation.

Selected-method single-seed test EERs, reported without choosing the best seed:

| Source | Seed 42 | Seed 43 | Seed 44 | Uniform ensemble |
| --- | ---: | ---: | ---: | ---: |
| HAR hybrid | 0.2130 | 0.2004 | 0.2288 | 0.2146 |
| WISDM batch-hard | 0.0790 | 0.1248 | 0.1398 | 0.0958 |

An earlier identical-method 36-run validation-only sweep is retained in `preliminary-validation/`. It was repeated after adding pre-training dataset fingerprints and budget guards, before new test results were read. Only the repeat determines the primary selection, even where its validation score is worse. GPU training is not promised bitwise reproducible. [Source provenance](source_manifest.json) distinguishes the training snapshot from later guard/report changes.

## Secondary HAR study

After observing the neural HAR failure, a [second plan](secondary_plan.json) fixed three descriptor methods: standardized cosine, automatic WCCN and WCCN shrinkage 0.2. Three disjoint subject folds cover the original **17 training identities** once. Every fold fits centering, scale and within-class covariance using only the other folds. The original four validation subjects are reserved for calibration and never select this feature method.

Mean fold EER selects shrinkage 0.2 at **0.0822**. The transform is refitted on all 17 training identities, its threshold calibrated on the four reserved identities, and artifacts locked before the secondary test evaluation. `har-feature-cv/` records every fold, subject coverage, calibration, hashes and results. The HAR test cohort was previously inspected; the 0.1044 result is an **exploratory follow-up requiring fresh-cohort replication**, not a new blind holdout or a reason to erase the neural failure. No further candidate was selected from test scores.

## Enrollment protocol

Prefix enrollment uses five windows per identity, skips the following two ordered windows, then compares every surviving probe against each normalized identity template. It is a separate protocol and cannot be directly compared with random-pair EER.

| WISDM method | Enrollment EER | ROC AUC | Fixed FAR | Fixed FRR |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.2837 | 0.7804 | 0.2111 | 0.3772 |
| Selected neural ensemble | 0.2042 | 0.9104 | 0.1078 | 0.3599 |

HAR enrollment also worsened for the primary neural winner; its complete results remain in `har/locked_test_results.json`. Secondary HAR descriptors were evaluated on the pair protocol only. Index purging assumes preserved within-subject order; raw recording offsets are unavailable, so raw nonoverlap and independent enrollment/probe sessions are **not established**.

## Runtime and model cost

Resident-input inference on Darkstar, two CPU threads, FP32, five warm-up batches and 30 timed repetitions. Times include normalization and all three ensemble members. They exclude input transfer, data loading and comparison scoring. These timings are separate from the earlier 15.72× equal-output preprocessing improvement.

| WISDM encoder | Stored parameters | CPU batch-1 median | CUDA batch-1 median | CUDA batch-128 median |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 174,720 | 0.253 ms | 0.525 ms | 0.443 ms |
| New first member | 28,972 | 0.897 ms | 1.658 ms | 1.390 ms |
| New full ensemble | 86,916 | 2.624 ms | 4.507 ms | 4.153 ms |

The ensemble has fewer stored parameters but is slower. The old count includes its unused DAE decoder; JSON also reports forward-used parameters separately. Many small multiscale operations and three forwards add dispatch cost. No blanket architecture speedup or phone energy claim is made. The HAR descriptor pipeline includes feature extraction plus the dense covariance projection: CPU median 0.515 ms for one window and 13.798 ms for 128. Its covariance and normalization tensors are learned state, not a parameter-free model.

Opt-in branch packing stacks centered zero-padded kernels into one convolution per multiscale block. In the same packing benchmark, CUDA ensemble latency fell from **4.693 to 3.687 ms** at batch one (**1.27×**) and 4.431 to 3.488 ms at batch 128. CPU batch one improved from 3.424 to 2.392 ms, while batch 128 worsened from 76.505 to 92.802 ms. These are within-run comparisons; raw repeats are retained. Packing increases stored ensemble parameters to 126,084 and remains slower than the old baseline. It is optional and never changes training checkpoints.

Synthetic checks and all 345 real WISDM test windows across three seeds passed numerical equivalence with cuDNN TF32 disabled; the real check observed zero embedding difference and unchanged EER/fixed decision counts. A preliminary real check with TF32 enabled exceeded the 1e-5 tolerance, despite passing synthetic checks. The helper therefore rejects CUDA forwards under that setting rather than silently accepting drift. Configure both original and packed inference identically and validate the serving backend. No universal bitwise guarantee follows from this measurement.

## Reproduction and validation

Use the [complete command guide](../../docs/research_experiments.md). It first creates frozen baselines, then runs `scripts.research_verification --stage train`, `--stage lock`, and `--stage evaluate`. Portable one-seed configurations are `configs/metric_har.yaml` and `configs/metric_wisdm.yaml`; single-seed results are distinct from ensemble results. For the secondary diagnostic run `scripts.cross_validate_gait_features --stage train` followed by `--stage evaluate`. Weights and source datasets remain local; compact JSON evidence is checked in.

Final local suite counts and logs are recorded in `validation.json`. Real HAR new-encoder training completed locally on CPU for one full epoch in 4.45 seconds, including validation and checkpoint creation; this is a smoke check, not a quality or speed comparison. A local CUDA smoke exited 137 under severe host memory pressure and produced no completed result. Full final CUDA experiments completed on Darkstar. Independent review closed budget, data-fingerprint and resume-checkpoint findings; the secondary fold protocol was reviewed separately.

The methods adapt [batch-hard mining](https://arxiv.org/abs/1703.07737), [supervised contrastive learning](https://arxiv.org/abs/2004.11362), [multiscale time-series CNNs](https://arxiv.org/abs/1909.04939), [squeeze/excitation](https://arxiv.org/abs/1709.01507) and [within-class covariance normalization](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html). They are adaptations across tasks/modalities, not reproductions of those papers' systems or performance. Independent days/devices, walking gating, continuous takeover delay and realistic enrollment remain unvalidated.

Statistical descriptors and neural embeddings can exploit placement, orientation or recording cues as well as locomotion. These datasets do not establish that the learned identity cue remains stable across sessions or resists imitation.
