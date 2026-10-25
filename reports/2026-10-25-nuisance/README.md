# Paired nuisance covariance: robustness gain, retain baseline

The fixed two-arm experiment retains **Canonical362 without nuisance loading**.
Paired gain/noise covariance loading reduces the mean genuine rejection rate
across gain0.9, gain1.1 and noise1% from **45.607280% to 15.241379%**—a
**30.365900 percentage-point** improvement—but fails **13 of 33 distinct** registered
promotion/safety gates. It is a useful exploratory robustness result, not a
replacement for the retained method or a deployment threshold.

All figures below are **equal-fold means** over twenty overlapping splits of
the same fifteen Santos DEV identities. The three-condition objective also
weights conditions equally. Repeated people, windows and directed claims are
dependent. The design adapted to earlier inspected DEV stress failures; it is
not independent confirmation. Original CALIB and TEST payloads stayed closed.
The earlier TEST operating point remains 3/200 false accepts (1.5%); this report
does not update it.

## Fixed method and paper fidelity

Both arms use unchanged Canonical362: sign-pooled gravity/horizontal-PCA
features plus the original raw-coordinate lag spectrum, automatic identity-
balanced WCCN, the equal mean/two-medoid enrollment score, and policy C. Four
nine-person inner fits provide pooled **clean** calibration scores; both cutoffs
freeze before either twelve-person refit. No stressed calibration is performed.

For each model's own TRAIN identities, four paired raw views are fixed:
gain0.9, gain1.1, and antithetic plus/minus Gaussian noise at 1% of that partition's
dynamic triad RMS. TRAIN noise uses seed 20261026; validation noise retains seed
20261025. Clean original rows alone determine centering, scaling and automatic
shrinkage. The candidate adds an identity-balanced **uncentered** second moment
of standardized paired feature displacements to the clean within-class
covariance, with coefficient one. Synthetic views never increase subject counts
or the shrinkage estimator's sample count.

[Hatch, Kajarekar and Stolcke (2006), §2](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf)
motivates inverse within-class covariance weighting. Our paired nuisance term
is an adaptation, not a reproduction of their speaker-recognition system.
[Um et al. (2017), §§3.2 and 4.3](https://arxiv.org/html/1706.00527v2)
motivates investigating scaling/jitter and warns that transformations need not
preserve task labels. Neither paper establishes efficacy for this gait study.
The [frozen protocol](../../docs/nuisance_covariance_study.md),
[metric mathematics](../../docs/nuisance_metric.md), and
[exact gates](../../docs/robustness_selection.md) provide the full specification.

## Complete fixed-condition results

Percentages; lower FAR, FRR and EER are better. Enrollment remains clean, and
every condition uses its arm's frozen clean threshold.

| Probe condition | Baseline FAR | Candidate FAR | Baseline FRR | Candidate FRR | Baseline EER | Candidate EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.500000 | 0.583333 | 14.787356 | 14.959770 | 5.876437 | 6.040230 |
| Common x rotation 30° | 0.500000 | 0.583333 | 14.787356 | 14.959770 | 5.876437 | 6.040230 |
| Common x rotation 90° | 0.500000 | 0.583333 | 14.787356 | 14.959770 | 5.876437 | 6.040230 |
| Gain0.9 | 0.000000 | 0.583333 | 45.206897 | 15.293103 | 7.129310 | 5.873563 |
| Gain1.1 | 0.166667 | 0.500000 | 39.827586 | 15.126437 | 6.451149 | 6.040230 |
| Noise1% TRAIN RMS | 0.250000 | 0.500000 | 51.787356 | 15.304598 | 7.054598 | 6.551724 |
| Noise5% TRAIN RMS | 0.000000 | 0.000000 | 100.000000 | 100.000000 | 33.899425 | 9.571839 |

The exact robustness objective is `23807/52200 → 221/1450`; its improvement is
`15851/52200`. Fourteen people's mean FRR across the three objective conditions
improves, one ties, and none worsens. Both prescribed rotations preserve their
own arm's ordered clean decisions. These constant common-coordinate changes
do not establish phone relocation or time-varying orientation robustness.

At 5% noise both arms still reject **596/596 genuine claims** at their frozen
cutoffs. Better secondary EER there does not repair that operating behavior;
no threshold was selected using the stress results.

## Why the candidate is not promoted

Clean mean FAR rises from `1/200` to `7/1200`; mean FRR rises from
`2573/17400` to `2603/17400`; mean EER rises from `409/6960` to `1051/17400`.
Actual repeated counts are **6/1192 → 7/1192 false accepts** and
**88/596 → 89/596 genuine rejects**. These pooled counts differ from equal-fold
means and do not represent additional independent observations.

The number of folds above 2% FAR stays 2/20 on clean data, but rises 0→2 under
each gain condition and 1→2 under noise1%. The candidate also increases false
accepts targeting some claimants in clean/rotation and all three objective
stresses. Its seven condition means are each below 1% FAR, but that alone does
not satisfy the registered fold-tail and claimant protections or certify a
population FAR bound.

The [exact summary](summary.json) preserves all 33 distinct gate decisions and all 13
failure names. No gate, coefficient, seed, condition or threshold was changed
after inspecting this outcome.

## Clean user outcomes

One person's clean rejection improves, twelve tie and two worsen. Both
worsening people gain one repeated rejection. Subject 9 remains completely
rejected. The clean false accepts targeting subject 24 remain five; subject 9
remains one; subject 17 gains one. Per-person EER/AUC and condition-specific
claimant counts are retained as exact fractions in [summary.json](summary.json).

| DEV person | Baseline clean rejects | Candidate clean rejects | Repeated genuine claims |
| --- | ---: | ---: | ---: |
| 1 | 1 | 1 | 40 |
| 3 | 3 | 3 | 40 |
| 5 | 17 | 17 | 40 |
| 7 | 0 | 0 | 40 |
| 9 | 40 | 40 | 40 |
| 11 | 1 | 1 | 40 |
| 13 | 13 | 14 | 40 |
| 14 | 10 | 11 | 40 |
| 16 | 0 | 0 | 40 |
| 17 | 2 | 1 | 40 |
| 19 | 0 | 0 | 36 |
| 24 | 0 | 0 | 40 |
| 25 | 0 | 0 | 40 |
| 27 | 0 | 0 | 40 |
| 29 | 1 | 1 | 40 |

## Verification, coverage and compute

The frozen source is `4c0b840226e55fba69058487c2baab1ad688a9da` in the main
repository; [provenance.json](provenance.json) and registration bind all **50**
source fingerprints and runtime versions. One registered run completes **200
metric fits**, **160 inner plus 280 outer score sets**, and **541 geometry
contexts**. Full strict verification rebuilds the DEV evidence without fitting,
checks model covariance/whitening algebra, and replays all 440 score sets with
identical frozen-threshold decisions. It retains exact cached metrics,
thresholds, assignments and selection. Source windows and claim coverage are
preserved; no invalid frames or noise references occurred.

The baseline's 100 model states, 80 inner score sets and 140 outer score sets
match the previous Canonical362 experiment **exactly**, including score metadata;
see [the parity record](nuisance-baseline-parity-20261025.json).
Source exclusions remain 28 continuity breaks, 33,810 tail rows and the
subject19/day2/trial1 zero-window recording. The prior
[official archive authenticity audit](../2026-10-23-cross-day/README.md)
documents source acquisition; this study verifies selected member hashes and
does not recompute the complete archive hash by reading TEST payloads.

The observed one-thread CPU run took **5m58.54s**, with peak RSS 360,144 KiB;
verification took **5m32.48s**. These are recorded run timings, not a speedup,
inference-throughput claim or independent benchmark. Raw resource logs accompany
this report. No GPU or remote compute was used for this study.

The final main suite passes **1,167 tests**, Student B **1,163**, each with
**1,592 subtests** and the same eight pre-existing export/quantization warnings.
Pinned Ruff/ty checks pass. Independent source review found and resolved a
zero-RMS failure-diagnostic gap before any real DEV fitting; synthetic tests
now retain all 541 assigned records and explicit blocked statuses without
inventing noise windows. Review scope is recorded in [audit.md](audit.md).

## Evidence and reproduction

[numeric-inventory.json](numeric-inventory.json) binds all **1,201** numeric
artifacts: 200 model/diagnostic states, 440 scores, 20 thresholds and 541 geometry
arrays. The compact Git publication includes **460** score/threshold artifacts,
compressed registration/development/verification and comparison metadata.
Dense enrollment coordinates are represented by metadata hashes. Full model
and geometry payloads remain in the local `artifacts/nuisance-covariance-v1`
archive; this compact publication alone is **not** a complete strict-replay
bundle. Reproduce with the registered source, environment, official source and
fresh output root as specified in the protocol, or use the original full local
archive. Do not alter frozen hashes to accept a different checkout.

The [synthetic diagnosis](synthetic-diagnosis/summary.json) independently
demonstrates gain-sensitive amplitude/log-trace coordinates and noise-sensitive
small lag modes. It does not identify what caused actual DEV errors. Its raw
[script snapshot](synthetic-diagnosis/diagnose.py) pins the original absolute
workspace and prior evidence checkout; it is preserved as historical diagnostic
provenance, not presented as a portable CLI. The registered experiment above
supports caller-supplied archive and output paths.

The remaining questions include severe-noise calibration/coverage, hard-user
rejection, claimant-specific false accepts, actual phone placement, independent
sessions and live single-window thresholds. These reused observations cannot
settle those questions or establish a state-of-the-art performance ceiling.
