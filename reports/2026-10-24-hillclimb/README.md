# Multisensor and calibrated operating-point hillclimb

All new Santos TEST comparisons are **exploratory reuse of five inspected
identities**, each enrolled on day1 and probed on day2. These controlled
leg-mounted sensor recordings do not establish phone authentication, population
FAR, placement robustness, or continuous takeover detection. The same fifty
genuine and two hundred directed impostor claims recur across comparisons.

## What the research changed

The first registered comparison tested thirty sensor/representation/enrollment
candidates with five development subject folds. Four neural families each ran
five folds and two seeds for forty epochs: forty CV fits. A complete prior
forty-fit attempt failed its strict cross-GPU replay audit and was discarded;
both attempts count in the historical compute cost: eighty CV fits, 3,200
run-epochs. Version two caches the exact validation embeddings used to select
the model and separately checks model replay within 1e-5. It preserves the
original numeric selection rather than changing scores to match another GPU.
The descriptor winner required no final neural training.

The selected method combines accelerometer and gyroscope axes with rotational
invariant descriptors, training-only automatic-shrinkage WCCN, and two angular
medoid enrollment templates scored by maximum cosine. Mean development EER
fell from 11.736% for the matched acceleration/mean-template control to 6.678%.
Neural losses, sensor alternatives and unsuccessful candidates remain in the
complete thirty-candidate evidence. See the exact registered definitions in
[the multisensor protocol](../../docs/multisensor_study.md).

A separate six-backend follow-up evaluated cohort score normalization,
paired-day nuisance projection and lagged covariance features. Four balanced
five-fold partitions give twenty unique held-out triples: each development
identity validates four times. The initial unbalanced random-triple pilot was
discarded and is excluded from selection. Lagged covariance won at mean EER
4.874%, versus 5.885% for the multisensor control and 8.897% for the historical
acceleration baseline under these same twenty splits. These means have a
different resampling schedule from the first five-fold comparison.

## Frozen-threshold test trade-off

Every method has its own threshold calibrated on five separate CALIB users at
empirical FAR at most 1%; TEST does not choose thresholds or methods.

| Method | EER | AUC | False accepts | Genuine rejections |
| --- | ---: | ---: | ---: | ---: |
| Historical acceleration + mean template | 7.0% | 0.9888 | 1/200 (0.5%) | 11/50 (22%) |
| Multisensor + two templates | 2.5% | 0.9932 | 6/200 (3.0%) | 1/50 (2%) |
| Add lagged covariance | 2.5% | 0.9958 | 7/200 (3.5%) | 0/50 (0%) |
| Lagged covariance + mean/two-template blend | 0.0% | 1.0000 | 3/200 (1.5%) | 0/50 (0%) |

The lagged features improved calibration rejection from 11/49 to 7/49 at the
same one false accept in 196 claims. Their test result still illustrates the
failure of a small calibration cohort to ensure the intended false-accept rate
on different people. Six of seven false accepts target identity 26. Neither
zero genuine rejections nor a lower EER makes that threshold deployment-ready.
The full per-person counts are retained in [summary.json](summary.json).

## Separate study of calibrated decisions

The next study changes the development criterion, rather than tuning the test
threshold. Its twenty splits use nine people for metric fitting, three
different people for calibration and three for validation. Every development
person fits twelve times, calibrates four times and validates four times.
Fifteen candidates cross three covariance shrinkages with five enrollment
scorers, using exactly sixty metric fits and three hundred comparisons.

Only the equal blend of mean-template and two-template scores with automatic
shrinkage met mean validation FAR at most 1%. Against the matched two-template
control, validation FAR fell from 3.178% to 0.917%, and FRR from 23.684% to
19.276%. EER slightly worsened from 5.299% to 5.457%; it is a diagnostic, not
this study's selection criterion. At final five-person calibration, both
methods have one false accept in 196 claims; blending reduces genuine
rejections from 7/49 to 3/49. The locked one-attempt TEST evaluation reduced
false accepts from 7/200 to 3/200, retaining zero rejections in fifty genuine
claims. Descriptive EER is 0% and AUC 1.0 on these five previously inspected
people. The frozen operating threshold still exceeds the intended 1% TEST FAR.
Perfect separation of this small reused score set does not establish perfect
authentication or authorize selecting a new threshold from the TEST gap.

The three-person inner calibration tail is especially coarse: only sixty
impostor claims, so an empirical FAR constraint of 1% allows zero false
accepts. Repeated overlapping splits do not provide independent replications
or population confidence intervals. Selection on this development population
can still overfit it. See [the operating protocol](../../docs/secure_operating_study.md)
and [scoring formulas](../../docs/secure_enrollment.md).

## Transfer and execution costs

The separate development-only HAR/WISDM invariance ablation improved mean EER
from 12.027% to 9.560% on HAR and 13.107% to 8.713% on WISDM. Those are
within-recording subject folds, not fresh cross-day or TEST results. Full rows
are in `transfer-results.json.gz`.

On Wiker, resident raw-window-to-embedding CPU latency at batch one is 1.107 ms
for the acceleration baseline and 5.731 ms for the multisensor winner. Batch
32 medians are 4.274 ms and 13.794 ms respectively. This is a measured accuracy
and cost trade-off, not an execution speedup. Thirty timed repetitions use two
CPU numerical threads. Feature extraction, fixed projection and normalization
are timed; sensor acquisition, file loading, cold start, enrollment/claim
scoring, transfer and phone energy are excluded. Full distributions and
provenance are in `runtime-wiker.json`.

The final lagcov/blend pipeline was also measured separately with
`scripts.benchmark_secure_operating`: batch-one and batch-32 raw-window
embedding medians are 3.816 ms and 15.230 ms. With raw features already
computed, complete DEV metric transformation, trial aggregation, enrollment and
2,250 directed claim scores take 54.165 ms median. Its matched lagcov control
takes 3.885 ms, 14.864 ms and 55.728 ms respectively; these small differences
are not evidence of a speedup. Both use the same embedding representation.
The 30-repeat distributions, locked-source hashes, exact inputs and timing
scope are in `runtime-operating-wiker.json`. These are separate warm CPU
benchmarks and must not be compared against the earlier timing as a controlled
speed improvement. Reproduce from the repository with the original study root:

```bash
uv run --no-project --python ../.venv/bin/python python -m scripts.benchmark_secure_operating \
  --archive-path ../datasets/santos/santos-v7.zip \
  --out-root ../artifacts/multisensor/secure-operating-v1 \
  --out ../artifacts/multisensor/runtime-operating-repeat.json --repeats 30
```

The first neural study used Wiker's GTX 1650 Ti, Darkstar's RTX 5070 Ti and the
lab p8 RTX A6000. The follow-ups fit small covariance models on Darkstar CPU;
additional GPU training is unnecessary. On resuming this round, p8 was busy
with another workload and was left alone.

## Evidence and validation

`multisensor/`, `backends/` and `operating/` publish original registration,
selection, calibration, attempt and result JSON bytes compressed losslessly
with gzip. Numeric final fitted states and calibration/test score arrays are
included. Each `artifact_inventory.json` records every original artifact's
SHA256, size and publication status, plus compressed-file digests where
published. Full CV checkpoint and covariance states remain in the local study
archives and are inventoried rather than all committed. Reproducing the full
replay audit requires those local states or rerunning the frozen stages on a
fresh root; the compact report alone is not a complete resumable study root.
No raw sensor data are copied into the repository.

The implementation and archive/source fingerprints are inside each original
registration. Decompressing a JSON must reproduce its inventory SHA256 exactly.
Validation caches, model/source checks, exact metric recomputation and exclusive
attempt guards are described in each protocol. Checksums are integrity
evidence, not externally signed experiment registration.

Both full local suites pass: **459 tests +759 subtests** in main and
**455 +759** in Student B. The eight existing ONNX/quantization warnings remain.
The 26 new focused tests include scalar scoring references, common claim
graphs, disjoint balanced roles, constrained selection, tampered calibration
replay and consumed failed TEST attempts. Mirrored code and evidence describe
one set of experiments, not another independent sample.

The methods adapt WCCN, supervised contrastive learning, rotation-invariant
features and cohort normalization; source links and distinctions from paper
reproductions are recorded in the corresponding method documentation. Next
confirmation requires a fresh repeated-session phone cohort and direct
measurement of false prompts, takeover delay and battery use.

Dataset attribution: Santos, Wanderley, Tavares and Rocha,
[*A multi-sensor human gait dataset captured through an optical system and
inertial measurement units*](https://www.nature.com/articles/s41597-022-01638-2),
[Figshare version 7](https://doi.org/10.6084/m9.figshare.14727231.v7), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Published derived
artifacts add windowing, descriptor extraction, fitted metrics and enrollment
scoring; the study does not reproduce the authors' original analysis.
