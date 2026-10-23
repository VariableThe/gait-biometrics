# Subject generalization and fresh-session evaluation

This study uses source-TRAIN subject cross-validation to compare development
candidates. The existing HAR and WISDM v1 test cohorts have already been
inspected in earlier work. They must not select candidates, widths, losses,
epochs, seeds, preprocessing, or thresholds, and another evaluation on them
would not constitute a fresh independent test. This round does not load them.
Development CV and a reserved threshold-calibration cohort do not establish
cross-session authentication or guarantee a deployment error rate.

## Fixed development protocol

The split interface is:

```python
from engine.subject_protocol import make_subject_folds

protocol = make_subject_folds(
    walking_train_subject_ids, calibration_subjects=4, n_folds=3, seed=2026
)
```

The caller obtains authentic encoded identities from **walking windows of the
original source TRAIN split**, before optimizer work. The helper cannot prove
source provenance or discover missing walking data. It validates a one-dimensional
nonnegative integer array, deduplicates and sorts it, and applies exactly one
`np.random.default_rng(2026).permutation`. The first four identities are reserved
for calibration. `np.array_split` partitions the remainder into three balanced
validation cohorts. All returned ID lists are sorted; IDs are never replaced by
contiguous ranks, fabricated from row numbers, or inferred from timestamps.

The JSON-safe result contains `development_subject_ids`,
`calibration_subject_ids`, and `folds`, with each fold holding `fold` (zero-based),
`train_subject_ids`, and `val_subject_ids`, plus seed/method metadata. Every
development identity validates exactly once. Each fold trains on every other
development identity; calibration identities appear in neither side. Each train
and validation cohort needs at least two identities, so the defaults require at
least ten unique walking identities. Insufficient cohorts fail without silently
reducing the number of folds. Identity counts alone do not prove that every
identity has sufficient distinct enrollment and probe windows.

Before training, freeze the candidate registry, folds, seeds, durations, metric
definitions, and selection/tie rules. The planned neural comparison is a 2 x 2
factorial: width 32 or 96, and a linear or ArcFace identity head, using the same
hybrid metric objective, training seeds 42/43, and 40 fixed final epochs.
Compare three fixed descriptor baselines on these same folds: seven candidates
per source. The candidate manifest is the executable source of truth for the
precise configuration. Wider networks and a margin head are hypotheses about
representation generalization, not evidence that either factor will improve it.

Every fitted normalizer, descriptor scaler, covariance/shrinkage estimator,
classifier vocabulary, and representation sees only the fold's training
identities. Validation subjects can rank registered candidates, but calibration
subjects must never enter optimizer batches, fold checkpoint selection, or
candidate selection. With fixed final epochs, save and score the prescribed
final checkpoint even when an earlier epoch has a better diagnostic validation
EER. Do not choose the better training seed; report both equally.

Select the candidate by the predeclared mean fold/seed development criterion,
then freeze its configuration. Refit on all development identities for the
prescribed duration. Only after this freeze may the four reserved identities
calibrate a fixed threshold. Calibration scores are not a model-ranking signal.
A future genuine external cohort is required to assess the fixed model and
threshold without reusing either development or calibration subjects.
Ranking seven candidates on these CV folds also makes the selected candidate's
CV mean an optimistic development estimate; these are not nested outer-test
folds. Calibration-cohort EER can be diagnostic, but errors measured on the same
scores that chose a threshold do not validate that threshold on new people or
days. Preserve the unsuccessful candidates and both seeds in the report.

## Summaries and subject diagnostics

`fold_summary(rows)` takes a separate candidate/source's complete grid of flat
`{"fold": 0, "seed": 42, "eer": 0.2, "roc_auc": 0.8}` rows. All rows must have
the same finite scalar metric columns; duplicate runs or missing fold/seed
combinations fail. It averages training seeds within each fold, then folds
equally. Window-rich folds and subjects do not receive additional weight.
`mean`, `per_fold`, `per_seed`, and `all_fold_results` preserve the overall and
individual evidence. `fold_sample_sd` and `seed_sample_sd` are descriptive
sample standard deviations; a single observation produces JSON `null`.

These standard deviations are not population confidence intervals or standard
errors. CV training cohorts overlap, random seeds repeat the same people, and
comparisons share windows and enrollment templates. A paired candidate contrast
must align the same fold and seed, report every difference, and retain those
dependence limits. No subject-cluster EER bootstrap or bootstrap interval is
implemented in this module. Relabelling repeated sampled identities as different
people can create false impostors; copying a row as a new genuine observation
can create invalid self-comparisons. Pair-level bootstrap uncertainty would also
mistake dependent scores for independent samples.

`per_subject_error_rates(labels, scores, claim_subject_ids, probe_subject_ids,
threshold)` reports fixed-threshold errors with one aligned claimed ID and one
aligned probe ID per score. Acceptance includes ties (`score >= threshold`).
Labels must equal authentic identity equality. `by_claim_subject` measures the
vulnerability of each enrolled identity to others' probes; `by_probe_subject`
measures the acceptance of each person's probes under others' identities. Both
report per-subject genuine/impostor counts, false accepts/rejects, FAR/FRR, and
equal-subject macro rates. `micro` reports comparison-weighted rates. Missing
denominators yield `null`, not an invented zero, and macro denominators report
how many identities contribute.

For the probe-major output of `engine.enrollment.enrollment_scores`, construct
aligned IDs as follows:

```python
import numpy as np

from engine.subject_protocol import per_subject_error_rates

claimed_ids = np.tile(metadata["claimed_subject_ids"], len(probe_idx))
probe_ids = np.repeat(subject_ids[probe_idx], len(metadata["claimed_subject_ids"]))
subject_report = per_subject_error_rates(
    labels, scores, claimed_ids, probe_ids, frozen_threshold
)
```

Use an independently calibrated threshold, and show difficult identities rather
than hiding them behind a pooled mean. The helper cannot establish exhaustive
coverage without probe-row identifiers or verify threshold provenance. Its
counts are descriptive, not confidence bounds on independent authentication
attempts. Even zero observed false accepts among dependent comparisons is not
a guaranteed 1% deployment FAR.

## Research basis and honest adaptations

**Subject separation and dependence.** Dehghani, Glatard, and Shihab,
[*Subject Cross Validation in Human Activity Recognition* (2019),
arXiv:1904.02666v2](https://arxiv.org/abs/1904.02666v2), show that within-person
correlation and overlapping windows can inflate ordinary window-level HAR
cross-validation. Our adaptation groups by authentic identity before fitting
any representation or transform. Their HAR findings do not establish gait
authentication performance, and subject separation does not create independent
sessions within each held-out identity. Bengio and Grandvalet,
[*No Unbiased Estimator of the Variance of K-Fold Cross-Validation* (JMLR,
2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html), explain why naive
variance estimates ignoring correlations can underestimate uncertainty. This
motivates the explicitly descriptive fold summaries; it does not rule out all
carefully designed inference procedures.

**Margin classification.** Deng et al.,
[*ArcFace: Additive Angular Margin Loss for Deep Face Recognition* (CVPR,
2019)](https://arxiv.org/abs/1801.07698), normalize features and class weights
and replace a training sample's target logit by `s * cos(theta_y + m)` while
non-target logits are `s * cos(theta_j)`. This is an angular margin inside
classification, not merely subtracting a constant from cosine similarity. An
identity head may regularize inertial embeddings; only optimizer-training IDs
belong in its class vocabulary. Unseen-subject verification uses the frozen
embedding and discards the head. The gait architecture, sensor channels, P x K
sampling, fixed training duration, and hybrid loss are adaptations, not a
reproduction of ArcFace's face benchmarks or proof of sensor robustness.

**Existing hybrid objective.** Hermans, Beyer, and Leibe,
[*In Defense of the Triplet Loss for Person Re-Identification*
(2017)](https://arxiv.org/abs/1703.07737), motivates within-batch hard-positive
and hard-negative metric learning. Khosla et al.,
[*Supervised Contrastive Learning* (2020)](https://arxiv.org/abs/2004.11362),
uses class labels to gather positives in a normalized embedding space.
Using authentic training identities as classes and distinct inertial windows
as positives is an adaptation of visual research. A hybrid of those objectives
with a linear or angular-margin identity loss is a registered experiment,
not a published gait recipe that promises improvement.

**Session shift must be measured.** Lee et al.,
[*Gait-Based Continuous Authentication Using a Novel Sensor Compensation
Algorithm and Geometric Features Extracted From Wearable Sensors* (IEEE
Access, 2022)](https://doi.org/10.1109/ACCESS.2022.3221813), reports separate
same-day and later-day evaluations with different error rates. Its sensor
compensation and geometric-feature method differs from this neural baseline.
The relevant inference is to preserve authentic day/recording metadata and
separate enrollment from later-session probes. A temporal purge within one
recording cannot support that claim.

## Fresh data acquisition

Official dataset access, licensing, subject identity, device identity, and
session identity are distinct requirements. A public download link alone does
not supply redistribution rights or a cross-day experiment. Availability was
researched on 2026-10-23 without downloading sensor archives, installing data,
or claiming access to an independent test cohort. Dataset-specific acquisition
options and constraints follow below.

### UCI WISDM Smartphone and Smartwatch: a fresh device-domain option

The [official UCI record, DOI 10.24432/C5HK59](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
offers the 51-participant dataset under **CC BY 4.0**. Its
[official description](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf)
documents participant IDs 1600–1650, phone-in-pocket and watch-on-dominant-wrist
accelerometer/gyroscope streams, nominally 20 Hz, and 18 three-minute activities;
walking is `A`. Raw records contain
`subject-id,activity-code,timestamp,x,y,z;`. Device/sensor provenance is encoded
by paths such as
`raw/{phone,watch}/{accel,gyro}/data_<id>_<sensor>_<device>.txt`.

Acquire from the record's download button or
[official raw archive](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip)
(listed at approximately 295.4 MB). Research checked a successful HTTP HEAD
response without downloading the archive. Preserve the stream paths and
authentic IDs when writing a future adapter, and filter raw walking records
before windowing. No day/session field or repeated-day protocol is documented.
Different device streams, separate sensor files, and timestamps must not be
invented into sessions. This supports a possible external population/device
study, **not a verified cross-day test**; cohort freshness must also be checked
against project history.

The earlier WISDM releases are distinct. Kwapisz, Weiss, and Moore's original
[*Activity recognition using cell phone accelerometers* (2011), DOI
10.1145/1964897.1964918](https://www.kdd.org/exploration_files/v12-02-19-SensorKDD-Kwapisz.pdf)
studied 29 participants with 20 Hz accelerometry, ten-second segments, and
ten-fold activity-classification CV. That paper, the later
[36-subject raw v1 release](https://www.cis.fordham.edu/wisdm/dataset.php),
and the 51-subject phone/watch dataset must not be treated as one
cohort or assigned each other's identity/session guarantees. The original
activity-classification experiment does not validate session-held-out
authentication.

### Santos multi-sensor gait: smaller, licensed cross-day acquisition

Santos, Wanderley, Tavares, and Rocha's
[*A multi-sensor human gait dataset captured through an optical system and
inertial measurement units* (Scientific Data 9, 545, 2022)](https://www.nature.com/articles/s41597-022-01638-2)
documents 25 adults, two sessions on different days, and ten trials per day.
Leg-mounted MCU data include accelerometer and gyroscope readings at 100 Hz;
a Nexus 5 supplies accelerometer data. The
[author-deposited Figshare v7 dataset, DOI 10.6084/m9.figshare.14727231](https://figshare.com/articles/dataset/A_multi-sensor_and_cross-session_human_gait_dataset_captured_through_an_optical_system_and_inertial_measurement_units/14727231)
publicly lists a 1.42 GB download and **CC BY 4.0**.

Documented filenames such as `capture_<userID>_day<1|2>_<trial>_imu.csv`
preserve authentic participant, day, and trial identity. IDs within 01–30 are
nonconsecutive: preserve the actual cohort rather than assuming 30 people.
Acquire from Figshare when authorized for the next study, retain raw/processed
provenance, and design day-one enrollment/day-two probes before scoring.
Limitations include short, controlled walkway trials, leg placement differing
from pocket HAR data, and phone alignment performed after capture rather than
hardware synchronization. This is a concrete future cross-day option, not an
external result already obtained here or a drop-in nine-channel HAR input.

### NONAN GaitPrint: deeper recordings with explicit day identifiers

Wiles et al.,
[*NONAN GaitPrint: An IMU gait database of healthy older adults* (Scientific
Data 12, 143, 2025)](https://www.nature.com/articles/s41597-024-04359-w),
describes 41 healthy adults aged 56+, 16 IMUs at 200 Hz, and nine four-minute
walking trials per visit, normally one week apart. The
[official dataset deposit](https://figshare.com/articles/dataset/NONAN_GaitPrint_An_IMU_gait_database_of_healthy_older_adults/27815034)
offers public **CC BY 4.0** per-subject archives; its full listed size is
53.79 GB, so select a documented subset for acquisition instead of fetching
everything. Names such as `S###_G03_D01_B01_T01` encode subject, population,
day, block, and trial. Subjects S104 and S126 lack a second visit and need
explicit cross-day eligibility handling.

These are Noraxon-exported kinematics with documented processing, rather than
untouched sensor packets. An adapter must verify signal units, coordinate
frames, body placement, and exported channels before comparisons. Preserve
the day codes and trial metadata. The older age group and full-body laboratory
equipment introduce further domain shifts. A separate young-adult counterpart
has 35 participants; its individual raw-file license was not verified in this
research and must be checked before acquisition or redistribution.

### WHUGait and OU-ISIR: access and protocol qualifications

Zou et al.,
[*Deep Learning-Based Gait Recognition Using Smartphones in the Wild* (IEEE
TIFS 15, 3197–3212, 2020)](https://arxiv.org/abs/1811.00338),
documents 118 participants with smartphone accelerometer/gyroscope data at
50 Hz; 20 were recorded across two days and 98 across one. The
[author repository](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones)
links [processed datasets #1–#8 in Google Drive](https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_).
Research could list that folder without login, but could not verify an authentic
day/session mapping in those processed files. The separate raw-data Baidu share
reported an expired link. The repository describes academic research use; no
explicit open data license was found. Confirm data terms and obtain the
day/recording mapping from the authors before using it as a cross-session
benchmark. Its supplied train/test splits must not be assumed to hold out days.

Ngo et al.,
[*The largest inertial sensor-based gait database and performance evaluation of
gait-based personal authentication* (Pattern Recognition 47(1), 228–237,
2014)](https://omilab.naist.jp/papers/PR2014-GaitDB.pdf),
describes up to 744 subjects with waist-mounted inertial sensors, including a
smartphone subset. The paper states that each person's sequences were captured
within one approximately one-minute session; separate walks, slopes, or sensors
therefore do not establish independent visits. The
[official OU-ISIR access page](https://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html)
timed out during research. Current acquisition and dataset-specific release
terms remain unverified; do not label it an immediately downloadable open
cross-session dataset.

Before any future external evaluation, preserve the official raw-file hashes,
README/license, participant IDs, device and sensor tags, activity labels, and
real recording/session identifiers. Split raw recordings before windowing and
record enrollment/probe disjointness. Freeze the model, required channel mapping,
sampling-rate policy, and threshold-transfer policy before looking at external
scores. If adaptation or target-domain threshold calibration is needed, use a
separate target development cohort and reserve untouched target identities and
later sessions for evaluation. Report zero-shot threshold transfer separately
from calibrated adaptation.

For now, the deliverable is development subject CV plus independent reserved
calibration. No fresh cross-session performance or successful domain transfer
has been measured.
