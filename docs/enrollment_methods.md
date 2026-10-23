# Opt-in enrollment protocol and diagnostic gait features

These modules add a separate, fixed enrollment/probe experiment. They do not
change the existing trainer, adapter, pair evaluator, configurations or historical
metrics. No dataset performance is established by the synthetic unit tests.

## Fixed protocol

Use source `har` (nine channels, 50 Hz) or `wisdm_v1` (three channels, 20 Hz),
walking activity label 0, 128 samples and stride 64. Preserve the adapter's source
row order after subject/activity filtering. Representation learning and feature
preprocessing use only optimizer-training subjects. Use the existing fixed
subject-held-out validation split (`val_subjects: 4`, `split_seed: 42`) and the
source's held-out test subjects. Verify all three subject sets are disjoint
before calling these low-level functions; they receive no training provenance
and cannot enforce that separation themselves.

`engine.enrollment.enrollment_split(subject_ids, enrollment_windows=5,
purge_windows=2)` returns `(enroll_idx, probe_idx, coverage)`:

1. For each identity, select its first five windows in supplied source order.
2. Discard its next two windows.
3. Use all its remaining windows as probes. An identity needs at least eight
   windows; otherwise exclude all its windows and report that exclusion.

Both index arrays are NumPy int64 and preserve global input order. Coverage is
JSON-safe and reports included/excluded subject IDs, per-subject counts, purged
indices and excluded-window totals. The split is deterministic without a seed.
Positive enrollment budgets and nonnegative purge budgets are required. Purge
zero is an explicit control with no overlap guard. Scoring needs at least two
eligible identities, each with both enrollment and probe windows.

### What the purge establishes

There are no shared *window rows* between enrollment and probes. `WindowData`
has no raw recording offsets, segment IDs or session IDs, so the implementation
reports `raw_offset_nonoverlap_verified: false`. It cannot discover shuffled
windows, duplicated source records, an incorrect stride, or externally reordered
input. The following is a conditional argument for the current adapters, not
an offset audit:

- Official HAR supplies source-ordered, 128-sample prewindows with 64-sample
  stride. Within a contiguous subject/activity block, keeping windows at least
  two source rows apart avoids shared samples. Filtering walking while retaining
  order can remove intervening windows, increasing the separation within that
  block; a different source block does not share its raw samples. Skipping two
  ordered same-subject windows places the first probe at least three window
  positions after the last enrollment window when both are in one block.
- WISDM's adapter emits source-ordered windows within contiguous valid segments
  and never bridges an activity/subject change, timestamp reset/gap or invalid
  record. Within a segment, the same stride argument applies. Across distinct
  segments the emitted windows cannot share raw record positions. Subject and
  walking filtering must preserve that emitted order.

For another T/stride configuration, choose a conservative gap of at least
`ceil(T / stride)` *before* evaluation. This API does not receive T or stride and
cannot validate the gap. The default is for 128/64 only. Enrollment windows may
overlap each other, and probes may overlap other probes; the guard concerns
enrollment versus probes. Enrollment shots are window counts, not independent
gait cycles or independent trials. A stronger future protocol should retain raw
offsets and recording IDs in the adapter and audit sample-range intersections.

This remains **held-out-subject, within-recording verification**. Prefix order
is not proof of a later visit, day, device placement or independent session.

### Templates and scores

`enrollment_scores(embeddings, subjects, enroll_idx, probe_idx)` returns
`(labels, scores, metadata)`. Embeddings may be NumPy arrays or torch tensors of
shape `(N,D)`; tensors are detached and copied to CPU for scoring. Each enrollment
vector is unit normalized, each identity's enrollment vectors are averaged, and
that mean is unit normalized into one frozen template. Zero vectors and
cancelling templates fail explicitly. No classifier is trained on held-out
identities and no probe enters its template.

Each probe is compared to every enrolled identity exactly once. The result is
flattened in probe-index order, then ascending claimed ID. Labels are 1 for a
genuine claim and 0 for an impostor claim. Scores are `cosine_similarity - 1`,
in [-2, 0]; higher means more genuine and acceptance is `score >= threshold`.
Metadata exposes claimed IDs, input indices, template norms, per-subject
enrollment/probe counts and actual comparison counts. It contains no calibrated
threshold. The scorer validates disjoint indices and identity coverage, but
cannot enforce chronological purging for arbitrary manually supplied indices.

With K identities and P probes there are P genuine and P*(K-1) impostor
comparisons. There is no artificial class balancing or repetition. Comparisons
share probes/templates and source recordings, so their number is not the number
of independent trials or users. The impostors here are the other enrolled
subjects, not a separate open-set attacker cohort. Report identity coverage,
per-subject errors and actual counts alongside pooled FAR/FRR. Subject-level
uncertainty is preferable to treating comparisons as independent Bernoulli draws.

## Validation selection and frozen test decisions

Choose the method, feature/PCA settings, enrollment shot count and purge policy
using training/validation only. Choose the threshold from validation
enrollment/probes, then apply that exact threshold to test enrollment/probes.
Enrollment establishes a held-out user's template; it must not refit the
representation, scaler/PCA or global operating threshold. Test labels are for
locked evaluation, never model, hyperparameter, seed or threshold selection.

The following illustrates the interface, assuming `z_val`, `ids_val`, `z_test`
and `ids_test` already come from disjoint, correctly ordered subject cohorts:

```python
from engine.enrollment import enrollment_split, enrollment_scores
from engine.evaluate import calibrate_threshold, fixed_threshold_metrics

ve, vp, val_coverage = enrollment_split(ids_val, 5, 2)
val_labels, val_scores, val_context = enrollment_scores(z_val, ids_val, ve, vp)
calibration = calibrate_threshold(val_labels, val_scores, target_far=0.01)

te, tp, test_coverage = enrollment_split(ids_test, 5, 2)
test_labels, test_scores, test_context = enrollment_scores(z_test, ids_test, te, tp)
test_fixed = fixed_threshold_metrics(
    test_labels, test_scores, threshold=calibration["threshold"]
)
```

Save both coverage dictionaries, scoring metadata, validation calibration,
preprocessing state, selected configuration and split provenance with results.
A descriptive test EER can be reported separately via `compute_eer`; its implied
test-derived threshold is not the frozen operational threshold. Validation FAR
is not a guaranteed test or deployment FAR. Reusing validation for candidate
selection and calibration is a limitation with only four validation identities.

Evaluate **both the existing neural baseline and every candidate** under this
same enrollment protocol. Keep old pair ROC results in a separate result family.
Comparing a candidate's enrollment EER against a baseline's old random-pair EER
does not isolate an architectural improvement. Do not overwrite or relabel old
HAR/WISDM pair metrics. No improvement is guaranteed, and an unfavorable locked
test outcome must remain in the report.

## Fast time/spectral feature diagnostic

`models.gait_features.extract_gait_features(X, sampling_rate)` consumes NumPy or
torch `(N,C,T)` windows and returns float32 `(N,D)`, where
`D = 26 * (C + floor(C/3))`. It adds the vector norm of each adjacent xyz triple
to the original channels: HAR's body acceleration, gyroscope and total
acceleration triples yield 312 features; WISDM xyz yields 104. Only the magnitude
features are invariant to an orthogonal rotation of their xyz triple; the entire
concatenated representation is not orientation invariant.

Each original channel, followed by each magnitude channel, contributes:

| Features | Count | Definition |
| --- | ---: | --- |
| Time statistics | 11 | Mean, population std, RMS, min, max, 10/25/50/75/90% quantiles, mean absolute first difference |
| Autocorrelation | 6 | Normalized products of demeaned lagged signals at 0.1/0.2/0.3/0.5/0.75/1.0 seconds; lags rounded and clipped into 1..T-1 |
| Relative spectral bands | 6 | Demeaned Hann-window rFFT power in [0,.5), [.5,1), [1,2), [2,3), [3,5), [5,Nyquist] Hz; DC excluded, normalized by total non-DC power |
| Spectral descriptors | 3 | Normalized power entropy, dominant frequency (Hz), power-weighted centroid (Hz) |

Bands above Nyquist are empty and give zero. Constant/zero-energy signals have
zero autocorrelation and spectral descriptors. Extraction has no labels, fitted
parameters or RNG. FFT temporaries are bounded by batches of 256 windows. It is a
diagnostic for whether learned embeddings discard useful amplitude, cadence or
spectral information, not an online authentication system.

`GaitFeatureBaseline(sampling_rate, *, standardize=True, pca_components=None)`
adds train-only population-mean/std scaling and optional deterministic,
unwhitened PCA. Its methods are `fit(X_train)`, `transform(X)`,
`fit_transform(X_train)`, `state_dict()` and `from_state_dict(state)`.
The class has no identity input and cannot independently verify the caller's
training cohort. Fit only on optimizer-training subjects, never validation,
test enrollment or test probes. `transform` enforces fitted channel/window
shape and never updates state. The JSON-safe state records training-window
count and all fitted numeric parameters; persist it before test scoring.
PCA rank must be at most `min(n_training_windows - 1, n_features)`.

```python
from models.gait_features import GaitFeatureBaseline

features = GaitFeatureBaseline(sampling_rate=20.0, pca_components=None)
features.fit(X_optimizer_train)
z_val = features.transform(X_val)
frozen_state = features.state_dict()
# Once validation chooses and locks this configuration:
z_test = GaitFeatureBaseline.from_state_dict(frozen_state).transform(X_test)
```

Raw extraction is untrained; the fitted wrapper is an unsupervised preprocessing
baseline. Neither API uses subject IDs for fitting. Train/validation split
selection, appropriate normalization controls and any feature/neural fusion are
the experiment controller's responsibility. There is no automatically registered
model or evaluator CLI switch in this additive implementation.

## Research basis: adaptations, not reproductions

Primary texts inspected on 2026-10-23:

1. Gadaleta and Rossi, **IDNet: Smartphone-based Gait Recognition with
   Convolutional Neural Networks**, [arXiv:1606.03238v3](https://arxiv.org/html/1606.03238v3),
   Sections 5–6. IDNet combines cycle embeddings, a user-specific one-class SVM
   and accumulation of successive cycle scores. It motivates using evidence
   from multiple observations. Our fixed mean enrollment template is an
   adaptation: there is no cycle segmentation, one-class SVM or sequential
   probability-ratio decision. Its reported results are not our expected metrics.
2. Fereidooni et al., **AuthentiSense: A Scalable Behavioral Biometrics
   Authentication Scheme using Few-Shot Learning for Mobile Platforms**,
   [arXiv:2302.02740v1](https://arxiv.org/html/2302.02740v1), Sections IV and V-D1,
   Table V. The paper varies n-shot enrollment evidence and authentication
   duration for mobile interaction behavior. This motivates explicitly fixing
   enrollment shots and evaluating unseen users. Our walking-only HAR/WISDM
   template-cosine protocol uses different data, feature aggregation and scoring;
   it does not reproduce AuthentiSense or transfer its reported performance.
3. Hoang, Choi and Nguyen, **On the Instability of Sensor Orientation in Gait
   Verification on Mobile Phone**, [arXiv:1602.03199v1](https://arxiv.org/html/1602.03199v1),
   Sections 3.2 and 4.3. The paper uses sensor-coordinate compensation,
   time/frequency features and PCA followed by supervised classifiers. It
   motivates retaining time/spectral diagnostics and measuring orientation
   sensitivity. Our xyz norms and compact fixed bands omit that coordinate
   compensation, cycle features and SVM, so this is an adapted diagnostic.

## Focused verification

Run the tiny CPU controls from the repository root; they set torch threads to
one and do no training or dataset/GPU experiments:

```bash
TMPDIR=/home/sherlock/Code/github/VariableThe/artifacts/tmp \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python /home/sherlock/Code/github/VariableThe/.venv/bin/python \
python -m unittest discover -s tests -p test_enrollment_protocol.py -v
```

The tests cover fixed purging with known synthetic raw offsets, disjoint indices,
coverage/exclusions, exhaustive genuine/impostor counts, normalized templates,
known two-identity scores, malformed budgets/inputs, cancelling templates,
synthetic spectral frequencies, xyz magnitude rotation, frozen train-only
scaling and serialized PCA round trips. They establish implementation invariants,
not real-dataset overlap provenance or authentication performance.
