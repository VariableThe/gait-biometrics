# Santos cross-day descriptor verification

This opt-in CPU descriptor study implements the population and session proposal in
[cross_session_acquisition.md](cross_session_acquisition.md). It does not change
the existing within-recording study or add a neural architecture. The runner is
[scripts/cross_day_verification.py](../scripts/cross_day_verification.py), using
the separate `data.santos.load_santos_windows` adapter.

The source is Santos, Wanderley, Tavares and Rocha, *A multi-sensor human gait
dataset captured through an optical system and inertial measurement units*,
[official Figshare v7](https://doi.org/10.6084/m9.figshare.14727231.v7), article
14727231, file 35882624, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Derived transformations are acceleration channel selection, nonoverlapping
windowing, handcrafted descriptors, a development-fitted metric and trial/day
aggregation. This is a new protocol, not a reproduction of the authors' analysis.

Official archive verification on 2026-10-23 matched 1,523,321,221
bytes, MD5 `35adf0860202f2aefa37560dc0748ebd`, SHA-256
`ed944fa88a98ce1058327ff4b3c8f919fbf58626fc7e3c5256360ee080a3a304`.
The runner independently checks all three at every stage. Reading compressed
bytes for integrity and listing filenames do not load numerical test signals.
The observed inventory has 499 processed MCU trials for the authentic 25 IDs.
`user12/day2/0009` is absent; no replacement trial is invented. The exact prefix
is `A multi-sensor human gait dataset/processed_data/user01/`, with recording
names such as `capture_user01_day1_0001_imu_walk.csv`.

Official development-only header inspection found exactly
`frame,acc x,acc y,acc z,gyro x,gyro y,gyro z`. This differs from the three-axis
companion repository. The primary experiment selects **only acceleration xyz**,
in m/s² at native nominal 100 Hz. The adapter validates the declared source
schema, numeric finiteness and frame boundaries; gyro values are not features.
No extra acceleration conversion, resampling, per-trial centering, axis rotation
or leg-side inference is applied. Leg placements are pooled and side is unknown.
The frame index is provenance, not a sensor channel.

## Fixed subjects and selection

Authentic IDs remain integers, including gaps. The lists are literal constants;
the runner never regenerates them with another PRNG.

| Role | IDs in prescribed permutation order |
| --- | --- |
| Development | 7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17 |
| Calibration | 21, 18, 23, 12, 30 |
| Untouched test | 26, 28, 22, 15, 6 |

Five folds validate consecutive groups of three development IDs; the other
twelve train the transform. Both days of those twelve identities may enter
fitting. Each fold uses held-out day1 enrollment and day2 probes. The fixed
candidate order is `standard`, `wccn_auto`, `wccn_0.2`. The winner minimizes the
unweighted mean of the five fold EERs; exact ties prefer that listed order.
Calibration identities never select the descriptor or metric.

All candidates use the existing `extract_gait_features(X, 100)` implementation.
Its channel-major descriptor has 26 features for each acceleration axis and its
xyz magnitude, hence 104 values: mean, standard deviation, RMS, extrema, five
quantiles, mean absolute first difference, six normalized autocorrelations at
0.1/0.2/0.3/0.5/0.75/1.0 s, six relative spectral bands with edges
0/0.5/1/2/3/5/∞ Hz, spectral entropy, dominant frequency and centroid. Spectrum
uses a Hann window after subtracting the mean, with DC removed. The code source
hash freezes exact equations, quantile conventions and numerical safeguards.

`WithinClassMetric` fits the training-window mean and population standard
deviation (floored at 1e-6). `standard` uses the identity projection. WCCN forms
within-identity residuals with equal total covariance weight per training
identity, then uses either Ledoit–Wolf estimated shrinkage or fixed 0.2 isotropic
shrinkage. It applies the symmetric inverse square root with eigenvalue floor
`max(max_eigenvalue * 1e-5, 1e-6)`. No normalization or covariance estimate is
fitted to validation, calibration or test values. After CV, the winning transform
is refitted once on all fifteen development identities and saved as NumPy NPZ.

## Windows, templates and scores

Keep each processed MCU trial separate. Windows contain 256 samples (2.56 s),
stride 256. Incomplete tails are discarded. Nothing crosses a subject, day,
trial or recording boundary, and all original frame starts and recording
metadata enter the fingerprint. Adapter exclusions are included in reports.

Every transformed window vector is unit-normalized. Within a trial, average
these window vectors and unit-normalize that average. For enrollment, average
the eligible **day1 trial vectors with equal trial weight**, then unit-normalize
the template. Each eligible day2 trial vector is one probe. A subject must have
at least one eligible trial on both days. At least two identities must survive
in an evaluated population. Missing days and zero-norm trial/template means are
reported as exclusions. A zero-norm window transform fails the stage because it
has no defined cosine direction (norm tolerance 1e-12).

Score every day2 probe against every eligible day1 identity template in the
same held-out cohort. The genuine claim uses its actual identity; all other
claims are directed impostor decisions. Score is `clip(cosine, -1, 1) - 1`, so
larger values mean greater similarity. Accept iff `score >= threshold`; ties
are accepted together. Trial counts, identity counts, genuine/impostor decision
counts, missing trials and subject exclusions are explicit in every report.

## Three stages and test-access gate

Run from the repository root after the environment setup in the project README.
The examples use a repository-local `.venv`; choose your existing interpreter
with `--python` if it lives elsewhere. Acquisition verifies the pinned archive
without extracting it or parsing signals.

```bash
mkdir -p artifacts/tmp
uv run --no-project --python .venv/bin/python python -m scripts.setup_santos_dataset
TMPDIR="$PWD/artifacts/tmp" uv run --no-project --python .venv/bin/python python -m scripts.cross_day_verification \
  --stage train --archive-path datasets/santos/santos-v7.zip \
  --out-root artifacts/santos-cross-day-primary

TMPDIR="$PWD/artifacts/tmp" uv run --no-project --python .venv/bin/python python -m scripts.cross_day_verification \
  --stage lock --archive-path datasets/santos/santos-v7.zip \
  --out-root artifacts/santos-cross-day-primary

# Inspect the concrete registry and lock and complete independent review here.
TMPDIR="$PWD/artifacts/tmp" uv run --no-project --python .venv/bin/python python -m scripts.cross_day_verification \
  --stage evaluate --archive-path datasets/santos/santos-v7.zip \
  --out-root artifacts/santos-cross-day-primary
```

`train` reads only development signal members, produces `registry.json` with all
fifteen CV results, and writes the winner's `metric.npz`. It records exact config,
source hashes, archive integrity/inventory, selected development member hashes
and a fingerprint of development arrays **and all metadata**. The registry also
records any declared prior test inspection. Existing study artifacts prevent
overwriting the output root.

`lock` checks those frozen inputs, reloads development to verify its fingerprint,
and loads only calibration signals. The saved metric is restored without fitting.
The existing `engine.evaluate.calibrate_threshold` selects the least restrictive
attainable threshold with empirical calibration FAR ≤ the fixed target 0.01.
The resulting `protocol_lock.json` freezes that threshold, attained FAR/FRR and
counts, calibration member and data/metadata hashes, exact split IDs, registry
and metric hashes, config and source hashes, and its own canonical checksum.
This file is written **before any test numerical loader runs**.

`evaluate` requires that lock and verifies its checksum, every source/config/
registry/metric/archive hash, and reloaded development/calibration fingerprints
and selected member hashes. It then exclusively creates `test_attempt.json`
**before** calling the test loader. A second evaluation refuses even when a
previous evaluation crashed or failed after reservation. The sole result is
exclusively created `test_result.json`; the CLI has no overwrite or retest flag.
The operator should preserve all output files. The checksum detects changes,
not malicious replacement of a lock and all its checksums.

An inspected test set can only be used in a separately declared follow-up study:
choose a fresh output root and pass `--prior-test-inspection 'what was inspected,
when, and how it influenced this follow-up'` to its `train` stage. That disclosure
is frozen in the registry, lock and result. A new directory cannot make previously
inspected identities untouched; the program cannot detect inspections outside
its own output root. Complete code/adapter changes and source review before
locking, because any source change invalidates the lock.

The script imports the existing evaluation helpers lazily. Their containing
module imports Torch, although this runner never constructs or trains a neural
network or requests a GPU. The synthetic tests extract the existing pure helper
functions from their AST to exercise the exact score/tie logic without importing
Torch. This test-only isolation does not replace production metric functions.

## Reporting limits and checks

The result uses `engine.evaluate.compute_eer`, `roc_auc_score` and
`fixed_threshold_metrics`. The operating FAR/FRR use the frozen calibration
threshold. EER is a descriptive interpolated ROC crossing; its implicit test
threshold is never reused. Per-subject errors are reported both by claim and by
probe identity via `engine.subject_protocol.per_subject_error_rates`.

With all test trials eligible there are 50 genuine and 200 directed impostor
decisions, hence FAR resolution 1/200 = 0.005. The known missing calibration
trial implies at most 49 genuine and 196 impostor calibration decisions before
further window exclusions. Report actual counts and attained FAR, since a 1%
target need not be exactly attainable. Reused people, templates and trials are
dependent; no confidence interval or low-FAR population assurance is claimed.
The scope is held-out-subject cross-day verification on controlled, leg-mounted
short walks, not evidence for cross-device or continuous passive authentication.

Synthetic tests cover the exact split/folds, day separation, equal trial weights,
fixed-order candidate ties, calibration ties, fitting only development identities,
lock-before-test access, metadata tampering and one-shot test access. They use
invented NumPy signals and a mocked archive audit; they do not inspect official
test values or initiate a real study.
