# Source-bound walking gate

`models/walking_gate.py` provides a CPU binary logistic walking detector. It uses
18 fixed magnitude, distribution, jerk, dynamic-energy, spectral and
autocorrelation features. Training standardizes features with **fit subjects
only** and uses one class-balanced L2 logistic fit (`C=1`, LBFGS, 1000 iterations,
`tol=1e-8`). There is no feature search, tuning grid, neural training or GPU use.
Walking means the authoritative metadata name `Walking`; Jogging, Upstairs,
Downstairs, Sitting and Standing are other activities. Label number zero alone
does not establish walking. Outputs are logistic scores in `(0,1)`, without a
claim of calibrated probability.

Prediction requires a `SourceContract` containing source, nominal sample rate,
ordered channels, per-channel units and exact window length. This version
supports three total-acceleration axes including gravity, with explicit
`m/s^2` or `wisdm_native_10_per_g` units.
The registered WISDM contract is `wisdm_v1`, 20 Hz, `acc_x,acc_y,acc_z`, 128
samples and `wisdm_native_10_per_g` units. The supplied raw-about metadata defines
10 raw units as 1g (9.81 m/s²); unconverted values are not labelled exact SI.
Mismatches fail before extraction; there is no implicit unit conversion,
axis selection, resampling or transfer. A WISDM phone gate cannot be applied to
Santos leg 100 Hz six-axis data. There is **no cross-device walking robustness
claim**, and matching metadata alone does not establish robustness for a new
placement, session or acquisition domain.

Each score uses a completed trailing window. There is no future-window
normalization, centered stream filtering, smoothing or state adaptation. A
128-sample window covers nominally 6.4 seconds at 20 Hz (first-to-last sample
span 6.35 seconds); startup needs a full window. Timestamp the score at its last
included sample. The caller must preserve acquisition continuity and exclude
windows crossing gaps, resets or recordings. This is causal availability, not
instantaneous activity recognition; the development adapter excludes transition
windows using authoritative labels, so the study does not measure transitions.

The spectral feature searches 0.5–4 Hz using a within-window Hann taper and
excludes DC. Sub-bands are `[0.5,1.5)` and `[1.5,4]` Hz. Autocorrelation searches
integer lags `ceil(fs/4)..floor(2fs)`, normalized by full-window zero-lag energy.
These peaks are periodicity proxies, not validated step-rate measurements.
Inference processes at most `batch_size` windows at a time (default 64), with
working memory proportional to that bound plus the output vector. The window
extractor rejects nonfinite and absolute acceleration above `1e6` source units.

The numeric JSON model contains the feature schema, source contract, fit-only
means/scales, coefficients, intercept, frozen threshold, fit subject provenance
and integrity digest. It contains no pickle or estimator object. Strict schema,
dimension, finiteness, positive-scale and digest checks reject numeric corruption.
SHA256 provides accidental-change detection, not authentication against an actor
who can rewrite all files and hashes. NumPy is the only inference dependency;
scikit-learn and threadpoolctl are imported only when fitting.

```python
from models.walking_gate import SourceContract, WalkingGate

gate = WalkingGate.load("/absolute/study/model.json")
contract = SourceContract(
    source="wisdm_v1", sample_rate=20.0,
    channels=("acc_x", "acc_y", "acc_z"), units=("wisdm_native_10_per_g",) * 3,
    window_length=128,
)
probabilities = gate.predict_proba(completed_windows, contract=contract)  # (N,3,128)
walking = probabilities >= gate.threshold
```

The existing runtime's external `walking_probability` API may consume these
scores **only after the caller checks this same contract**. The runtime does not
validate acquisition metadata itself. Its policy walking threshold is separate
from this study's selected detector threshold; using `gate.predict` implements
the detector operating point. This change does not alter runtime or score policy.

## Registered development study

The fixed protocol below separates implementation checks from measured source
performance. The follow-up report records which registered runs were completed
and their results. Run from the repository root with the existing environment;
installation is unnecessary:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run --no-project --python ../.venv/bin/python -m scripts.walking_gate_study \
  --stage run --threads 1 --data-root /absolute/datasets \
  --out-root /absolute/new-walking-gate-study

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run --no-project --python ../.venv/bin/python -m scripts.walking_gate_study \
  --stage verify --threads 1 --out-root /absolute/new-walking-gate-study
```

`run` requires an absent output directory and never overwrites/resumes. It checks
the official AR v1.1 raw SHA256, then calls the existing provenance adapter with
`source='wisdm_v1', split='train', window_size=128, stride=64`. The common raw file
is hashed and parsed source-wide by that adapter; only source TRAIN subjects are
windowed. The source partition is the repository's reproducible subject split,
not a published official WISDM split: sorted IDs 1–36, `default_rng(42)`, first
28 TRAIN and remaining eight TEST. No TEST windows are requested or materialized.
No Santos payload or metadata is read.

Before any fit, TRAIN IDs are shuffled with independent seed 20261024 and fixed
as 16 fit, six calibration, six held DEV validation subjects. Balancing is by
subject count, without examining labels or fitting alternatives. No person can
cross roles. A role lacking either class fails without resplitting. All surviving
windows remain, including people missing a class; undefined per-person rates
are reported as JSON null. Overlap is allowed only inside one person's same
role/activity/segment, and does not create independent observations.

`registration.json` closes source-code hashes, official raw manifest, original
window digest, numeric-cache hashes, real subject roles, all class/person counts,
fixed settings and software versions **before the single fit**. `features.npz`
contains all fit/calibration/validation fixed features, activity/binary labels,
real IDs, role codes 0/1/2 and exact raw support/segment/timestamp provenance.
Calibration maximizes window-weighted balanced accuracy over unique scores plus
0 and 1, with `p>=threshold`. Exact ties prefer fewer other-activity passes, then
the largest threshold. Validation cannot change features, weights or threshold.

`model.json`, `scores.npz` and `results.json` preserve the numeric model, both
held score sets, confusion counts/rates and per-person/per-activity reports.
Performance records wall and CPU time for five repeated end-to-end inference
passes over validation windows, after one warmup, batch size 64 and one thread.
It excludes loading, fitting, imports and disk I/O; it does not claim deployment
latency or improvement over another method. Repeated inference adds no fits or
validation-driven choices. Counts/rates are descriptive; no independent-window
confidence interval or real-world/cross-session guarantee is asserted.

`verify` reads registration, current source code and numeric artifacts with
`allow_pickle=False`; it never opens the raw source or invokes the adapter. It
checks fingerprints, roles/support leakage, fit-only statistics and provenance,
replays calibration and validation scores, recomputes the calibration threshold
and confusion reports, and writes fresh `verification.json`. It requires the
registered software versions, performs zero fits and does not remeasure timing.
It cannot independently reconstruct raw features without reopening raw data.
Repeated verification refuses to overwrite its previous result.

Synthetic checks (no real datasets):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run --no-project --python ../.venv/bin/python -m pytest \
  -p no:cacheprovider -q tests/test_walking_gate.py
```
