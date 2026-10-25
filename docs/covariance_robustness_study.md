# DEV covariance comparison and sensor sensitivity

This new experiment compares three fixed representations using the existing
Santos DEV identities. Original CALIB/TEST member payloads are never requested.
It is an exploratory extension after earlier DEV and TEST inspection. Reusing
these people does not provide independent confirmation or a population FAR
guarantee. No published TEST threshold or deployment adapter changes.

The control is fusion336 plus the existing26 lag-spectrum features. The two
new494-dimensional arms retain the complete regularized per-triad lag covariance,
with and without its matrix logarithm. See [lag_covariance.md](lag_covariance.md)
for the formula and independent mathematical checks. Barachant et al.'s
[ESANN2012 paper](https://www.esann.org/sites/default/files/proceedings/legacy/es2012-130.pdf),
sections2.1 and2.3, supports the covariance/log-map mathematics; its EEG experiment
is not reproduced by this gait adaptation. Full covariance versus log covariance
isolates the logarithmic map. Comparison with the spectrum control also changes
regularization and parameterization, so a gain cannot be attributed solely to
retaining covariance directions.

Twenty registered balanced outer folds each use four nine-person inner fits and
one twelve-person refit per representation: **300 actual WCCN fits**. Each arm
keeps automatic shrinkage, the equal mean/two-medoid enrollment blend, raw scores
and threshold policy C. Four held-triple score sets calibrate empirical FAR<=1%;
that cutoff is applied unchanged to the refit and its separate outer triple.
Every person's validation observations recur in four folds. Scores and fold
rates are dependent; reported rates are equal-fold means, not pooled counts.

Clean-only nomination uses exact count fractions: mean FAR<=1%, no increase in
folds above2% FAR, strictly lower mean FRR and at least10/15 person means no worse
than control. Ties use FRR, FAR and registered arm order. Failure retains control.
The complete grid is retained; no unregistered early stopping or extra tuning.

Seven diagnostic conditions apply to day2 probes only: clean,30/90-degree common
triad rotations about x, gain0.9/1.1 and Gaussian noise1%/5%. Noise scales are
triad dynamic RMS from the twelve refit TRAIN identities only: center each window
channel over time, then average squared residuals over its triad/windows/time.
Scale repeats across xyz. Seed20261025 and original subject/day/trial/start keys
make noise reproducible. Enrollment remains bit-identical. Synthetic results
never participate in nomination or threshold calibration. Values return to the
source float32 storage before descriptor extraction.

[Um et al.2017](https://arxiv.org/abs/1706.00527), section3.2, motivates controlled
rotation/scaling/noise tests and cautions that transformations need not preserve
labels. [Hoang et al.2015](https://arxiv.org/abs/1602.03199) motivates orientation
sensitivity specifically for gait verification. Stress magnitudes here are fixed
engineering probes, not measured phone error distributions, spoofing experiments
or new independent walking recordings. They do not reproduce MODA or a physical
sensor-displacement simulator. Full matrix features are orientation-sensitive.

The runner writes source/runtime/plan registration **before loading DEV**, then
source-derived trial inventories and directed graphs **before features/fitting**.
Zero-window source trials are recorded by the existing adapter; no scorer can
silently remove further trials. Every numeric model/score artifact is hashed.
Verification reloads only DEV, checks source/runtime/member fingerprints, all
model assignments and TRAIN means/scales, replays all660 score sets with1e-6
tolerance, and requires unchanged fresh/cache decisions at frozen thresholds.
Thresholds and metrics recompute from lossless caches exactly; verification fits
no model. It is numerical reproduction, not independent validation of raw truth.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --python ../.venv/bin/python python \
  -m scripts.covariance_robustness_study --stage run \
  --archive-path datasets/santos/santos-v7.zip \
  --out-root ../artifacts/covariance-robustness-v1
# Repeat with --stage verify, identical arguments, frozen source and runtime.
```

Each run requires a fresh empty output directory. A failed attempt remains on
disk. Do not edit its registration to match later code; use a new version/root
for corrected experiments and report discarded work. No GPU work or download is
required. Synthetic tests count actual fit calls and enforce role separation,
coverage, exact ties, the fold-tail safety gate and frozen-threshold decisions.
