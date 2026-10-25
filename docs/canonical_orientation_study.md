# Frozen DEV canonical orientation study

This bounded, exploratory experiment compares three predeclared representations
on the same fifteen Santos DEV identities and twenty crossfit folds as
[`covariance_robustness_study.py`](../scripts/covariance_robustness_study.py).
Earlier DEV and TEST inspection is acknowledged. Original CALIB and TEST member
payloads are never requested by this runner. Repeated DEV use provides neither
independent confirmation nor a population FAR guarantee. No DEV results are
included in this protocol guide. Run only after the canonical module and study
reviews pass; measured outcomes belong in a separate frozen evidence report.

## Freeze brief and representations

The implementation is [`canonical_orientation_study.py`](../scripts/canonical_orientation_study.py).
The fixed order and dimensions are:

1. `control` (362): the existing `prior.features(data)["lagcov"]` unchanged.
2. `canonical336` (336): `extract_canonical_features(X, 100).require_all_valid()`.
   Both proper gravity/PCA sign frames receive equal descriptor weight **before**
   any learned standardization, metric projection, or scoring.
3. `canonical362` (362): `canonical336` followed by the **original raw-window**
   26D lag-spectrum block, exactly `control[:, 336:]`. The lag block is never
   re-extracted from canonical coordinates and is not the newer full/log
   covariance representation.

The [canonical adapter specification](gait_canonical.md) defines its one shared
frame for both native-unit triads, PCA sign ambiguity and numerical guards.
This runner adds no filtering, amplitude normalization, learned axes, tuning,
new sensor data, neural training, GPU work or dependencies. Every source window
has shape `(6, 256)` at 100 Hz, stride 256, acceleration in m/s² and gyro in
degrees/sec. The three arm dimensions are checked at extraction.

Every arm uses automatic identity-balanced WCCN, raw `mean_multi` scores (equal
mean/two-medoid enrollment blend), and policy C. Each outer fold has four
nine-person inner fits scored on four disjoint held triples. Their **clean**
pooled scores calibrate an empirical FAR <= 1% cutoff. A twelve-person clean
TRAIN refit then scores the disjoint three-person outer validation set at that
unchanged cutoff. No stressed feature enters training or calibration.

The complete fixed budget is **300 actual fits, 240 inner score sets and 420
outer score sets**: `20 × 3 × (4 + 1)` fits and `20 × 3 × 7` outer conditions.
`budget()` independently derives these totals before DEV is loaded. Run-time
fit counting and complete artifact assignment checks enforce them again. Every
DEV person validates four times; observations, claims and folds are dependent.

## Geometry and coverage before features or fits

`registration.json` freezes `plan()`, all source hashes, Python version and
NumPy/SciPy/scikit-learn versions **before DEV loading**. Its source closure
includes all old helper `SOURCES`, the three new study files and the canonical
module, tests and documentation. `plan()["origins"]` names the copied protocol
and directly reused helpers. Historical study globals and source are unchanged;
there is no dynamic module cloning or arm monkeypatching.

The source-derived claim graphs are built from original day/trial inventories.
Next, `estimate_canonical_frames` evaluates all clean DEV windows, then every
fold's complete validation subset for each of the seven conditions, including
clean. These **141 contexts** are completed before the first descriptor or fit.
Noise scales come solely from that fold's twelve refit TRAIN identities.

`development.json` preserves the data/member fingerprint, complete loader
metadata (including zero-window/excluded source recordings), exact source claim
graphs, and the geometry manifest. Each context has a lossless NPZ containing:

- Every ordered `(subject, day, trial, start)` row key, including invalid rows.
- Both frames, validity mask, reason strings, gravity ratio, horizontal energy
  ratio and relative eigengap, including undefined NaN diagnostics.
- JSON counts by reason, source-window and row-key fingerprints, and a hash of
  the NPZ. The manifest also stores every fold's six repeated triad scales and
  the exact twelve TRAIN IDs supplying them.

If **any** context contains an unidentifiable frame, the complete geometry and
development manifests remain on disk and the run aborts **before any features
or fits**. No window is dropped, imputed, given a fallback frame or removed from
the raw claim graph. `require_all_valid()` also guards feature extraction.
Registered relative guards stay strictly at `1e-6` gravity, `1e-12` horizontal
energy and `1e-6` eigengap; none may be changed after DEV inspection.

Every actual score is checked against its original exact claim/enrollment graph.
Even a finite metric transform that causes a trial mean to cancel cannot
silently shrink the denominator.

## Seven fixed diagnostic conditions

The exact inherited order is `clean`, `rotation30`, `rotation90`, `gain09`,
`gain11`, `noise01`, `noise05`. Perturbations act on day2 probes only. Day1
enrollment windows stay bit-identical. Rotations are common proper 30°/90°
rotations about x for both triads; gains are 0.9/1.1 for both triads. Noise is
Gaussian at 1%/5% of the twelve-person TRAIN dynamic triad RMS: center each
TRAIN window/channel over time, average squared residuals over windows, xyz and
time per triad, take the square root, and repeat that scale across xyz.
Seed 20261025 and the original row keys make noise reproducible. The inherited
perturbation helper returns values to the original float32 sensor storage.

All conditions use the same refit model, fixed clean enrollment and fixed
clean-calibrated threshold. Stress results are diagnostic only and cannot
nominate an arm, choose a cutoff or expand this grid. These transformations
measure synthetic sensitivity; they are not measured placement distributions,
physical relocation, new walks, or evidence of noise robustness.

## Exact clean-only nomination

All gates use rational fractions of integer errors and claim counts. Equal-fold
mean FAR must be <= 1%; the number of folds with FAR > 2% cannot exceed control;
mean FRR must be strictly lower; and at least 10 of the 15 people must have
four-appearance mean FRR no worse than control. Eligible candidates tie by lower
FRR, then lower FAR, then the fixed arm order above. Otherwise retain control.
Display floats and all stress outcomes are excluded from these decisions.

## API, artifacts and verification

The CLI accepts exactly `--stage {run,verify}`, `--archive-path`, `--out-root`,
matching the covariance study. Python callers may pass an object with
`archive_path` and `out_root` to `run(args)` or `verify(args)`; `run` returns its
summary and `verify` returns its verification report. `plan()`, `budget()`,
`hashes()`, `representations(data)` and `summarize(rows)` expose the frozen
contract without loading an archive automatically.

Successful runs persist 300 lossless numeric metric states (mean, scale,
projection, training IDs and shrinkage diagnostics), 660 lossless score NPZs
(labels/scores), exact score metadata, ordered per-fold threshold JSON,
per-person/pair error counts and exact selection. Numeric model records include
training row/feature fingerprints, window counts and dimensions. All artifacts
are hashed and assignment inventories are complete and unique.

Verification checks source/runtime registration **before DEV loading**. It
rebuilds source coverage, scales and all geometry contexts, checks every row and
diagnostic exactly, and compares the full development/member manifest before
features. It restores models without fitting, verifies every TRAIN assignment,
dimension, training row/feature fingerprints, fitted mean/scale and shrinkage
record, then replays all 660 score sets. Score tolerance is the inherited
`atol=rtol=1e-6`, with exact labels and metadata and **identical decisions at the
frozen threshold**, even inside that numerical tolerance. Pooled thresholds,
cached decision counts, all rows and exact selection recompute exactly. Model
projection integrity is established by its artifact hash and score replay;
verification does not estimate a new covariance or refit WCCN.

Each run needs a fresh empty root. Failed attempts and their registrations must
be retained. Do not rewrite registration to accept changed source, tolerances,
runtime or geometry. Use a separately versioned experiment/root and disclose
discarded attempts if an audited implementation correction is necessary.

## Validation and parent execution prompt

All tests in
[`test_canonical_orientation_study.py`](../tests/test_canonical_orientation_study.py)
construct synthetic `(N,6,256)` signals. Real canonical extraction and all seven
real perturbations are integrated without fitting. A separate tiny five-feature
TRAIN fixture exercises 300 actual CPU WCCN fits and the complete run/verify
inventory; descriptors are substituted only in that fixture. Tests also cover
exact gates/ties, runtime/source/tolerance guards before DEV, source graph
cancellation, a final-fold stress geometry failure before the first fit, and
geometry/model/threshold/count/assignment/selection tampering. No test opens an
archive or existing artifact payload.

Run the implementation checks with the existing environment, no installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 \
uv run --no-project --python ../.venv/bin/python python -B -m pytest \
  -p no:cacheprovider tests/test_canonical_orientation_study.py
ruff check --no-cache scripts/canonical_orientation_study.py tests/test_canonical_orientation_study.py
ruff format --check --no-cache scripts/canonical_orientation_study.py tests/test_canonical_orientation_study.py
ty check --python ../.venv/bin/python scripts/canonical_orientation_study.py tests/test_canonical_orientation_study.py
```

After independent review, freeze the exact three-arm, seven-condition protocol
and its complete source/runtime closure. Run once in a fresh output root, then
verify with the identical interpreter, source, environment and archive argument.
Retain failed attempts and their geometry diagnostics. Do not change numerical
guards after DEV inspection, add arms/conditions, recalibrate on stress or use
stress for nomination. The source registration binds the canonical adapter's
local 64-epsilon spectral tie rule; the historical control extractor is unchanged.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 \
uv run --no-project --python ../.venv/bin/python python -B \
  -m scripts.canonical_orientation_study --stage run \
  --archive-path "$SANTOS_ARCHIVE_PATH" --out-root "$CANONICAL_STUDY_ROOT"
# Repeat the exact command/environment with --stage verify.
```
