# Window-local gravity/PCA canonical descriptors

`models/gait_canonical.py` adds a deterministic orientation adapter for complete
256-sample windows. It has no fitted parameters and makes no authentication or
noise robustness claim. Implementation checks use independently constructed
synthetic signals. A separate registered DEV study measures verification and
sensitivity performance; its reports retain the empirical results.

## Research basis and scope

[IDNet, section 3.3, equations (5)–(13)](https://arxiv.org/html/1606.03238#S3.SS3)
estimates gravity from mean acceleration, projects acceleration onto its
orthogonal plane, and uses the leading horizontal covariance eigenvector as a
forward axis. Gravity cross forward supplies the lateral axis. Both inertial
triads use this frame. The variance direction is a modeling assumption; it does
not establish the physical direction of travel.

[Section 3.1](https://arxiv.org/html/1606.03238#S3.SS1) describes interpolation
to 200 Hz and 40 Hz FIR filtering. [Section 3.4](https://arxiv.org/html/1606.03238#S3.SS4)
resamples cycles to 200 samples and normalizes their amplitudes. This adapter
instead uses existing fixed 256-sample windows in native units. It adds no
filter, cycle extraction, interpolation, amplitude normalization, CNN, or
classifier. It is not a reproduction of full IDNet.

## API and coordinate convention

```python
from models.gait_canonical import (
    estimate_canonical_frames,
    extract_canonical_features,
)

# X: finite real NumPy-compatible (N, 6, 256) array
# channels: acceleration x/y/z, then gyroscope x/y/z
# Both triads must already share the same sensor coordinate system.
geometry = estimate_canonical_frames(X)
result = extract_canonical_features(X, sampling_rate=100)

# For a consumer that requires every window to have a frame:
features = result.require_all_valid()  # (N, 336), or ValueError
```

`estimate_canonical_frames` returns `CanonicalFrames` with row-aligned fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `frames` | `(N, 2, 3, 3)` | Two bases, with columns forward, lateral, gravity |
| `valid` | `(N,)` | Frame identifiability mask |
| `reason` | `(N,)` | `valid`, `gravity_unidentifiable`, `horizontal_unidentifiable`, or `horizontal_degenerate` |
| `gravity_ratio` | `(N,)` | Norm of acceleration mean divided by acceleration RMS norm |
| `horizontal_energy_ratio` | `(N,)` | Horizontal sample covariance trace divided by raw acceleration mean-square norm |
| `eigengap_ratio` | `(N,)` | Largest-minus-second eigenvalue divided by the largest |

`extract_canonical_features` returns `CanonicalFeatures`, containing
`features`, `geometry`, and a `valid` property forwarding `geometry.valid`.
Both result classes provide `require_all_valid()`: geometry raises on missing
frames; the feature result raises or returns the complete feature matrix.
Empty batches return correctly shaped empty arrays and pass this check.

For each valid basis `B = geometry.frames[n, s]`, canonical coordinates are
`B.T @ X[n, :3]` and `B.T @ X[n, 3:]`. Reconstruction uses `B @ canonical`.
The same proper rotation acts on both triads, preserving each vector norm and
cross-triad inner products. Gravity/DC acceleration remains in the transformed
signal. Acceleration and gyro retain their distinct original units, with no
conversion between g and m/s² or between degrees/s and radians/s.

An arbitrary fixed proper sensor rotation within a window transforms the
unordered frame pair equivariantly. This requires the same coordinate change
for both triads. Independently rotated triads, reflections, time-varying device
orientation, altered placement, dynamic mean acceleration mistaken for gravity,
and actual changes in walking motion are outside this guarantee.

## Explicit PCA sign ambiguity and feature order

A covariance eigenvector defines a line, not a directed forward vector. The
implementation does not choose a sign using any sensor-axis component.
For unit gravity `z` and either leading eigenvector sign `f`, it constructs

```text
l = z cross f
B0 = [ f,  l, z]
B1 = [-f, -l, z] = B0 diag(-1, -1, 1)
```

After numerical orthogonalization, both bases are orthonormal and have
determinant +1. Their order is arbitrary. Neither is asserted to face forward.
For each basis, transform both triads, compute the existing six-channel
descriptors, and concatenate in this exact order:

1. Axis-plus-norm208 in `extract_gait_features` order, with a local stable
   dominant-spectral-peak rule described below.
2. `extract_invariant_features`: 128 vector-triad features, in its existing order.

Average the two 336D vectors in float64 and return float32. Both signs have
equal weight. Sign flips and pair reordering leave the pooled descriptor
unchanged. Rotation invariance holds up to floating-point error for identifiable
frames. Descriptor pooling precedes **all** learned standardization, projection,
or metric inference; averaging nonlinear metric outputs defines another method.
The invariant portion is intentionally retained to match the existing fusion336
interface. No lag covariance block is appended by this module.

The inherited axis extractor uses an exact spectral `argmax`. On balanced peaks,
roundoff from an otherwise equivariant coordinate change can switch its reported
frequency. Only this canonical adapter replaces each axis/norm dominant-frequency
column: select the lowest-frequency bin whose power is at least
`(1 - 64 * float64_eps)` times the maximum. Retain the original zero-energy
criterion `total_power > 1e-24`. Other axis features remain unchanged. The existing
invariant128 block already uses this relative tie convention. This guard resolves
the audited balanced-peak fixture; it is not a global continuity bound at every
possible numerical threshold. The shared historical extractor and control are
unchanged, and this local adaptation is part of the canonical method definition.

## Fixed numerical guards

Write `a(t)` for acceleration, `m = mean(a)`, and
`E = mean(||a(t)||²)`. Center horizontal acceleration and form its sample
covariance with denominator 255. Let its eigenvalues be
`lambda0 <= lambda1 <= lambda2`; the small vertical eigenvalue is included only
to evaluate the ordinary 3×3 covariance. Validity requires all of:

```text
||m|| / sqrt(E)                      > 1e-6
trace(horizontal covariance) / E     > 1e-12
(lambda2 - lambda1) / lambda2         > 1e-6
```

Zero denominators fail the corresponding guard. The amplitude resolution floor
is one part per million relative to the window's own acceleration RMS; the
horizontal energy floor is its square. The relative eigengap floor rejects
nearly tied horizontal directions. These are fixed conservative numerical
conditioning guards, well above float64 rounding at this window length. They
are not empirically validated quality thresholds and provide no bound on
frame error under sensor noise. Pure noise can pass these numerical guards; a
barely separated horizontal eigenspace can rotate substantially under small
noise while remaining valid. Walking/quality gating remains external. No
observations, labels, or tuning selected these tolerances.

One common scalar rescales acceleration internally before covariance arithmetic
to avoid overflow. It cancels in the ratios and does not alter output units or
select the eigenvector sign. Roundoff-negative covariance eigenvalues are
clamped to zero. The selected eigenvector is reprojected horizontally and
orthogonalized with cross products. Changing acceleration units by any positive
representable scalar preserves the mathematical validity tests; descriptor
amplitude features continue to reflect those units.

## Failure, coverage, timing, and memory contract

Every source window retains its original row. Unidentifiable rows contain
**only NaNs** in `frames` and `features`. Horizontal diagnostics are NaN when
gravity is unidentifiable; zero acceleration has gravity ratio zero. Strong
gyro signals cannot rescue an unidentifiable acceleration frame. There is no
identity frame, last-known frame, zero feature vector, or raw-axis fallback.

The caller must carry row identity/provenance into its original trial table,
record invalid endpoints as non-evaluable or abstained according to its
predeclared protocol, and report invalid counts and source coverage. A trial
requiring an invalid window cannot receive authenticated evidence from this
adapter. Selecting only valid rows and reporting accuracy on that reduced
denominator, imputing NaNs, or silently deleting trials violates this contract.
`require_all_valid()` provides a strict entry point for pipelines that have no
explicit abstention path. This module supplies no scores or decisions.

Inputs must have exactly `(N, 6, 256)` shape and real float/integer dtype;
boolean, complex, object, nonnumeric, NaN, and infinity inputs raise
`ValueError`, including invalid gyro in a window lacking gravity. Values must
fit float64. Unrepresentable projected values or descriptor outputs raise
`ValueError` rather than clipping. Sampling rate must be positive with a finite
float64 reciprocal and float32-representable frequency range. Descriptor
underflow conventions are inherited from the existing feature functions.

Inference occurs at the end of the current window. The entire current window
may estimate its own frame; no preceding/future windows, labels, source data,
fitted state, random state, or persistent cache is read. This is window-end
causality, not a promise of sample-by-sample causal coordinates within a window.
Source arrays, including read-only arrays, are never modified. Frame/feature
outputs require O(N) storage; signal temporaries use at most 64 source windows
and 128 sign-transformed windows. Validation also operates in bounded slices.

## Synthetic verification

`tests/test_gait_canonical.py` checks known axes and scalar-reference diagnostics;
independent Rodrigues SO(3) rotations; orthonormality, determinant +1, and
reconstruction; one shared triad frame and native units; all 24 proper signed
axis permutations; eigensolver sign and frame-order invariance; pooling before
a nonlinear metric; current-window independence; bounded batch sizes and
repeatability on read-only inputs; relative tolerance boundaries at varied
scales; rotated degeneracies; source coverage and fail-closed behavior; empty
and integer arrays; and strict finite/range validation. These tests do not
establish biometric performance or solve the previously observed noise issue.

Run without installing dependencies, using the existing interpreter:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 \
uv run --no-project --python ../.venv/bin/python python -B -m pytest \
  -p no:cacheprovider tests/test_gait_canonical.py
ruff check --no-cache models/gait_canonical.py tests/test_gait_canonical.py
ruff format --check --no-cache models/gait_canonical.py tests/test_gait_canonical.py
ty check --python ../.venv/bin/python models/gait_canonical.py tests/test_gait_canonical.py
```
