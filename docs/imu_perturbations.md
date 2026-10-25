# Evaluation-only IMU stress tests

`data/imu_perturbations.py` applies deterministic synthetic stresses to NumPy
`float32`/`float64` arrays shaped **(N, 6, T)**, with channels
`[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`. The second triad is **angular
velocity**, not an orientation matrix. `T >= 1`; empty `N` is supported.
There are no training hooks, dataset reads, model calls, fits, or reference
estimators. `data/gait_transforms.py` is a separate training augmentation module
and is unchanged.

These stresses measure sensitivity of a frozen pipeline on existing windows.
They do **not** create new independent data, reproduce MODA, or simulate realistic
phone relocation, placement changes, sensor attachment dynamics, or gait.
The 90-degree rotation is a deliberately stronger orientation challenge, not a
claim about how often such changes occur in deployment.

## Public API

```python
IMUPerturbation(
    *, name="clean", rotation_axis=(1, 1, 1), rotation_degrees=0,
    gain=(1, 1), bias=(0, 0, 0, 0, 0, 0),
    units=("caller/native", "caller/native"),
    noise_fraction=(0, 0), reference_std=None, reference_id=None,
    hold_every=None, hold_offset=None, time_scale=1,
)
apply_imu_perturbation(
    windows, spec, *, window_keys, seed=0, batch_size=128, timestamps=None,
) -> PerturbationResult
iter_imu_perturbation_batches(
    windows, spec, *, window_keys, seed=0, batch_size=128, timestamps=None,
) -> Iterator[PerturbationResult]
modest_stress_grid(
    *, reference_std, reference_id, units=("caller/native", "caller/native"),
) -> tuple[IMUPerturbation, ...]
```

`IMUPerturbation` is a frozen dataclass. Sequence parameters are copied into
tuples; axes are normalized on construction. Unknown settings fail.
`spec.to_dict()` returns detached JSON-native parameters, and
`spec.parameter_hash` returns their canonical-JSON SHA-256.

`PerturbationResult` contains `.windows`, `.timestamps`, and `.metadata`.
Windows and supplied timestamps are copied, retain their shape/dtype, and never
alias the inputs. Metadata records parameters, parameter hash, seed, run hash,
keys, local row range, shape, dtype, conventions and evaluation-only status.
`run_hash` covers the parameter hash and seed. **Neither hash identifies source
signals, timestamps, or the full experiment.** Retain source manifests, keys,
reference provenance, software versions, and frozen protocol alongside results.

## Parameter conventions and validation

Transforms execute in this fixed order: time interpolation, rotation, gain,
bias, Gaussian noise, sample hold.

- **Rotation:** one fixed, right-handed active SO(3) matrix, `v_out = R @ v_in`,
  shared across all timesteps and **both triads** in every window. The axis must
  contain three finite real values and be nonzero; its magnitude is irrelevant.
  Signed degrees must be in `[-180, 180]`. No random axes or per-axis reflections.
- **Gain:** two dimensionless, positive factors, one isotropic factor per triad.
  Each must be in `[1e-6, 10]`. Use `(g, g)` for a common sensor gain.
- **Bias:** six constant channel offsets, added in the caller's input/native
  units. Each is finite and in `[-1e12, 1e12]`. `units` contains two nonempty
  labels (acceleration, angular velocity). Defaults explicitly mean
  `caller/native`; no conversion between g and m/s² or deg/s and rad/s occurs.
- **Noise:** two dimensionless fractions in `[0, 1]`, one per triad. Supply six
  fixed TRAIN-derived channel standard deviations in `reference_std`, plus a
  nonempty `reference_id` identifying the frozen TRAIN reference. Each standard
  deviation is finite and in `[0, 1e12]`; a zero reference channel gets zero
  noise. For channel `c`, noise standard deviation is
  `noise_fraction[c // 3] * reference_std[c]`, in native units. Reference values
  are never estimated, normalized, updated, or checked against probe statistics.
  The caller guarantees TRAIN-only provenance and consistent units. A reference
  identifier without values, values without an identifier, or active noise
  without both fails. Supplying a reference with zero noise is allowed.
- **Sample drop/hold:** declare `hold_every` as an integer in `[2, 2**31-1]`.
  `hold_offset` defaults to `hold_every - 1` and must be an integer in
  `[1, hold_every - 1]`. Replace all six channels at zero-based indices
  `offset + k*every` with their previous sample. Thus `hold_every=10` holds indices
  `9, 19, 29, ...`; the first sample is retained. This represents a declared
  missing-sample/last-value policy. It removes no timestamps, changes no length,
  and performs no cross-window hold. A period longer than `T` may have no effect.
- **Optional time/rate stress:** `time_scale` is in `[0.5, 2]`. On the sample
  grid, define `center=(T-1)/2` and
  `q[t]=clip(center + time_scale*(t-center), 0, T-1)`. Linearly interpolate each
  channel at `q[t]`; clipping holds endpoints. Values above one compress motion
  within the window; values below one expand the central portion. There is no
  anti-alias filter, acceleration/velocity amplitude correction, biomechanical
  model, or physical trajectory reconstruction. This is a **signal stress, not
  physically faithful gait simulation**. Supplied timestamps are retained and
  are not used as interpolation coordinates, including on nonuniform grids.

All numerical parameters reject booleans (including NumPy booleans), strings,
NaN and infinity. Integer parameters reject fractional values and integer-valued
floats. Seeds and integer keys must be in `[0, 2**64-1]`; batch sizes must be
integers in `[1, 4096]`. Text labels/keys are nonempty, have no control characters,
and use at most 1024 UTF-8 bytes. Input arrays reject nonfinite values, integers,
booleans, complex numbers, objects and unsupported floating dtypes. Intermediate
math uses float64; results preserve input dtype. Overflow raises `ValueError`
rather than returning nonfinite values. Numeric limits are API safety bounds,
not empirical sensor calibrations. Default settings return a **bit-exact copy**,
including signed zeros.

Optional `timestamps` must be a numeric NumPy array of shape `(T,)` (shared) or
`(N,T)`, finite and strictly increasing within each window. They retain their
values, shape, and dtype. No window reorder, circular shift, reversal, reslicing,
or subject/recording mixing occurs. Time stress samples monotonically within the
same window; hold substitutes values at existing positions. The caller guarantees
source subject, recording, window, triad and timestamp alignment, and must retain
that provenance when applying downstream scoring.

## Frozen modest grid

`MODEST_STRESS_NAMES` fixes the order below. Each condition changes only the
listed setting; default axis is normalized `(1,1,1)` and all other settings are
identity. References and units are supplied once when creating the grid.

- `clean`: exact identity.
- `rotation15`, `rotation45`, `rotation90`: fixed signed positive angles in degrees.
- `gain0.9`, `gain1.1`: shared gain on both triads.
- `noise0.01`, `noise0.05`: respectively 1% and 5% of the supplied TRAIN channel
  standard deviations, added in native units to both triads.
- `hold_every10`: retain length and hold every tenth sample, starting at index 9.

Freeze the whole tuple, TRAIN reference, seed, stable keys and analysis plan
**before** inspecting evaluation data. No bias or time stress is hidden in this
grid. Optional bias/time conditions require separately declared, frozen settings.
DEV-only exploration must remain clearly labeled; do not use held-out results to
choose strengths or refit a frozen scorer, threshold, scaler, or feature extractor.

```python
from data.imu_perturbations import modest_stress_grid, iter_imu_perturbation_batches

# These six numbers and identifier come from an already frozen TRAIN reference.
# The module provides no estimator and performs no I/O.
grid = modest_stress_grid(
    reference_std=train_channel_std,
    reference_id=train_reference_identifier,
    units=("m/s^2", "rad/s"),
)
frozen_parameters = [spec.to_dict() for spec in grid]
for spec in grid:
    for result in iter_imu_perturbation_batches(
        dev_windows, spec, window_keys=source_window_keys,
        timestamps=dev_timestamps, seed=197, batch_size=128,
    ):
        # Feed result.windows through the existing frozen evaluation pipeline.
        # Retain result.metadata and source provenance alongside aligned scores.
        pass
```

Prefer perturbing native signals before the frozen preprocessing pipeline. If
inputs use another representation, declare its units and supply TRAIN reference
standard deviations in exactly that representation; raw-unit references cannot
be silently reused after normalization.

## Determinism and memory

Every window needs a stable, unique string or nonnegative integer key, preferably
encoding subject/recording/window identity. Integer `1` and string `"1"` differ.
Noise uses a local NumPy `Generator(PCG64)` seeded by SHA-256 of the version,
explicit seed and typed key. It never touches the global RNG. The same key and
seed use common standard-normal draws across stress conditions, permitting paired
comparisons. No subject ID or noise seed is inferred from signal values.
Pin NumPy and record versions for reproducibility across environments.

Changing batch size, splitting externally, or selecting a window with its
original key produces exactly the same output. **Never reset window keys to
`range(len(batch))` across external batches.** Duplicate keys are rejected within
each call; checking uniqueness across calls is the caller's responsibility.

The iterator allocates signal workspace proportional to `batch_size * 6 * T`,
with one window of float64 working storage. Key validation keeps O(N) scalar
metadata. It does not allocate a transformed dataset or materialize all signal
validation masks; it can consume an already supplied memory-mapped array without
opening files itself. Consume/release yielded batches to retain the bound.
Value/timestamp validation is incremental: a later invalid batch may raise after
earlier batches were yielded. The collecting `apply_imu_perturbation` additionally
allocates the full output; its temporary computation remains bounded by batches.

## Motivation and limits from the primary paper

Wu, Guo and Qin, [MODA: Motion-Drift Augmentation for Inertial Human Motion
Analysis, CVPR 2025, pp. 27771–27781](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_MODA_Motion-Drift_Augmentation_for_Inertial_Human_Motion_Analysis_CVPR_2025_paper.pdf),
§§3–4, motivates checking physical consistency rather than applying arbitrary
time-series operations to inertial modalities. Its §4.2 synthesizes readings
using time-dependent sensor self-rotation and sliding offsets tied to motion.
The paper's rotation-valued modality consists of SO(3) orientation matrices;
its restrictions on directly adding noise to those matrices must not be
conflated with this module's Euclidean angular-velocity vectors.

This implementation is **not a MODA reproduction**: it has neither motion/bone
trajectories nor calibration/offset dynamics. A shared constant proper rotation
is a consistent coordinate-frame stress for the two vector triads. Gain, bias,
noise, hold and interpolation remain declared synthetic measurement stresses.
Any robustness result concerns these existing, correlated source windows under
the specified stresses; it establishes neither realism of phone relocation nor
generalization to independent users, recordings, placements, or real device drift.

Tests use synthetic CPU arrays, independent SciPy rotations/interpolation, norm
and cross-triad invariants, coordinate equivariance, exact streaming equivalence,
and validation/overflow cases. They require no dataset or experiment artifacts.
