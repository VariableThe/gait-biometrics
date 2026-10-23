# Six-axis Santos input and deterministic vector invariants

This opt-in extension exposes authentic gyroscope samples and a separate,
NumPy-only descriptor. It leaves `extract_gait_features` and the default
acceleration-only Santos signal, row eligibility, window boundaries, identities,
and source-member hashes unchanged. Historical reports and their pinned source
commits remain historical evidence; this change does not rewrite them.

## Loader API and units

```python
from data.santos import load_santos_windows

# Existing callers retain [N,3,T] acceleration in m/s^2.
accel = load_santos_windows(archive_path, subject_ids)

# Explicit opt-in: [N,6,T], acceleration xyz then gyroscope xyz.
six = load_santos_windows(
    archive_path, subject_ids, window_size=256, stride=256,
    channels="accel_gyro",
)
assert six["metadata"]["channel_names"] == [
    "acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z",
]
assert six["metadata"]["channel_units"] == ["m/s^2"] * 3 + ["degrees/sec"] * 3
```

`channels` accepts exactly `"accel"` (default) or `"accel_gyro"`. Both modes
validate the same exact seven-column source schema and all six sensor values.
An invalid gyro value therefore still breaks an acceleration-only segment.
Both modes retain the same original member SHA-256, canonical
`members_sha256`, frame integers, data-row offsets, coverage, and exclusions.
Empty six-axis outputs have shape `[0,6,T]`. Unselected payloads remain unopened.

Both modes add parallel `channel_names` and `channel_units` lists. The legacy
`channels` metadata remains a list of selected column names. Legacy `units`
remains `"m/s^2"` for default calls; six-axis calls report
`"mixed; see channel_units"`. The loader performs no unit conversion.

The official [Santos raw MCU table](https://www.nature.com/articles/s41597-022-01638-2/tables/3)
declares acceleration in m/s² and gyro in °/s. The descriptor's Methods,
Data processing section describes extracting walking sections into processed
CSVs without describing a gyro unit conversion. Carrying °/s through to those
processed CSVs is an inference from that processing description and source
schema, not a conclusion from inspecting signal magnitudes. The source is
[Figshare v7](https://doi.org/10.6084/m9.figshare.14727231.v7),
licensed CC BY 4.0; preserve the dataset authors' attribution and provenance
specified in [santos_dataset.md](santos_dataset.md).

If a study requires radians/sec, convert only an explicit derived copy and
record the conversion and resulting units in that study's configuration:

```python
import numpy as np

derived = six["X"].astype(np.float64)
derived[:, 3:6, :] *= np.pi / 180.0
derived_channel_units = ["m/s^2"] * 3 + ["rad/s"] * 3
```

Passing the original six-axis windows to the extractor keeps degrees/sec.
Amplitude features change with unit conversion; dimensionless features do not
for a positive scalar conversion applied to a complete triad. Acceleration and
gyro norms are never combined into one mixed-unit norm.

## Extractor contract

```python
from models.gait_invariants import (
    extract_invariant_features, invariant_feature_names,
)

groups = ((0, 1, 2), (3, 4, 5))
features = extract_invariant_features(six["X"], 100, vector_groups=groups)
names = invariant_feature_names(6, vector_groups=groups)
assert features.shape == (len(six["X"]), 128)
assert len(names) == 128
```

`extract_invariant_features(X, sampling_rate, *, vector_groups=None)` accepts
finite real arrays `[N,C,T]` with `T >= 4`, `C` in `{3,6,9}`, and a positive
finite sampling rate in Hz. It returns float32 `[N,64*G]`, including empty
`N=0` inputs. `FEATURES_PER_VECTOR_GROUP` is 64. Default groups are adjacent
xyz triples. Explicit groups are an ordered iterable of three integer channel
indices each, partitioning every channel exactly once. Repeated, missing,
out-of-range, boolean, and noninteger indices fail. The caller must ensure that
each triple is one physical vector in a consistent unit.

| Input | Physical groups in existing channel order | Output dimension |
| --- | --- | --- |
| C3 WISDM/Santos default | acceleration `(0,1,2)` | 64 |
| C6 Santos opt-in | acceleration `(0,1,2)`, gyro `(3,4,5)` | 128 |
| C9 HAR | body acceleration `(0,1,2)`, gyro `(3,4,5)`, total acceleration `(6,7,8)` | 192 |

There are no labels, random draws, fitted parameters, gravity estimators,
canonical-frame transforms, or learned encoders. Processing is independent per
window and triad, in batches of 256. Temporary float64 scaling protects squared
computations; the 31 amplitude features are restored to source units. Inputs or
computed features outside the finite float32 range raise `ValueError` rather
than being clipped. Extremely small amplitude features may underflow on the
requested float32 output. Scaling, dimensionality reduction, metric fitting,
and template selection remain responsibilities of the study runner.

## Exact 64-column block

For a vector sequence `v[t]`, define `c[t] = v[t] - mean_t(v[t])`,
`r[t] = ||v[t]||`, `s[t] = ||c[t]||`, and
`d[t] = ||v[t+1] - v[t]||`. The ten-statistic order is always
`mean, std, rms, min, max, q10, q25, q50, q75, q90`.
Standard deviations use population normalization (`ddof=0`); quantiles use
NumPy's default linear interpolation.

| Zero-based offsets | Exact suffixes, in order | Definition/units |
| --- | --- | --- |
| 0–9 | `norm.{stat}` | Ten statistics of `r`; source triad units |
| 10–19 | `centered_norm.{stat}` | Ten statistics of `s`; source triad units |
| 20–29 | `difference_norm.{stat}` | Ten statistics of `d`; source triad units, **not** a rate-normalized derivative |
| 30 | `mean_vector_norm` | `||mean_t(v[t])||`; source triad units |
| 31–33 | `covariance_eigen_fraction.0`, `.1`, `.2` | Ascending eigenvalues of `mean_t(c c^T)`, each divided by their sum; dimensionless |
| 34–39 | `lag_cosine.{lag}s` | Mean of pointwise normalized `v[t] dot v[t+k]`; dimensionless |
| 40–45 | `centered_lag_correlation.{lag}s` | Sum of `c[t] dot c[t+k]`, divided by the product of the two segment Frobenius norms; dimensionless |
| 46–54 | `vector_spectrum.{spectral}` | Spectrum summed across three centered vector axes, described below |
| 55–63 | `norm_spectrum.{spectral}` | Spectrum of centered `r`, described below |

The six lag suffixes are exactly `0.1s, 0.2s, 0.3s, 0.5s, 0.75s, 1s`.
For each, `k = min(T-1, max(1, round(lag * sampling_rate)))`, using Python's
rounding. At short windows or low rates some lags intentionally coincide.
For pointwise cosine, a pair containing a zero vector contributes zero.
All zero-energy normalized features are zero, including covariance fractions,
centered lag correlation, and spectral descriptors. Nonzero constant vectors
have raw lag cosines of one and centered descriptors of zero.

The nine spectral suffixes, in exact order, are:

```text
power_0_0.5hz, power_0.5_1hz, power_1_2hz, power_2_3hz,
power_3_5hz, power_5_inf_hz, entropy, dominant_hz, centroid_hz
```

For vector power, compute `sum_xyz |rfft(c * hanning(T))|^2`; for norm
power, compute `|rfft((r - mean(r)) * hanning(T))|^2`. Set the DC bin to zero.
The first six features are fractions of remaining power in the half-open bands
`[0,0.5), [0.5,1), [1,2), [2,3), [3,5), [5,infinity)` Hz. Empty bands are zero;
nonzero spectra have band fractions summing to one. Entropy is
`-sum(p log(p)) / log(max(2, F-1))`, where `F=floor(T/2)+1` and `p` is the
normalized non-DC power. Dominant and centroid frequencies are in Hz. Dominant
frequency selects the lowest bin within `64*float64_epsilon` of the normalized
peak. Centered vector or norm energies at or below
`(32*float64_epsilon)^2` of their corresponding uncentered energy are treated as
numerically constant to suppress roundoff-only spectra.

Columns are group-major. Every suffix is prefixed by, for example,
`group0[0,1,2].` or `group1[3,4,5].`; `invariant_feature_names` is the executable
column schema. Feature names reveal group order and indices, not inferred
physical sensor labels.

## What is invariant, and what is discarded

For a constant proper rotation `R`, norms and inner products satisfy
`||Rv||=||v||` and `(Ru) dot (Rv)=u dot v`. Differences commute with a constant
rotation, covariance changes by `R C R^T`, and the vector Fourier transform
changes by `R` at each frequency. These identities justify every block above.
The directional lag and vector spectrum blocks preserve temporal information
that norm-only features lose: a constant vector and a vector tracing a circle
can have identical norms and different descriptors.

No cross-triad dot products are used, so the implementation is also invariant
to a separate constant rotation for each triad. This is a mathematical property
of the representation. A real co-located accelerometer/gyro coordinate change
normally applies a common rotation; independently rotating HAR body and total
acceleration is not a physical sensor simulation. The stronger algebraic
invariance deliberately discards relative directions and phase relationships
between different physical vectors. The descriptors also discard handedness
and are mathematically invariant to reflections.

Use the same temporal processing on all three axes before extraction. Unequal
per-axis standardization of raw input generally breaks rotation invariance.
Train-only standardization of the already extracted scalar features preserves
the property because the same fixed transform receives identical invariants.

Time-varying coordinate rotations do **not** commute with differences, temporal
inner products, or spectra. Physical sensor rotation can add measured angular
velocity; changing placement can change acceleration through lever arms and
attachment dynamics. This API therefore makes no claim of invariance to
moving sensors, left/right leg placement, different sessions, translations or
biases, time warping, changed gait, or changed units. Centering selected blocks
does not make the entire descriptor bias-invariant.

## Research adaptation and evaluation boundary

[Gadaleta and Rossi, IDNet, Section 3.2, Eq. 1](https://arxiv.org/html/1606.03238)
uses acceleration magnitude for orientation-independent gait-cycle extraction.
Its Section 3.3 instead estimates a gravity/PCA reference frame for subsequent
signals, and Section 4.2 studies accelerometer/gyro fusion. This implementation
adopts the motivation for magnitude and authentic separate gyro input, then
adds explicit temporal invariants. It does not reproduce IDNet's cycle
extraction, PCA frame, neural network, classifier, or reported accuracy.

[Kobayashi, Hasida and Otsu, ICASSP 2011, Sections 2.1–2.2, Eqs. 4 and 7–10](https://staff.aist.go.jp/takumi.kobayashi/publication/2011/ICASSP2011.pdf)
derive a rotation-invariant Fourier correlation matrix. Its diagonal is the
xyz-summed spectral power used here. This descriptor summarizes that diagonal;
it omits the paper's off-diagonal cross-frequency correlations, log mapping,
and classifier. The shared-axis Hann taper preserves the constant-rotation
argument but does not establish temporal-shift invariance for this finite-window
descriptor.

[Zhong and Deng, IJCB 2014, Sections 3.1–3.3, Eqs. 3 and 6–7](https://www.baesystems.com/en-us/dam/jcr:aa05526e-72e2-472c-be8c-fe7b1bebbefb/TS-2014-070-Sensor-Orientation-Invariant.pdf)
construct normalized pairwise inner products and organize them by temporal lag.
The six lag-cosine means here are sparse scalar summaries of that principle.
They omit the full gait dynamic image, distribution modeling, and i-vector
learning. The paper's static-relative-frame assumption and Section 6 placement
limitation match the narrower invariance claim above.

[Guimarães, Sousa and Correia, Sensors 2021, Sections 2.4.1, 2.4.4 and 2.5.3](https://www.mdpi.com/1424-8220/21/11/3940)
uses angular-rate magnitude energy, acceleration magnitude for gait events,
and synthetic rotations to assess orientation robustness. Those choices support
magnitude processing and rotation tests here. Their foot-worn gait measurement
task does not validate biometric identification or arbitrary placement changes.

[Baek et al., TRI-HAR, ISWC 2026, Sections 3.1–3.4 and 5.4](https://arxiv.org/html/2608.15621)
groups vector streams by physical IMU location, learns equivariant features,
then produces invariant scalars before location fusion. Its Eq. 5 uses the same
orthogonality identity for inner-product invariants. This implementation uses
fixed statistics with an independently invariant block for each physical vector,
discarding more cross-vector geometry; it does not implement TRI-HAR's learned
frame or network. TRI-HAR studies activity classification under fixed per-location
rotations, not cross-day identity verification, and leaves comparison with
handcrafted invariant pipelines to future work.

The invariant identities above are mathematical design arguments, not evidence
of improved identity verification. Fixed descriptors are intentionally
complementary to the existing axis/magnitude baseline; which works better is
an empirical question under the existing subject/session protocol.

Select descriptor and template choices using the five development folds only.
Keep any fitted scaler/PCA/metric within each fold's optimizer-training
identities. Refit the chosen configuration on development identities and
calibrate each candidate separately on CALIB before its one TEST evaluation.
All previous HAR, WISDM, and Santos TEST cohorts have already been inspected:
new follow-ups are exploratory and must never be described as fresh blind
tests. This implementation read no real signal payloads and ran no data jobs.
