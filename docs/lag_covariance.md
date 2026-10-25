# Full and log lag covariance features

`models.lag_covariance.extract_lag_covariance_features` is a stateless extractor
for `(N, C, T)` windows with adjacent xyz triads and `C` equal to 3, 6, or 9.
It returns float64 features and supports an empty `N` dimension. It has no
training, labels, fitted reference matrix, or dataset access.

```python
from models.lag_covariance import extract_lag_covariance_features

fullcov = extract_lag_covariance_features(X, logarithm=False)
logcov = extract_lag_covariance_features(X, logarithm=True)
```

The study fixes `lags=(0, 5, 10, 20)` and `shrinkage=0.2` in advance. The API
allows other settings for mathematical checks; these are not parameters to
select on calibration or test identities. The existing spectrum control is
`models.session_robust.extract_lagged_covariance_features`. That extractor is
unchanged. Dataset assembly, the 336-dimensional base feature extractor,
concatenation, training, and comparison belong to the parent study.

## Definition and layout

Treat each xyz triad independently, preserving its physical units. In
particular, accelerometer and gyroscope channels never share a covariance
matrix or trace normalization. For a triad `A`, let `L = T - max(lags)` and
stack channels in the supplied lag order, then xyz order:

```text
Z[3*k + axis, t] = A[axis, t + lags[k]],  t = 0, ..., L-1.
```

All lag slices therefore use the same aligned base times. Center each row of
`Z` over time. With `d = 3 * len(lags)`, define the sample covariance and its
regularized, trace-normalized matrix as

```text
G   = Z_centered @ Z_centered.T / (L - 1)
tau = max(trace(G), 1e-8)
S   = (1 - shrinkage) * G / tau + shrinkage * I_d / d.
```

At the frozen settings, `d=12` and `S = 0.8*G/tau + 0.2*I_12/12`.
If the trace is below the floor, `trace(S)` can be less than one. A constant
triad gives `S = 0.2*I_12/12` and `log(tau) = log(1e-8)`.

For `fullcov`, vectorize `S`. For `logcov`, vectorize its symmetric matrix
logarithm, `Q @ diag(log(lambda)) @ Q.T`, using a symmetric eigendecomposition.
This is a matrix logarithm, not an elementwise logarithm. The implementation
diagonalizes `G/tau`, which has the same eigenvectors as `S`, and applies the
shrinkage to its eigenvalues in log space. Negative eigenvalues caused by
roundoff in the positive semidefinite covariance are clamped to zero before
that affine transform. This preserves the mathematical positive shrinkage
floor even when `shrinkage/d` underflows in float64.

Both branches use the row-major upper triangle, including the diagonal:
`(0,0), (0,1), ..., (0,d-1), (1,1), ..., (d-1,d-1)`. Multiply each
off-diagonal entry by `sqrt(2)` and append `log(tau)` as the last coordinate
of that triad. Concatenate triad blocks in input channel order. The weighted
upper triangle preserves Frobenius inner products and distances between
symmetric matrices; the appended log trace is a separate energy coordinate.

Each default block has `12*13/2 + 1 = 79` coordinates. Output shapes are
`(N,79)` for three channels, `(N,158)` for six, and `(N,237)` for nine. In the
six-channel study, concatenating the 336 base coordinates yields `(N,494)`.
With custom lags the width is `(C/3) * (d*(d+1)/2 + 1)`.

Centering removes channel translations within floating-point precision.
Above the trace floor, uniform positive scaling within one triad preserves
its matrix coordinates and shifts its energy coordinate by twice the log
scale. Full matrix coordinates retain orientation information: a fixed
rotation changes their entries in general. The existing eigenvalue-only
spectrum control discards that information under a fixed orthogonal triad
rotation. No orientation invariance is claimed for `fullcov` or `logcov`.

## Numeric contract

`X` must contain finite real floating-point or integer values. Boolean,
complex, object, and string arrays are rejected. `logarithm` is required and
must be a Python or NumPy boolean. Lags must be a nonempty sequence of distinct
nonnegative integers; booleans and integer-valued floats are rejected. Lag
order is preserved and zero need not be present. Require `T-max(lags) >= 2`.
`shrinkage` must be a finite real scalar in `(0,1]` that remains positive
when converted to float64; booleans and array-valued settings are rejected.
Zero shrinkage is excluded so rank deficiency has a defined matrix logarithm.
Invalid inputs raise `ValueError`; omitting the required keyword raises
`TypeError`.

Processing uses at most 128 windows per batch, one triad at a time. Scaling
before centering follows the approach in `models/session_robust.py`; a second
scaling of centered deviations protects their covariance calculation from
underflow. The raw covariance and raw trace never need to be represented:
the log trace is accumulated from scale logarithms, and the normalized
covariance is calculated with bounded coordinates. Consequently even finite
signals whose squared amplitudes overflow float64 are supported. Extended
floating-point inputs are scaled before conversion to float64 where that
dtype is available. An unrepresentable final feature is rejected rather than
returned as infinity. Zero, constant, rank-deficient, and extremely small
signals produce finite features. Input arrays are never modified.

## Research attribution and scope

Alexandre Barachant, Stéphane Bonnet, Marco Congedo, and Christian Jutten,
[“BCI Signal Classification using a Riemannian-based kernel,” ESANN 2012,
pp. 97–102](https://www.esann.org/sites/default/files/proceedings/legacy/es2012-130.pdf),
provides the mathematical connection. Section 2.1 discusses sample covariance
features and regularization. Section 2.3 develops the tangent mapping and the
identity-reference special case in Equation 6, immediately before Section 3:
the kernel becomes the Frobenius inner product of matrix logarithms. Section 3
compares vectorized covariance and kernel methods, including the identity and
geometric-mean reference choices, on motor-imagery EEG.

Here, `logcov` is an identity-reference log-Euclidean mathematical adaptation
to lagged gait triads. Lag stacking, separate triad units, trace normalization,
fixed shrinkage, and the appended log trace are this feature design. It neither
reproduces that EEG experiment nor implements its full classifier or a learned
geometric-mean reference. A downstream concatenated-feature metric is not
claimed to be a pure Riemannian classifier. The citation supplies mathematical
provenance, not evidence of gait-authentication performance.

`tests/test_lag_covariance.py` uses only synthetic signals. It checks independent
sample-covariance calculations, diagonal matrix logarithms, weighted Frobenius
geometry, logarithm/exponential consistency, finite degenerate and extreme
inputs, translation behavior, separate units, orientation sensitivity, batches,
input immutability, and strict validation. No raw data, saved artifacts, or
fitting are needed.
