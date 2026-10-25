# Paired nuisance metric

`models.nuisance_metric.fit_nuisance_metric(features, subject_ids,
view_features=None)` returns `(metric, diagnostics)`. `metric` is the existing
`WithinClassMetric` with automatic clean Ledoit–Wolf shrinkage. The registered
study fixes Canonical362 features and compares clean WCCN against the same clean
covariance plus an identity-balanced paired nuisance second moment with coefficient
exactly **1**. This module accepts any positive feature width so small synthetic
tests can verify the mathematics; the caller enforces Canonical362 in the study.

## Inputs and isolation

`features` must be a nonempty finite real numeric `(N,D)` matrix. `subject_ids`
must contain one nonnegative int64-compatible integer per original row and at
least two distinct subjects. Boolean, complex, string, and object arrays are
rejected. IDs are validated like `WithinClassMetric`; arbitrary nonconsecutive
IDs and unequal row counts are supported.

`view_features=None` selects the unchanged clean automatic WCCN baseline. On
ordinary inputs, its mean, scale, projection, and shrinkage are bitwise identical
to `WithinClassMetric().fit(features, subject_ids)`.

Otherwise, `view_features` must be a finite real numeric `(N,4,D)` array. Its
second axis has exactly four paired views, in caller-supplied order:

1. Gain 0.9.
2. Gain 1.1.
3. Noise plus.
4. Noise minus.

Every view belongs to the clean row at the same first-axis position. Views are
never independent training rows or identities. The API checks shape and numerical
validity, but feature values alone cannot establish semantic view order or row
pairing. The caller owns TRAIN-only identity isolation, feature extraction,
perturbation geometry, deterministic view construction, and row-key pairing. No
data loading, perturbation generation, protocol selection, or study registration
occurs here. Inputs are not modified.

## Frozen mathematics

Let `x_i` denote clean rows, `K` the number of original identities, and `n_j` the
original row count for identity `j`. Compute clean columnwise statistics using
NumPy's population standard deviation:

```text
mu = mean_i(x_i)
s = max(std_i(x_i, ddof=0), 1e-6)        # elementwise
z_i = (x_i - mu) / s
r_i = (z_i - mean_{h: subject_h=j}(z_h)) * sqrt(N / (K*n_j))
C_B = LedoitWolf(assume_centered=True).fit(r).covariance_
```

The clean expression order matches `models/gait_metric.py`. The estimator sees
exactly the original `N` residual rows, with equal total covariance weight per
identity. Only these residuals determine automatic shrinkage. Clean mean, scale,
covariance, shrinkage, and training identity IDs are identical between arms.
`C_B` is computed directly; it is never reconstructed by inverting an already
floored clean projection.

For paired view `v` of clean row `i`, use the clean scale to compute displacement:

```text
delta_iv = (view_features[i,v] - x_i) / s
V = (1/K) * sum_j [ (1/(4*n_j)) * sum_{i: subject_i=j} sum_{v=0}^3
                    delta_iv delta_iv^T ]
C = C_B + 1 * V
```

`V` is intentionally **uncentered**: systematic corruption bias contributes to
the penalty, even if every displacement is the same vector. Each original
identity receives weight `1/K`, regardless of its row count. The four-view
average appears only inside that identity's paired second moment. Clean means
and shrinkage are never estimated from views. For the baseline, `V` is exactly
zero.

Whiten using the existing eigendecomposition and floor rule:

```text
lambda, Q = eigh(C)
floor = max(max(lambda) * 1e-5, 1e-6)
P = (Q * maximum(lambda, floor)**-0.5) @ Q.T
```

The returned metric uses `mu`, `s`, and `P` in the existing transform: standardize
each query, multiply by `P`, L2-normalize each row, and return float32 descriptors.
The existing transform rejects a query exactly equal to the training center.
Whitening regularizes eigenvalues below the floor; in those directions
`P @ C @ P` need not be the identity.

## Diagnostics and serialization

The diagnostics mapping contains exactly these lossless NumPy arrays. Matrix and
eigenvalue arrays are float64; counts are int64. Scalars are zero-dimensional
arrays, not JSON numbers or rounded lists.

| Key | Shape | Meaning |
| --- | --- | --- |
| `clean_covariance` | `(D,D)` | Direct clean Ledoit–Wolf covariance `C_B`, before the eigenvalue floor. |
| `nuisance_second_moment` | `(D,D)` | Uncentered identity-balanced `V`; all zeros for the baseline. |
| `eigenvalues` | `(D,)` | Ascending raw eigenvalues of `C_B + V`, before flooring. |
| `eigenvalue_floor` | `()` | Float64 floor used for whitening. |
| `original_rows` | `()` | Int64 original clean `N`. |
| `original_subjects` | `()` | Int64 original distinct identity count `K`. |
| `views_per_row` | `()` | Int64 `0` for baseline or `4` for the paired arm. |

The metric exposes `mean_`, `scale_`, `projection_`, `training_subject_ids_`, and
`shrinkage_`. Its unchanged `state_dict()` and
`WithinClassMetric.from_state_dict(state)` preserve the inference transform and
training IDs. The existing state format does not serialize `shrinkage_` or the
diagnostics; callers that need these training records must preserve them
separately.

Invalid input types, dimensions, labels, nonfinite values, float64 conversion
overflow, nonfinite intermediate arithmetic, invalid shrinkage, or failed
eigendecomposition raise `ValueError` before a result is returned. Clean,
nuisance, and combined covariance matrices are checked for finiteness, symmetry,
and positive semidefiniteness. The symmetry and PSD tolerance is
`64 * eps_float64 * D`, relative to `max(1, max(abs(matrix)))`; numerical errors
within this tolerance are allowed, and the matrices are not repaired or centered.
The prescribed whitening floor is applied only when forming the projection.

## Research provenance and limits

[Hatch, Kajarekar, and Stolcke (Interspeech 2006), section 2, equations 1–2](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf)
motivates inverse within-class covariance weighting and covariance smoothing.
Its speaker-recognition procedure uses a different smoothing formulation. Adding
the paired uncentered `V`, choosing coefficient 1, using automatic clean
Ledoit–Wolf shrinkage, and using Canonical362 gait features are this study's
mathematical adaptation. This is not a reproduction of that system and provides
no performance guarantee. Synthetic tests establish estimator arithmetic and
validation behavior; they do not establish gait recognition or robustness gains.

## Focused checks without installation

From `gait-biometrics`, using the existing shared interpreter and installed tools
whose versions match the lock (`ruff==0.17.1`, `ty==0.0.90`):

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-project ../.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_nuisance_metric.py tests/test_gait_metric.py
ruff format --check --no-cache models/nuisance_metric.py tests/test_nuisance_metric.py
ruff check --no-cache models/nuisance_metric.py tests/test_nuisance_metric.py
ty check --python ../.venv/bin/python models/nuisance_metric.py tests/test_nuisance_metric.py
```

All test arrays are tiny and synthetic. No real-data or NPZ artifacts are read.
