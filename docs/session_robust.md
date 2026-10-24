# Exploratory session-robust backends

This follow-up is **exploratory**: all old TEST results have already been seen.
There is no fresh confirmatory TEST claim. Fit only DEV training identities;
never fit means, scales, nuisance directions, covariance, or hyperparameters on
CALIB or TEST. The parent owns compute, integration, protocol selection, and
evaluation. These backends introduce no network, Torch dependency, or data job.

## Lagged covariance spectrum

`extract_lagged_covariance_features(X, *, lags=(0,5,10,20))` accepts finite
real numeric `(N,C,T)` arrays with `C` equal to 3, 6, or 9. Adjacent xyz triples
are independent physical vectors. Empty batches are supported. Lags are
distinct nonnegative integer sample offsets; booleans and fractional values
are rejected. At least two aligned observations must remain.

For each triad, stack `x[t+lag]` over the requested lags using the **same** base
times `t=0,...,T-max(lags)-1`. Center each stacked channel across these base
times and compute sample covariance with denominator `T-max(lags)-1`.
Let `L=3*len(lags)` and `tau=trace(covariance)`. Emit the `L` ascending
eigenvalues of `covariance/max(tau,1e-8)`, each floored at `1e-8` then logged,
followed by `log(max(tau,1e-8))`. Output is float32, group-major, shape
`(N,(L+1)*G)`: defaults give 13, 26, or 39 columns. Covariance is evaluated in
scaled float64 coordinates and its raw trace in the log domain to avoid
overflow for extreme finite float64 signals. Constant windows give thirteen
copies of `log(1e-8)` per default triad.

A constant proper rotation `Q` of a triad applies `diag(Q,...,Q)` to its lag
stack. Channel centering commutes with that rotation, so covariance undergoes
orthogonal conjugation. Its eigenvalues and trace are invariant. Independent
constant rotations of different triads also preserve the output because there
are no cross-triad products. This proof does not cover time-varying rotations,
changed sensor placement, or changed timing. Numerical checks use tolerance.

**Raw units are declared, not inferred.** No six-channel gyro heuristic, implicit
unit conversion, or resampling is performed for any channel count. Acceleration,
gyro, or other triads must each have internally consistent units supplied by the
caller. The logged trace depends on those units (squared raw units before the
log); changing amplitude by `a` adds `2*log(abs(a))` when above the floor.
The trace floor also breaks scale invariance at very small energy.

## Paired-day nuisance projection followed by WCCN

`NuisanceProjectedMetric(nuisance_directions=2, shrinkage='auto')` exposes
`fit(features, subjects, days)`, `transform(features)`, `state_dict()`, and
`from_state_dict(state)`. Features must be nonempty finite real matrices;
subjects and days are integer vectors. Fit requires at least two training
identities, exactly day labels 1 and 2, and both days for every identity.

Fit first estimates the mean and population standard deviation of all supplied
training rows, with a `1e-6` standard deviation floor. For each training identity,
form standardized `mean(day2)-mean(day1)` and stack these rows into `S x D`.
The leading right singular vectors form an orthonormal nuisance basis `B`.
Use at most the requested number and the numerical rank of the difference
matrix; zero singular directions are not arbitrarily removed. Project every
standardized training row as `z-(z@B.T)@B`. Then fit the existing
`WithinClassMetric('wccn', shrinkage)` on all projected training rows and their
identities. That backend performs its own train standardization and regularized
within-class covariance estimation. Transformation follows the identical chain
and returns the backend's float32 unit cosine vectors. A descriptor at the WCCN
training center has no cosine direction and retains the backend's explicit error.

This is a supervised **paired-day adaptation inspired by NAP**, not a
reproduction of the cited speaker-verification systems. Two directions and the
default lag list are hypotheses to investigate for FRR; no improvement is
claimed. Removing directions can also discard identity information.

`training_subject_ids_` preserves sorted training identities, while
`training_day_counts_` records each identity's day-1/day-2 row counts. State is
a flat dictionary of numeric arrays with version, requested direction count,
shrinkage setting (numeric `-1` means auto), fitted shrinkage, standardization,
basis, singular values, counts, IDs, and the WCCN state. Save with
`np.savez(path, **metric.state_dict())`; load with `np.load(path,
allow_pickle=False)` and `NuisanceProjectedMetric.from_state_dict(archive)`.
Restoration and transformation never refit. The caller remains responsible for
recording its DEV split and confirming that supplied IDs belong to it; the API
cannot infer CALIB/TEST membership from numeric labels.

## Verified primary references

- Alex Solomonoff, William M. Campbell, and Ian Boardman, “Advances in channel
  compensation for SVM speaker recognition,” *ICASSP 2005*, vol. 1,
  pp. I-629–I-631, March 19–23, 2005.
  [MIT Lincoln Laboratory publication record](https://www.ll.mit.edu/r-d/publications/advances-channel-compensation-svm-speaker-recognition),
  [DOI: 10.1109/ICASSP.2005.1415192](https://doi.org/10.1109/ICASSP.2005.1415192).
  The authors' institution describes projection-based removal of nonspeaker
  variation learned from multisession/channel variation. This supports the
  method's motivation; it does not validate this gait adaptation.
- Alex Solomonoff, William M. Campbell, and Carl B. Quillen, “Nuisance attribute
  projection,” May 2007. [MIT Lincoln Laboratory primary publication
  record](https://www.ll.mit.edu/r-d/publications/nuisance-attribute-projection).
  The institutional record labels this a chapter in *Speech Communication*;
  no unverified volume, pagination, or DOI is supplied here. The exact title
  belongs to this 2007 record, not an invented 2005 paper of that title.
- Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke, “Within-class
  covariance normalization for SVM-based speaker recognition,” *Interspeech
  2006*, [ISCA primary archive](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html),
  [DOI: 10.21437/Interspeech.2006-183](https://doi.org/10.21437/Interspeech.2006-183).
  This is the existing WCCN backend's reference, not a new implementation claim.

Tiny synthetic tests are in `tests/test_session_robust.py`; they cover manual
aligned covariance, common and independent triad rotations, zero/constant and
extreme signals, manual nuisance removal, strict validation, and numeric NPZ
round trips with refitting disabled. No real observations are used.
