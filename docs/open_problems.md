# Work that does not need another dataset

This register tracks the 24 October engineering follow-up. Published biometric
results remain descriptive; new software does not retroactively validate them.

| Problem | Implementation / evidence | Status |
| --- | --- | --- |
| Evaluation precision, finite values and checkpoint shape validation | `numerical_validation.md` | Implemented; integration checks pending |
| Recoverable metric-training interruptions | `metric_resume.md` | Implemented; integration checks pending |
| Causal decisions, silence expiry and latched fallback | `continuous_auth.md` | Policy implemented; stream integration pending |
| Fractional training options and late validation failures | `training_validation.md` | In progress |
| Partial HAR extraction mistaken for complete setup | `training_validation.md` | In progress |
| Verification windows sharing raw sensor samples | `window_provenance.md` | In progress for official WISDM v1 |
| Claimant score scales at a low FAR operating point | Fixed raw/Z/S blend comparison, DEV only | In progress |
| Reproducible CPU/CUDA environments and automated checks | `pyproject.toml`, `uv.lock` and CI | In progress |

## Limits that software cannot remove

The anonymous legacy WISDM CSVs have no identity mapping; the authentic v1 loader
is the supported identity source. Released HAR windows do not provide enough
source offsets to certify independent sessions or raw-sample separation.
Contiguous WISDM segments do not establish independent recording days.

Previously inspected TEST users cannot provide a fresh confirmation. Small
cohorts and dependent directed claims cannot certify a population FAR of 1%.
Controlled leg recordings cannot establish phone placement robustness, battery
cost, real device takeover behavior or spoof resistance. Those claims require
appropriate additional observations or hardware measurements.

## Historical reproduction

Numerical validation fixes change source fingerprints intentionally. Keep every
published registration and failed attempt intact. To replay the earlier
hillclimb and pooled-calibration reports, use the source commit recorded in
their report, or the main repository's complete pre-follow-up merge commit
`4014d85e3c1962c3d27f5f425b2b6cc6e7fc6175`. Run that checkout in its own
environment; do not overwrite historical hash manifests with current hashes.
