# Work that does not need another dataset

This register tracks the 24 October engineering follow-up. Published biometric
results remain descriptive; new software does not retroactively validate them.

| Problem | Implementation / evidence | Status |
| --- | --- | --- |
| Evaluation precision, finite values and checkpoint shape validation | [Numerical validation](numerical_validation.md) | Implemented and tested |
| Recoverable metric-training interruptions | [Metric resume](metric_resume.md) | Exact CPU and local deterministic FP32 CUDA equivalence verified |
| Causal decisions, silence expiry and latched fallback | [Policy](continuous_auth.md), [strict stream/evaluation](stream_authentication.md) | Implemented and tested |
| Fractional training options and late validation failures | [Training validation](training_validation.md) | Fixed before optimization |
| Partial HAR extraction mistaken for complete setup | [Dataset repair](training_validation.md) | Fixed; generated-archive regressions pass |
| Verification windows sharing raw sensor samples | [Raw provenance](window_provenance.md) | Implemented; existing WISDM TRAIN parity/nonoverlap verified |
| Claimant score scales at a low FAR operating point | [Fixed raw/Z/S comparison](claimant_normalization.md) | 100 fits replayed; alternatives fail gates, retain raw |
| Walking activity gate | [Source-bound detector](walking_gate.md) | One registered TRAIN-only fit and held DEV validation |
| Raw window → gate → encoder → template score → decision | [Window adapter](window_authentication.md) | Implemented; explicit single-window calibration required |
| Reproducible CPU/CUDA environments and automated checks | [Environment/CI](reproducible_environment.md) | Frozen CPU installation and both repository suites pass; dated audits pass |

The [follow-up report](../reports/2026-10-24-engineering/README.md) retains all
negative/positive DEV evidence, per-person limits, hashes, source snapshots and
validation logs. These are opt-in protocols; old metrics are not rewritten.

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
