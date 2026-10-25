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

The [25 October follow-up](../reports/2026-10-25-robustness/README.md) adds frozen
full/log-covariance baselines, deterministic source-bound sensor perturbations
and timed takeover simulations. Neither covariance arm passes promotion. Probe
rotation/noise substantially increases rejection; orientation/noise robustness
remains a measurable research problem using existing data. Simulated recording
resets demonstrate bounded expiry and preserve short-follow-up censoring.

The [canonical orientation comparison](../reports/2026-10-25-canonical/README.md)
nominates Canonical362 under the clean low-FAR gates and substantially reduces
the measured constant-coordinate-rotation sensitivity. Mean EER worsens, five
users have higher rejection, and gain/noise sensitivity remains unresolved.
This is a DEV nomination; real phone relocation, time-varying orientation and
single-window live thresholds remain unvalidated.

The [paired nuisance-covariance study](../reports/2026-10-25-nuisance/README.md)
reduces the fixed gain/noise objective FRR from 45.607% to 15.241%, with fourteen
people improving and one tied. It fails thirteen safety gates: clean error
rates/EER and some stressed fold/claimant false accepts worsen. Retain the
canonical baseline; severe noise and the fully rejected DEV person remain open.
All 200 fits, 440 scores and 541 geometry contexts verify.

The [cycle feasibility follow-up](../reports/2026-10-25-cycle/README.md) tests
1,280 fixed synthetic windows before fitting a cycle baseline. Short support,
missing boundaries, degenerate channels and half-stride aliases require a new
protocol. The [exact gated ROC utility](gated_roc.md) now retains failed claims,
reports undefined crossings honestly and avoids floating-point equality errors.
Longer contiguous support and matched validity-gated canonical controls remain
research options using existing DEV recordings; no cycle DEV fit is claimed.

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
