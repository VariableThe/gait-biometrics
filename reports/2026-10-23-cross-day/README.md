# First locked cross-day study — Santos v7

This is held-out-subject, cross-day verification on controlled leg-mounted
walking, measured once after a frozen development/calibration protocol. It is
separate from the HAR/WISDM within-recording results; their EERs are not controls
for this dataset. It does not establish phone authentication or deployment safety.

## Result and exact support

- Selected method: **wccn_auto**, chosen by five development subject folds.
- Descriptive test EER: **7.000%**; ROC AUC **0.988800**.
- Frozen-threshold test FAR: **0.500%** (1/200).
- Frozen-threshold test FRR: **22.000%** (11/50).
- Test support: 50 genuine and 200 directed impostor claims.
- Calibration used five other users: 49 genuine,
  196 impostor claims; attained FAR
  0.510% and FRR 36.735%.
- Threshold **-0.6393263152747406**, accept iff `cosine - 1 >= threshold`.
  It came only from calibration and was not changed after test scoring.

The test has only five people. Decisions reuse people, enrollment templates and
trials, so 250 comparisons do not constitute 250 independent observations.
FAR resolution is 0.500 percentage points; calibration
resolution is 0.510.
No confidence interval, extreme-low-FAR assurance, cross-device robustness,
online takeover delay or continuous passive authentication is established.
Full per-claimed-user and per-probe-user denominators/errors are in `summary.json`.
User 6 accounts for 8 of the 11 false rejects (8/10 genuine claims rejected).
Users 15 and 22 have none, user 26 has one and user 28 has two. The aggregate
rate therefore hides a severe usability failure for one evaluated person.

## Frozen design

The exact participant split was specified using directory metadata before signal
inspection: fifteen development, five calibration, five untouched test users.
Each of five folds trains on twelve development identities and validates on
three other identities. All feature centering/scaling and covariance fitting
use that training population only. Both days of training identities may fit;
every validation/calibration/test identity enrolls only from day1 and probes
only from day2. Five calibration identities select only an empirical threshold.
Candidate selection uses equal-fold mean EER; ties follow declared method order.

| Candidate | Mean development-fold EER |
| --- | ---: |
| standard | 18.023% |
| wccn_auto | 11.736% |
| wccn_0.2 | 12.736% |

The transform is refitted once on all fifteen development identities, frozen
as `metric.npz`, and never fitted to calibration or test data. Window descriptors
are 104 dimensions at native 100 Hz. Windows are 256 samples/stride256, never
crossing a trial/day/person boundary. Unit window vectors average into unit
trial vectors; day1 trial vectors receive equal weight in identity templates.
Every day2 trial vector is scored against every eligible template in its cohort.

The first actual DEV load produced 588 windows across 300 recordings. The
official archive has 499 processed MCU trials: user12/day2/trial9 is absent,
reducing calibration support to 49 genuine/196 impostor claims. No trial or
identity is reconstructed. Exact exclusions, frames, member hashes and coverage
are retained in the complete artifacts.

Source commit `4cbd10b` was pushed before first test access. Independent source
review passed; a second review checked the concrete calibration lock before
evaluation. Config, implementation, archive, fitted metric and DEV/calibration
member/data hashes were rechecked before the exclusive `test_attempt.json`
reservation. The test was loaded/scored once, with no overwrite/retry flag.
Lock checksum: `fa605237ef4f7420cbf423bf3ead094207878f095ef22a7308118a803c12c911`.

## Provenance and reproduction

Santos, Wanderley, Tavares and Rocha,
[*A multi-sensor human gait dataset captured through an optical system and
inertial measurement units*](https://www.nature.com/articles/s41597-022-01638-2),
[Figshare v7, DOI 10.6084/m9.figshare.14727231.v7](https://doi.org/10.6084/m9.figshare.14727231.v7),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Derived operations: acceleration selection, nonoverlapping windows, handcrafted
descriptors, training-only metric fitting and trial/day aggregation. This is a
new verification protocol rather than a reproduction of the authors' analysis.
MCU acceleration is m/s²; seven-column files also contain gyro, which was
validated but excluded from the primary input. No resampling/scaling or
leg-side inference was introduced; placements are pooled with unknown side.

Follow [the full stage guide](../../docs/cross_day_verification.md). The downloader
verifies pinned bytes/MD5/SHA256; the reader never extracts archive members.
Original registry/lock/test-result bytes are gzip-compressed losslessly to avoid
repeating several megabytes of recording metadata in review diffs. For example,
`gzip -dc registry.json.gz > registry.json` restores the exact bytes referenced
by the lock; compare SHA256 to `validation.json`. Compact outcomes and all fifteen
development fold results remain directly readable as JSON. Metric NPZ contains
only numeric arrays and loads with `allow_pickle=False`.

Local validation: **331 tests +311 subtests** main, **327 +311** Student B;
focused cross-day **41 +96**. Eight existing ONNX/quantization warnings remain.
Mirroring delivers the same experiment, not another independent test sample.
All three real stages ran on Wiker CPU; GPU compute was unnecessary here.
