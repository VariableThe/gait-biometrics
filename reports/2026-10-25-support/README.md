# Saved-metadata continuity inventory — 25 October 2026

**The proposed 768-sample protocol fails cohort support. Retain Canonical362.**
Across 300 source-present DEV trials, only one continuity segment supports a
768-sample block. It belongs to subject 9, day 1, trial 8. No identity has such
support on both enrollment and probe days. This stops that protocol before any
cycle extraction or model fitting.

Complete continuity segments retain raw support for all fifteen identities and
all historical trials. They include 33,810 valid rows omitted by the old tiling,
a **22.4609375% increase over the old covered rows**. This is an observation-support
finding, not a measured improvement in features or authentication.

## Frozen comparison and observed support

The [registered tool](../../scripts/santos_support_inventory.py) uses only two
public metadata snapshots from the [canonical study](../2026-10-25-canonical/README.md).
It freezes source, runtime, input bytes, definitions and flags before decoding
those snapshots. Its [protocol document](../../docs/santos_support_inventory.md)
and [tests](../../tests/test_santos_support_inventory.py) belong to the source
closure. The main and Student B freeze commits are recorded in
[provenance](provenance.json).

| Registered support definition | Units | Unique valid rows covered | Unused valid rows | Supported trials | Identities supported on both days |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 256-sample tiles | 588 | 150,528 | 33,810 | 299/300 | 15/15 |
| Fixed 768-sample blocks | 1 | 768 | 183,570 | 1/300 | 0/15 |
| Complete continuity segments | 328 | 184,338 | 0 | 300/300 | 15/15 |

Units in the three rows have different lengths and are not independent samples.
The complete-segment row counts every nonempty segment conceptually, including
very short fragments; it makes no extraction or trainability claim. Source rate
is 100 Hz. Blocks restart at each original segment boundary; trials and segments
are never concatenated.

There are **328 segments**: 29 shorter than 256 samples, 298 from 256 through 767,
and one at least 768. Minimum/median/maximum lengths are **1/605/931 samples**.
Six recordings are fragmented and 294 are unbroken. The fixed cohort includes
all fifteen identities and all twenty original outer folds. The proposed
minimum-two-units count flag passes for all 100 inner/refit training owners
under historical tiles and complete segments, and for none under fixed768.
This count flag does not establish covariance rank or extraction feasibility.

Subject 19, day 2, trial 1 is present with 470 valid rows, split by 21 gaps into
22 segments. It produces no historical tile. Its source presence is retained
separately; the original **299-trial graph is unchanged**, so the new raw-support
ledger does not silently add evaluation claims. The metadata reconciles the
historical 588 windows, 28 continuity breaks and 33,810 tails.

## Integrity and execution

One exclusive actual attempt completed, followed by saved-only replay in both
repositories. The [full result](evidence/result.json.gz), both exact input
snapshots, registration, diagnostics and attempt reservation are published.
The 12,181,414-byte result is losslessly compressed; restored bytes are bound
by SHA-256 in provenance. [Summary JSON](summary.json) provides the tables,
subject/day ledgers and fixed flags without replacing the full result.

The standard-library job ran with the frozen **59/60 CPU-second soft/hard limit,
512 MiB address-space limit and one CPU**. The additional
[Python access guard](guarded_inventory.py) permits only the source closure,
declared metadata and saved attempt files, and blocks archive/model/data imports.
Run and replay access reports show zero denied accesses. This guard is not an
OS sandbox or a whole-archive authenticity audit. Historical member and signal
hashes are carried as assertions; their payloads were not opened or rehashed.
Original CALIB and TEST remain unopened.

| Job | Process CPU seconds | Job wall seconds | Peak RSS KiB |
| --- | ---: | ---: | ---: |
| One actual inventory | 3.117781770 | 9.431101372 | 120,592 |
| Main saved-only replay | 5.191227397 | 10.434046485 | 135,556 |
| Student B saved-only replay | 5.191657447 | 10.467731086 | 135,280 |

These timings cover the guarded job after setup. Full scientific suites ran
concurrently; this table supports no speedup claim. Student B replays the same
inventory and is not an independent data replication.

Validation: **1,272 main tests / 1,268 Student B tests**, with **1,805 subtests
each**, pass. Each suite retains eight earlier warnings. Ruff 0.17.1 and
ty 0.0.90 pass across the 53-file CI list; the two new Python source files pass
formatting. [Audit history](audit.md) records the memory/schema fixes and the
shared-runner resource-isolation fix before source freeze.

## Reproduce the saved inventory

This publication preserves the three registered source files. Use the matching
Python runtime in provenance. From the repository, choose a fresh absolute output
path:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-project --offline --python ../.venv/bin/python \
  python reports/2026-10-25-support/replay.py \
  --out-root /absolute/path/to/a-new-restored-support-attempt
```

The helper verifies the publication inventory, restores all six original files
byte-for-byte, then invokes strict saved-only replay behind the published
guard. It refuses an existing output directory and retains a failed restored
attempt. It reads no original metadata path, sensor archive or model. Source
or runtime mismatch is a refusal, not permission to refit or relax the checks.
After later source changes, use a separate checkout at a recorded source freeze
and copy this report directory into that checkout before replaying.

## Research implication

[IDNet, section 3.2](https://arxiv.org/html/1606.03238v3#S3.SS2) uses iterative
acceleration-magnitude template matching across a motion trace. Its normalized
cycle length is distinct from the duration supplied to segmentation. That
motivates investigating complete continuity segments; it does not establish
successful cycle extraction on these Santos recordings. The earlier
[synthetic cycle studies](../2026-10-25-cycle-length/README.md) also remain
adaptations rather than reproductions of the full IDNet pipeline.

A later DEV comparison needs a new registration: cycle selection, endpoint and
plateau rules, minimum successful support, partial failures, segment/trial and
enrollment weights, and a canonical control with matching observation support.
Canonical362 accepts 256 samples; a tail-covering control must explicitly handle
endpoint overlap and fragments shorter than 256. Failed claims must retain
their denominators through [gated ROC](../../docs/gated_roc.md). Any learned
quality rule stays within its own TRAIN identities; each changed scoring system
needs its own policy-C calibration.

This inventory stops at the support report. It authorizes no automatic cycle
experiment, cohort reduction or graph expansion. Authentication results,
thresholds and the retained Canonical362 nominee remain unchanged.
