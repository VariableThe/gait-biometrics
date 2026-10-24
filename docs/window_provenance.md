# Raw-sample provenance for WISDM windows

`data.window_provenance.load_windows_with_provenance` is an opt-in wrapper for
official-format WISDM AR v1.1 raw records. It returns a `ProvenancedWindowData`,
a subclass of the existing `data.adapter.WindowData`, with unchanged tensors,
labels, global subject mapping, sampling rate, source split, and window order.
Existing loaders and callers remain usable. The original adapter and enrollment
implementation are not changed by this addition.

Only `source="wisdm_v1"` is supported. Official UCI HAR windows lack the raw
source offsets needed to establish sample support. Their order, 128-sample
length, or nominal overlap must not be used to fabricate offsets. Legacy WISDM
CSVs lack authentic subject identities. Both sources fail clearly before being
loaded by this wrapper.

## Coordinates and alignment

Each window has aligned read-only arrays on `data.provenance`:

- `start_rows`, `stop_rows`: its half-open raw support `[start, stop)`, in
  zero-based ordinals among **valid parsed records in the entire raw file**.
- `segment_ids`: the first valid-record ordinal of its contiguous preprocessing
  run. This is a continuity identifier, **not an independent session**.
- `first_timestamps_ns`, `last_timestamps_ns`: optional exact integer nanoseconds
  for the first and last **included** samples. The timestamp endpoint is not
  exclusive. Disable both with `include_timestamps=False`.
- `source_sha256`: the SHA256 of the raw file bytes. A schema-valid synthetic
  fixture or subset has its own digest; loading it does not certify that it is
  the complete official distribution.

`_read_wisdm_v1` assigns `_source_row` after discarding invalid records. It is
neither a physical line number nor an ordinal among all attempted records.
Several records can occur on one semicolon-separated text line. Invalid records
consume no valid-row ordinal but change `_segment`, so windows cannot bridge
them. The wrapper reuses `_wisdm_segments` and `_window_wisdm`, including their
subject/activity changes, timestamp resets/duplicates, gaps over one second,
source-row gaps, and stride restarts. No sorting occurs during loading. A run's
identifier remains stable across source subject splits.

The wrapper rechecks the source digest against the parser's cached digest. A
file whose bytes changed while its size and modification time remained the same
is rejected instead of receiving misleading source metadata. Treat source files
and the adapter's cached frame as immutable during a load.

Use `data.take(selection)` for filtering, permutation, sampling, or subsets.
It selects tensors and provenance together, retaining exact requested order,
duplicate rows, empty subsets, and the global subject vocabulary. Integer index
vectors, boolean masks, and slices are supported. `provenance.take(selection)`
is available when embeddings are already stored separately. Applying an
unrelated tensor permutation, reconstructing plain `WindowData`, or calling a
legacy subset helper that drops additional fields does not preserve provenance.
Caller-created `WindowProvenance` validates coordinates and shapes but cannot
prove that supplied embeddings or raw intervals describe the claimed source.

## Enrollment and probe separation

```python
from data.window_provenance import load_windows_with_provenance
from engine.enrollment import enrollment_scores
from engine.provenance_verification import provenance_enrollment_split

data = load_windows_with_provenance("wisdm_v1", "train", data_root="datasets")
walking = data.take(data.y_activity == 0)
# Obtain z with the authorized encoder, in exactly walking.X's row order.
enroll, probe, coverage = provenance_enrollment_split(
    walking.y_subject, walking.provenance,
    enrollment_windows=5, purge_samples=128,
)
# labels, scores, details = enrollment_scores(z, walking.y_subject, enroll, probe)
```

The split recovers each subject's source order from `start_rows`, even for a
shuffled subset, and reserves its first `enrollment_windows` windows. Remaining
windows are retained as probes only when their actual raw supports avoid every
reserved enrollment prefix by at least `purge_samples`. A positive gap counts
valid parsed raw records between supports, including across segment boundaries;
it is neither elapsed time nor a guessed number of windows. At gap zero,
`[0,128)` and `[128,256)` are disjoint; `[0,128)` and `[64,192)` conflict.

All subjects with more than the prefix budget reserve their prefix in one pass.
Subjects with no safe probe are excluded completely; their reserved prefixes
are not subsequently released to readmit other subjects' probes. This is
conservative if caller-created metadata assigns overlapping support to different
identities. Correctly derived WISDM subjects have disjoint raw support.

Returned indices address the supplied arrays in their original row order.
Coverage includes actual purged indices, per-subject counts, excluded subjects,
source provenance, `raw_offset_nonoverlap_verified=True`, and
`cross_session=False`. An empty or insufficient cohort produces empty indices
with explicit coverage; downstream verification still needs enough enrolled
identities and genuine probes. This split supplies no threshold or calibration.

## Verification pairs without shared raw support

```python
from engine.provenance_verification import provenance_verification_pairs

pairs, labels, context = provenance_verification_pairs(
    walking.y_subject, walking.provenance, n_pairs=10_000, seed=42,
    activity_ids=walking.y_activity, purge_samples=0,
)
# pairs[:, 0] and pairs[:, 1] index aligned windows/embeddings.
```

Both genuine and impostor pairs must have disjoint half-open supports and meet
the raw-sample gap. Duplicate subset entries cannot create a valid genuine pair
from a window and another copy of itself. Segment changes and timestamp resets
never exempt a pair from the support check. `support_conflicts` exposes the same
elementwise check for equally sized left/right index vectors.

The sampler divides the requested count into floor(n/2) genuine and the
remaining impostor comparisons. When activity labels are supplied, it cycles
over strata having both kinds of safe pairs with matching budgets (apart from
the possible extra impostor). Activities without coverage are reported as
excluded. A cohort without any viable stratum fails with an actionable error.

Within a genuine stratum, the sampler chooses an eligible subject uniformly,
then an admissible unordered pair uniformly. Within an impostor stratum, it
chooses an admissible unordered different-subject pair uniformly. These are
explicitly different sampling semantics from the legacy subject-uniform
impostor sampler. Both classes sample **with replacement**; context reports
available, excluded, unique, and repeated pair counts. Excluding shared support
within each pair does not make repeated comparisons or reused windows
statistically independent.

Pair-index preprocessing and storage are O(N log N) and O(N), respectively;
returned pairs and duplicate-count bookkeeping additionally use O(n_pairs)
storage. Sorted interval starts and suffix counts permit direct selection by
pair rank, with O(log² N) bounded searches per impostor pair. The sampler neither builds an N² pair
table nor retries until a rare genuine pair happens to be drawn. A population
with only one admissible genuine pair remains usable with replacement, and its
limited unique coverage is visible in the report.

Window sizes, strides, prefix/pair budgets, raw purge gaps, and seeds require
integer values, excluding booleans and floats such as `2.0` or `2.5`. Coordinates
and parameters must be representable in int64; gap checks saturate rather than
wrapping at the integer limit. Raw support establishes within-recording
separation only. It supplies no independent session, cross-day, cross-device,
or biometric-performance claim.

## Synthetic checks and historical replay

No real data is required for the tests:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_window_provenance.py tests/test_wisdm_identity_windows.py \
  tests/test_enrollment_protocol.py tests/test_verification_protocol.py
```

The synthetic controls cover semicolon-concatenated records, invalid-record gaps,
timestamp resets/duplicates, long gaps, digest changes, exact loader parity,
shuffled/duplicated subsets, raw gaps, both classes' support exclusions,
exhaustive ragged-interval pair tables, scarce genuine pairs, rejected fractional
parameters, and int64 boundaries. They establish software behavior, not measured
dataset performance. No reserved TEST/CALIB payload is used.

This opt-in addition does not rewrite frozen study scripts, registrations, or
artifacts. If fixes elsewhere change shared source hashes recorded by a
historical study, replay it from a separate historical source checkout whose
files match the registered hashes. Recover the original source snapshot if no
commit matches; do not update registrations or bypass source checks to make a
current checkout appear historical. New provenance-aware comparisons require a
separately identified protocol and appropriately authorized calibration data.
