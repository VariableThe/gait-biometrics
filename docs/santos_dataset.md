# Santos v7 acceleration loader

`data/santos.py` is an opt-in NumPy-only loader for the official walking-only,
processed MCU CSVs. It does not change `data.adapter` or `load_windows`, extract
files, contact the network, import Torch, or use a GPU.

## Source, license, and acquisition

Geise Santos, Marcelo M. Wanderley, Tiago Tavares, and Anderson Rocha,
*A multi-sensor human gait dataset captured through an optical system and
inertial measurement units*, Figshare article 14727231, version 7:
[10.6084/m9.figshare.14727231.v7](https://doi.org/10.6084/m9.figshare.14727231.v7).
The data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Preserve this attribution, source version, license, and the transformations below
with derived artifacts. The accompanying descriptor is
[Scientific Data, s41597-022-01638-2](https://www.nature.com/articles/s41597-022-01638-2).

The parent verified the official archive on 2026-10-23 before the development
header inspection:

| Property | Verified value |
| --- | --- |
| File ID | 35882624 |
| Bytes | 1,523,321,221 |
| MD5 | `35adf0860202f2aefa37560dc0748ebd` |
| SHA-256 | `ed944fa88a98ce1058327ff4b3c8f919fbf58626fc7e3c5256360ee080a3a304` |
| Central-directory members | 3,048 |
| Processed MCU walking CSVs | 499 |

The metadata inventory covers the 25 authentic subject IDs below. It contains
no `user12` day2 trial `0009` MCU walking CSV; this missing recording is reported
as missing, never synthesized. These inventory facts came from filenames.
The parent inspected one official development CSV header; the implementation
and its tests did not inspect official signal payloads.

The caller must verify its archive against the byte count and checksums before
calling the loader. The loader deliberately does not recompute the complete
archive hash, and its metadata does not claim local integrity verification.

## Verified schema and selection

The exact accepted path is:

```text
A multi-sensor human gait dataset/processed_data/userXX/capture_userXX_dayD_TTTT_imu_walk.csv
```

`XX` is a real two-digit subject ID, `D` is `1` or `2`, and `TTTT` is `0001`
through `0010`. Directory and filename subject IDs must agree. The exact,
case-sensitive official header is:

```csv
frame,acc x,acc y,acc z,gyro x,gyro y,gyro z
```

Only `acc x`, `acc y`, and `acc z`, in that order, become model input. Units are
already **m/s²**: no additional g conversion, centering, scaling, resampling,
or normalization occurs. The nominal acquisition rate is 100 Hz. Gyroscope
channels are omitted from the output but checked when validating rows; any
nonfinite or float32-overflowing value among all six sensors invalidates its
row. No gyroscope study candidate is introduced by this loader.

The author's pinned companion development CSV has the different four-column
header `frame,acc x,acc y,acc z`; it is not an official-v7 schema and is rejected
by this production reader. Unknown headers, channel counts, spelling/case, and
encoding are rejected. CSVs must be UTF-8. Unrecoverable CSV syntax raises an
error rather than silently discarding an ambiguous span.

All central-directory paths are checked before any payload is opened. Unsafe
absolute/traversal paths, backslashes, control characters, symlinks, duplicate
members, and inconsistent processed-MCU identities fail. Raw, Nexus, and optical
payloads are never opened. The reader opens the validated `ZipInfo`, and only
for the explicitly selected subjects. It never calls `testzip()` or reads
unselected subjects to inspect their headers or compute checksums.

## API and provenance

```python
from data.santos import fingerprint_santos_members, load_santos_windows

development = [7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17]
data = load_santos_windows(archive_path, development, window_size=256, stride=256)
manifest = fingerprint_santos_members(archive_path, development)
assert data["metadata"]["members_sha256"] == manifest["members_sha256"]
```

The return value is a dict:

| Key | Type/shape | Meaning |
| --- | --- | --- |
| `X` | NumPy float32 `[N,3,T]` | Unscaled acceleration windows |
| `subjects` | NumPy int64 `[N]` | Authentic numeric IDs, e.g. 1 and 3 |
| `days` | NumPy int64 `[N]` | Original day label 1 or 2 |
| `trials` | NumPy int64 `[N]` | Original trial number 1–10 |
| `starts` | NumPy int64 `[N]` | Original zero-based CSV data-row offset |
| `frames` | NumPy int64 `[N,T]` | Every original MCU frame integer |
| `recording_indices` | NumPy int64 `[N]` | Index into `metadata["recordings"]` |
| `metadata` | JSON-serializable dict | Schema, paths, hashes, coverage, exclusions |

`starts` excludes the header but counts blank and invalid data rows. Frames need
not begin at zero or one; signed int64 values are parsed directly as integers,
without a floating-point intermediate. Subject labels are never remapped to
consecutive indices. Missing requested subjects and invalid/duplicate requested
IDs raise errors. Requested subjects and recordings are sorted deterministically
by real numeric subject, day, and trial. Empty outputs retain the promised shapes
and dtypes, and zero-window recordings remain in the metadata.

Each recording includes its exact member path, source SHA-256, byte count,
recording key `[7, subject, day, trial, "MCU", "processed_data"]`, source row
count, invalid row offsets, continuity breaks, segment boundaries, and window
coverage. Source SHA-256 covers all original uncompressed CSV bytes, including
invalid rows and dropped tails, before decoding or conversion.
`members_sha256` hashes canonical JSON of the ordered source-member manifest;
it is distinct from a complete ZIP checksum. `fingerprint_santos_members`
returns the same source manifest and digest without parsing signals or fitting
anything. It still **reads selected payload bytes**, so fingerprinting test IDs
is subject to the same protocol lock as loading test signals.

## Windows, exclusions, and study separation

The defaults are 256 samples and stride 256 at native 100 Hz (2.56 seconds).
Each recording and each valid contiguous segment is windowed independently,
starting the stride again at the segment's beginning. Incomplete tails are
dropped. Blank records, wrong field counts, invalid frame integers, invalid
sensor numbers, NaNs, infinities, and overflow end a segment. Gaps, duplicate
frames, and resets also end a segment; their current valid row starts the next
one. Nothing is filled or stitched across these boundaries or across recordings.

Coverage distinguishes invalid rows, continuity breaks, unused tails, and rows
skipped when stride exceeds window size. `covered_rows` counts unique rows
covered by windows, including when windows overlap. Missing expected recordings
and zero-window recordings are explicit exclusions. Per-subject coverage lists
present and eligible trials on each day and `eligible_both_days`; a subject
requires at least one eligible trial on each day for the downstream study.
The loader reports coverage without choosing evaluation exclusions or replacing
subjects. Placement side remains unknown; pooling trials mixes leg placements.

The fixed split and study rules are specified in
[cross_session_acquisition.md](cross_session_acquisition.md). The exact sets are:

- Development: `7,16,11,9,29,27,5,13,3,1,25,24,14,19,17`.
- Calibration: `21,18,23,12,30`.
- Untouched test: `26,28,22,15,6`.

Only the twelve fold-training development identities may fit transforms inside
each development fold. Refit the selected descriptor/metric on all fifteen
development identities. Calibrate the threshold on the five calibration
identities, then freeze the protocol, fitted settings, code/config hashes,
aggregation, acceptance inequality/ties, and threshold before any test payload
or test-member fingerprint is read. The caller enforces this sequencing; this
low-level loader does not infer authorization from a filename or ID.

Enrollment uses day1 trial means with equal weight per trial; each day2 trial
mean is a primary probe. Window embedding/descriptor aggregation belongs to the
study code and must group by the preserved subject/day/trial or recording index.
The loader does not fit descriptors, choose metrics, merge days, or aggregate
trials. The proposed minimal descriptor/standard/WCCN candidates remain the
parent study's responsibility.

## Validation

`tests/test_santos.py` uses tiny synthetic ZIPs with the exact official header.
It checks authentic labels, units, source hashes, selected-only payload access,
empty outputs, missing coverage, trial/day boundaries, bad rows (including gyro
values), frame gaps/resets/duplicates, int64 precision/limits, unsafe paths,
unknown schemas, duplicate entries, and fingerprint invalidation. It imports
neither Torch nor official dataset values.
