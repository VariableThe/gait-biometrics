# WISDM identity and recording boundaries

The supplied `wisdm/train.csv` and `test.csv` expose only accelerometer axes,
timestamps in seconds, and activity labels. They cannot support person-level
verification. No code reconstructs a person from row order, activity, timestamp,
or a source split name. `source="wisdm"` remains usable for activity recognition
and returns `y_subject=None`; identity training/evaluation must reject it.

## Authentic source and schema

Use `source="wisdm_v1"` and put the official extracted file at
`<data_root>/wisdm/WISDM_ar_v1.1_raw.txt`. The adapter also accepts the official
`wisdm/WISDM_ar_v1.1/` subdirectory. Its six native fields are
`user,activity,timestamp,x,y,z;`. The included raw-about document identifies
users as masked integer IDs 1–36, timestamps as phone uptime in nanoseconds,
and sampling rate as 20 Hz. Integer timestamps are parsed without float rounding.

Primary sources:

- [Fordham WISDM dataset page](https://www.cis.fordham.edu/wisdm/dataset.php)
  distinguishes controlled AR v1.1 from real-world Actitracker v2 and documents
  masked subject IDs. Its distribution asks redistributors to retain `readme.txt`.
- [Official AR archive](https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz)
  includes `WISDM_ar_v1.1_raw_about.txt`, the schema authority read for this implementation.
- [Kwapisz, Weiss and Moore, Activity Recognition using Cell Phone Accelerometers](https://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf)
  describes the controlled collection protocol. This is an activity-recognition
  study, not evidence of this project's verification performance.
- [UCI dataset 507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
  is a different WISDM collection with 51 IDs 1600–1650 and 18 activities.
  This v1.1 parser intentionally rejects that schema rather than relabeling it.

Measured official download fingerprints (2026-10-23):

```text
AR archive SHA256: e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7
raw text SHA256:   9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e
```

The raw reader computes the actual SHA256 and records schema version, source URL,
record counts, and whether it matches the official fingerprint. A schema-valid
subset or synthetic test fixture can be loaded but does not match that fingerprint.
No download contents or subject IDs are inferred from a filename alone.

## Labels and splits

Activity labels are fixed across splits:

| Label | Activity |
| --- | --- |
| 0 | Walking |
| 1 | Jogging |
| 2 | Upstairs |
| 3 | Downstairs |
| 4 | Sitting |
| 5 | Standing |

For walking verification set `activity_ids: [0]` in the trainer configuration.
The adapter emits native 20 Hz, unscaled, 3-channel `(N,3,T)` float32 windows.
It does not invent gyroscopes or add train/test-dependent scaling.

The source split sorts all observed real IDs, permutes them with NumPy
`default_rng(42)`, allocates floor(80%) to training and the remainder to test
(at least one subject per side). Filtering activities happens after this split.
This is a project-defined reproducible subject split, not an official WISDM split.
Training validation must be carved only from source-training subjects.

`WindowData.y_subject` contains globally contiguous int64 indices into
`WindowData.subject_ids`, which holds the same sorted original real IDs in both
splits. Thus `subject_ids[y_subject[i]]` recovers the source identity. Source IDs
are not independently remapped per split. HAR's existing mapping is unchanged.

## Boundaries and invalid data

Windows remain inside runs with the same subject, activity, and explicit session
if supplied. Source order is preserved. A timestamp reset or duplicate, gap above
one second, missing/nonfinite sensor record, or removed source-row interval breaks
continuity. The one-second gap threshold is an explicit conservative policy;
it does not establish a session identity. Semicolon-concatenated records are read
separately. Missing/foreign subject IDs and unknown activity names fail clearly.
Window strides restart at each boundary; mixed-activity majority labels are removed.

The official raw file produced 1,073,630 valid records and 24,579 excluded records
with this policy. At `window_size=128, stride=64`, measured native-rate windows were:

| Source split | All activity windows | Walking windows | Walking subjects |
| --- | ---: | ---: | ---: |
| train | 2,155 | 927 | 27 |
| test | 718 | 345 | 8 |

The source subject partition has 28/8 subjects; one training subject has no surviving
walking windows, and real subject 7 has only one. Training pair/triplet sampling
must handle subjects without two genuine examples. These are data-retention counts,
not authentication accuracy measurements. Overlapping windows remain correlated;
subject separation does not establish cross-day or cross-device performance.
The supported evaluation claim is held-out-subject verification within these
recordings. No explicit cross-session split or test-based tuning is established
by this preprocessing; validation-selected thresholds must remain fixed on test.

The reader caches one raw frame per process, keyed by resolved path, size, and
modification time in nanoseconds. Treat the cached frame as read-only. The cache
invalidates when file metadata changes and is never deserialized from pickle.
Measured on the primary CPU: first train load 10.586 s, subsequent test load
0.107 s, repeated train load 0.297 s. These timings preceded a small parser cleanup;
they compare cold parsing against warm reuse, not independent-machine performance.

## Student A processed output

Run the standalone pipeline with `--wisdm-raw /absolute/path/to/WISDM_ar_v1.1_raw.txt`
to select authentic identities. Without it, the original CSV zip remains the
activity-only input. The original NPZ keys `X`, `y`, `subject`, file names, and
`(N,T,3)` shape are preserved. Real-source `subject` stores the original WISDM IDs;
legacy unknown identities store integer `-1`, replacing misleading `train`/`test`
strings. Consumers must inspect `identity_available` in the JSON sidecar.

Resampling and filtering occur separately within each valid segment. Short
segments safely skip FIR filtering if too short for its padding. Normalization
statistics come exclusively from source-training windows and are reused for test.
JSON sidecars include the input fingerprint, source URL, split IDs/protocol,
boundary policy, identity availability, output counts, and filter availability.
Upsampling to the standalone pipeline's default 50 Hz does not add measured
information beyond the original 20 Hz sensors. The UCI-HAR export is unchanged.

Boundary, label, cache-invalidation, missing-identity, subject-disjointness, and
train-only normalization regression tests use explicitly synthetic fixtures.
Real-source retention counts above were measured separately on the official file.

The complete standalone WISDM export was also run with its default 50 Hz,
90-sample windows and 45-sample stride. Real v1.1 produced 28,885 training and
7,776 test windows in 23.67 s. The supplied legacy zip produced 28,903 training
and 7,759 test windows in 13.74 s, with every exported identity explicitly `-1`.
Both used training-only statistics and wrote schema/provenance JSON sidecars.
These wall times include parsing, preprocessing, and compressed output under
concurrent primary-machine load, with OpenBLAS limited to one thread; they are
not a controlled before/after speed benchmark. Local artifacts are under
`artifacts/student_a_identity_validation/` in the parent workspace.
