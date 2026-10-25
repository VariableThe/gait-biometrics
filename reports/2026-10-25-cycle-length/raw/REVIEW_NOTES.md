# Review handoff: one locked synthetic attempt

The measurement completed once. All frozen sources, manifest, cached reference,
original report, attempt marker, measured JSONL and arrays remain unchanged.
This note and postrun_review.json summarize already saved evidence. They are not
an independent root review and do not authorize publication or real DEV work.

All 1,280 N256 exact-rule records match all original cached fields exactly,
including float bits, failures, minima, pairs and guard statistics. All 1,280
six-axis N768 prefixes match independently generated N256 fixtures bitwise.
The frozen source/manifest/runtime hashes were recorded before measurement.

Both rules have identical outcomes, selected lags, pairs and finite vectors on
both grid lengths. Each rule has these results out of 1,280 fixtures:

- N256: 848 finite, 192 missing pairs, 240 channel-degenerate failures; 752 finite
  one-P intervals and 96 finite half-P aliases.
- N768: 959 finite, one missing pair, 320 channel-degenerate failures; 863 finite
  one-P intervals and 96 finite half-P aliases.

All 5,120 record rows and 2,560 fixture/length combinations remain. Aliases are
finite extraction successes, never additional extraction failures. Every weak
near-flat gyro fixture at N768 fails the frozen guard; there is no imputation.
No quality cutoff was trained or applied, and there are no authentication scores.

The 18 fixed vectors (tile-sine and exactly equal synthetic maxima, each with
nine fixed ULP variants) all preserve candidate selection 100 and nearset [100,200]
under reversed enumeration. Exact selection differs from the candidate in nine
vectors, as expected. The unfiltered tile-sine baseline has r100=
0.9999999999999999 and r200=1.0: exact chooses 200, candidate chooses 100.
This result does not imply stability at arbitrary 1e-10 membership boundaries.

At N256, independently enumerated continuous prescribed-minimum geometry has
1,086 available selected-lag pairs and 194 phase holes per rule: 81 holes each
for gravity and weak-gyro families, 32 alternating-family holes, and none for
balanced peaks. The exact phase indices are saved by family/period/rule/length.
All 192 observed missing-pair failures are geometric holes. Two observed
extractions lie in geometric holes: gravity and weak families, P160 phase30.
Their continuous first raw minimum lies at 8.9834468260952, outside the margin;
their filtered detected pair is [10,169]. Thus continuous raw-minimum geometry
and discrete filtered extraction need not agree. The weak-family extraction
still fails its gyro guard.

At N768 all 1,280 reference geometries have a pair at the selected lag, with
zero continuous geometric phase holes. However alternating_step_harmonic/P180/
phase20 has strict detected minima [67,157,608,698]. None of their pair differences
is within +/-10% of selected lag 180. The continuous reference minima are
[67.5,157.5,247.5,337.5,427.5,517.5,607.5,697.5], admitting six eligible 180 pairs.
Both rules retain this failure. The saved record proves the disagreement;
no extra filtering or new measurement was run to assign a further numerical
cause. A geometric length argument is not a strict-minima guarantee.

All 1,280 N768 diagnostics preserve four channels (raw/filtered magnitudes for
both sensors) with the same 348 anchors at every lag 40..400. Balanced acceleration
has P/2 ambiguity, while its gyro magnitude contains P; this joint-magnitude
fixture is not unidentifiable. The raw balanced acceleration half-P alternative
is in the nearmax set in all 320 cases, and the raw gyro half-P alternative in
none. Raw gyro P is nearmax in all 320. Filtered diagnostics and all harmonic
gaps remain in diagnostics.jsonl/NPZ; they never rescue an extraction. A unique
nearmax never asserts anatomical stride, and there are no general gait claims.

The main run used 64.069864588 process CPU seconds and 66.66214266908355 wall timer
seconds on CPU0, one numerical thread, CUDA disabled. The saved-array review used
2.834259962 CPU seconds. These are not speed-comparison measurements. Python 3.14.8,
NumPy 2.5.4 and SciPy 1.18.1 were used from the existing shared environment.

## Preserved fingerprint defect

The frozen audit.py initially stops at its combined scientific fingerprint check.
The measurement's individual source/manifest/runtime, JSONL, NPZ and partial-array
file hashes all verify. The combined in-memory fingerprint was computed when
`diagnostic_summary[*].nearmax_set_size_counts` had integer keys. JSON converts
these to strings; sorted canonical serialization then sorts e.g. 10 before 2,
changing the byte order used for a naive post-save fingerprint recomputation.

The original combined fingerprint is recovered exactly by restoring only those
count keys to integers. A separate reproducible fingerprint of the fields as
actually saved in JSON is in fingerprint_serialization_note.json. The frozen
report and sources were not edited and the measurement was not retried.
postrun_audit.py retains every other frozen audit check and explicitly verifies
both fingerprint forms; its pass with documented defect is in postrun_audit.json.
The bug is metadata serialization, not a scientific numeric retuning.

The portable saved-JSON scientific fingerprint is:
`9061bf1d4df7f88a56ef9dd156e55625a4b44933867e87fd55bc5083cf890112`.
The preserved report SHA256 is:
`bd890b2065009c961c7f8dc53624d85435e50ea44186d86465323673b45761da`.

Named audit_arrays.npz contains raw six axes, lossless extraction/diagnostic
correlations, feature vectors, validity/presence masks, fixed vectors and explicit
row maps. postrun_audit.json confirms 5,120 extraction rows,5,120 channel diagnostics,
1,280 exact cached records and 18 numerical vectors. artifact_index.json binds all
files present at handoff, including this note and the documented audit defect.

IDNet section3 includes eight channels, adaptive templates and orientation
processing absent here. Derawi's cited DOI is 10.1109/IIHMSP.2010.84 (2010).
Four noiseless analytic fixtures cannot validate gait or authentication. The
researcher's N768 rationale is the first multiple of 256 greater than 620, with
geometric motivation only. Root independently reviews before any publication or
real DEV plan. No real-data, fitting, installs, Git, SSH or child work was done.
