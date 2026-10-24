# Numerical validation of current verification APIs

These checks use synthetic arrays and tiny CPU models only. They do not open
datasets, reserved CALIB/TEST payloads, or historical result artifacts.

`engine.evaluate.fixed_threshold_metrics` compares scores and thresholds in
float64 and accepts ties (`score >= threshold`). A strict float64 threshold
above a float32 score must reject that score, even when rounding the threshold
to float32 would accept it. Regression coverage includes a complete tied ROC
table, checking each operating point against actual decision counts.

Scores must be finite real numeric arrays; complex, boolean, text, and object
scores are rejected before conversion. Labels must be binary and contain both
classes. Boolean binary labels remain supported. Thresholds and calibration
targets must be finite real scalars: Python/NumPy numbers and numeric
zero-dimensional arrays are accepted; booleans, complex/text values, and
non-scalar arrays are rejected with `ValueError`. Targets must lie in `[0, 1]`.

ROC-based APIs retain a finite reject-all threshold above the largest score.
If that score is float64's maximum, no such threshold exists and the API raises
`ValueError` without an overflow warning. The immediate predecessor is supported
with the maximum as its reject-all threshold. Direct fixed-threshold decisions
can still use the maximum itself; tied scores at that threshold are accepted.

Classification metrics require equal-length one-dimensional integer labels in
`[0, n_classes)` and a positive integer class count. Negative labels, fractional
labels, and mismatched lengths cannot silently alter confusion counts or
accuracy. Empty classification inputs retain their zero-accuracy result.
Verification subject/activity IDs must be nonnegative integer vectors.

Pair scoring rescales rows when their squared norms would overflow or lose
precision through underflow. Enrollment scoring already uses this approach.
Finite nonzero directions from float64 subnormals through its maximum therefore
remain usable. Complex/boolean embeddings are rejected; exact zero rows and
canceling enrollment templates remain invalid. Tensor inputs remain unchanged
and no gradients are retained.

`WithinClassMetric` preserves its ordinary centering, scaling, projection, and
normalization arithmetic. Its normalization now recovers tiny/huge finite
directions instead of rejecting small vectors or returning zero vectors after
norm overflow. Exact zero directions remain invalid. Training statistics and
projected descriptors that cannot be represented finitely in float64 fail
explicitly. This is not arbitrary-precision fitting or matrix multiplication.

Numeric metric checkpoints require a nonempty one-dimensional mean, matching
strictly positive scales, a square projection, finite real numeric values, and
at least two unique nonnegative integer training IDs representable in int64.
Validation occurs before coercing IDs, preventing fractional truncation and
unsigned integer wraparound. Numeric `np.savez` archives remain supported with
`allow_pickle=False`, and restored arrays are copied.

Run the focused checks from the project directory using the existing shared
environment, without installing dependencies:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_numerical_regressions.py tests/test_evaluate_metrics.py \
  tests/test_verification_protocol.py tests/test_enrollment_protocol.py \
  tests/test_gait_metric.py tests/test_gait_inference.py
```

## Historical study replay

Frozen studies register hashes of shared engine/model source. Fixing a current
public API can legitimately make a historical source-hash check fail, even when
ordinary arithmetic is unchanged. Historical research scripts and frozen result
artifacts are not modified to erase that mismatch or replace old metric values.

For an exact historical replay, check out the historical source commit matching
the study's provenance in a separate checkout and verify its files against the
registered hashes before running its replay command. A report's creation date
or the current merged branch is not sufficient to establish a matching source
revision. If no committed revision matches, recover the original source snapshot;
do not rewrite the registration or bypass its source audit. Replays requiring
reserved CALIB/TEST payloads need separate authorization and were not performed
for these fixes.

A historical helper that retained float32 scores may have rounded a strict
float64 cutoff during comparison. Its recorded results remain historical
evidence under that implementation. Use the current public engine API for new
decisions; any corrected historical analysis must be separately identified and
must not silently replace a frozen artifact.
