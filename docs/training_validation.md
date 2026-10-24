# Training preflight and partial dataset repair

The current activity and Siamese trainers validate integer settings before any
coercion. Epochs, batch size, subset size, thread count, worker count, prefetch
factor, seed, window size, stride, validation subject count, activity IDs, and
configured channel/class/subject dimensions reject booleans, strings, fractions,
and nonfinite values. Python and NumPy integers remain supported. Positive
budgets must be at least one; worker counts and validation counts may be zero
where their existing protocol permits it. Seeds are in `[0, 2**32)` so Python,
NumPy, and Torch use the same accepted seed. YAML epochs reach the validator
unchanged, including when no CLI override is present.

Loaded windows require a finite positive real sampling rate in addition to the
existing shape, dtype, label, and finite-sensor checks. Explicit dimensions must
match the loaded windows exactly. Activity filtering retains original class
indices and names. Optimizer learning rate and weight decay must be finite and
nonnegative; zero retains its previous accepted behavior. Invalid class-weight
policies and chronological validation settings fail before loading data.

Siamese triplet multipliers and validation pair counts must be positive integers
(`val_pairs >= 2`), including on the contrastive path. Embedding dimensions must
be positive integers; dropout must be finite and in `[0, 1]`, and the loss margin
must be finite and nonnegative. These static checks run before loading data.

After the TRAIN-only subject split and before allocating a model or optimizer,
Siamese validation checks both global identity coverage and the evaluator's
actual pair eligibility rule. With `same_activity`, at least one activity must
have two subjects and an identity with two windows in that activity. Global
genuine pairs across different activities do not satisfy this condition. The
preflight uses the evaluator's sampler with only two probe pairs; it does not
embed windows or generate the full validation budget. Unsupported activities
remain excluded by the evaluator when another activity is viable. Legacy
unrestricted validation and `val_subjects: 0` training-loss selection retain
their existing behavior. Training subsets still follow subject splitting.

`scripts/setup_datasets.py` repairs the bundled local `CI-2026.zip` without
network access. HAR completeness requires README, feature and activity metadata,
both splits' feature matrices, subject/activity labels, and all nine inertial
channels per split. Legacy WISDM requires both CSV splits. Every required path
must be an ordinary nonempty file; a README, empty file, directory, or symlink
cannot stand in for a complete dataset. Complete datasets skip archive reads.

For partial datasets, archive members and required entries are checked before
writing. Repair preserves existing ordinary nonempty files and fills missing or
empty files from the local archive. The shared ZIP extractor keeps its ordinary
overwrite behavior for other callers, using a temporary sibling file and atomic
replacement after the member's CRC succeeds. Path traversal, symlink components,
nonregular entries, and duplicate destinations are rejected. A malformed later
path cannot change an earlier file, and a CRC failure leaves that destination
unchanged. A failure on a later member can leave earlier successful members in
place; rerunning repairs the remaining files. Completeness is structural, not a
content hash or numerical validation of existing nonempty files.

All focused tests use generated CPU tensors and temporary nested ZIP fixtures;
they do not download data or load real TEST/CALIB payloads:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_training_validation_regressions.py tests/test_training_runtime.py \
  tests/test_validation_protocol.py tests/test_siamese_training_protocol.py \
  tests/test_archive_extraction.py tests/test_public_dataset_setup.py
ruff check engine/train.py engine/train_siamese.py scripts/setup_datasets.py \
  tests/test_training_validation_regressions.py
ty check --python ../.venv/bin/python engine/train.py engine/train_siamese.py \
  scripts/setup_datasets.py tests/test_training_validation_regressions.py
```

The metric trainer imports the shared activity-filter, split, and window
validators. Its synthetic compatibility checks are:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_metric_learning.py tests/test_metric_cohorts.py tests/test_metric_resume.py
```

These fixes change shared training source hashes. Historical study scripts,
registrations, and result artifacts remain frozen. For exact historical replay,
use a separate checkout of the source commit recorded by that study and verify
its files against the registered hashes before running it. If no commit matches,
recover the original source snapshot. Do not rewrite historical registrations
or bypass source-hash checks to make current source appear identical. No
historical replay or reserved-payload evaluation was performed for these fixes.
