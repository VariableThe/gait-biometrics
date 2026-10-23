# Model size and identity objective study

This study tests whether more model capacity improves identity generalization.
It compares a fixed 2 × 2 design: multiscale encoder width 32 or 96, with a linear
identity head or ArcFace head. The metric objective remains hybrid batch-hard
triplet/SupCon, so the size and classifier comparisons have matched controls.
Three fixed descriptor metrics provide inexpensive competing hypotheses.
ArcFace is adapted from face recognition; it is not an established gait result.
See [the original paper](https://arxiv.org/abs/1801.07698).

Only walking windows from the source TRAIN partition enter this program. The
source TEST partition has been inspected in previous studies and is never
loaded here. New development results cannot establish fresh test improvement.

Before training, the program records sensor-value fingerprints, implementation
hashes, every effective run configuration and all subject cohorts. Four users
are reserved with seed 2026 for threshold calibration. All remaining users are
partitioned into three subject folds. Each neural candidate runs both seeds
42/43 and exactly 40 epochs per fold: 24 runs per dataset, 48 in total. Normalizer
moments and identity heads fit each fold's training users only. Fixed final
epoch selection prevents choosing the best epoch on a reporting fold. Model
ranking uses the equal fold/seed mean EER, retaining every individual outcome.
Fold/seed standard deviations describe variability; they are not population
confidence intervals because the training cohorts overlap.

The selected descriptor fits all development users. A selected neural method
uses all six fold/seed encoders as a uniform concatenation ensemble. That
ensemble is distinct from a single fold prediction; its calibration result
must not be represented as its cross-validation performance. Only after the
choice is frozen may calibration representations determine an attainable
empirical threshold at FAR <= 1%. Calibration accuracy is not a fresh test
result, and its target does not promise a deployment FAR. The four calibration
users are excluded from both optimizer and validation cohorts of every run.

```bash
uv run python -m scripts.subject_scaling_study --stage train --device cuda
uv run python -m scripts.subject_scaling_study --stage lock --device cuda
```

Use `--sources har` or `--sources wisdm_v1` to schedule the two datasets in
separate GPU processes. Keep their output roots identical and monitor memory.
The default device is CPU; no process is launched implicitly. Reproduction on
another machine can choose a new output root. Resume requires matching source
values, implementation, device/configuration, checkpoint seeds, cohorts and
complete histories. Locking rechecks all 24 checkpoints per source and
recomputes both neural and descriptor scores before threshold calibration.
Missing runs cannot trigger training during locking. Smoke runs belong in
separate directories and cannot satisfy the registered study budget.

Outputs in `runs/subject_scaling/<source>/` are `pretraining_protocol.json`,
per-run safe checkpoints and complete histories, `development_results.json`,
`frozen_choice.json`, `calibration.npz` and `selection_lock.json`. The lock
contains artifact hashes, the fixed threshold and explicit limitations. Full
neural ensembles have six inference passes; an accuracy benefit alone does not
establish that this is suitable for a phone. Model scaling does not create more
independent participants, sessions or device placements.
