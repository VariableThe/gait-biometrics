> Historical project document. Current source contracts, validation behavior, measured results and limits are in [README.md](README.md) and [23 October evidence](reports/2026-10-23/README.md). Earlier results/protocols are retained here for provenance and are not directly comparable to the new walking-only runs.

# Student B — first measured results (HAR activity classification, HAR verification, WISDM activity classification)

These are the **first measured numbers for this repo**; before this, only
synthetic unit tests had been run. Every figure below was machine-copied or
arithmetically derived from the JSON artifacts written by
`engine/train.py`, `engine/train_siamese.py` and `engine/evaluate.py` under
`runs/real/`; no number was typed from memory or estimated. The extraction
script is throwaway and lives outside the repo (`/tmp/opencode/extract_results.py`).

**What was measured**

* Activity classification on the UCI HAR test split (2947 windows) for five
  registry models: `cnn_lstm`, `cnn_baseline`, `se_convnet`, `gait_vit`,
  `multitask`.
* Activity classification on the WISDM test split (2958 windows) for
  `cnn_lstm`.
* Gait verification on the UCI HAR test split for the Siamese
  `EmbeddingNet` (`engine/train_siamese.py` + `--task verification`).

**What was NOT measured**

* No validation-split result of any kind — no validation split exists, and
  `best.pt` is selected on **training** loss (see "Interpretation / limits").
* No GPU result: the code has no `.to(device)` call anywhere in `engine/`,
  `models/`, `scripts/` or `data/`; every run above is CPU-only.
* No WISDM verification result: WISDM carries no subject ids (`y_subject=None`
  in `data/adapter.py`), so no genuine/impostor pairs can be formed.
* No held-out subject-recognition accuracy for the multitask model: the only
  identity number produced is the train-split `final_subject_acc` (0.6949).
* No ONNX export / pruning / quantization numbers for these checkpoints:
  `onnx` is not installed in this environment, so `scripts/export_onnx.py`
  (B6) was not run. (`runs/smoke_verification/lightweight/metrics.json` is a
  pre-existing synthetic smoke run, unrelated to this document.)
* Nothing from the B1–B7 papers. See the final section.

## Provenance

| Field | Value |
|---|---|
| `git_commit` recorded in every artifact | `a445dc67855619ac868f6e8d74aec68566c75caa` |
| Working-tree state at run time | **dirty** — the runs executed against uncommitted changes |
| `git diff` fingerprint at run time (sha256 of `git diff` output) | `7cda3c8737d66bb2ac7d92ec0e7a7d6759e6d3df0ba2513663036e7449747cd7` |
| Torch | `2.12.1+cu130` |
| Python | `3.14.8` |
| Compute | CPU-only (RTX 5080 16 GiB present; `torch.cuda.is_available()` is `True`, but no code path moves a model or batch to the device) |
| Threads | `torch.set_num_threads(8)` from `torch_threads: 8` |

The recorded `git_commit` is the checked-out commit; because the tree was dirty,
the commit alone does not identify the code that ran. The `git diff` fingerprint
above does: it was verified to reproduce exactly on the current worktree
(`git diff | sha256sum`), i.e. the tree has not changed since the runs.

**Derived run-config recipe.** Each run used a copy of the corresponding
`configs/*.yaml` with four edits, nothing else:

1. delete the `subset` key (train/evaluate on the full split),
2. `epochs: 20` (instead of the smoke-test 2),
3. `torch_threads: 8`,
4. `out_dir: runs/real/<name>`.

The derived configs were written to `runs/real_configs/real_<name>.yaml`
(gitignored, not committed). `configs/` in the repo is unchanged by this work.

All artifacts referenced below live under the gitignored `runs/` tree and are
therefore **not in the repository**; the document records their paths and the
commands that regenerate them.

## Measured activity-classification results

Split sizes are the counts the frozen adapter actually loaded
(`data.adapter.load_windows(source, split, window_size=128, stride=64)`),
re-measured for this document because the artifacts record only `n_samples`
for the test split; `n_samples` in each `eval_activity.json` was asserted equal
to the re-measured test size during extraction.

| model | source | n_train | n_test | test accuracy | macro-F1 | params | epochs | best epoch | selection rule | artifact |
|---|---|---|---|---|---|---|---|---|---|---|
| `cnn_lstm` (B3) | har | 7352 | 2947 | 0.930098 | 0.931380 | 350406 | 20 | 16 | `train_loss` | `runs/real/cnn_lstm/eval_activity.json` |
| `cnn_baseline` (B1) | har | 7352 | 2947 | 0.895487 | 0.892737 | 132230 | 20 | 20 | `train_loss` | `runs/real/cnn_baseline/eval_activity.json` |
| `se_convnet` (B5) | har | 7352 | 2947 | 0.929420 | 0.929386 | 146090 | 20 | 20 | `train_loss` | `runs/real/se_convnet/eval_activity.json` |
| `gait_vit` (B7) | har | 7352 | 2947 | 0.923651 | 0.924022 | 813958 | 20 | 18 | `train_loss` | `runs/real/gait_vit/eval_activity.json` |
| `multitask` (B2) | har | 7352 | 2947 | 0.908381 | 0.910602 | 345243 | 20 | 20 | `train_loss` | `runs/real/multitask/eval_activity.json` |
| `cnn_lstm` (B3) | wisdm | 11771 | 2958 | 0.866802 | 0.859304 | 348486 | 20 | 20 | `train_loss` | `runs/real/wisdm_cnn_lstm/eval_activity.json` |

* "epochs" is the configured epoch count (20 in every run); "best epoch" is the
  epoch whose `train_loss` was lowest and whose weights are in `best.pt`.
* **Macro-F1 is derived post-hoc, not measured directly.** It is
  `mean(2PR/(P+R))` over the six classes, computed from the measured
  per-class precision/recall in `per_class` of each `eval_activity.json`. The
  per-class precision, recall and F1 values are in the appendix.
* Accuracy is quoted at the precision of the extraction script (6 dp); the
  logged value is the same number at 4 dp.

## Measured verification result (HAR, Siamese)

| source | EER | FAR | FRR | ROC AUC | n_pairs | n_subjects | embed_dim | params | epochs | artifact |
|---|---|---|---|---|---|---|---|---|---|---|
| har | 0.4162 | 0.4162 | 0.4162 | 0.62435346 | 10000 | 9 | 128 | 275712 | 20 | `runs/real/siamese/eval_verification.json` |

Training provenance of the checkpoint: `loss_type: triplet`, `margin: 1.0`,
`seed: 42`, `selection_metric: train_loss`, `best_epoch: 20` (from
`runs/real/siamese/metrics.json`).

**Pair protocol (as implemented in `engine/evaluate.py::verification_metrics`,
`n_pairs=10000`, `seed=42`):**

* genuine pair = two **different windows of the same subject** (drawn from
  subjects with ≥ 2 windows);
* impostor pair = two windows from **different subjects**;
* 10000 pairs total: 5000 genuine + 5000 impostor, sampled with a
  `random.Random(42)` instance;
* score = cosine distance on L2-normalized 128-d embeddings, negated so higher
  means more similar; ROC AUC is computed on those similarities;
* EER/FAR/FRR are linearly interpolated at the single crossing point of the
  FAR and FRR curves.
* `n_subjects = 9` is the number of distinct subjects in the HAR **test** split
  (21 in the train split), re-measured from the adapter for this document.

**`FAR == FRR == EER == 0.4162` here.** This is one measured quantity reported
three times, not three independent measurements: the implementation
interpolates a single crossing point and returns `far`, `frr` and
`0.5 * (far + frr)` from it. Nothing should be inferred from their agreement.

## Confusion matrices (as measured)

Rows are true classes, columns predicted classes; order is the
`activity_names` list stored in each artifact. `LAYING` is also the last row
and last column everywhere.

### `cnn_lstm`, HAR — accuracy 0.930098

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 495 | 1 | 0 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 0 | 452 | 19 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 6 | 1 | 413 | 0 | 0 | 0 |
| SITTING | 0 | 2 | 0 | 420 | 62 | 7 |
| STANDING | 0 | 1 | 0 | 107 | 424 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

Reading: errors are concentrated in SITTING↔STANDING (62 and 107 confused
windows, plus 7 SITTING→LAYING); all walking classes and LAYING are otherwise
near-clean.

### `cnn_baseline`, HAR — accuracy 0.895487

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 467 | 13 | 16 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 2 | 449 | 20 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 22 | 67 | 331 | 0 | 0 | 0 |
| SITTING | 0 | 5 | 0 | 416 | 70 | 0 |
| STANDING | 0 | 0 | 0 | 93 | 439 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

Reading: errors split between SITTING↔STANDING (70 and 93) and the two
walking-stair classes (67 DOWNSTAIRS→UPSTAIRS, 22→WALKING); LAYING is perfect.

### `se_convnet`, HAR — accuracy 0.929420

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 496 | 0 | 0 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 16 | 444 | 10 | 1 | 0 | 0 |
| WALKING_DOWNSTAIRS | 0 | 0 | 420 | 0 | 0 | 0 |
| SITTING | 0 | 17 | 0 | 358 | 109 | 7 |
| STANDING | 0 | 1 | 0 | 47 | 484 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

Reading: SITTING↔STANDING dominates (109 and 47) and is the worst of the five
HAR models on SITTING (recall 0.7291); walking classes and LAYING are clean.

### `gait_vit`, HAR — accuracy 0.923651

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 481 | 3 | 12 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 9 | 458 | 3 | 1 | 0 | 0 |
| WALKING_DOWNSTAIRS | 4 | 4 | 412 | 0 | 0 | 0 |
| SITTING | 1 | 19 | 0 | 380 | 86 | 5 |
| STANDING | 3 | 4 | 0 | 71 | 454 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

Reading: SITTING↔STANDING dominates (86 and 71); with 813958 parameters it is
the largest model here and not the most accurate.

### `multitask`, HAR — accuracy 0.908381

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 492 | 4 | 0 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 8 | 455 | 8 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 9 | 3 | 408 | 0 | 0 | 0 |
| SITTING | 0 | 3 | 0 | 406 | 82 | 0 |
| STANDING | 0 | 3 | 0 | 123 | 406 | 0 |
| LAYING | 4 | 23 | 0 | 0 | 0 | 510 |

Reading: SITTING↔STANDING dominates (82 and 123) and it is the only model with
a visible LAYING error (23 LAYING→WALKING_UPSTAIRS, 4→WALKING), consistent with
its worse overall accuracy.

### `cnn_lstm`, WISDM — accuracy 0.866802

| true \ pred | activity_0 | activity_1 | activity_2 | activity_3 | activity_4 | activity_5 |
|---|---|---|---|---|---|---|
| activity_0 | 265 | 6 | 1 | 0 | 7 | 9 |
| activity_1 | 17 | 548 | 1 | 0 | 9 | 17 |
| activity_2 | 0 | 0 | 121 | 33 | 0 | 0 |
| activity_3 | 0 | 0 | 0 | 120 | 1 | 0 |
| activity_4 | 18 | 4 | 1 | 0 | 396 | 11 |
| activity_5 | 46 | 104 | 7 | 0 | 102 | 1114 |

Reading: errors are diffuse across the label space. The class labels are the
adapter's generic `activity_0..5` (WISDM label semantics are Student A's, not
invented here), so the readable statements are limited to the matrix itself:
activity_2↔activity_3 (33), activity_5 absorbing errors from activity_1 (104)
and activity_4 (102), and activity_3 being nearly perfect (120/121).

## Interpretation / limits

(a) **No validation split.** Both trainers select `best.pt` on the *lowest
training loss* (`selection_metric: "train_loss"`, recorded in the checkpoint,
in `metrics.json` and in `STUDENT_B_SCOPE.md`). No validation split exists and
no split protocol has been agreed, so these are **not validation-selected**
results; the test split is used only by `engine/evaluate.py` and never for
selection or tuning. The reported test numbers are therefore one honest
measurement of one training run with 20 fixed epochs, not a model-selection
procedure.

(b) **The activity accuracy is not an authentication result.** The goal metric
for this project is verification, and the only verification measurement here is
EER 0.4162 with ROC AUC 0.62435346 — i.e. close to the 0.5/0.0-population-
inversion chance level, so the current Siamese embedding is **unusable for
authentication**. 0.93 activity accuracy and 0.42 EER are consistent: the
embedding separates activities, not identities. No claim about the other five
models' authentication utility can be made at all; they were not evaluated for
verification.

(c) **WISDM has no verification metric.** `data/adapter.py` returns
`y_subject=None` for WISDM (it ships no subject ids), so genuine/impostor pairs
cannot be formed and no EER/FAR/FRR exists for that source. Only its activity
accuracy (0.866802) is measured.

(d) **The multitask model's identity number is not a test result.**
`final_subject_acc = 0.6949129488574538` is the last **epoch's training-split**
subject accuracy from `runs/real/multitask/metrics.json` — it is the fraction of
the *training* windows whose subject the head predicted correctly at the end of
epoch 20. WISDM has no subject ids and no WISDM multitask run exists, and the
test split is never fed to the subject head by `engine/evaluate.py` (the
`--task activity` path uses only the activity logits), so there is no held-out
subject accuracy for this model anywhere in these artifacts. Do not cite 0.6949
as a recognition result.

(e) **`window_size` / `stride` are no-ops for HAR.** UCI HAR ships
pre-segmented 128-sample windows; `data/adapter.py::_load_har` ignores both keys
and loads one row per file row. The `window_size: 128` / `stride: 64` values in
the HAR run configs therefore had no effect on the data. They are meaningful
only for WISDM, where `_load_wisdm` slides a 128-sample window with stride 64
and refuses windows crossing a session boundary.

## Literature-quoted values (NOT measured)

none recorded

## Reproduce

Prerequisites (both required; `datasets/` is gitignored and `CI-2026.zip` is a
Git LFS pointer in the repo):

```bash
git lfs pull
python scripts/setup_datasets.py
```

Nothing has to be authenticated or tuned — no epoch/loss options were passed.
Build the run configs, then train and evaluate each model. The configs were
derived from `configs/*.yaml` by dropping `subset`, setting `epochs: 20`,
`torch_threads: 8` and `out_dir: runs/real/<name>`; that resolved config is what
`runs/real_configs/real_<name>.yaml` held at run time.

```bash
# per HAR classification model in {cnn_lstm, cnn_baseline, se_convnet, gait_vit, multitask}
python -m engine.train --config runs/real_configs/real_<name>.yaml
python -m engine.evaluate --checkpoint runs/real/<name>/best.pt \
    --config runs/real_configs/real_<name>.yaml --task activity

# WISDM (source: wisdm in the config)
python -m engine.train --config runs/real_configs/real_wisdm_cnn_lstm.yaml
python -m engine.evaluate --checkpoint runs/real/wisdm_cnn_lstm/best.pt \
    --config runs/real_configs/real_wisdm_cnn_lstm.yaml --task activity

# Siamese verification (HAR only — needs subject ids)
python -m engine.train_siamese --config runs/real_configs/real_siamese.yaml
python -m engine.evaluate --checkpoint runs/real/siamese/best.pt \
    --config runs/real_configs/real_siamese.yaml --task verification
```

`engine.evaluate` defaults `--out-dir` to the checkpoint's parent directory, so
the eval JSONs land next to `best.pt` as listed above. `--subset` must be
omitted to reproduce these numbers (all artifacts record `subset: null`).

## Verification investigation (negative result)

**What was tested and why.** The committed verification baseline
(`runs/real/siamese/eval_verification.json`) is EER 0.4162 / ROC AUC 0.62435346
— unusable for authentication. That run had no validation split, was trained
with a single triplet-sampler pass per epoch (20 epochs = 1880 Adam steps), and
selected `best.pt` on the **training** loss (`selection_metric: "train_loss"`,
`best_epoch: 20`), so the checkpoint was never selected on anything that
measures the goal. The diagnosis was therefore undertraining + train-loss-based
selection + a low effective rank (participation ratio 5.4/128, 17 components for
99% of test variance). Two levers were added to `engine/train_siamese.py`:
`triplets_per_epoch` (raise the optimiser budget) and `val_subjects` (carve a
subject-disjoint validation split and select the checkpoint on validation EER
instead of training loss). This section records whether they worked. **They did
not.**

**Diagnosis evidence.** Measured with a read-only throwaway script
(`/tmp/opencode/diag_siamese.py`), re-run for this section, with every row
scored by the same `engine.evaluate.verification_metrics` call
(`n_pairs=10000`, `seed=42`) so the rows are directly comparable:

| score source | EER | ROC AUC |
|---|---|---|
| trained net (committed baseline, 128-d) | 0.4162 | 0.62435346 |
| raw flattened windows (1152-d) | 0.4894 | 0.51319638 |
| per-channel mean / std / energy (27-d) | 0.4822 | 0.52972028 |
| randomly-initialised net, same architecture, seed 42 | 0.4744 | 0.53649116 |
| activity one-hot control (6-d) | 0.1698 | 0.5003 |

The one-hot row is a **trap, not a result**. Its EER 0.1698 comes from a
two-valued score set (1 when the two windows share an activity label, 0
otherwise): the FAR and FRR curves are step functions and their linear
interpolation lands on the same-activity collision rate of the sampled impostor
pairs — measured 849/5000 = 0.1698 under the identical seed-42 pair sampling.
Its ROC AUC is 0.5003, i.e. chance. Quoting 0.1698 as a verification result
would be quoting an interpolation artefact.

The trained embedding is neither collapsed nor activity-driven: 0 of 128
dimensions have variance below 1e-6 (test), and nearest-centroid identity
decodability on the test split (0.3627) exceeds activity decodability (0.3549)
by 0.0078, so there is no activity leakage. The low effective rank is real, but
the representation is not degenerate.

**Sweep.** Six runs; `triplets_per_epoch` and `model_args.embed_dim` vary per
run, `epochs` came from the `--epochs` CLI override (read from each
`metrics.json` `epochs` field), and all six used a subject-disjoint validation
carve (`val_subjects: 4`, 1383 validation windows, `n_train_windows = 5969`).
Optimiser steps are derived arithmetically, once:

```
steps = epochs × ceil(triplets_per_epoch × n_train_windows / batch_size)
      = epochs × ceil(triplets_per_epoch × 5969 / 64)
```

with `n_train_windows = 5969` (7352 HAR train windows minus the 1383-window
validation carve) and `batch_size = 64` (one Adam step per 64-triplet chunk,
`engine/train_siamese.py:239`).

| run | triplets/epoch | embed_dim | epochs | optimiser steps | best_epoch | val_eer | val_roc_auc | test EER | test ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 128 | 20 | 1880 | 14 | 0.4048 | 0.6500 | 0.4184 | 0.6234 |
| **B** | 5 | 128 | 20 | 9340 | 16 | 0.3676 | 0.6688 | 0.4258 | 0.6079 |
| C | 5 | 32 | 20 | 9340 | 9 | 0.4138 | 0.6196 | 0.4272 | 0.5989 |
| D | 2 | 32 | 20 | 3740 | 10 | 0.3892 | 0.6374 | 0.4174 | 0.6089 |
| E | 5 | 32 | 40 | 18680 | 40 | 0.4040 | 0.6329 | 0.4318 | 0.5953 |
| F | 10 | 64 | 30 | 27990 | 3 | 0.4190 | 0.6212 | 0.4178 | 0.6126 |

`best_epoch`, `val_eer` and `val_roc_auc` are the selected epoch's recorded
validation values (`metrics.json`); `test EER` / `test ROC AUC` are the
`eval_verification.json` fields. **Run B is the validation-selected winner**
(lowest `val_eer`, 0.3676) — and the test columns played no part in the
selection, because `best.pt` was written whenever the per-epoch validation EER
improved and the test split is never loaded during training. All six artifacts
record `git_commit: d73e095cfc4a4fbbbae01462ce066bffc6390112`, which is the
current branch head.

**The result.** No run met the success bar (EER < 0.35 **and** ROC AUC > 0.70).
The best test EER across the sweep is **0.4174** (run D, 3740 steps, dim 32) and
the best test ROC AUC is **0.6234** (0.62340680, run A, 1880 steps, dim 128) —
neither beats the committed baseline (0.4162 EER / 0.62435346 ROC AUC). The
validation-selected winner (run B)
was **worse** on test than the committed baseline: test EER **0.4258** vs
**0.4162** (+0.0096) and test ROC AUC **0.6079** vs **0.62435346**
(−0.01642910).

**What it refutes.**

* More optimiser steps did not help, and at fixed `embed_dim` made test EER
  monotonically worse. Dim 32: 3740 steps → 0.4174, 9340 → 0.4272, 18680 →
  0.4318. Dim 128: 1880 → 0.4184, 9340 → 0.4258.
* Lower `embed_dim` did not help: at equal budget (9340 steps) dim 128 gave
  0.4258 and dim 32 gave 0.4272.
* Training loss and the goal metric are decoupled: run F's training loss fell
  from 0.6036 (epoch 1) to 0.1668 (epoch 30) while its validation EER oscillated
  in 0.4190–0.4620 with no downward trend.
* Across a 14.9× span in optimiser steps (1880 → 27990) the test EER moved only
  within 0.4174–0.4318, with no trend.

**What this implies.** Neither the optimiser budget nor the embedding width is
the binding constraint, and the sweep provides no evidence for which lever
would be. **Hypothesis, explicitly not measured:** UCI HAR is a daily-activity
dataset recorded from a waist-mounted phone, not a gait-walking protocol, so
the identity signal in the windows may be too weak for cross-subject
verification regardless of training budget or capacity. The measured genuine-vs-
impostor cosine margin of the baseline embedding — 0.1869 on test versus 0.5899
on train, a 3.16× collapse — is consistent with fitting the training identities
rather than learning a transferable gait embedding. That is a hypothesis to
test, not a finding: this sweep varied optimiser steps and width only, and does
not isolate data quality from model capacity.

**Blocker for the data owner (Student A).** Verification requires
subject-labelled walking data, and the WISDM data delivered here is not
subject-labelled. The CSV pair `datasets/wisdm/{train,test}.csv` has columns
`acc_x,acc_y,acc_z,timestamp,Activity` and no subject/user column, so
`data/adapter.py` returns `y_subject=None` for WISDM
(`data/adapter.py:119`) and no verification metric (EER/FAR/FRR) is computable
on that source. The public WISDM v1.1 raw data carries a `user` column, so the
subject ids can be restored by re-deriving these CSVs from the raw files; that
re-derivation is Student A's deliverable, and until it lands the goal metric is
unmeasurable on WISDM — this blocks Student B's verification work and Student
C's dependent work. WISDM activity labels likewise arrive as anonymous
`activity_0..activity_5` (`WISDM_ACTIVITY_NAMES` in `data/adapter.py`), so the
activity semantics are Student A's to supply as well.

**Reproduce the sweep.** The six derived configs live under the gitignored
`runs/sweep_configs/`; each sets `val_subjects: 4`, `triplets_per_epoch` and
`model_args.embed_dim` per the table above, and `out_dir: runs/sweep/<letter>`.

```bash
python -m engine.train_siamese --config runs/sweep_configs/siamese_A.yaml --epochs 20 --out-dir runs/sweep/A
python -m engine.evaluate --checkpoint runs/sweep/A/best.pt --config runs/sweep_configs/siamese_A.yaml --task verification --out-dir runs/sweep/A
python -m engine.train_siamese --config runs/sweep_configs/siamese_B.yaml --epochs 20 --out-dir runs/sweep/B
python -m engine.evaluate --checkpoint runs/sweep/B/best.pt --config runs/sweep_configs/siamese_B.yaml --task verification --out-dir runs/sweep/B
python -m engine.train_siamese --config runs/sweep_configs/siamese_C.yaml --epochs 20 --out-dir runs/sweep/C
python -m engine.evaluate --checkpoint runs/sweep/C/best.pt --config runs/sweep_configs/siamese_C.yaml --task verification --out-dir runs/sweep/C
python -m engine.train_siamese --config runs/sweep_configs/siamese_D.yaml --epochs 20 --out-dir runs/sweep/D
python -m engine.evaluate --checkpoint runs/sweep/D/best.pt --config runs/sweep_configs/siamese_D.yaml --task verification --out-dir runs/sweep/D
python -m engine.train_siamese --config runs/sweep_configs/siamese_E.yaml --epochs 40 --out-dir runs/sweep/E
python -m engine.evaluate --checkpoint runs/sweep/E/best.pt --config runs/sweep_configs/siamese_E.yaml --task verification --out-dir runs/sweep/E
python -m engine.train_siamese --config runs/sweep_configs/siamese_F.yaml --epochs 30 --out-dir runs/sweep/F
python -m engine.evaluate --checkpoint runs/sweep/F/best.pt --config runs/sweep_configs/siamese_F.yaml --task verification --out-dir runs/sweep/F
```

`--subset` must be omitted (all twelve artifacts record `subset: null`), and
`runs/sweep_configs/` is gitignored: the config files are not in the repository
and must be re-derived from the table's `triplets_per_epoch` / `embed_dim`
values plus `val_subjects: 4` before these commands are reproducible.

## Appendix — measured per-class precision / recall, and F1 derived from them

F1 = `2PR/(P+R)`, computed by the extraction script from the artifact
`per_class` values below; precision/recall are printed at 4 dp, F1 at 4 dp.

| run | class | precision | recall | F1 |
|---|---|---|---|---|
| `cnn_lstm` (har) | WALKING | 0.9880 | 0.9980 | 0.9930 |
| `cnn_lstm` (har) | WALKING_UPSTAIRS | 0.9891 | 0.9597 | 0.9741 |
| `cnn_lstm` (har) | WALKING_DOWNSTAIRS | 0.9560 | 0.9833 | 0.9695 |
| `cnn_lstm` (har) | SITTING | 0.7970 | 0.8554 | 0.8251 |
| `cnn_lstm` (har) | STANDING | 0.8724 | 0.7970 | 0.8330 |
| `cnn_lstm` (har) | LAYING | 0.9871 | 1.0000 | 0.9935 |
| `cnn_baseline` (har) | WALKING | 0.9511 | 0.9415 | 0.9463 |
| `cnn_baseline` (har) | WALKING_UPSTAIRS | 0.8408 | 0.9533 | 0.8935 |
| `cnn_baseline` (har) | WALKING_DOWNSTAIRS | 0.9019 | 0.7881 | 0.8412 |
| `cnn_baseline` (har) | SITTING | 0.8173 | 0.8473 | 0.8320 |
| `cnn_baseline` (har) | STANDING | 0.8625 | 0.8252 | 0.8434 |
| `cnn_baseline` (har) | LAYING | 1.0000 | 1.0000 | 1.0000 |
| `se_convnet` (har) | WALKING | 0.9688 | 1.0000 | 0.9841 |
| `se_convnet` (har) | WALKING_UPSTAIRS | 0.9610 | 0.9427 | 0.9518 |
| `se_convnet` (har) | WALKING_DOWNSTAIRS | 0.9767 | 1.0000 | 0.9882 |
| `se_convnet` (har) | SITTING | 0.8818 | 0.7291 | 0.7982 |
| `se_convnet` (har) | STANDING | 0.8162 | 0.9098 | 0.8604 |
| `se_convnet` (har) | LAYING | 0.9871 | 1.0000 | 0.9935 |
| `gait_vit` (har) | WALKING | 0.9659 | 0.9698 | 0.9678 |
| `gait_vit` (har) | WALKING_UPSTAIRS | 0.9385 | 0.9724 | 0.9552 |
| `gait_vit` (har) | WALKING_DOWNSTAIRS | 0.9649 | 0.9810 | 0.9728 |
| `gait_vit` (har) | SITTING | 0.8407 | 0.7739 | 0.8059 |
| `gait_vit` (har) | STANDING | 0.8407 | 0.8534 | 0.8470 |
| `gait_vit` (har) | LAYING | 0.9908 | 1.0000 | 0.9954 |
| `multitask` (har) | WALKING | 0.9591 | 0.9919 | 0.9752 |
| `multitask` (har) | WALKING_UPSTAIRS | 0.9267 | 0.9660 | 0.9459 |
| `multitask` (har) | WALKING_DOWNSTAIRS | 0.9808 | 0.9714 | 0.9761 |
| `multitask` (har) | SITTING | 0.7675 | 0.8269 | 0.7961 |
| `multitask` (har) | STANDING | 0.8320 | 0.7632 | 0.7961 |
| `multitask` (har) | LAYING | 1.0000 | 0.9497 | 0.9742 |
| `wisdm_cnn_lstm` (wisdm) | activity_0 | 0.7659 | 0.9201 | 0.8360 |
| `wisdm_cnn_lstm` (wisdm) | activity_1 | 0.8278 | 0.9257 | 0.8740 |
| `wisdm_cnn_lstm` (wisdm) | activity_2 | 0.9237 | 0.7857 | 0.8491 |
| `wisdm_cnn_lstm` (wisdm) | activity_3 | 0.7843 | 0.9917 | 0.8759 |
| `wisdm_cnn_lstm` (wisdm) | activity_4 | 0.7689 | 0.9209 | 0.8381 |
| `wisdm_cnn_lstm` (wisdm) | activity_5 | 0.9679 | 0.8114 | 0.8827 |
