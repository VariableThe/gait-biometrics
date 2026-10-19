# Student B — Scope Map (paper tracks B1–B7)

**Status: integration documentation, not a specification.** The source for the
B-track mapping below is `Research_Reading_Plan_Gait.docx`, which is a **reading
plan**: it lists papers and study annotations. It is *not* an implementation task
list, API spec, or deliverable contract. Where this file states what the code
*does*, that claim was verified against this repository; where the plan only
suggests a topic, it is marked as a reading item. The per-paper notes in
`reading_notes/` are title/annotation-derived and remain **UNVERIFIED**: the
B1–B7 PDFs do exist in the companion minor-project checkout (under
`papers/student_b/`), but they were **not reviewed** for this code mapping, so
nothing here should be read as a claim about the papers' content.

Student B's plan section is "Deep Learning Model Development" with seven paper
tracks. All seven have corresponding code and/or config in this repo:

| Track | Plan topic / working title | Reading-note venue | Code | Config |
|-------|----------------------------|--------------------|------|--------|
| B1 | IMU gait recognition with CNNs + multi-sensor fusion | Dehzangi et al., 2017, Sensors | `models/cnn_baseline.py` (`MultiSensorFusionCNN`, `_SensorBranch`) | `configs/cnn_baseline.yaml` |
| B2 | METIER: multi-task activity + user recognition (CNN+LSTM) | Chen et al., 2020, ACM IMWUT | `models/multitask.py` (`MultiTaskNet`, `multitask_loss`) | `configs/multitask.yaml` |
| B3 | DeepConvLSTM for continuous authentication from activity patterns | Sensors (MDPI) | `models/cnn_lstm.py` (`DeepConvLSTM`) — primary classification baseline | `configs/cnn_lstm.yaml` |
| B4 | Siamese denoising autoencoders for robust gait recognition | Sheng & Li, 2020, Neurocomputing | `models/siamese.py` (`EmbeddingNet`, `SiameseNet`, `triplet_loss`, `contrastive_loss`, `TripletWindowSampler`); trained by `engine/train_siamese.py` | `configs/siamese.yaml` |
| B5 | Wearable behavioural authentication with squeeze-and-excitation (SE-DeepConvNet) | Computers (MDPI), 2024 | `models/attention.py` (`SEBlock`, `DepthwiseSeparableConv1d`, `SEConvNet`) | `configs/se_attention.yaml` |
| B6 | Real-time multi-person gait recognition at the edge (pruning/quantization) | Scientific Reports, 2025 | `models/compress.py` (counting, pruning, dynamic quantization, ONNX export); `scripts/export_onnx.py` | `configs/lightweight.yaml` |
| B7 | SEGait-ConViT: ViT for efficient gait recognition | Procedia CS, 2025 | `models/vit.py` (`PatchEmbed1d`, `ConViTBlock`, `GaitViT`) | `configs/vit.yaml` |

Registry keys (`models/registry.py`): `cnn_baseline`, `cnn_lstm`, `multitask`,
`se_convnet`, `gait_vit`. The Siamese model is **not** a registry key: `SiameseNet`
wraps an `EmbeddingNet`, so it is trained by `engine/train_siamese.py` and
evaluated as `--task verification` by `engine/evaluate.py` (`configs/siamese.yaml`
keeps `model: cnn_lstm` only to satisfy the shared config shape).

## Compression support (B6) — verified

`models/compress.py::apply_pruning(model, amount, mode=...)`:

* **Structured** (`mode="structured"`) is topology-specific. It is implemented —
  index-for-index, and covered by exact-index plus mask-equivalent regression
  tests (`tests/test_compress_deepconvlstm_alignment.py`) — for exactly one
  topology: `DeepConvLSTM` (B3, registry key `cnn_lstm`, the model
  `configs/lightweight.yaml` selects). One channel keep-set is propagated along
  its Conv1d -> BatchNorm1d -> LSTM -> Linear order: conv input columns come from
  the producer's kept indices, BatchNorm `weight`/`bias`/`running_mean`/
  `running_var` are indexed by those same indices, LSTM gate rows /
  `weight_hh`/`weight_ih` columns / both biases follow the kept hidden units
  (layer 0's `weight_ih` columns follow the conv's kept channels), and the head
  Linear takes exactly the kept hidden-unit columns.
* **Structured pruning is not implemented** for `MultiSensorFusionCNN` (B1),
  `MultiTaskNet` (B2), `EmbeddingNet` (B4), `SEConvNet` (B5) or `GaitViT` (B7);
  `apply_pruning(..., "structured")` raises `NotImplementedError` *before* any
  mutation (no partial/half-pruned model is ever returned) and
  `structured_prune_support(model)` returns the reason. B1 has one independent
  branch per sensor channel (no single channel flow), B2 has two parallel Linear
  heads off the same trunk vector, and B7 ties the patch-embedding width to
  `embed_dim` because the CLS/positional embeddings are added to it.
  **B4 and B5 are deferred, not supported**: a previous generic cascade accepted
  them while slicing downstream tensors by *prefix* rather than by the producer's
  kept indices, which silently attached BatchNorm affine/running statistics,
  classifier columns and (for B5) SE bottleneck / depthwise filters to the wrong
  channels. They now fail closed until each gets topology-specific slicing with
  exact-index tests. Use `mode="unstructured"` as the supported fallback for all
  five.
* Independently of the class-name allowlist, a structural guard
  (`_check_prunable_structure`) rejects any model containing a grouped/depthwise
  Conv1d, an SE block, a non-affine/non-tracking BatchNorm1d, a bidirectional or
  projected LSTM, or no LSTM at all — so a model named `DeepConvLSTM` whose
  topology differs still fails closed instead of being mis-sliced.
* **Unstructured** (`mode="unstructured"`) applies L1 pruning to every Conv1d and
  Linear weight and then **materializes** the mask, so the zeros are genuine
  `weight` values. `pruned_weight_sparsity(model)` reports the mask-induced
  sparsity (it reads `weight_orig * weight_mask` while a reparametrization is
  live); a reparametrized model previously reported ~0% and could not be
  deep-copied or quantized.
* Pruning/quantization options for the export script are **config-driven**:
  `configs/lightweight.yaml` (`prune: 0.3`, `quantize: true`) supplies the
  defaults and the CLI flags (`--prune`/`--no-prune`, `--quantize`/`--no-quantize`)
  override them. An unsupported model with pruning enabled fails with a clear
  error instead of writing a broken ONNX graph.

## Checkpoint selection and load (cross-cutting)

* Both trainers (`engine/train.py`, `engine/train_siamese.py`) select `best.pt`
  on **training loss**, and record `"selection_metric": "train_loss"` in the
  checkpoint metrics and in `metrics.json`. There is **no validation split**
  (no split protocol has been agreed), and the held-out test split is used only
  by `engine/evaluate.py` — never for selection or tuning. Adding a real
  validation split is deferred until Student A's data protocol exists.
* Checkpoints are loaded with `weights_only=True` via
  `models.common.load_checkpoint` (`engine/evaluate.py`, `scripts/export_onnx.py`),
  so an untrusted `.pt` cannot execute code during load. Rich checkpoints
  (`save_checkpoint`) and bare `state_dict`s both load.
* `scripts/setup_datasets.py` rejects archive members that escape the destination
  (`../`, absolute POSIX/Windows/UNC paths) and symlink members, while keeping the
  normal-file path and the existing macOS-metadata filters.

## Open integration dependencies (not owned by Student B)

These are **unresolved and deliberately untouched** — do not read this file as a
claim that they are fixed:

* **Student A — `data/adapter.py` (frozen shared contract).** `load_windows`,
  the WISDM label/subject semantics and any resampling/windowing live there.
  WISDM currently carries no subject ids (`y_subject=None`) and generic activity
  names (`activity_0..5`); the mapping to real WISDM labels/subjects must come
  from Student A, not be invented here.
* **Student A — window size / time dimension.** The fixed `T=128` (and every
  model's pooled/shape assumption, e.g. `SEConvNet`'s `128 * 16` classifier input
  and `GaitViT`'s `n_patches = 128 // patch_len`) depends on the adapter's window
  size. If that changes, those shapes must be revisited together with Student A.
* **Student C — evaluation framework.** EER semantics (including tie handling at
  the equal-error operating point) are Student C's definition. `engine/evaluate.py`
  implements a linear-interpolation EER; it was **not** changed in this work.
* **Data availability.** `datasets/` is absent and `CI-2026.zip` is a Git LFS
  pointer (not fetched here), so `scripts/smoke_test.py` and any dataset-backed
  run could not be executed in this environment.

## Verification performed

`python -m unittest discover -s tests -p 'test_*.py'` covers: exact-index
alignment and mask-equivalent equivalence of structured pruning for the one
supported topology (`DeepConvLSTM`), plus early failure without mutation for
B1/B2/B4/B5/B7 and for structurally-uncovered shapes (implemented, run, and
passing: 51 tests); unstructured sparsity accuracy and downstream
forward/deep-copy/quantize; config-default vs CLI-override resolution for the
lightweight export options; archive traversal/absolute/symlink rejection with
normal extraction intact; and safe checkpoint loading (rich dict, bare
`state_dict`, malicious pickle rejected).
