# Minor Specialization Project — Gait-Based User Authentication

**Team**: Student A (Data Pipeline), Student B (Model Development), Student C (Evaluation Framework)  
**Mentor**: Manjunath Hegde (MAHE)  
**Focus**: Wearable-sensor gait recognition for continuous user authentication

---

## Project Overview

This project builds a **gait-based continuous user authentication system** using wearable sensor data (accelerometer/gyroscope from smartphones). The pipeline uses deep learning models to recognize users by their unique walking patterns, enabling continuous authentication without explicit user interaction.

The work is divided across three students:

| Student | Track | Responsibility |
|---------|-------|----------------|
| **A** | Data Pipeline & Preprocessing | Data loading, windowing, resampling, sensor fusion, dataset preparation |
| **B** | Deep Learning Model Development | CNN/LSTM/ViT architectures, multi-task learning, Siamese verification, edge compression |
| **C** | Evaluation Framework | Metrics (EER, FAR, FRR), verification protocols, benchmarking, reporting |

---

## Repository Structure

```
minor-specialization-project/
├── README.md                    # This file
├── Research_Reading_Plan_Gait.docx   # Reading plan (7 Student-B papers + 5 common)
├── link.md                      # OneDrive link to datasets (CI-2026)
├── papers/
│   ├── student_b/               # 7 papers for Student B (B1–B7)
│   └── common/                  # 5 common papers (C1–C5)
│
├── gait-biometrics-student-b/   # Student B's model repo — separate clone inside this
│   │                            # directory (not a submodule, not tracked here)
│   ├── configs/                 # YAML configs for each model track
│   ├── data/                    # Frozen adapter interface (Student A owns this)
│   ├── engine/                  # Training, evaluation, Siamese training
│   ├── models/                  # Model implementations (B1–B7 tracks)
│   ├── scripts/                 # Setup, export, smoke test scripts
│   ├── tests/                   # Unit tests (67 passing; synthetic, not dataset-backed results)
│   ├── reading_notes/           # Paper notes (UNVERIFIED)
│   ├── gait_biometrics_summary.md
│   ├── requirements.txt
│   └── STUDENT_B_SCOPE.md       # Detailed scope & integration notes
```

---

## Quick Start

### Prerequisites

- Python: no floor pinned in the repo (the measured runs used Python 3.14.8)
- `git lfs` (for dataset)
- PyTorch 2.14.1, the pin in `gait-biometrics-student-b/requirements.txt` (the measured runs used torch 2.12.1+cu130)

### Clone & Setup

```bash
# Clone the main project
git clone https://github.com/VariableThe/minor-specialization-project.git
cd minor-specialization-project

# Clone Student B's model repo (inside this directory)
git clone https://github.com/VariableThe/gait-biometrics-student-b.git
cd gait-biometrics-student-b

# Install dependencies
pip install -r requirements.txt

# Install git-lfs and pull the dataset (~68 MiB)
git lfs install
git lfs pull

# Extract datasets
python scripts/setup_datasets.py
```

This creates:
```
datasets/
├── wisdm/
│   ├── train.csv
│   └── test.csv
└── uci_har/
    └── UCI HAR Dataset/
```

---

## Student A — Data Pipeline (Your Starting Point)

### What You Own

**File**: `gait-biometrics-student-b/data/adapter.py`

This is the **frozen shared contract**. Student B's and C's code imports `load_windows` and `build_loader` from here. **Do not change the function signatures or return types.**

### Interface Contract

```python
# data/adapter.py
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader

@dataclass
class WindowData:
    X: torch.Tensor              # (N, C, T) float32 — sensor windows
    y_activity: torch.Tensor     # (N,) int64 — activity labels
    y_subject: Optional[torch.Tensor]  # (N,) int64 or None — subject IDs
    activity_names: list[str]    # HAR: display names read from activity_labels.txt;
                                 # WISDM: ["activity_0", ..., "activity_5"]
    subject_ids: list[int]       # e.g. [1, 2, 3, ...]
    sampling_rate: float         # Hz (50.0 for HAR, 20.0 for WISDM)

def load_windows(source: str, split: str, window_size: int, stride: int, data_root: str) -> WindowData:
    """Load preprocessed windows for HAR or WISDM."""
    ...

def build_loader(data: WindowData, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Wrap WindowData in a DataLoader."""
    ...
```

### What You Need to Implement

1. **`load_windows(source, split, window_size, stride, data_root)`**
   - `source`: `"har"` or `"wisdm"`
   - `split`: `"train"` or `"test"`
   - Returns `WindowData` with correctly shaped tensors
   - **Current stub** loads pre-windowed HAR Inertial Signals and windows WISDM CSV with session boundaries

2. **`build_loader(data, batch_size, shuffle, num_workers)`**
   - Wraps `WindowData` in a `DataLoader`
   - Should handle `y_subject=None` gracefully (WISDM has no subject IDs)

### Datasets You'll Work With

| Dataset | Channels | Sampling Rate | Window Size | Subjects | Activities |
|---------|----------|---------------|-------------|----------|------------|
| **UCI HAR** | 9 (acc/gyro x/y/z) | 50 Hz | 128 (fixed) | 30 | 6 |
| **WISDM** | 3 (acc x/y/z) | 20 Hz | configurable | — | 6 |

**Key Notes:**
- HAR uses **pre-windowed** Inertial Signals (128 samples, fixed)
- WISDM requires **session-aware windowing** (your implementation)
- WISDM **has no subject IDs** in the raw data — `y_subject=None` is correct
- Activity labels: the HAR source labels are 1–6, but the adapter applies `- 1`, so `y_activity` is 0-indexed 0–5; WISDM labels are 0-indexed 0–5

### Testing Your Adapter

```bash
cd gait-biometrics-student-b

# Quick sanity check
python -c "
from data.adapter import load_windows, build_loader
data = load_windows('har', 'train', 128, 64, '.')
print('X:', data.X.shape)           # (N, 9, 128)
print('y_activity:', data.y_activity.shape)
print('activity_names:', data.activity_names)
loader = build_loader(data, batch_size=64, shuffle=True, num_workers=0)
print('Loader batches:', len(loader))
"
```

---

## Student B — Model Development (Already Implemented)

### Model Tracks (B1–B7)

| Track | Model | Registry Key | Config | Paper |
|-------|-------|--------------|--------|-------|
| B1 | Multi-Sensor Fusion CNN | `cnn_baseline` | `configs/cnn_baseline.yaml` | Dehzangi 2017 |
| B2 | Multi-Task Activity+Identity | `multitask` | `configs/multitask.yaml` | METIER 2020 |
| B3 | DeepConvLSTM | `cnn_lstm` | `configs/cnn_lstm.yaml` | Sensors 2021 |
| B4 | Siamese + Denoising AE | `siamese` (via `cnn_lstm`) | `configs/siamese.yaml` | Sheng & Li 2020 |
| B5 | SE-DeepConvNet | `se_convnet` | `configs/se_attention.yaml` | Computers 2024 |
| B6 | Edge Compression (Prune/Quant) | `cnn_lstm` | `configs/lightweight.yaml` | Scientific Reports 2025 |
| B7 | SEGait-ConViT | `gait_vit` | `configs/vit.yaml` | Procedia CS 2025 |

### Key Files

| File | Purpose |
|------|---------|
| `models/registry.py` | `build_model(key, args, n_channels, n_classes)` |
| `models/compress.py` | Pruning, quantization, ONNX export |
| `engine/train.py` | Classification training (selects `best.pt` on **train loss**) |
| `engine/train_siamese.py` | Siamese training (triplet/contrastive) |
| `engine/evaluate.py` | Evaluation: activity + verification (EER, FAR, FRR, AUC) |
| `scripts/export_onnx.py` | Export pruned/quantized ONNX models |

### Training

```bash
cd gait-biometrics-student-b

# Classification (e.g., DeepConvLSTM on HAR)
python -m engine.train --config configs/cnn_lstm.yaml

# Siamese verification
python -m engine.train_siamese --config configs/siamese.yaml

# Edge export (prune + quantize + ONNX)
python -m scripts.export_onnx \
    --config configs/lightweight.yaml \
    --checkpoint runs/smoke_classification/cnn_lstm/best.pt \
    --out model.onnx \
    --prune 0.3 --quantize
```

---

## Measured Results

These are dataset-backed numbers (UCI HAR / WISDM test splits), not unit-test output. Full provenance, per-class precision/recall and confusion matrices are in `STUDENT_B_RESULTS.md` in the Student B repo.

**HAR activity classification** (UCI HAR test split, 2947 windows):

| Model | Registry key | Test accuracy | Macro-F1 |
|-------|--------------|---------------|----------|
| B3 DeepConvLSTM | `cnn_lstm` | 0.930098 | 0.931380 |
| B5 SE-DeepConvNet | `se_convnet` | 0.929420 | 0.929386 |
| B7 SEGait-ConViT | `gait_vit` | 0.923651 | 0.924022 |
| B2 Multi-task | `multitask` | 0.908381 | 0.910602 |
| B1 Fusion CNN | `cnn_baseline` | 0.895487 | 0.892737 |

**WISDM activity classification** (WISDM test split, 2958 windows): B3 DeepConvLSTM `cnn_lstm` accuracy **0.866802** (macro-F1 0.859304).

**HAR verification (Siamese, goal metric)**: EER **0.4162**, ROC AUC **0.6244** (10000 pairs, 9 test subjects, 128-d embeddings). This is close to chance and **not usable for authentication yet**.

**Negative sweep**: six configs varying the optimiser budget (`triplets_per_epoch`) and embedding width (`embed_dim`); the best test EER was **0.4174**, so neither lever is the bottleneck.

**Blocker (Student A)**: WISDM ships without a subject/user column, so `y_subject=None` and no verification metric is computable on that source. Restoring subject ids from the raw WISDM v1.1 files is Student A's deliverable and is what unblocks the goal metric.

---

## Student C — Evaluation Framework

### Your Interface

File: `gait-biometrics-student-b/engine/evaluate.py`

Key functions you own/define:

```python
def verification_metrics(embeddings, subject_ids, n_pairs: int = 10000,
                         seed: int = 42) -> dict:
    """EER, FAR, FRR, ROC-AUC from genuine/impostor pairs.

    Requires `subject_ids` (one per embedding row) to form the pairs; L2-normalizes
    `embeddings` internally, so pass raw embeddings.
    """
    ...

def _eer_operating_point(labels, scores) -> tuple[float, float, float]:
    """Return `(eer, far, frr)` at the single interpolated FAR/FRR crossing.
    `labels`: 1 = genuine, 0 = impostor; higher `scores` = more similar."""
    ...
```

### Current Status

- EER computation uses linear interpolation between FAR/FRR operating points
- **Known issue**: FAR, FRR and EER are all reported from one interpolated FAR/FRR crossing point, so `FAR == FRR == EER` — they are not three independent measurements
- Checkpoint loading uses `weights_only=True` (safe)
- The classification trainer has no validation split (selection on train loss); the Siamese trainer carves a subject-disjoint `val_subjects` split and selects `best.pt` on validation EER

---

## Integration Points

### Data Flow

```
Student A (adapter.py)
    └── load_windows() → WindowData
            │
            ├── Student B: engine.train() / train_siamese()
            │       └── build_loader() → DataLoader → Model
            │
            └── Student C: evaluate.py
                    └── verification_metrics() → EER, FAR, FRR, AUC
```

### Shared Constants

| Constant | Value | Location |
|----------|-------|----------|
| Window size (T) | 128 | `configs/*.yaml` `window_size` |
| HAR channels | 9 | `configs/*.yaml` `n_channels` |
| WISDM channels | 3 | no shipped config — all `configs/*.yaml` set `n_channels: 9` (HAR); a WISDM run needs a derived config with `source: wisdm` and `n_channels: 3` |
| HAR activities | 6 | `datasets/uci_har/UCI HAR Dataset/activity_labels.txt` (read at runtime by `data/adapter.py`) |
| WISDM activities | 6 | `data/adapter.py` `WISDM_ACTIVITY_NAMES` (`activity_0`..`activity_5`) |
| Sampling rate (HAR) | 50 Hz | `data/adapter.py` |
| Sampling rate (WISDM) | 20 Hz | `data/adapter.py` |

---

## Datasets

### CI-2026 Dataset (OneDrive)

The raw dataset is stored in the mentor's OneDrive. Link: see `link.md`

```
CI-2026.zip (68 MiB, Git LFS)
├── CI-2026/Dataset/
│   ├── WISDM_Dataset.zip
│   └── human+activity+recognition+using+smartphones.zip
│       └── UCI HAR Dataset.zip
```

Run `python scripts/setup_datasets.py` to extract to `datasets/`.

---

## Development Workflow

### For Student A (Data)

1. Implement `load_windows` / `build_loader` in `data/adapter.py`
2. Test with `python -c "from data.adapter import load_windows; ..."`
3. Ensure `y_subject=None` for WISDM, correct activity encoding
3. Commit changes — Student B/C will pull and test

### For Student B (Models)

1. Pull Student A's adapter changes
2. Train models: `python -m engine.train --config configs/*.yaml`
3. Export ONNX: `python -m scripts.export_onnx ...`
4. Document results in `reading_notes/` (upgrade UNVERIFIED notes)

### For Student C (Evaluation)

1. Define EER/FAR/FRR protocols in `engine/evaluate.py`
2. Ensure `verification_metrics` matches Student B's Siamese outputs
3. Produce benchmark reports

---

## Common Commands

```bash
# Install deps
pip install -r requirements.txt

# Run all tests (Student B's test suite)
python -m unittest discover -s tests -p 'test_*.py' -v

# Quick smoke test (requires datasets)
python scripts/smoke_test.py

# Export ONNX for edge deployment
python -m scripts.export_onnx --config configs/lightweight.yaml \
    --checkpoint runs/.../best.pt --out model.onnx

# Extract datasets (run once)
python scripts/setup_datasets.py
```

---

## Known Limitations & Blockers

| Issue | Owner | Status |
|-------|-------|--------|
| Classification trainer has no validation split (checkpoint on train loss); the Siamese trainer selects `best.pt` on validation EER via `val_subjects` | A/B/C | Partially addressed — classification protocol still needs agreeing |
| WISDM label encoding unverified | A | Blocked on Student A's preprocessing |
| FAR, FRR and EER all come from one interpolated crossing point, so `FAR == FRR == EER` | C | Reported as measured; not three independent metrics |
| Structured pruning only for DeepConvLSTM (B3) | B | Others fail closed (use unstructured) |
| No early stopping in either trainer (both run the configured epochs; `best.pt` is selected post hoc) | A/B/C | Deferred |
| Dataset not in repo (68 MiB LFS) | All | Run `git lfs pull` |

---

## Paper References (Student B Tracks)

| ID | Title | Venue | Year |
|----|-------|-------|------|
| B1 | IMU-based gait recognition using CNNs and multi-sensor fusion | Sensors | 2017 |
| B2 | METIER: Multi-task activity + user recognition | ACM IMWUT | 2020 |
| B3 | DeepConvLSTM for continuous authentication | Sensors | 2021 |
| B4 | Siamese denoising autoencoders for gait recognition | Neurocomputing | 2020 |
| B5 | Wearable auth with SE-DeepConvNet | Computers | 2024 |
| B6 | Real-time multi-person gait recognition at edge | Scientific Reports | 2025 |
| B7 | SEGait-ConViT: Vision Transformer for gait | Procedia CS | 2025 |

Common papers (C1–C5) cover surveys, edge computing, implicit authentication, and smartphone-based gait recognition.

---

## Getting Help

- **Slack/Teams**: Project channel (ask mentor to add you)
- **Issues**: Open GitHub issues in respective repos
- **Mentor**: Manjunath Hegde (MAHE) — `manjunath_hegde@manipal.edu`

---

## License

Academic research project. See individual paper licenses for reuse.

---

**Last Updated**: October 2026  
**Maintainer**: Aditya (Student B) — `VariableThe` on GitHub
