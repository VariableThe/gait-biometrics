# Reading Notes — Papers B1–B3 (Student B classification stack)

Scope: the three papers behind the core classification stack in
`models/cnn_baseline.py` (B1), `models/multitask.py` (B2) and
`models/cnn_lstm.py` (B3). None of these papers is available locally — only
the reading plan and an unrelated 2024 survey PDF are — so every section below
is derived from the paper title and the plan's annotation only.

## B1

**Title:** IMU-based gait recognition using convolutional neural networks and multi-sensor fusion
**Venue:** Dehzangi et al., 2017, Sensors (MDPI)
**Plan annotation:** "Foundational CNN approach for IMU-based gait recognition. Good starting architecture to understand."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper addresses gait recognition — identifying individuals from their
walking patterns — using inertial measurement unit (IMU) sensor data, with a
convolutional neural network that fuses information from multiple sensors.
This relates to our project because our classification stack consumes exactly
this kind of data: multi-channel IMU windows (9-channel HAR, 3-channel WISDM)
that must be mapped to per-window labels. The multi-sensor fusion idea is
directly reusable when several accelerometer/gyroscope channels must be
combined into one decision.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a CNN architecture in which features extracted
from multiple IMU sensors are fused. We adapt this as one 1D convolutional
branch per sensor channel (`_SensorBranch` in `models/cnn_baseline.py`), with
three selectable fusion strategies — concatenation, element-wise summation, and
a learned softmax attention weighting over branches — before the activity
classifier.

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/cnn_baseline.py` implements
the per-sensor-branch idea with configurable `fusion` ("concat" | "sum" |
"attention"), so the B1 concept can be recreated and ablated against the other
baselines on HAR/WISDM windows.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.

## B2

**Title:** METIER: A deep multi-task learning based activity and user recognition model using wearable sensors
**Venue:** Chen et al., 2020, ACM IMWUT
**Plan annotation:** "Multi-task learning combining CNN and LSTM. Shows how to extract robust features from sensor data."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper solves joint activity recognition and user (identity) recognition
from wearable-sensor data using a single deep multi-task model — i.e. one
network that predicts both *what* the user is doing and *who* the user is.
This maps directly onto our project: the HAR dataset ships both activity
labels and subject ids, so our multi-task stack can train one shared
representation against both targets instead of two separate models.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a shared CNN+LSTM trunk that extracts robust
temporal features from the sensor window, with task-specific heads for
activity and user prediction trained jointly under a multi-task objective. We
adapt this in `models/multitask.py` as a shared Conv1D+LSTM trunk with an
activity head and a subject head, plus `multitask_loss` (alpha-weighted sum of
the two cross-entropies). The subject head is only meaningful when subject
labels exist, so `engine/train.py` fails fast if it is used with a source that
has none (e.g. WISDM).

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/multitask.py` mirrors the
METIER design — shared CNN+LSTM features with two heads and a weighted
multi-task loss — and `configs/multitask.yaml` wires it to HAR with
`n_subjects: 21` (the 21 subjects of the HAR train split).

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.

## B3

**Title:** Deep Learning Approaches for Continuous Authentication Based on Activity Patterns Using Mobile Sensing
**Venue:** Various, 2021, Sensors (MDPI)
**Plan annotation:** "DeepConvLSTM architecture achieving strong continuous authentication results. Directly relevant to your model design."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper surveys deep-learning approaches to continuous authentication —
verifying a user's identity on an ongoing basis from their activity patterns
captured by mobile-sensing hardware — and presents the DeepConvLSTM
architecture as a strong performer. This relates to our project because our
primary classification baseline, DeepConvLSTM, follows the same design
principle: convolutional feature extraction followed by recurrent temporal
modeling of mobile/wearable sensor windows.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a DeepConvLSTM architecture — stacked
convolutional layers over the raw sensor time series feeding stacked LSTM
layers and a classifier. We adapt this in `models/cnn_lstm.py` as three
Conv1d+BatchNorm+ReLU blocks (each halving the time axis, so T=128 becomes 16
steps) followed by a 2-layer LSTM and a linear classifier, accepting
(N, C, 128) input directly.

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/cnn_lstm.py` is the primary
baseline of our stack and the first entry point for any later
attention/ViT/compression variants (B4–B7), since those all build on the same
conv-then-recurrent feature extractor contract.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.
