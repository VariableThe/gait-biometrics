# Reading Notes — Papers B4–B7 (Student B verification + efficiency stack)

Scope: the four papers behind the verification and efficiency stack in
`models/siamese.py` (B4), `models/attention.py` (B5), `models/compress.py`
(B6) and `models/vit.py` (B7). None of these papers is available locally —
only the reading plan and the paper titles/annotations are — so every
section below is derived from the paper title and the plan's annotation only.

## B4

**Title:** Siamese denoising autoencoders for joints trajectories reconstruction and robust gait recognition
**Venue:** Sheng, Li, 2020, Neurocomputing
**Plan annotation:** "Siamese network architecture for gait verification. Key reference for your embedding network design."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper addresses gait verification — confirming whether two gait
trajectories belong to the same person — using a siamese network over
joint-trajectory reconstructions, with a denoising autoencoder providing a
robust pretext representation. This relates to our project because our
verification stack needs an embedding network that maps sensor windows to
a metric space where same-subject pairs are closer than different-subject
pairs, and the HAR dataset ships exactly the subject labels needed to form
genuine/impostor pairs.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a siamese architecture with two
weight-sharing branches embedding gait trajectories, trained with a
triplet or contrastive objective, plus a denoising-autoencoder pretext head
that reconstructs clean trajectories from noisy inputs. We adapt this in
`models/siamese.py` as `EmbeddingNet` (a Conv1d feature extractor with a
projection head, L2-normalized output), `SiameseNet` (the siamese wrapper),
`triplet_loss` and `contrastive_loss`, and `TripletWindowSampler` (forms
anchor/positive/negative window triples from `WindowData`). The DAE pretext
head (`dae_decoder`) is included but kept OFF the default forward path.

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/siamese.py` implements the
siamese embedding design with triplet/contrastive losses and a window-level
triplet sampler; `configs/siamese.yaml` wires it to HAR (the only source
with subject labels) and selects the triplet objective.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.

## B5

**Title:** Wearable Sensor-Based Behavioral User Authentication Using a Hybrid Deep Learning Approach with Squeeze-and-Excitation Mechanism
**Venue:** Various, 2024, Computers (MDPI)
**Plan annotation:** "SE-DeepConvNet for continuous authentication. Shows attention mechanisms for channel recalibration."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper addresses behavioral user authentication — verifying a user's
identity from wearable-sensor data — using a hybrid deep learning approach
with a squeeze-and-excitation mechanism for channel recalibration. This
relates to our project because our classification stack must extract
discriminative features from multi-channel IMU windows, and SE blocks provide
a lightweight way to weight channels by importance, improving the
feature extractor without a large parameter increase.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a deep convolutional network in which each
convolutional block is followed by an SE block that squeeze-excites channels
— global-average-pooling the time axis, passing the channel descriptor
through a bottleneck MLP, and rescaling channels by a sigmoid gate. We adapt
this in `models/attention.py` as `SEBlock`, `DepthwiseSeparableConv1d`, and
`SEConvNet` (three conv+SE blocks with max-pooling, then a linear
classifier). The depthwise-separable option keeps the parameter count under
the 158676 cap (1.2 x `MultiSensorFusionCNN`'s 132230).

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/attention.py` implements
the SE-recalibrated deep convnet; `configs/se_attention.yaml` wires it to HAR
with depthwise-separable convolutions and a reduction of 16.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.

## B6

**Title:** Real-time multiple people gait recognition in the edge
**Venue:** Various, 2025, Scientific Reports
**Plan annotation:** "Model compression, pruning, and quantization for edge deployment. Essential for your lightweight model goal."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper addresses real-time gait recognition for multiple people on edge
devices, where compute and memory are constrained. The core problem is making
a gait-recognition model small and fast enough for edge deployment without
sacrificing too much accuracy. This relates to our project because our
lightweight model goal requires compressing the CNN+LSTM baseline — pruning
redundant channels/weights and quantizing to int8 — so it can run on
resource-constrained hardware.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: model compression via pruning (removing
redundant channels or weights) and quantization (reducing numerical
precision), targeting edge deployment. We adapt this in `models/compress.py`
as `count_parameters`, `count_flops`, `apply_pruning` (structured channel
removal by L1 norm and unstructured L1 pruning), `dynamic_quantize`
(int8 dynamic quantization of Linear/LSTM layers), `export_onnx`, and
`flops_report`. `configs/lightweight.yaml` combines the CNN+LSTM baseline
with `prune: 0.3` and `quantize: true`.

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/compress.py` provides the
pruning/quantization/export pipeline for the edge goal;
`configs/lightweight.yaml` wires the CNN+LSTM baseline to 30% structured
pruning plus dynamic quantization.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.

## B7

**Title:** Efficient Gait Recognition with SEGait-ConViT: A Vision Transformer Model
**Venue:** Various, 2025, Procedia CS
**Plan annotation:** "Vision Transformer approach to gait recognition. Understand if Transformer elements can improve your model."

**Problem Statement: What problem does the paper solve? How does it relate to our project?**
The paper addresses gait recognition with a vision-transformer model
(SEGait-ConViT), exploring whether transformer elements — self-attention
over patch tokens — can improve recognition efficiency and accuracy over
CNN-based approaches. This relates to our project because we want to know
whether a transformer over 1D sensor patches can beat our CNN/LSTM and
SE-DeepConvNet baselines on HAR/WISDM windows, and at what parameter cost.

**Methodology: What is the technical approach? Can we reuse or adapt any components?**
From the title and annotation: a ViT that splits the input into patches,
embeds each patch, prepends a CLS token with learned positional embeddings,
processes the sequence with pre-LN transformer blocks, and classifies from
the pooled token representation. We adapt this in `models/vit.py` as
`PatchEmbed1d` (Conv1d with kernel=stride=patch_len), `ConViTBlock` (pre-LN
MHA + MLP), and `GaitViT` (8 patches from T=128 at patch_len=16, CLS token,
learned positional embeddings, 4 pre-LN blocks, mean-pooled head).
Pre-LN blocks train stably at lr 0.001 without warm-up.

**Datasets: What datasets were used? Are they available? What preprocessing was done?**
Not determinable from available materials.

**Results: What accuracy/EER was achieved? What are the baseline comparisons?**
Not determinable from available materials.

**Limitations: What are the acknowledged weaknesses?**
Not determinable from available materials.

**Code Availability: Is there a GitHub repository?**
Not determinable from available materials.

**What this implies for our model code:** `models/vit.py` implements the
1D ViT classifier; `configs/vit.yaml` wires it to HAR with patch_len=16,
embed_dim=128, depth=4, and 4 heads.

Status: UNVERIFIED — paper not available locally; derived from title and the plan's annotation only.
