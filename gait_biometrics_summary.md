> Historical project document. Current source contracts, validation behavior, measured results and limits are in [README.md](README.md) and [23 October evidence](reports/2026-10-23/README.md). Earlier results/protocols are retained here for provenance and are not directly comparable to the new walking-only runs.

### Summary of the Research Paper
**"Journey into gait biometrics: Integrating deep learning for enhanced pattern recognition" (2024)**

This paper is a comprehensive survey that explores the evolution of gait recognition through Deep Learning (DL) pipelines. It serves as a technical roadmap for researchers, detailing every stage of a robust DL system for gait analysis:
*   **Motivation:** Gait is highlighted as a superior biometric for surveillance because it is non-intrusive, works at a distance, and is difficult to conceal compared to face or fingerprint biometrics.
*   **Pipelines:** It breaks down the architecture of gait recognition into datasets (e.g., CASIA, OU-ISIR), feature extraction (CNNs, Encoders), feature reduction (PCA, Autoencoders), and classification (SVM, Softmax).
*   **Key Findings:** The study emphasizes that **Adam** is the most preferred optimizer, **ReLU** is the dominant activation function, and **Dropout** is critical for regularization to prevent overfitting across diverse datasets. It concludes that while vision-based gait recognition is mature, real-time edge computing and sensor-based (wearable/smartphone) gait recognition are the next frontiers.

---

### Expansion on CNN and LSTM Theory in Gait Recognition

In the context of behavioral biometrics like gait, CNNs and LSTMs serve as the "eyes" and the "memory" of the system, respectively.

#### 1. Convolutional Neural Networks (CNN): The Spatial Feature Extractors
CNNs are designed to recognize patterns in space. In gait recognition, they translate visual or signal data into a high-dimensional feature map.
*   **Theory:** Through layers of kernels (filters), CNNs perform "convolutions" that detect edges, then shapes, and finally complex structures like the silhouette of a leg or the tilt of a torso.
*   **Application in Gait:**
    *   **Silhouette Analysis:** CNNs process Gait Energy Images (GEI)—a single image representing a full walking cycle—to extract the unique "shape" of a person's stride.
    *   **Automatic Feature Engineering:** Unlike traditional methods where researchers had to manually define "leg length" or "step width," a CNN learns which pixels are most discriminative for identity.
    *   **3D-CNNs:** By adding a third dimension (time), CNNs can process several frames at once, capturing short-term motion "blur" that represents a person's unique movement speed.

#### 2. Long Short-Term Memory (LSTM): The Temporal Dynamics
Gait is not a static image; it is a periodic sequence of movements. LSTMs are a specialized type of Recurrent Neural Network (RNN) designed to remember information over long durations.
*   **Theory:** LSTMs solve the "vanishing gradient" problem (where neural networks forget the beginning of a sequence). They use a **Cell State** (a long-term memory track) and **Gates** (input, forget, and output) to decide what movement data is important to keep and what to discard.
*   **Application in Gait:**
    *   **Sequence Modeling:** In sensor-based biometrics (like smartphone accelerometers), LSTMs analyze the *rhythm* of the walk. They can identify the subtle delay in a person's left-foot strike vs. their right.
    *   **CNN-LSTM Hybrid (Spatial-Temporal):** This is the "gold standard" for gait. A CNN extracts the features of each frame (spatial), and the LSTM tracks how those features change over 30–60 frames (temporal).
    *   **Behavioral Biometrics:** LSTMs are particularly good at catching "idiosyncrasies"—the tiny, repetitive hitches or swings in a walk that are unique to an individual's skeletal structure and habit, making them nearly impossible to spoof.

### Application in Behavioral Biometrics
Combining these two allows for **Robust Identification**:
*   **CNNs** handle the **Physical Biometric** (What you look like while walking).
*   **LSTMs** handle the **Behavioral Biometric** (How you move over time).

This dual approach allows gait recognition to remain accurate even if a subject changes clothes (changing the CNN's "shape" input) because the LSTM still recognizes the underlying "rhythm" of the motion.
