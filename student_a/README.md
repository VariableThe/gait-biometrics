# Inertial Sensor Data Pipeline

This project prepares inertial-sensor datasets for gait recognition, activity recognition, and authentication experiments. The implementation is contained in [`gait_pipeline.py`](gait_pipeline.py).

The pipeline was designed from the preprocessing methods described in the supplied papers. It is intended to produce clean, reproducible model inputs rather than reproduce any paper's complete neural-network architecture or reported results.

## Included datasets

The `Dataset/` directory contains:

- `WISDM_Dataset.zip`
- `human+activity+recognition+using+smartphones.zip`, containing the UCI HAR dataset

The source archives are read directly. They are not modified or extracted permanently by the script.

## What the pipeline does

### WISDM

For each WISDM train and test split, the script:

1. Reads the three accelerometer axes: `acc_x`, `acc_y`, and `acc_z`.
2. Uses timestamps to resample the signal to a regular 50 Hz grid.
3. Applies a low-pass FIR filter with an 8 Hz cutoff when SciPy is available.
4. Creates sliding windows of 90 samples, corresponding to 1.8 seconds at 50 Hz.
5. Uses a 50% overlap, so the step size is 45 samples.
6. Assigns each window the majority activity label.
7. Standardizes the windows using statistics calculated from the training split only.
8. Saves the processed arrays as compressed NumPy files.

The 50 Hz resampling, timestamp handling, filtering, and fixed-window approach follow the general procedures described in the supplied gait-preprocessing papers. The 1.8-second window and 50% overlap are practical defaults for the available WISDM data.

### UCI HAR

UCI HAR is already distributed as fixed samples with labels and subject identifiers. The script therefore preserves its provided representation and exports:

- training features and labels;
- test features and labels;
- subject identifiers for subject-aware evaluation.

The UCI HAR feature arrays have shape `(samples, 561)` because they are the dataset's existing engineered feature representation, not raw 3-axis windows.

## Running the pipeline

From the project directory, run:

```powershell
python gait_pipeline.py --out processed
```

The default configuration is:

```text
Sampling rate: 50 Hz
Window length: 90 samples / 1.8 seconds
Window step: 45 samples / 50% overlap
Low-pass cutoff: 8 Hz
Output directory: processed/
```

The main parameters can be changed from the command line:

```powershell
python gait_pipeline.py `
  --out processed_64 `
  --hz 50 `
  --window 64 `
  --step 32 `
  --cutoff 8
```

The 64-sample configuration corresponds to the 1.28-second window used in the supplied AE-GDI work.

## Output files

Each `.npz` file contains:

- `X`: input samples;
- `y`: integer labels;
- `subject`: source split or subject identifier.

The generated files are:

```text
processed/
├── wisdm_train.npz
├── wisdm_test.npz
├── wisdm_train.json
├── wisdm_test.json
├── wisdm_train_normalization.npz
├── wisdm_test_normalization.npz
├── uci_har_train.npz
├── uci_har_test.npz
├── uci_har_train.json
└── uci_har_test.json
```

For WISDM, `X` has shape `(number_of_windows, window_length, 3)`. With the default settings, the generated shapes are approximately:

```text
wisdm_train: (232662, 90, 3)
wisdm_test:  (232237, 90, 3)
```

The exact counts can change if the source archives or command-line parameters change.

## Loading the data

```python
import numpy as np

train = np.load("processed/wisdm_train.npz")
X_train = train["X"]
y_train = train["y"]

test = np.load("processed/wisdm_test.npz")
X_test = test["X"]
y_test = test["y"]
```

## Important limitations

- The WISDM archives do not expose a subject identifier in the columns used by this pipeline. The `subject` field therefore records `train` or `test`, not individual people. Do not use it as a person-level identity label.
- The WISDM pipeline operates on accelerometer channels only because those are the channels present in the supplied CSV files. It does not invent gyroscope data.
- The UCI HAR export preserves the dataset's 561 engineered features. It does not reconstruct the original six raw inertial channels.
- The supplied papers use different datasets, window definitions, and experimental protocols. The code captures reusable preprocessing ideas, but it should not be interpreted as an exact reproduction of every paper.
- For model evaluation, keep the official train/test split and avoid fitting normalization statistics on the test set.

## Dependencies

The script requires Python with:

```text
numpy
pandas
scipy (recommended for FIR filtering)
```

Install them with:

```powershell
python -m pip install numpy pandas scipy
```

If SciPy is unavailable, the script still performs timestamp-based linear resampling but skips the FIR filtering step.

