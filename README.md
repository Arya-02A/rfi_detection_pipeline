# AI-Powered RFI Detection with Analysis of Radio Astronomy Data

An explainable, visual pipeline for detecting and classifying Radio Frequency Interference (RFI) in GMRT radio astronomy data. The project combines FITS-data processing, robust statistics, feature engineering, a Random Forest classifier, and a Streamlit dashboard to turn complex radio-observation data into an interpretable workflow.

> **Project status:** research/educational prototype. The current implementation is designed for demonstration and preliminary data-quality analysis, not for production observatory operations.

## Overview

Radio telescopes measure extremely weak radio signals from astronomical sources. Human-made transmissions and electronic activity can contaminate those observations as **Radio Frequency Interference (RFI)**. This project explores a student-friendly pipeline that first detects suspicious high-power samples using robust statistics and then uses patch-level statistical features with a Random Forest model to categorize detected regions as:

- **0 — Clean**
- **1 — Narrowband**
- **2 — Broadband**

The system also reconstructs patch predictions into an overlay view and exposes feature importance to make the model easier to inspect.

The accompanying project report describes the same overall pipeline: FITS loading → power conversion → waterfall extraction → robust RFI masking → patch construction → feature extraction → Random Forest classification → overlay generation → analysis and visualization.

## What the system does

1. **Loads GMRT FITS data** and reads the visibility `DATA` field.
2. **Converts complex samples to power** using `Real² + Imag²` and selects a 2D time–frequency representation.
3. **Detects candidate RFI** with a Median Absolute Deviation (MAD)-based threshold.
4. **Builds an ML dataset** by slicing the waterfall into 64×64 patches with stride 32.
5. **Creates binary patch labels** from the fraction of masked pixels: RFI if the fraction exceeds 0.20, otherwise Clean.
6. **Extracts statistical features** such as mean, standard deviation, maximum, energy, skewness, kurtosis, time variance and frequency variance.
7. **Converts RFI patches to heuristic subclasses** using frequency-axis vs. time-axis variance.
8. **Trains a 100-tree Random Forest** inside a preprocessing pipeline.
9. **Maps patch predictions back** onto the waterfall grid for an interpretable overlay.
10. **Visualizes the pipeline** through a Streamlit dashboard, including an embedded FITS structure explorer.

## Project architecture

```text
rfi_detection_pipeline/
├── main.py                         # Main Streamlit dashboard
├── detection.py                    # Standalone robust RFI detector/demo
├── analysis.py                     # FITS structure and signal analysis helpers
├── dataset_builder.py              # Patch extraction, labeling and splitting
├── model_training.py
│                                   # Feature extraction, RF training, overlay map
├── dataset/
│   ├── .gitkeep                    # Keep folder in Git; do not commit large raw FITS files
│   └── README.md                   # Add dataset provenance/instructions here
├── dataset_ml/
│   ├── .gitkeep
│   └── README.md                   # Generated ML arrays live here locally
├── models/
│   ├── .gitkeep
│   └── README.md                   # Generated .pkl model lives here locally
├── docs/
│   └── THEORY.md                   # Theory and methodology
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_REPOSITORY_URL>
cd rfi_detection_pipeline
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Dataset setup

The application expects a GMRT FITS file containing a primary-HDU table with a `DATA` field compatible with the indexing used in the current code.

Because radio-astronomy FITS datasets can be very large, Keep large observation files outside the repository and document their provenance, access instructions, and expected filename/path in `dataset/README.md`.

For a reproducible public repository, add one of the following:

- a small, redistributable sample FITS file;
- a documented external-download workflow;
- or a synthetic/example dataset for demonstration.

## Running the dashboard

Update the FITS path in the sidebar, then run:

```bash
streamlit run main.py
```

The dashboard contains two major sections:

### RFI Pipeline

- raw GMRT waterfall
- binary RFI mask
- ML dataset builder
- Random Forest training
- feature-importance visualization
- AI classification overlay
- RFI distribution analysis
- scientific-yield style summary

### FITS Analysis

The FITS explorer reports file structure, parameter names, header metadata, and basic signal statistics. The analysis module samples records rather than loading every value for every plot.

## Methodology

### 1. Complex visibility to power

For a complex sample with real component `R` and imaginary component `I`, the project computes:

```text
P = R² + I²
```

The resulting 2D array is treated as a time × frequency waterfall.

### 2. Robust RFI detection

For each frequency channel, the detector computes:

```text
median = median(X)
MAD    = median(|X - median|)
σrobust = 1.4826 × MAD
threshold = median + k × σrobust
```

A sample is marked as RFI when its power exceeds the threshold. The dashboard exposes `k` as a sigma-threshold setting; the current default is 7.

### 3. Patch-based dataset generation

The waterfall is divided into overlapping 64×64 patches with stride 32. The patch-level binary label is determined from the RFI-mask density:

```text
RFI fraction > 0.20  → RFI (1)
otherwise             → Clean (0)
```

The current preprocessing then applies global z-score normalization and a stratified 80/20 train-test split.

### 4. Feature engineering

Each patch is summarized by eight features:

| Feature | Intuition |
|---|---|
| Mean | Overall signal level |
| Standard deviation | Spread/variability |
| Maximum | Strongest local peak |
| Energy | Aggregate squared magnitude |
| Skewness | Distribution asymmetry |
| Kurtosis | Peakiness/heavy-tail behavior |
| Time variance | Variation across the time axis |
| Frequency variance | Variation across the frequency axis |

### 5. Multiclass heuristic

Binary RFI patches are converted to three classes by comparing the two directional variances:

```text
Clean                         → 0
RFI and frequency variance > time variance → Narrowband (1)
RFI and otherwise                         → Broadband (2)
```

This is a **heuristic labeling rule**, not an independently ground-truthed physical annotation.

### 6. Random Forest

The classifier uses a `StandardScaler` followed by a `RandomForestClassifier` with 100 trees. Feature importance is exposed in the dashboard as a simple model-inspection mechanism.

### 7. Overlay reconstruction

Patch predictions are placed back on the original time-frequency grid. Overlapping patches are accumulated and averaged to form the overlay.

## Results documented by the project

The project report documents the following qualitative outputs:

- visible vertical narrowband carriers;
- visible horizontal broadband bursts;
- RFI-percentage estimation;
- patch-level classification accuracy;
- feature-importance visualization;
- AI overlay maps;
- scientific-yield style analysis;
- FITS-structure exploration.

The report's dashboard screenshots also show an example model-training accuracy of approximately **77.85%** and an example RFI analysis distribution of **54.7% Clean, 12.5% Narrowband, and 32.8% Broadband**. These should be treated as **example outputs from the reported run**, not as a general benchmark for the model.

## Important scientific / implementation caveats

This repository is best presented honestly as a **prototype**. Before calling the results a production-grade ML evaluation, several parts of the current implementation should be tightened:

### 1. The overlay currently needs correction

The dashboard predicts on the saved **training patches** and feeds those predictions to the full-grid overlay function. Because the training split changes patch order and contains only a subset of patches, this can misalign predictions with their original waterfall locations.

**Recommended fix:** reconstruct the complete deterministic patch list from the full waterfall in sliding-window order, extract features for those patches, predict all patches, and only then build the overlay.

### 2. Evaluation currently has data-leakage / split-definition issues

`dataset_builder.py` normalizes the complete patch array before the train/test split, and the training function then performs another internal split of the already-selected training data. Therefore, the accuracy shown by the current application is not a clean evaluation on the saved test set.

**Recommended fix:** split first, fit preprocessing on the training set only, evaluate once on the untouched test set, and report accuracy plus a confusion matrix and per-class precision/recall/F1.

### 3. Narrowband/Broadband labels are heuristic

The code derives Narrowband vs. Broadband from a comparison of frequency variance and time variance. This is useful for a prototype, but it should not be described as ground-truth source classification without labelled observations or expert annotation.

### 4. Terminology should be precise

The binary MAD mask is a **detector/flagging mask**, not a scientific ground-truth mask. Likewise, the percentage derived from model predictions is a **model-based patch distribution**, not automatically a measured physical percentage of contaminated telescope data.

### 5. The current project is not real-time

The report itself lists fixed patch size, manual sigma tuning, lack of multi-telescope validation, and non-real-time operation as limitations.

## Future work

The project report proposes several extensions:

- real-time RFI monitoring;
- software-defined-radio (SDR) integration;
- CNN-based segmentation;
- transfer learning across telescopes;
- pulsar-detection integration;
- conversational assistance for radio-data questions.

The strongest next technical step is to improve dataset provenance and evaluation first, then move toward multi-observation validation and a better-defined segmentation/classification model.

## Theory

See [`docs/THEORY.md`](docs/THEORY.md) for the project theory, equations, pipeline explanation, and interpretation guide.

## Authors

Arya Madiwale
Developed as a student research project focused on AI, radio astronomy and interpretable scientific data analysis.
