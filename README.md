# Ocular Disease Recognition — Multi-Label Fundus Image Classifier

---

## Overview

This project implements an end-to-end deep learning pipeline for automated 
detection of ocular diseases from retinal fundus images. The system performs 
**multi-label classification** across 8 diagnostic categories using the 
ODIR-5K dataset, and is designed with clinical applicability in mind — 
reporting sensitivity, specificity, and per-class AUC alongside standard 
accuracy metrics.

---

## Disease Categories

| Code | Disease |
|------|---------|
| N | Normal fundus |
| D | Diabetes (diabetic retinopathy) |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration (AMD) |
| H | Hypertension |
| M | Myopia |
| O | Other pathologies |

---

## Dataset

**ODIR-5K** — Ocular Disease Intelligent Recognition  
Source: [Kaggle — andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

- 5,000 patients, 7,000 training images (left and right fundus per patient)
- Significant class imbalance: Normal and Diabetes ~2,100 cases vs Hypertension ~200
- Labels assigned at patient level; both eyes share the same label

---

## Pipeline
```
Data loading & EDA
        ↓
Patient-level 5-fold split (prevents data leakage)
        ↓
TFRecord serialisation
        ↓
Preprocessing + augmentation (flip, brightness, contrast, saturation)
        ↓
Transfer Learning — EfficientNetV2B2 (frozen backbone)
        ↓
Fine-tuning — top 50% layers unfrozen
        ↓
Threshold optimisation on validation set (F1 and recall variants)
        ↓
Evaluation — AUC, F1, sensitivity, specificity, confusion matrices
```

---

## Model Architecture

- **Backbone:** EfficientNetV2B2 (pretrained on ImageNet)
- **Head:** GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(8, sigmoid)
- **Loss:** Weighted Binary Cross-Entropy (pos_weight clipped to [1, 10])
- **Activation:** Sigmoid (multi-label)
- **Input:** 224×224 RGB, pixel range [0, 255] (EfficientNet internal rescaling)

### Training strategy

| Phase | LR | Loss | Monitor |
|-------|----|------|---------|
| Transfer Learning | 1e-4 | BinaryCrossentropy | val_loss |
| Fine-tuning | 5e-5 | Weighted BCE | val_recall |

---

## Results

| Metric | Value |
|--------|-------|
| Test AUC (macro, multi-label) | 0.8112 |
| Test Accuracy | 0.8778 |
| Test Recall (sensitivity) | 0.782 |
| Test Precision | 0.309 |
| Macro F1 | 0.50 |
| Micro F1 | 0.53 |

### Per-class AUC

| Class | AUC |
|-------|-----|
| Myopia | 0.985 |
| Cataract | 0.950 |
| AMD | 0.897 |
| Glaucoma | 0.900 |
| Hypertension | 0.783 |
| Normal | 0.722 |
| Diabetes | 0.711 |
| Other | 0.568 |

---

## Requirements
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow kaggle
```

---

## Usage

1. Set up Kaggle credentials and download the dataset
2. Run cells sequentially in `ocular_disease.ipynb`
3. TFRecords are cached to avoid reprocessing on re-runs
4. Best models are saved as `.keras` checkpoints during training

---

## Project Structure
```
├── ocular_disease.ipynb
├── data/
│   ├── dataset_remapped.csv
│   ├── labels.txt
│   └── plots/
│       ├── class_distribution.png
│       ├── confussion_matrices.png
│       ├── fine_tubed_plots.png
│       ├── sample.png
│       └── transfer_learning_plots.png
└── README.md
```

---

## Course
Design of AI Systems  
**Authors:** Josue Valenzuela Perez · Ludwig Alexandersson
