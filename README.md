# OAG-CAE
**Orthogonal Autoencoder for Interpretable Brain-Age Prediction from Functional Connectivity**

This repository implements a **two-stage, interpretable brain-age prediction framework** based on an **Orthogonal Autoencoder (OAG-CAE)** trained on functional connectivity (FC) matrices, with **post-hoc explainability via PCA + KernelSHAP and back-projection to FC space**.

The code is designed for **neuroimaging research** on:

+ Normal aging
+ Latent disentanglement (age-related vs. nuisance representations)  
潜在纠缠解缚（年龄相关表征与有害表征）
+ Cross-validated brain-age modeling
+ Multi-level interpretability (latent → PCA → FC edges)

## 1. Overview of the Framework
### Core idea
The pipeline explicitly separates **age-related information** from other latent factors using an **orthogonality constraint**, and then explains model predictions at the **functional connectivity level**.

```bash
FC (278×278)
   ↓
Orthogonal Autoencoder
   ├── z_age   (age-related latent)
   └── z_noise (non-age latent)
        ↓
Age Regressor
        ↓
Predicted Age

```

 For interpretability, predictions are explained via:  

```bash
FC → PCA → KernelSHAP → inverse PCA → FC-level importance map

```

 2. Project Structure  

```bash
OAG-CAE/
│
├── scripts/
│   ├── main.py
│   │   Entry point for training with cross-validation
│   │
│   └── explain_age2latentpca2fc_shap.py
│       PCA → KernelSHAP → FC-level explanation
│
├── src/
│   ├── models/
│   │   ├── OAG_CAE.py
│   │   │   Orthogonal Autoencoder (encoder + decoder + age heads)
│   │   │
│   │   └── regressors.py
│   │       Attention-based and convolutional age regressors
│   │
│   ├── training/
│   │   ├── train_pipeline.py
│   │   │   Cross-validation orchestration
│   │   ├── stage1_train.py
│   │   │   Stage 1: autoencoder + orthogonality + age supervision
│   │   ├── stage2_train.py
│   │   │   Stage 2: latent → age regression
│   │   ├── losses.py
│   │   │   Orthogonal-guided composite loss
│   │   ├── metrics.py
│   │   │   MAE / R² metrics
│   │   ├── io_training.py
│   │   │   Training log I/O
│   │   └── utils.py
│   │   │   Seeding, age grouping, utilities
│   │   └── datasetFC.py
│   │       FC datasets, masking, augmentation, label handling
│   │
│   └── explain/
│       ├── model_adapters.py
│       │   SHAP-compatible model adapters
│       ├── pca_shap.py
│       │   PCA + KernelSHAP logic
│       └── io.py
│           Explanation result saving utilities
│
└── BN278_FC/
    ├── BN278_FC_1.npy
    ├── BN278_FC_2.npy
    ├── ...
    └── label*.npy

```

## 3. Data Format
### Functional Connectivity (FC)
+ File format: `.npy`
+ Shape per subject:

```bash
(278, 278)

```

+  Dataset shape:  

```bash
(N, 278, 278)

```

+  Only one triangle (upper by default) is kept internally; the rest is masked.  

### Labels
+ File format: `.npy`
+ Shape:

```bash
(N,)

```

+ Units:
    - Default: **months**, automatically converted to years (`/12`)
    - Configurable via `LabelConfig`

## 4. Training Pipeline
### Stage 1 — Orthogonal Autoencoder
**Objective**  
Learn disentangled latent representations:

+ `z_age`: age-related information
+ `z_noise`: nuisance / non-age information

**Loss components**

+ Masked FC reconstruction loss
+ Age regression loss (Huber)
+ Age-group classification loss
+ Orthogonality loss between `z_age` and `z_noise`

Implemented in:

+ `src/models/OAG_CAE.py`
+ `src/training/losses.py`
+ `src/training/stage1_train.py`

---

### Stage 2 — Age Regression Refinement
+ Encoder is fixed
+ A convolutional regressor predicts age from `z_age`

Implemented in:

+ `src/models/regressors.py`
+ `src/training/stage2_train.py`

---

### Cross-Validation
+ Fold-based CV using group-wise splits
+ Normalization parameters are computed **only on training data**
+ Implemented in `train_pipeline.py`

---

## 5. Running Training
### Example
```bash
python scripts/main.py \
  --input_dir ./BN278_FC \
  --out_dir ./outputs \
  --device cuda \
  --seed 2574

```

 Outputs (example):  

```bash
outputs/
├── fold1_summary.json
├── fold2_summary.json
├── ...
└── cv_summary.json

```

## 6. Explainability: PCA → KernelSHAP → FC
### Motivation
Direct SHAP on FC matrices (278×278) is infeasible.  
We therefore perform explainability in a reduced PCA space and project explanations back to FC.

### Script
```bash
python scripts/explain_age2latentpca2fc_shap.py \
  --fc_all all_folds_fc_combined.npy \
  --fold_sizes 117 120 120 119 123 \
  --out_root ./shap_results \
  --encoder_template ./outputs/fold{fold}/best_encoder.pth \
  --regressor_template ./outputs/fold{fold}/best_regressor.pth

```

python scripts/explain_age2latentpca2fc_shap.py \

  --fc_all all_folds_fc_combined.npy \

  --fold_sizes 117 120 120 119 123 \

  --out_root ./shap_results \

  --encoder_template ./outputs/fold{fold}/best_encoder.pth \

  --regressor_template ./outputs/fold{fold}/best_regressor.pth

## 7. Environment Requirements
### Recommended (Conda)
```bash
name: oag-cae
channels:
  - pytorch
  - nvidia
  - conda-forge

dependencies:
  # --- core ---
  - python=3.9

  # --- pytorch ---
  - pytorch=2.2.*
  - torchvision=0.17.*
  - torchaudio=2.2.*
  - pytorch-cuda=11.8


  # --- scientific stack ---
  - numpy<2.0
  - scipy<1.12
  - scikit-learn>=1.2,<1.7
  - pandas>=2.0
  - matplotlib>=3.7

  # --- explainability ---
  - shap>=0.44

  # --- neuroimaging utils ---
  - nibabel
  - nilearn
  - networkx

  # --- misc ---
  - sympy
  - joblib
  - pyyaml
  - tqdm

  # --- pip-only packages ---
  - pip
  - pip:
      - monai==1.3.0
      - nipype==1.8.6

```
## Saved latent representations

After training, the encoder-generated latent spaces are saved for each fold:

outputs/
├── fold1/
│   ├── latent_train.npz
│   ├── latent_train.csv
│   ├── latent_val.npz
│   ├── latent_val.csv
│   ├── latent_test.npz
│   └── latent_test.csv

Each file contains:
- z_age: age-related latent representation
- z_noise: residual / nuisance latent representation
- age_true: chronological age
- age_pred: predicted age from the Stage 2 regressor
## Computational requirements and runtime report

For each fold, the pipeline saves:

outputs/fold*/runtime_report.json

The report includes:
- CPU/GPU information
- PyTorch/CUDA version
- number of model parameters
- train/validation/test sample sizes
- Stage 1 runtime
- Stage 2 runtime
- total fold runtime
- peak GPU memory usage
### Important Notes
+ **Do not use NumPy ≥ 2.0** (ABI incompatibility with PyTorch/SciPy)
+ If CUDA issues occur, start with CPU-only installation
+ `sympy` may be required for some PyTorch builds

---

## 8. Design Principles
+ **Explicit stage separation** → easier debugging and ablation
+ **Orthogonality constraint** → interpretable latent structure
+ **Adapter-based explainability** → SHAP compatibility without modifying core models
+ **Config-driven training** → reproducibility

This is **research-grade code**, not a minimal demo.

---

## 9. Intended Extensions
+ Normal aging vs. neurodegenerative disease comparison
+ Multimodal inputs (FC + VBM)
+ Latent-wise SHAP / LRP
+ Graph-aware reconstruction losses

---

## 10. License & Citation
Code is provided for research use.  
If you use this framework, please cite the relevant methodological literature.  
(A paper describing this framework is in preparation.)

