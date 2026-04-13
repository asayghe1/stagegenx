# StageGenX: Process-Stage-Aware Generative Data Augmentation for Cyber-Physical Attack Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIFS%20Submission-blue)](https://github.com/asayghe1/stagegenx)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-iTrust%20SWaT%2FWaDi-orange)](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

Official implementation of **StageGenX**, a stage-conditioned generative framework for cyber-physical attack data augmentation in Industrial Control Systems (ICS).

---

## Overview

Existing anomaly detection frameworks for ICS (MAD-GAN, USAD, LSTM-VAE, TranAD, CAE-T) treat multivariate sensor data as a flat feature vector, ignoring the physically distinct process stages of multi-stage systems like SWaT.

**StageGenX** addresses this with four core contributions:

1. **FiLM conditioning** — Feature-wise Linear Modulation injects process-stage embeddings as multiplicative scale and additive shift transformations at *every* LSTM hidden state
2. **Physics Consistency Loss** — 10 verified SWaT process invariants enforced during generation (92.9% mean constraint satisfaction, +29.2 pp over unconditioned baseline)
3. **CrossOver Normal Model** — First use of the 100-hour coupled SWaT–WaDi CrossOver dataset for ML training
4. **Formal theoretical framework** — Three propositions proving stage conditioning is *necessary* when I(X;S) > 0

![StageGenX Architecture](assets/fig1_stagegenx_architecture.svg)

---

## Results

| Method | F1 | AUC-ROC | FPR |
|--------|-----|---------|-----|
| MAD-GAN | 0.783 | 0.871 | 0.182 |
| USAD | 0.846 | 0.912 | 0.126 |
| CAE-T (best baseline) | 0.878 | 0.936 | 0.097 |
| **StageGenX (ours)** | **0.912** | **0.961** | **0.069** |

All improvements over CAE-T are statistically significant (p < 0.01, Wilcoxon signed-rank, Bonferroni-corrected).

**Per-stage gains** — Stage-5 reverse osmosis attacks yield the largest improvement (+13.1 F1 points), consistent with the high physical specificity of RO dynamics.

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| Python  | ≥ 3.9   |
| PyTorch | ≥ 2.0   |
| numpy   | ≥ 1.24  |
| pandas  | ≥ 2.0   |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| scipy   | ≥ 1.11  |
| tqdm    | ≥ 4.65  |

---

## Dataset Access

StageGenX uses the **iTrust SWaT and WaDi** datasets from the Singapore University of Technology and Design (SUTD).

> These datasets are **not publicly available** without a data-sharing agreement.  
> Request access at: https://itrust.sutd.edu.sg/itrust-labs_datasets/

**Required datasets:**
- `SWaT.A1_Dec_2015` — 51 features, 41 labeled attacks
- `SWaT.A12_Mar_2026` — 86 features, stage annotations (P1_STATE–P6_STATE)
- `WaDi.A3_Dec_2023` — 127 features, 100-hour CrossOver mode

Once approved, place the CSV files in:
```
data/
├── SWaT_A1_normal.csv
├── SWaT_A1_attack.csv
├── SWaT_A12_normal.csv
└── WaDi_A3_crossover.csv
```

---

## Project Structure

```
stagegenx/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/                        # Place dataset CSVs here (not tracked by git)
│   └── .gitkeep
│
├── src/
│   ├── preprocessing.py         # Data cleaning, scaling, windowing
│   ├── preprocessing_a12.py     # SWaT.A12-specific loader and schema alignment
│   ├── stagegenx.py             # Full model: SCVAE, SCWGAN-GP, CO-VAE, IDS
│   └── utils.py                 # MMD, TSTR, silhouette, plotting helpers
│
├── configs/
│   └── default.yaml             # All hyperparameters
│
├── scripts/
│   ├── train.sh                 # Full training pipeline
│   └── evaluate.sh              # Evaluation against baselines
│
├── assets/                      # Figures for README
│   ├── fig1_architecture.png
│   ├── fig4_film_mechanism.png
│   └── fig6_scada_deployment.png
│
└── paper/                       # LaTeX source
    ├── stagegenx_final_v2.tex
    └── stagegenx_final.bib
```

---

## Quickstart

### 1. Preprocessing

```bash
python src/preprocessing_a12.py \
    --swat_a12 data/SWaT_A12_normal.csv \
    --swat_a1_atk data/SWaT_A1_attack.csv \
    --wadi_co data/WaDi_A3_crossover.csv \
    --output_dir data/processed/
```

### 2. Train StageGenX

```bash
python src/stagegenx.py \
    --mode train_all \
    --data_dir data/processed/ \
    --output_dir checkpoints/ \
    --epochs_scvae 200 \
    --epochs_scwgan 300 \
    --lambda_phys 0.1 \
    --beta 4 \
    --latent_dim 32 \
    --window_size 60
```

### 3. Evaluate

```bash
python src/stagegenx.py \
    --mode eval \
    --checkpoint checkpoints/stagegenx_best.pt \
    --test_data data/processed/test.pt
```

### 4. Run Ablation

```bash
python src/stagegenx.py \
    --mode ablation \
    --data_dir data/processed/
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window W | 60 s | Sliding window length |
| Latent dim d_z | 32 | VAE latent dimension |
| Stage emb dim d_e | 16 | Stage embedding size |
| LSTM hidden H | 128 | Hidden dimension |
| β (KL weight) | 4 | β-VAE disentanglement |
| λ_phys | 0.1 | Physics loss weight |
| Critic steps n_c | 5 | WGAN-GP critic steps |
| GP penalty λ | 10 | Gradient penalty weight |

---

## Physics Invariants

StageGenX enforces 10 verified SWaT process invariants during generation:

| k | Invariant | Stage |
|---|-----------|-------|
| c1 | FIT101 ≈ 0 when P101 OFF | P1 |
| c8 | PIT501 ≥ p_min^RO when P501 ON | P5 |
| c9 | FIT501 ≈ FIT502 (permeate balance) | P5 |
| ... | (full list in paper Appendix D) | — |

Constraint satisfaction: **92.9%** (StageGenX) vs **63.7%** (unconditioned), **+29.2 pp**.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@article{sayghe2026stagegenx,
  title     = {{StageGenX}: Process-Stage-Aware Generative Data Augmentation 
               for Cyber-Physical Attack Detection in Water Critical Infrastructure},
  author    = {Sayghe, Adel and {co-authors}},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

Datasets provided by **iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)**.

Required credit: *"iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design."*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
