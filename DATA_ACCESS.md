# Dataset Access

## Why the data is not included in this repository

The SWaT and WaDi datasets are provided by **iTrust, Centre for Research in Cyber Security,
Singapore University of Technology and Design (SUTD)** under a data-sharing agreement.
They may **not** be redistributed publicly. Uploading them to GitHub would violate the
iTrust terms of use.

## How to request access

1. Go to: **https://itrust.sutd.edu.sg/itrust-labs_datasets/**
2. Click **"Request dataset"**
3. Fill in the form — approval typically takes 3–5 business days
4. You will receive download links for the specific datasets you request

## Datasets used in this paper

| Dataset | Version | Features | Notes |
|---------|---------|----------|-------|
| SWaT | A1 (Dec 2015) | 51 | 41 labeled attacks — used for attack window training |
| SWaT | A12 (Mar 2026) | 86 | Stage annotations (P\_STATE) — used for normal windows |
| WaDi | A3 (Dec 2023) | 127 | 100-hour CrossOver mode — used for CO-VAE training |
| WaDi | A1 (test) | 127 | Used for cross-dataset evaluation only |

The file `data/WaDi_A3_dataset_info.txt` contains the official iTrust characterisation
of the WaDi.A3 dataset included with permission.

## Required filenames

Once approved, rename your downloaded CSV files to match these names and place them
in the `data/` folder:

```
data/
├── SWaT_A1_normal.csv          # SWaT.A1 normal operation (7 days)
├── SWaT_A1_attack.csv          # SWaT.A1 attack period (4 days, 41 attacks)
├── SWaT_A12_normal.csv         # SWaT.A12 normal operation (includes P_STATE columns)
└── WaDi_A3_crossover.csv       # WaDi.A3 CrossOver-mode run (100 hours)
```

## SWaT.A12 schema

SWaT.A12 (March 2026) introduced four schema changes over A1:

| Column type | Suffix | Example | Notes |
|------------|--------|---------|-------|
| Continuous sensor | `.Pv` | `LIT101.Pv` | Float, MinMax-scaled |
| Actuator status | `.Status` | `P101.Status` | Integer 0/1 |
| Alarm channel | `.Alarm` | `P101.Alarm` | String: Inactive/Active/Bad Input |
| Stage label | `P_STATE` | `P1_STATE` | Integer 1–6 per second |

The preprocessing pipeline in `src/preprocessing_a12.py` handles all schema alignment
between A1 and A12 automatically via the 31 shared sensor IDs.
