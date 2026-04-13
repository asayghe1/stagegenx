#!/bin/bash
set -e
DATA_DIR="${1:-data/processed/}"
OUT_DIR="${2:-checkpoints/}"
echo "=== Preprocessing ==="
python src/preprocessing_a12.py \
    --swat_a12    data/SWaT_A12_normal.csv \
    --swat_a1_atk data/SWaT_A1_attack.csv \
    --wadi_co     data/WaDi_A3_crossover.csv \
    --output_dir  "$DATA_DIR"
echo "=== Training ==="
python src/stagegenx.py \
    --mode train_all \
    --data_dir  "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --config configs/default.yaml
