#!/bin/bash
set -e
CKPT="${1:-checkpoints/stagegenx_best.pt}"
DATA_DIR="${2:-data/processed/}"
python src/stagegenx.py \
    --mode eval \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --config configs/default.yaml
