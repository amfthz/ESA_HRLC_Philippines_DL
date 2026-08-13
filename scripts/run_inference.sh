#!/bin/bash

# =============================================================
# RUN DL INFERENCE FOR AMAZON TILES (2019)
# =============================================================
#
# This script launches the ESA CCI SAR DL inference pipeline
# for a list of tiles.
#
# IMPORTANT:
# The pipeline now automatically:
#   1) fills missing SAR features (up to 28 expected features)
#   2) creates a water map if missing
#   3) detects water type (seasonal vs non-seasonal)
#   4) generates the merge_config JSON
#   5) runs the merging stage
#   6) sliding window parameters (patch size / stride / batch size) can be overridden here
#
# Therefore this script ONLY needs to call src.infer once per tile.
#
# -------------------------------------------------------------
# FIRST TIME ONLY
# -------------------------------------------------------------
# Make the script executable:
#   chmod +x scripts/run_inference_2019_amazon.sh
#
# -------------------------------------------------------------
# HOW TO RUN
# -------------------------------------------------------------
# From the project root (ESA_CCI_UNIPV):
#   bash scripts/run_inference_2019_amazon.sh
#
# -------------------------------------------------------------
# EXPECTED INPUT STRUCTURE
# -------------------------------------------------------------
# /tiles_io_egeos/input/static/2019/Amazon/<TILE>/S1
#
# ├── features
# │     ├── *.tif  (can be < 28; missing ones are auto-generated)
# │
# └── water_map
#       └── *.tif  (optional; created automatically if missing)
#
# =============================================================

source ~/miniconda3/etc/profile.d/conda.sh

# Automatically select conda environment (torch or torch-gpu)
if conda env list | grep -q "^torch-gpu "; then
    echo "Activating environment: torch-gpu"
    conda activate torch-gpu
elif conda env list | grep -q "^torch "; then
    echo "Activating environment: torch"
    conda activate torch
else
    echo "No suitable conda environment found (torch / torch-gpu)."
    exit 1
fi

TILES=(
"18MYS"
"19LEL"
"20LRP"
"20MQB"
"20MRB"
"20NQG"
"20NQJ"
"21LTJ"
"21MWV"
"22MCT"
"22MHB"
"22MHT"
"22NEG"
"23KPQ"
"23LNE"
"23LRF"
"23MKN"
"23MKS"
"23MLU"
"24KUE"
"24MVV"
)

for TILE in "${TILES[@]}"; do
    echo "======================================"
    echo "Running inference for tile: $TILE"
    echo "======================================"

    python -m src.infer \
        -c configs/inference.yaml \
        --input-dir /home/silvia/Desktop/GIGI/ESA_CCI_PROJECT/tiles_io/input/static/2015/Amazon/${TILE}/S1 \
        --polarization VH \
        --checkpoint /home/silvia/Desktop/GIGI/ESA_CCI_PROJECT/training_checkpoints_cls_whts_50_tiles_dataset/amazonia/2019/30m/amazonia/best_model.pt \
        --output-base-dir /home/silvia/Desktop/GIGI/ESA_CCI_PROJECT/tiles_io/output \
        --type static \
        --year 2015 \
        --area Amazon \
        --tile ${TILE} \
        --source S1 \
        --patch-size 128 \
        --stride 64 \
        --batch-size 4 \
    || {
        echo "Tile $TILE failed. Continuing..."
        continue
    }

done