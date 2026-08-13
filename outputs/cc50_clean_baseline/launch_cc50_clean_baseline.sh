#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/home/tlcrs/Philippines_Project/09_Luigi Russo/03-Swin_Unet_Pipeline"
PYTHON_BIN="/home/tlcrs/anaconda3/envs/swinunet_philippines/bin/python"

RUN_DIR="$PROJECT_DIR/outputs/cc50_clean_baseline"
LOG_FILE="$RUN_DIR/training.log"
LOCK_FILE="$RUN_DIR/training.lock"

CHECKPOINT_BASE="/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/checkpoints_cc50_clean_baseline"

mkdir -p "$RUN_DIR"

exec 9>"$LOCK_FILE"

if ! flock -n 9; then
    echo "ERROR: CC50 baseline training is already running."
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

{
    echo "============================================================"
    echo "[START] $(date)"
    echo "[PROJECT] $PROJECT_DIR"
    echo "[PYTHON] $PYTHON_BIN"
    echo "[CHECKPOINT] $CHECKPOINT_BASE"
    echo "[SPLIT HASH]"
    sha256sum configs/splits/philippines/tile_split.json
    echo "============================================================"
} | tee "$LOG_FILE"

"$PYTHON_BIN" -u -m src.train \
    --checkpoint-base-dir "$CHECKPOINT_BASE" \
    2>&1 | tee -a "$LOG_FILE"

status=${PIPESTATUS[0]}

echo "============================================================" | tee -a "$LOG_FILE"
echo "[PROCESS EXIT] code=$status at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

exit "$status"
