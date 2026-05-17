#!/usr/bin/env bash
# ------------------------------------------------------------------------
# RF-DETR — pick-and-roll data pipeline (filter → graphs → split → norm stats)
# Run from repo root with venv active, or: bash scripts/run_pick_and_roll_pipeline.sh
# ------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Data directory (override: DATA_DIR=raw_data_3 bash scripts/run_pick_and_roll_pipeline.sh)
DATA_DIR="${DATA_DIR:-raw_data_4}"

# Val clips (comma-separated, no spaces). Override: VAL_CLIPS="Dame2,ChrisPaul7,..."
VAL_CLIPS="${VAL_CLIPS:-Dame2,ChrisPaul7,Harden6,Curry3,TyreseMaxey2}"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "=== Pick-and-roll pipeline ==="
echo "DATA_DIR=$DATA_DIR"
echo "VAL_CLIPS=$VAL_CLIPS"
echo ""

# 1. Filter detections
echo "[1/4] Filter annotations..."
python scripts/filter_tactical_jsonl.py \
    "${DATA_DIR}/annotations.jsonl" \
    --output "${DATA_DIR}/annotations_filtered.jsonl"

# 2. Build graph features (111-D with current graph.py)
echo "[2/4] Build graph features..."
python scripts/run_tactical_graph_from_jsonl.py \
    "${DATA_DIR}/annotations_filtered.jsonl" \
    --save-graphs "${DATA_DIR}/graph_all.jsonl"

# 3. Train/val split by clip_id
echo "[3/4] Split train/val..."
python scripts/split_clips_train_val.py \
    "${DATA_DIR}/graph_all.jsonl" \
    --train "${DATA_DIR}/graph_train.jsonl" \
    --val "${DATA_DIR}/graph_val.jsonl" \
    --val-clips "${VAL_CLIPS}" \
    --labels "${DATA_DIR}/labels.csv" \
    --labels-train "${DATA_DIR}/labels_train.csv" \
    --labels-val "${DATA_DIR}/labels_val.csv"

# 4. Norm stats on train only (after split)
echo "[4/4] Fit normalization on train split..."
python scripts/compute_graph_feature_stats.py \
    "${DATA_DIR}/graph_train.jsonl" \
    --output "${DATA_DIR}/graph_feature_norm.json" \
    --only-valid

echo ""
echo "Done. Next: label any blank rows in ${DATA_DIR}/labels.csv, then train:"
echo "  PYTHONPATH=src python scripts/train_pick_and_roll.py \\"
echo "    --graphs ${DATA_DIR}/graph_train.jsonl \\"
echo "    --labels ${DATA_DIR}/labels_train.csv \\"
echo "    --norm-stats ${DATA_DIR}/graph_feature_norm.json \\"
echo "    --epochs 50 \\"
echo "    --device cuda \\"
echo "    --save-model ${DATA_DIR}/checkpoints/pick_roll_model.pt \\"
echo "    --log-csv ${DATA_DIR}/logs/train_metrics.csv"
