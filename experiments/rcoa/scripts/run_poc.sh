#!/bin/bash
#
# R-CoA Proof-of-Concept (PoC) Pipeline
# Week 1: Anchor Head with InfoNCE Loss
#
# Usage: ./run_poc.sh [step]
#   step 1: Data preprocessing
#   step 2: Train Anchor Head (quick test)
#   step 3: Train Anchor Head (full)
#   step 4: Evaluate
#   step all: Run all steps

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  R-CoA Proof-of-Concept Pipeline"
echo "  Anchor Head + InfoNCE for Cross-lingual Alignment"
echo "======================================================================"
echo ""

STEP=${1:-all}

# Step 1: Data Preprocessing
if [ "$STEP" == "1" ] || [ "$STEP" == "all" ]; then
    echo -e "${GREEN}[STEP 1] Data Preprocessing${NC}"
    echo "----------------------------------------------------------------------"
    python scripts/preprocess/data_preprocessing.py
    echo ""
fi

# Step 2: Quick Training Test (1 epoch, small data)
if [ "$STEP" == "2" ]; then
    echo -e "${GREEN}[STEP 2] Quick Training Test${NC}"
    echo "----------------------------------------------------------------------"
    python scripts/train/anchor_train.py \
        --train-data data/splits/train_pairs.jsonl \
        --val-data data/splits/val_pairs.jsonl \
        --batch-size 16 \
        --epochs 1 \
        --lr 2e-5 \
        --lora-r 8 \
        --lora-alpha 16 \
        --projection-dim 256 \
        --temperature 0.07 \
        --output-dir checkpoints/quick_test \
        --device cuda
    echo ""
fi

# Step 3: Full Training
if [ "$STEP" == "3" ] || [ "$STEP" == "all" ]; then
    echo -e "${GREEN}[STEP 3] Full Training${NC}"
    echo "----------------------------------------------------------------------"
    python scripts/train/anchor_train.py \
        --train-data data/splits/train_pairs.jsonl \
        --val-data data/splits/val_pairs.jsonl \
        --batch-size 32 \
        --epochs 10 \
        --lr 2e-5 \
        --lora-r 8 \
        --lora-alpha 16 \
        --projection-dim 256 \
        --temperature 0.07 \
        --warmup-ratio 0.1 \
        --weight-decay 0.01 \
        --output-dir checkpoints/anchor_head_full \
        --save-interval 2 \
        --device cuda
    echo ""
fi

# Step 4: Evaluation
if [ "$STEP" == "4" ] || [ "$STEP" == "all" ]; then
    echo -e "${GREEN}[STEP 4] Evaluation${NC}"
    echo "----------------------------------------------------------------------"

    # Find best checkpoint
    CHECKPOINT="checkpoints/anchor_head_full/best_model.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found at $CHECKPOINT"
        echo "Please run training first (step 3)"
        exit 1
    fi

    python scripts/eval/anchor_evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --test-data data/splits/val_pairs.jsonl \
        --max-samples 1000 \
        --batch-size 32 \
        --retrieval-k 10 \
        --output-dir results \
        --device cuda
    echo ""
fi

echo "======================================================================"
echo -e "${BLUE}[COMPLETE] R-CoA PoC Pipeline Finished${NC}"
echo "======================================================================"
echo ""
echo "Output directories:"
echo "  - Data: $SCRIPT_DIR/data/"
echo "  - Checkpoints: $SCRIPT_DIR/checkpoints/"
echo "  - Results: $SCRIPT_DIR/results/"
echo ""
echo "Next steps:"
echo "  1. Check results in results/evaluation_results.json"
echo "  2. Visualize embeddings (t-SNE, UMAP)"
echo "  3. Run ablation studies"
echo "  4. Implement Chain Head (Week 3)"
echo ""
