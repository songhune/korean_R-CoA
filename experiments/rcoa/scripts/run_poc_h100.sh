#!/bin/bash
#
# R-CoA H100 Large Model Training
# Optimized for H100 GPU (79GB VRAM)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "  R-CoA H100 Large Model Training"
echo "  Model: XLM-RoBERTa-XL (3.5B parameters)"
echo "======================================================================"
echo ""

# Model selection
MODEL_SIZE=${1:-xl}  # Options: large, xl, xxl

case $MODEL_SIZE in
    large)
        MODEL_NAME="FacebookAI/xlm-roberta-large"
        BATCH_SIZE=64         # Conservative
        LORA_R=16
        PROJECTION_DIM=256
        GRAD_ACCUM=4
        EPOCHS=15
        echo "Model: XLM-RoBERTa-Large (550M params)"
        ;;
    xl)
        MODEL_NAME="FacebookAI/xlm-roberta-xl"
        BATCH_SIZE=32         # Conservative for large model
        LORA_R=16
        PROJECTION_DIM=512
        GRAD_ACCUM=6
        EPOCHS=20
        echo "Model: XLM-RoBERTa-XL (3.5B params)"
        ;;
    xxl)
        MODEL_NAME="FacebookAI/xlm-roberta-xxl"
        BATCH_SIZE=16         # Very conservative
        LORA_R=8
        PROJECTION_DIM=256
        GRAD_ACCUM=8
        EPOCHS=20
        echo "Model: XLM-RoBERTa-XXL (10.7B params)"
        ;;
    *)
        echo "Invalid model size. Use: large, xl, or xxl"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch Size: $BATCH_SIZE (per step)"
echo "  Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA Rank: $LORA_R"
echo "  Projection Dim: $PROJECTION_DIM"
echo "  Epochs: $EPOCHS"
echo "  Mixed Precision: BF16 (AMP)"
echo "  Gradient Checkpointing: Enabled"
echo "  Gradient Accumulation: $GRAD_ACCUM steps"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Training
echo -e "${GREEN}[TRAIN] Starting H100 Large Model Training${NC}"
echo "----------------------------------------------------------------------"

python scripts/train/anchor_train.py \
    --train-data data/splits/train_pairs.jsonl \
    --val-data data/splits/val_pairs.jsonl \
    --model-name "$MODEL_NAME" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 1e-5 \
    --lora-r $LORA_R \
    --lora-alpha $((LORA_R * 2)) \
    --projection-dim $PROJECTION_DIM \
    --temperature 0.05 \
    --warmup-ratio 0.05 \
    --weight-decay 0.01 \
    --num-workers 2 \
    --use-amp \
    --gradient-checkpointing \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --output-dir "checkpoints/anchor_head_${MODEL_SIZE}" \
    --save-interval 5 \
    --device cuda

echo ""
echo -e "${GREEN}[EVAL] Evaluating Model${NC}"
echo "----------------------------------------------------------------------"

python scripts/eval/anchor_evaluate.py \
    --checkpoint "checkpoints/anchor_head_${MODEL_SIZE}/best_model.pt" \
    --test-data data/splits/val_pairs.jsonl \
    --model-name "$MODEL_NAME" \
    --max-samples 2000 \
    --batch-size 64 \
    --retrieval-k 10 \
    --output-dir "results/h100_${MODEL_SIZE}" \
    --device cuda

echo ""
echo -e "${BLUE}[VISUALIZE] Creating Embeddings Visualization${NC}"
echo "----------------------------------------------------------------------"

python scripts/visualize/visualize_embeddings.py \
    --checkpoint "checkpoints/anchor_head_${MODEL_SIZE}/best_model.pt" \
    --data data/splits/val_pairs.jsonl \
    --model-name "$MODEL_NAME" \
    --max-samples 1000 \
    --output-dir "results/figures_${MODEL_SIZE}" \
    --device cuda

echo ""
echo "======================================================================"
echo -e "${BLUE}[COMPLETE] H100 Training Pipeline Finished${NC}"
echo "======================================================================"
echo ""
echo "Results:"
echo "  Checkpoints: checkpoints/anchor_head_${MODEL_SIZE}/"
echo "  Metrics: results/h100_${MODEL_SIZE}/evaluation_results.json"
echo "  Figures: results/figures_${MODEL_SIZE}/"
echo ""
