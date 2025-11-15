#!/bin/bash
#
# R-CoA Full Experimental Pipeline
# Week 1-4 전체 자동화
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "  R-CoA Full Experimental Pipeline"
echo "  Week 1-4 자동화 실험"
echo "======================================================================"
echo ""

# ============================================================================
# Week 1: Anchor Head
# ============================================================================
echo -e "${GREEN}[WEEK 1] Anchor Head 학습${NC}"
echo "----------------------------------------------------------------------"

# 데이터 전처리
if [ ! -f "data/splits/train_pairs.jsonl" ]; then
    echo "[1/3] 데이터 전처리..."
    python scripts/preprocess/data_preprocessing.py
else
    echo "[SKIP] 데이터 이미 존재"
fi

# Anchor Head 학습
echo "[2/3] Anchor Head 학습 (10 epochs)..."
python scripts/train/anchor_train.py \
    --train-data data/splits/train_pairs.jsonl \
    --val-data data/splits/val_pairs.jsonl \
    --batch-size 32 \
    --epochs 10 \
    --lr 2e-5 \
    --output-dir checkpoints/anchor_head \
    --device cuda

# 평가
echo "[3/3] Anchor Head 평가..."
python scripts/eval/anchor_evaluate.py \
    --checkpoint checkpoints/anchor_head/best_model.pt \
    --test-data data/splits/val_pairs.jsonl \
    --max-samples 1000 \
    --output-dir results/week1 \
    --device cuda

echo ""

# ============================================================================
# Week 2: Visualization & Analysis
# ============================================================================
echo -e "${GREEN}[WEEK 2] 시각화 및 분석${NC}"
echo "----------------------------------------------------------------------"

# 임베딩 시각화
echo "[1/2] 임베딩 시각화 (t-SNE)..."
python scripts/visualize/visualize_embeddings.py \
    --checkpoint checkpoints/anchor_head/best_model.pt \
    --data data/splits/val_pairs.jsonl \
    --max-samples 500 \
    --output-dir results/figures \
    --device cuda

# 성능 그래프 (추가 구현 필요)
echo "[2/2] 성능 분석..."
echo "  [TODO] Baseline 비교, Ablation study"

echo ""

# ============================================================================
# Week 3: Chain Head (향후 구현)
# ============================================================================
echo -e "${YELLOW}[WEEK 3] Chain Head (향후 구현)${NC}"
echo "----------------------------------------------------------------------"
echo "  [TODO] TransE + Chain Loss 학습"
echo "  [TODO] KG triple 데이터 준비"
echo "  [TODO] Citation chain 추출"
echo ""

# ============================================================================
# Week 4: 발표자료 (향후 구현)
# ============================================================================
echo -e "${YELLOW}[WEEK 4] 발표자료 준비 (향후 구현)${NC}"
echo "----------------------------------------------------------------------"
echo "  [TODO] Marp 슬라이드"
echo "  [TODO] Results aggregation"
echo "  [TODO] Demo notebook"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================"
echo -e "${BLUE}[COMPLETE] 실험 완료${NC}"
echo "======================================================================"
echo ""
echo "생성된 결과물:"
echo "  - checkpoints/: 모델 체크포인트"
echo "  - results/: 평가 결과"
echo "  - figures/: 시각화 그래프"
echo ""
echo "다음 단계:"
echo "  1. results/week1/evaluation_results.json 확인"
echo "  2. figures/week2/ 시각화 확인"
echo "  3. Week 3 Chain Head 구현"
echo "  4. Week 4 발표자료 준비"
echo ""
