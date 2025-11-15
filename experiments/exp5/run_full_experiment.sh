#!/bin/bash
# Full Experiment: Generate all predictions and confusion matrices
# 전체 808개 샘플 × 7개 모델

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "FULL CONFUSION MATRIX EXPERIMENT"
echo "======================================================================"
echo ""
echo "Total samples: 808 (classification task)"
echo "Total models: 7"
echo "Estimated time: ~2.5 hours"
echo ""
echo "Models:"
echo "  [API]"
echo "    1. GPT-4 Turbo"
echo "    2. GPT-3.5 Turbo"
echo "    3. Claude 3.5 Sonnet"
echo "    4. Claude 3 Opus"
echo "  [Open Source]"
echo "    5. Llama 3.1 8B Instruct"
echo "    6. Qwen 2.5 7B Instruct"
echo "    7. EXAONE 3.0 7.8B Instruct"
echo ""
echo "======================================================================"
echo ""

read -p "시작하시겠습니까? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo "======================================================================"
echo "STEP 1/2: Generating Predictions (전체 808개 샘플)"
echo "======================================================================"
echo ""

# API Models
echo "[1/7] GPT-4 Turbo (ETA: 25분)..."
python save_full_predictions.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --temperature 0.0

echo ""
echo "[2/7] GPT-3.5 Turbo (ETA: 25분)..."
python save_full_predictions.py \
    --model-type api \
    --model-name gpt-3.5-turbo \
    --temperature 0.0

echo ""
echo "[3/7] Claude 3.5 Sonnet (ETA: 25분)..."
python save_full_predictions.py \
    --model-type api \
    --model-name claude-3-5-sonnet-20241022 \
    --temperature 0.0

echo ""
echo "[4/7] Claude 3 Opus (ETA: 25분)..."
python save_full_predictions.py \
    --model-type api \
    --model-name claude-3-opus-20240229 \
    --temperature 0.0

# Opensource Models
echo ""
echo "[5/7] Llama 3.1 8B Instruct (ETA: 15분)..."
python save_full_predictions.py \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.0

echo ""
echo "[6/7] Qwen 2.5 7B Instruct (ETA: 15분)..."
python save_full_predictions.py \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0

echo ""
echo "[7/7] EXAONE 3.0 7.8B Instruct (ETA: 15분)..."
python save_full_predictions.py \
    --model-type opensource \
    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
    --temperature 0.0

echo ""
echo "======================================================================"
echo "STEP 2/2: Generating Confusion Matrices (평균 포함)"
echo "======================================================================"
echo ""

python generate_classification_confusion_matrix.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --results-dir ../../results/full_predictions \
    --output-dir ../../results/confusion_matrices_full \
    --temperature 0.0 \
    --use-full-predictions \
    --full-predictions-dir ../../results/full_predictions

echo ""
echo "======================================================================"
echo "실험 완료!"
echo "======================================================================"
echo ""
echo "결과 위치:"
echo "  - Full Predictions: ../../results/full_predictions/"
echo "  - Confusion Matrices: ../../results/confusion_matrices_full/"
echo "  - Average Matrix: ../../results/confusion_matrices_full/confusion_matrix_AVERAGE_all_models.png"
echo ""
echo "생성된 파일:"
ls -lh ../../results/confusion_matrices_full/*.png 2>/dev/null | wc -l | xargs echo "  - PNG 이미지:"
ls -lh ../../results/confusion_matrices_full/*.txt 2>/dev/null | wc -l | xargs echo "  - 리포트:"
echo ""
echo "======================================================================"
