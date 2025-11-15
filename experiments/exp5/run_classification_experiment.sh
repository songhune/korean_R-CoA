#!/bin/bash
# Classification Only Experiment (훨씬 빠름!)
# 808개 classification 샘플 × 7개 모델

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "CLASSIFICATION CONFUSION MATRIX EXPERIMENT"
echo "======================================================================"
echo ""
echo "Classification samples: 808"
echo "Models: 7"
echo "Estimated time: ~2 hours (classification만!)"
echo ""
echo "Models:"
echo "  [API]"
echo "    1. GPT-4 Turbo (~20분)"
echo "    2. GPT-3.5 Turbo (~20분)"
echo "    3. Claude 3 Opus (~20분)"
echo "    4. Claude 3 Haiku (~20분)"
echo "  [Open Source]"
echo "    5. Llama 3.1 8B Instruct (~13분)"
echo "    6. Qwen 2.5 7B Instruct (~13분)"
echo "    7. EXAONE 3.0 7.8B Instruct (~13분)"
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
echo "STEP 1/2: Generating Predictions (808 classification samples)"
echo "======================================================================"
echo ""

# API Models
echo "[1/7] GPT-4 Turbo..."
python save_classification_predictions.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --temperature 0.0

echo ""
echo "[2/7] GPT-3.5 Turbo..."
python save_classification_predictions.py \
    --model-type api \
    --model-name gpt-3.5-turbo \
    --temperature 0.0

echo ""
echo "[3/7] Claude 3 Opus..."
python save_classification_predictions.py \
    --model-type api \
    --model-name claude-3-opus-20240229 \
    --temperature 0.0

echo ""
echo "[4/7] Claude 3 Haiku..."
python save_classification_predictions.py \
    --model-type api \
    --model-name claude-3-haiku-20240307 \
    --temperature 0.0

# Opensource Models
echo ""
echo "[5/7] Llama 3.1 8B Instruct..."
python save_classification_predictions.py \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.0

echo ""
echo "[6/7] Qwen 2.5 7B Instruct..."
python save_classification_predictions.py \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0

echo ""
echo "[7/7] EXAONE 3.0 7.8B Instruct..."
python save_classification_predictions.py \
    --model-type opensource \
    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
    --temperature 0.0

echo ""
echo "======================================================================"
echo "STEP 2/2: Generating Confusion Matrices (with Average)"
echo "======================================================================"
echo ""

python generate_classification_confusion_matrix.py \
    --benchmark ../../benchmark/kls_bench_classification.json \
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
echo "  - Predictions: ../../results/full_predictions/"
echo "  - Confusion Matrices: ../../results/confusion_matrices_full/"
echo ""
echo "생성된 파일:"
ls -1 ../../results/confusion_matrices_full/*.png 2>/dev/null | wc -l | xargs echo "  - PNG 이미지:"
echo ""
echo "주요 파일:"
echo "  - confusion_matrix_AVERAGE_all_models.png (⭐ 평균)"
echo "  - confusion_matrix_AVERAGE_all_models_stats.txt (⭐ 통계)"
echo ""
echo "======================================================================"
