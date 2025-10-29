#!/bin/bash

# KLSBench 전체 모델 평가 스크립트
# 사용법: ./run_all_evaluations.sh [MODE] [RATIO]
#
# MODE:
#   - test: 각 태스크당 10개 샘플로 테스트
#   - sample: 전체의 일정 비율 샘플링 (RATIO 필요)
#   - full: 전체 벤치마크 평가
#
# RATIO: sample 모드에서 샘플링 비율 (예: 0.1, 0.3, 0.5)
#
# 예시:
#   ./run_all_evaluations.sh test          # 10개 샘플 테스트
#   ./run_all_evaluations.sh sample 0.3    # 전체의 30% 샘플링
#   ./run_all_evaluations.sh full          # 전체 평가

MODE=${1:-test}  # 기본값: test
RATIO=${2:-0.3}  # 기본값: 0.3 (30%)

if [ "$MODE" = "test" ]; then
    MAX_SAMPLES="--max-samples 10"
    echo "[TEST MODE] Sampling 10 items per task"
elif [ "$MODE" = "sample" ]; then
    MAX_SAMPLES="--sample-ratio $RATIO"
    echo "[SAMPLING MODE] Ratio: ${RATIO} ($(echo "$RATIO * 100" | bc)%)"
    echo "   - Classification: $(echo "808 * $RATIO" | bc | cut -d. -f1) items"
    echo "   - Retrieval: $(echo "1209 * $RATIO" | bc | cut -d. -f1) items"
    echo "   - Punctuation: $(echo "2000 * $RATIO" | bc | cut -d. -f1) items"
    echo "   - NLI: $(echo "1854 * $RATIO" | bc | cut -d. -f1) items"
    echo "   - Translation: $(echo "2000 * $RATIO" | bc | cut -d. -f1) items"
    echo "   - Total: $(echo "7871 * $RATIO" | bc | cut -d. -f1) items"
else
    MAX_SAMPLES=""
    echo "[FULL MODE] Evaluating all 7,871 items"
fi

echo "========================================"
echo "KLSBench Model Evaluation"
echo "========================================"
echo ""

# Check API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY not set"
    echo "   export OPENAI_API_KEY='your-key'"
else
    echo "[OK] OpenAI API Key found"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "[WARNING] ANTHROPIC_API_KEY not set"
    echo "   export ANTHROPIC_API_KEY='your-key'"
else
    echo "[OK] Anthropic API Key found"
fi

echo ""
echo "========================================"

# 1. API Models
echo ""
echo "[1/3] API Models"
echo "========================================"

# GPT-4 Turbo
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "[GPT-4 Turbo] Evaluating..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name gpt-4-turbo \
        --api-key $OPENAI_API_KEY \
        $MAX_SAMPLES

    echo ""
    echo "[GPT-3.5 Turbo] Evaluating..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name gpt-3.5-turbo \
        --api-key $OPENAI_API_KEY \
        $MAX_SAMPLES
fi

# Claude
if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "[Claude 3.5 Sonnet] Evaluating..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name claude-3-5-sonnet-20241022 \
        --api-key $ANTHROPIC_API_KEY \
        $MAX_SAMPLES

    echo ""
    echo "[Claude 3 Opus] Evaluating..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name claude-3-opus-20240229 \
        --api-key $ANTHROPIC_API_KEY \
        $MAX_SAMPLES
fi

# 2. Open Source Models
echo ""
echo "========================================"
echo "[2/3] Open Source Models"
echo "========================================"

# Llama 3.1 8B
echo ""
echo "[Llama 3.1 8B Instruct] Evaluating..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    $MAX_SAMPLES

# Qwen 2.5 7B
echo ""
echo "[Qwen 2.5 7B Instruct] Evaluating..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    $MAX_SAMPLES

# EXAONE 3.0 7.8B
echo ""
echo "[EXAONE 3.0 7.8B Instruct] Evaluating..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
    $MAX_SAMPLES

# 3. Supervised Learning Models
echo ""
echo "========================================"
echo "[3/3] Supervised Learning Models"
echo "========================================"

# Tongu (Not implemented yet)
echo ""
echo "[WARNING] Tongu model - Implementation needed"
# python exp5_benchmark_evaluation.py \
#     --model-type supervised \
#     --model-name tongu \
#     $MAX_SAMPLES

# GwenBert (Not implemented yet)
echo ""
echo "[WARNING] GwenBert model - Implementation needed"
# python exp5_benchmark_evaluation.py \
#     --model-type supervised \
#     --model-name gwenbert \
#     $MAX_SAMPLES

echo ""
echo "========================================"
echo "[COMPLETE] All evaluations finished"
echo "========================================"
echo ""
echo "Results location:"
echo "   - Directory: ../../benchmark/results/"
echo "   - JSON files: results_*_*.json"
echo "   - CSV summary: summary_*_*.csv"
echo ""
