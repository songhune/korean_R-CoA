#!/bin/bash

# Python environment setup
export PYTHONPATH="${PYTHONPATH}:${HOME}/.local/lib/python3.12/site-packages"

# Load environment variables from .env if exists
if [ -f .env ]; then
    echo "[CONFIG] Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
    echo "[OK] Environment variables loaded"
else
    echo "[WARNING] .env file not found"
fi
echo ""

# KLSBench Full Model Evaluation Script
# Usage: ./run_all_evaluations.sh [MODE] [RATIO]
#
# MODE:
#   - test: Test with 10 samples per task
#   - sample: Sample a ratio of full benchmark (requires RATIO)
#   - full: Evaluate full benchmark
#
# RATIO: Sampling ratio for sample mode (e.g., 0.1, 0.3, 0.5)
#
# Examples:
#   ./run_all_evaluations.sh test          # Test with 10 samples
#   ./run_all_evaluations.sh sample 0.3    # Sample 30% of data
#   ./run_all_evaluations.sh full          # Full evaluation

MODE=${1:-test}  # Default: test
RATIO=${2:-0.3}  # Default: 0.3 (30%)

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

# TongGu
echo ""
echo "[TongGu-7B-Instruct] Evaluating..."
python exp5_benchmark_evaluation.py \
    --model-type supervised \
    --model-name SCUT-DLVCLab/TongGu-7B-Instruct \
    $MAX_SAMPLES

# GwenBert (encoder model, not suitable for generation tasks)
echo ""
echo "[WARNING] GwenBert model - Encoder model, not suitable for generation tasks"
# python exp5_benchmark_evaluation.py \
#     --model-type supervised \
#     --model-name ethanyt/guwenbert-base \
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
