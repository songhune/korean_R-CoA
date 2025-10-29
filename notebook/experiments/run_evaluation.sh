#!/bin/bash

# KLSBench Unified Evaluation Runner
# ====================================
# Unified script for both zero-shot and few-shot evaluations
# Configuration is managed via config.yaml
#
# Usage: ./run_evaluation.sh [MODE] [OPTIONS]
#
# Modes:
#   test              - Test with 10 samples per task
#   sample [RATIO]    - Sample evaluation (default: 0.3)
#   full              - Full benchmark evaluation
#   fewshot [SHOTS]   - Few-shot evaluation (default: 1,3)
#
# Examples:
#   ./run_evaluation.sh test                    # Test mode
#   ./run_evaluation.sh sample 0.3              # 30% sampling
#   ./run_evaluation.sh full                    # Full evaluation
#   ./run_evaluation.sh fewshot "1 3"           # Few-shot (1-shot, 3-shot)

set -e  # Exit on error

# ============================================================================
# Configuration Loading
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Parse YAML (simple grep-based parser for basic YAML)
parse_yaml() {
    local prefix=$1
    local file=$2
    local s='[[:space:]]*'
    local w='[a-zA-Z0-9_]*'
    sed -ne "s|^\($s\):|\1|" \
         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1\2: \3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1\2: \3|p" "$file" |
    awk -F': ' '{
        indent = length(match($0, /^ */))
        key = $1
        gsub(/^ */, "", key)
        if (indent == 0) {
            section = key
        }
        if (length($2) > 0) {
            printf("%s_%s=%s\n", section, key, $2)
        }
    }'
}

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

# ============================================================================
# Parse Arguments
# ============================================================================

MODE=${1:-test}
OPTION=${2:-""}

# Parse mode-specific options
case "$MODE" in
    test)
        MAX_SAMPLES="--max-samples 10"
        EVAL_TYPE="zero-shot"
        echo "[TEST MODE] Sampling 10 items per task"
        ;;
    sample)
        RATIO=${OPTION:-0.3}
        MAX_SAMPLES="--sample-ratio $RATIO"
        EVAL_TYPE="zero-shot"
        echo "[SAMPLING MODE] Ratio: ${RATIO} ($(python3 -c "print(f'{$RATIO*100:.0f}%')"))"
        echo "   - Classification: $(python3 -c "print(int(808 * $RATIO))") items"
        echo "   - Retrieval: $(python3 -c "print(int(1209 * $RATIO))") items"
        echo "   - Punctuation: $(python3 -c "print(int(2000 * $RATIO))") items"
        echo "   - NLI: $(python3 -c "print(int(1854 * $RATIO))") items"
        echo "   - Translation: $(python3 -c "print(int(2000 * $RATIO))") items"
        echo "   - Total: $(python3 -c "print(int(7871 * $RATIO))") items"
        ;;
    full)
        MAX_SAMPLES=""
        EVAL_TYPE="zero-shot"
        echo "[FULL MODE] Evaluating all 7,871 items"
        ;;
    fewshot)
        SHOTS=${OPTION:-"1 3"}
        EVAL_TYPE="few-shot"
        MAX_SAMPLES="--max-samples 50"
        echo "[FEW-SHOT MODE] Shots: $SHOTS"
        echo "[INFO] Limited to 50 samples per task for faster evaluation"
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo ""
        echo "Usage: ./run_evaluation.sh [MODE] [OPTIONS]"
        echo ""
        echo "Modes:"
        echo "  test              - Test with 10 samples per task"
        echo "  sample [RATIO]    - Sample evaluation (default: 0.3)"
        echo "  full              - Full benchmark evaluation"
        echo "  fewshot [SHOTS]   - Few-shot evaluation (default: '1 3')"
        exit 1
        ;;
esac

echo "========================================"
echo "KLSBench Model Evaluation ($EVAL_TYPE)"
echo "========================================"
echo ""

# ============================================================================
# API Key Checking
# ============================================================================

check_api_keys() {
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[WARNING] OPENAI_API_KEY not set"
        echo "   export OPENAI_API_KEY='your-key'"
        OPENAI_ENABLED=false
    else
        echo "[OK] OpenAI API Key found"
        OPENAI_ENABLED=true
    fi

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "[WARNING] ANTHROPIC_API_KEY not set"
        echo "   export ANTHROPIC_API_KEY='your-key'"
        ANTHROPIC_ENABLED=false
    else
        echo "[OK] Anthropic API Key found"
        ANTHROPIC_ENABLED=true
    fi

    echo ""
    echo "========================================"
}

check_api_keys

# ============================================================================
# Evaluation Functions
# ============================================================================

run_zero_shot() {
    local model_type=$1
    local model_name=$2
    local api_key=$3

    echo ""
    echo "[$model_name] Evaluating..."

    if [ "$model_type" = "api" ]; then
        python exp5_benchmark_evaluation.py \
            --model-type api \
            --model-name "$model_name" \
            --api-key "$api_key" \
            $MAX_SAMPLES
    else
        python exp5_benchmark_evaluation.py \
            --model-type "$model_type" \
            --model-name "$model_name" \
            $MAX_SAMPLES
    fi
}

run_few_shot() {
    local model_type=$1
    local model_name=$2
    local api_key=$3

    echo ""
    echo "[$model_name] Evaluating (few-shot)..."

    if [ "$model_type" = "api" ]; then
        python exp6_fewshot_evaluation.py \
            --benchmark ../../benchmark/kls_bench/kls_bench_full.json \
            --model-name "$model_name" \
            --model-type api \
            --api-key "$api_key" \
            --shots $SHOTS \
            --tasks classification nli \
            $MAX_SAMPLES \
            --output ../../benchmark/results/fewshot
    else
        python exp6_fewshot_evaluation.py \
            --benchmark ../../benchmark/kls_bench/kls_bench_full.json \
            --model-name "$model_name" \
            --model-type "$model_type" \
            --shots $SHOTS \
            --tasks classification nli \
            $MAX_SAMPLES \
            --output ../../benchmark/results/fewshot
    fi
}

# Select evaluation function based on type
if [ "$EVAL_TYPE" = "zero-shot" ]; then
    RUN_EVAL=run_zero_shot
else
    RUN_EVAL=run_few_shot
fi

# ============================================================================
# Model Evaluation - API Models
# ============================================================================

echo ""
echo "[1/3] API Models"
echo "========================================"

# OpenAI Models
if [ "$OPENAI_ENABLED" = true ]; then
    $RUN_EVAL "api" "gpt-4-turbo" "$OPENAI_API_KEY"
    $RUN_EVAL "api" "gpt-3.5-turbo" "$OPENAI_API_KEY"
fi

# Anthropic Models
if [ "$ANTHROPIC_ENABLED" = true ]; then
    $RUN_EVAL "api" "claude-3-5-sonnet-20241022" "$ANTHROPIC_API_KEY"
    $RUN_EVAL "api" "claude-3-opus-20240229" "$ANTHROPIC_API_KEY"
fi

# ============================================================================
# Model Evaluation - Open Source Models
# ============================================================================

echo ""
echo "========================================"
echo "[2/3] Open Source Models"
echo "========================================"

$RUN_EVAL "opensource" "meta-llama/Llama-3.1-8B-Instruct" ""
$RUN_EVAL "opensource" "Qwen/Qwen2.5-7B-Instruct" ""
$RUN_EVAL "opensource" "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct" ""

# ============================================================================
# Model Evaluation - Supervised Learning Models
# ============================================================================

echo ""
echo "========================================"
echo "[3/3] Supervised Learning Models"
echo "========================================"

$RUN_EVAL "supervised" "SCUT-DLVCLab/TongGu-7B-Instruct" ""

echo ""
echo "[WARNING] GwenBert model - Encoder model, not suitable for generation tasks"
# $RUN_EVAL "supervised" "ethanyt/guwenbert-base" ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "[COMPLETE] All evaluations finished"
echo "========================================"
echo ""

if [ "$EVAL_TYPE" = "zero-shot" ]; then
    echo "Results location:"
    echo "   - Directory: ../../benchmark/results/"
    echo "   - JSON files: results_*_*.json"
    echo "   - CSV summary: summary_*_*.csv"
else
    echo "Results location:"
    echo "   - Directory: ../../benchmark/results/fewshot/"
    echo "   - JSON files: fewshot_*_*.json"
    echo "   - CSV summary: summary_*_*.csv"
    echo ""
    echo "To analyze improvements:"
    echo "   python exp6_analyze_improvements.py"
fi

echo ""
