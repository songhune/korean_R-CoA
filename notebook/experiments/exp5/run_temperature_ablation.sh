#!/bin/bash

# Temperature Ablation Study for KLSBench
# =========================================
# This script evaluates models across different temperature settings
# to understand the impact of temperature on task performance.
#
# Usage: ./run_temperature_ablation.sh [MODE]
#
# Modes:
#   test    - Test with 10 samples per task (default)
#   sample  - Sample evaluation (30% of data)
#   full    - Full benchmark evaluation
#
# Example:
#   ./run_temperature_ablation.sh test

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCHMARK_FILE="$PROJECT_ROOT/benchmark/kls_bench/kls_bench_full.json"
RESULTS_DIR="$PROJECT_ROOT/results/temperature_ablation"
mkdir -p "$RESULTS_DIR"

# Python environment setup
export PYTHONPATH="${PYTHONPATH}:${HOME}/.local/lib/python3.12/site-packages"

# Load environment variables from .env if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[CONFIG] Loading environment variables from .env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "[OK] Environment variables loaded"
else
    echo "[WARNING] .env file not found at $PROJECT_ROOT/.env"
fi
echo ""

# ============================================================================
# Parse Arguments
# ============================================================================

MODE=${1:-test}

case "$MODE" in
    test)
        MAX_SAMPLES="--max-samples 10"
        echo "[TEST MODE] Sampling 10 items per task"
        ;;
    sample)
        MAX_SAMPLES="--sample-ratio 0.1"
        echo "[SAMPLING MODE] Using 10% of data"
        ;;
    full)
        MAX_SAMPLES=""
        echo "[FULL MODE] Evaluating all 7,871 items"
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo ""
        echo "Usage: ./run_temperature_ablation.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  test    - Test with 10 samples per task"
        echo "  sample  - Sample evaluation (10% of data)"
        echo "  full    - Full benchmark evaluation"
        exit 1
        ;;
esac

echo "========================================"
echo "Temperature Ablation Study"
echo "========================================"
echo ""

# ============================================================================
# Temperature Settings
# ============================================================================

TEMPERATURES=(0.0 0.3 0.7)

echo "Temperature values to test: ${TEMPERATURES[@]}"
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
# Model Selection for Ablation
# ============================================================================

# We test with representative models:
# 1. GPT-4-turbo, GPT-3.5-turbo (OpenAI)
# 2. Claude-3.5-Sonnet, Claude-3-Opus (Anthropic)
# 3. Llama-3.1-8B (Open source representative)
# 4. Qwen2.5-7B (Open source representative)
# 5. Exaone-3.0-7.8B (Open source representative)

API_MODELS=()
OPENSOURCE_MODELS=()

if [ "$OPENAI_ENABLED" = true ]; then
    API_MODELS+=("gpt-4-turbo")
    API_MODELS+=("gpt-3.5-turbo")
fi

if [ "$ANTHROPIC_ENABLED" = true ]; then
    API_MODELS+=("claude-sonnet-4-5-20250929")  # Claude Sonnet 4.5 (latest)
    API_MODELS+=("claude-3-opus-20240229")       # Claude 3 Opus
fi

# Open source models (requires GPU)
# Only add if HF_TOKEN is set (indicates GPU environment)
if [ ! -z "$HF_TOKEN" ]; then
    OPENSOURCE_MODELS+=("meta-llama/Llama-3.1-8B-Instruct")
    OPENSOURCE_MODELS+=("Qwen/Qwen2.5-7B-Instruct")
    OPENSOURCE_MODELS+=("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
    echo "[OK] HF_TOKEN found - OpenSource models enabled"
else
    echo "[INFO] HF_TOKEN not set - Skipping OpenSource models"
fi

echo ""
echo "Models to evaluate:"
echo "  API Models: ${API_MODELS[@]:-None}"
echo "  Open Source Models: ${OPENSOURCE_MODELS[@]:-None}"
echo ""

# ============================================================================
# Evaluation Function
# ============================================================================

run_evaluation() {
    local model_type=$1
    local model_name=$2
    local temperature=$3
    local api_key=$4

    echo ""
    echo "========================================"
    echo "Model: $model_name"
    echo "Temperature: $temperature"
    echo "========================================"

    # Check if results already exist for this model/temperature combination
    local existing_results=$(python3 -c "
import json
import os
import sys

results_dir = '$RESULTS_DIR'
model_name = '$model_name'
temperature = float('$temperature')

for f in os.listdir(results_dir):
    if f.startswith('results_') and f.endswith('.json'):
        try:
            with open(os.path.join(results_dir, f)) as file:
                data = json.load(file)
                if data.get('model_name') == model_name and abs(data.get('temperature', -1) - temperature) < 0.001:
                    print(f)
                    sys.exit(0)
        except:
            pass
" 2>/dev/null)

    if [ ! -z "$existing_results" ]; then
        echo "[SKIP] Results already exist: $existing_results"
        echo ""
        return
    fi

    if [ "$model_type" = "api" ]; then
        python3 "$SCRIPT_DIR/exp5_benchmark_evaluation.py" \
            --model-type api \
            --model-name "$model_name" \
            --api-key "$api_key" \
            --temperature "$temperature" \
            --output "$RESULTS_DIR" \
            --save-samples \
            --num-samples-to-save 5 \
            $MAX_SAMPLES
    else
        python3 "$SCRIPT_DIR/exp5_benchmark_evaluation.py" \
            --model-type "$model_type" \
            --model-name "$model_name" \
            --temperature "$temperature" \
            --output "$RESULTS_DIR" \
            --save-samples \
            --num-samples-to-save 5 \
            $MAX_SAMPLES
    fi

    echo ""
    echo "[OK] Completed: $model_name (temp=$temperature)"
    echo ""
}

# ============================================================================
# Run Ablation Study
# ============================================================================

echo ""
echo "========================================"
echo "Starting Temperature Ablation"
echo "========================================"
echo ""

# API Models
for model in "${API_MODELS[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        if [[ "$model" == "gpt"* ]]; then
            run_evaluation "api" "$model" "$temp" "$OPENAI_API_KEY"
        elif [[ "$model" == "claude"* ]]; then
            run_evaluation "api" "$model" "$temp" "$ANTHROPIC_API_KEY"
        fi
    done
done

# Open Source Models
for model in "${OPENSOURCE_MODELS[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        run_evaluation "opensource" "$model" "$temp" ""
    done
done

# ============================================================================
# Generate Comparison Report
# ============================================================================

echo ""
echo "========================================"
echo "Generating Comparison Report"
echo "========================================"
echo ""

# Run analysis if results exist
if ls "$RESULTS_DIR"/results_*.json 1> /dev/null 2>&1; then
    python3 "$SCRIPT_DIR/analyze_temperature_ablation.py" "$RESULTS_DIR"
else
    echo "[INFO] No results to analyze yet"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "[COMPLETE] Temperature Ablation Study"
echo "========================================"
echo ""
echo "Results location: $RESULTS_DIR"
echo ""
echo "Files generated:"
echo "  - results_*.json: Raw evaluation results"
echo "  - summary_*.csv: Per-evaluation summaries"
echo "  - temperature_ablation_summary.csv: Combined summary"
echo "  - temperature_ablation_*.pdf: Visualization plots"
echo ""
echo "To re-run analysis:"
echo "  python3 $SCRIPT_DIR/analyze_temperature_ablation.py $RESULTS_DIR"
echo ""
