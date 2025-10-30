#!/bin/bash
#
# Experiment 6: Few-shot Evaluation Runner
# =========================================
#
# This script runs few-shot evaluations for models that achieved 0% in zero-shot.
# Focus: Classification and NLI tasks with 1-shot and 3-shot configurations.
#
# Priority models:
# - Claude 3.5 Sonnet (NLI 0%, Classification 10%)
# - Llama 3.1 8B (NLI 0%, Classification 20%)
# - Qwen 2.5 7B (both 0%)
# - EXAONE 3.0 7.8B (both 0%)
#
# Expected timeline: 1-2 days (50 samples per task, rate-limited)

set -e  # Exit on error

# Configuration
BENCHMARK_PATH="../../benchmark/k_classic_bench/k_classic_bench_full.json"
OUTPUT_DIR="../../results/fewshot"
MAX_SAMPLES=50  # Reduced for faster evaluation
SHOTS="1 3"     # Few-shot configurations

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/exp6_run_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Experiment 6: Few-shot Evaluation" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Function to run evaluation
run_fewshot() {
    local MODEL_NAME=$1
    local MODEL_TYPE=$2
    local API_KEY=$3

    echo "" | tee -a "$LOG_FILE"
    echo "Evaluating: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    if [ "$MODEL_TYPE" = "api" ]; then
        python exp6_fewshot_evaluation.py \
            --benchmark "$BENCHMARK_PATH" \
            --model-name "$MODEL_NAME" \
            --model-type "$MODEL_TYPE" \
            --api-key "$API_KEY" \
            --shots $SHOTS \
            --tasks classification nli \
            --max-samples $MAX_SAMPLES \
            --output "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    else
        python exp6_fewshot_evaluation.py \
            --benchmark "$BENCHMARK_PATH" \
            --model-name "$MODEL_NAME" \
            --model-type "$MODEL_TYPE" \
            --shots $SHOTS \
            --tasks classification nli \
            --max-samples $MAX_SAMPLES \
            --output "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    echo "Completed: $MODEL_NAME" | tee -a "$LOG_FILE"
}

# =============================================================================
# Priority 1: API Models with 0% NLI
# =============================================================================

# Claude 3.5 Sonnet (NLI 0%, Classification 10%)
# Uncomment and add your API key
# echo "Priority 1-A: Claude 3.5 Sonnet" | tee -a "$LOG_FILE"
# run_fewshot "claude-3-5-sonnet-20241022" "api" "$ANTHROPIC_API_KEY"

# Claude 3 Opus (NLI 30%, Classification 0%)
# Uncomment and add your API key
# echo "Priority 1-B: Claude 3 Opus" | tee -a "$LOG_FILE"
# run_fewshot "claude-3-opus-20240229" "api" "$ANTHROPIC_API_KEY"

# =============================================================================
# Priority 2: Open-Source Models with 0% on both tasks
# =============================================================================

# Llama 3.1 8B (NLI 0%, Classification 20%)
echo "Priority 2-A: Llama 3.1 8B" | tee -a "$LOG_FILE"
run_fewshot "meta-llama/Llama-3.1-8B-Instruct" "opensource" ""

# Qwen 2.5 7B (NLI 0%, Classification 0%)
echo "Priority 2-B: Qwen 2.5 7B" | tee -a "$LOG_FILE"
run_fewshot "Qwen/Qwen2.5-7B-Instruct" "opensource" ""

# EXAONE 3.0 7.8B (NLI 0%, Classification 0%)
echo "Priority 2-C: EXAONE 3.0 7.8B" | tee -a "$LOG_FILE"
run_fewshot "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct" "opensource" ""

# =============================================================================
# Summary
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Experiment 6 Completed!" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

# Generate aggregated comparison
echo "" | tee -a "$LOG_FILE"
echo "Generating aggregated comparison..." | tee -a "$LOG_FILE"

python -c "
import pandas as pd
import json
from pathlib import Path

results_dir = Path('$OUTPUT_DIR')
all_summaries = []

# Collect all summary CSVs
for csv_file in results_dir.glob('summary_*.csv'):
    df = pd.read_csv(csv_file)
    all_summaries.append(df)

if all_summaries:
    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)

    # Save combined results
    output_path = results_dir / 'exp6_fewshot_summary_combined.csv'
    combined.to_csv(output_path, index=False)
    print(f'Combined summary saved to: {output_path}')

    # Print comparison table
    print('\n' + '='*60)
    print('Few-shot Performance Summary')
    print('='*60)

    # Pivot table for easier comparison
    for task in combined['task'].unique():
        print(f'\nTask: {task}')
        print('-'*60)
        task_data = combined[combined['task'] == task]
        pivot = task_data.pivot_table(
            index='model',
            columns='n_shots',
            values=['accuracy', 'f1'],
            aggfunc='mean'
        )
        print(pivot.to_string())

    # Compare with zero-shot (if available)
    print('\n' + '='*60)
    print('Performance Improvement (few-shot vs zero-shot)')
    print('='*60)
    print('Note: Compare these results with zero-shot baseline from exp5')
else:
    print('No summary files found yet.')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "To compare with zero-shot results, run:" | tee -a "$LOG_FILE"
echo "python exp6_analyze_improvements.py" | tee -a "$LOG_FILE"
