#!/bin/bash
# Update all script paths to use reorganized results directory

set -e

echo "==========================================="
echo "Updating Script Paths"
echo "==========================================="
echo ""

BASE_DIR="/Users/songhune/Workspace/korean_eda/notebook/experiments"
cd "$BASE_DIR"

# Paths to update
OLD_RESULTS_PATH="/home/work/songhune/korean_R-CoA/results"
OLD_BENCHMARK_PATH="../../benchmark/results"
OLD_GRAPHS_PATH="../graphs"

NEW_RESULTS_PATH="../../results/raw_evaluation"
NEW_AGGREGATED_PATH="../../results/aggregated"
NEW_FIGURES_PATH="../../results/figures"
NEW_TABLES_PATH="../../results/tables"

# Update exp6/exp6_result_aggregation.py
echo "Updating exp6/exp6_result_aggregation.py..."
sed -i.bak "s|default='/home/work/songhune/korean_R-CoA/results'|default='../../results/raw_evaluation'|g" exp6/exp6_result_aggregation.py
sed -i.bak "s|default='/home/work/songhune/korean_R-CoA/results/aggregated'|default='../../results/aggregated'|g" exp6/exp6_result_aggregation.py

# Update exp6/exp6_analyze_improvements.py
echo "Updating exp6/exp6_analyze_improvements.py..."
sed -i.bak "s|../../benchmark/results/aggregated|../../results/aggregated|g" exp6/exp6_analyze_improvements.py

# Update exp6/exp6_fewshot_evaluation.py
echo "Updating exp6/exp6_fewshot_evaluation.py..."
sed -i.bak "s|../../benchmark/results|../../results/raw_evaluation|g" exp6/exp6_fewshot_evaluation.py

# Update exp7 scripts
echo "Updating exp7 scripts..."
sed -i.bak "s|--output-dir ../../results/figures|--output-dir ../../results/figures|g" exp7/*.py
sed -i.bak "s|output_dir = Path(__file__).parent.parent / 'graphs'|output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'|g" exp7/*.py

# Update run_exp7.sh
echo "Updating exp7/run_exp7.sh..."
sed -i.bak 's|OUTPUT_DIR="../../results/figures"|OUTPUT_DIR="../../results/figures"|g' exp7/run_exp7.sh

# Clean up backup files
echo ""
echo "Cleaning up backup files..."
find . -name "*.bak" -delete

echo ""
echo "✅ Path updates complete!"
echo ""
echo "Updated paths:"
echo "  Results: ../../results/raw_evaluation"
echo "  Aggregated: ../../results/aggregated"
echo "  Figures: ../../results/figures"
echo "  Tables: ../../results/tables"
echo ""
