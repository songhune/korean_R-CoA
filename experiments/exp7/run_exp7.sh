#!/bin/bash
# Experiment 7: Run all appendix generation scripts

set -e  # Exit on error

echo "========================================"
echo "Experiment 7: KLSBench Appendix Generation"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import pandas, numpy, matplotlib, seaborn, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required packages"
    echo "Install with: pip install pandas numpy matplotlib seaborn pyyaml"
    exit 1
fi
echo "✓ All dependencies found"
echo ""

# Create output directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figures"
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# Run main appendix generation
echo "========================================"
echo "Step 1: Generating Appendix Materials"
echo "========================================"
python3 exp7_generate_appendix.py
if [ $? -ne 0 ]; then
    echo "Error: Appendix generation failed"
    exit 1
fi
echo ""

# Run detailed analysis
echo "========================================"
echo "Step 2: Running Detailed Analysis"
echo "========================================"
python3 exp7_detailed_analysis.py
if [ $? -ne 0 ]; then
    echo "Error: Detailed analysis failed"
    exit 1
fi
echo ""

# Summary
echo "========================================"
echo "Experiment 7 Complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  📊 Appendix A: Task examples and distributions"
echo "  📊 Appendix B: Per-class performance and error analysis"
echo "  📊 Detailed: Comprehensive statistics"
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""

# List generated files
echo "Generated files (*.pdf):"
ls -lh "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l | xargs echo "  Total PDF files:"
echo ""

echo "Generated files (*.csv):"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l | xargs echo "  Total CSV files:"
echo ""

echo "✅ All done! Check the output directory for results."
