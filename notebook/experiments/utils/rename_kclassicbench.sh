#!/bin/bash

# K-ClassicBench를 KLSBench로 일괄 변경
# KLS = Korean Literary Style (또는 Korean Literature & Style)

echo "Renaming K-ClassicBench to KLSBench..."

# Python files
sed -i '' 's/K-ClassicBench/KLSBench/g' exp5_benchmark_evaluation.py
sed -i '' 's/KClassicBench/KLSBench/g' exp5_benchmark_evaluation.py
sed -i '' 's/k_classic_bench/kls_bench/g' exp5_benchmark_evaluation.py

sed -i '' 's/K-ClassicBench/KLSBench/g' k_classic_bench_generator.py
sed -i '' 's/KClassicBench/KLSBench/g' k_classic_bench_generator.py
sed -i '' 's/k_classic_bench/kls_bench/g' k_classic_bench_generator.py

# Shell scripts
sed -i '' 's/K-ClassicBench/KLSBench/g' run_all_evaluations.sh
sed -i '' 's/K-ClassicBench/KLSBench/g' setup_exp5.sh

# Markdown files
sed -i '' 's/K-ClassicBench/KLSBench/g' README_exp5.md
sed -i '' 's/K-ClassicBench/KLSBench/g' SUMMARY_exp5.md
sed -i '' 's/K-ClassicBench/KLSBench/g' SAMPLING_GUIDE.md
sed -i '' 's/k_classic_bench/kls_bench/g' README_exp5.md
sed -i '' 's/k_classic_bench/kls_bench/g' SUMMARY_exp5.md
sed -i '' 's/k_classic_bench/kls_bench/g' SAMPLING_GUIDE.md

echo "Done!"
