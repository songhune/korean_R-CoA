# KLSBench Complete Guide

**Last Updated:** 2025-10-31
**Version:** 2.0 (Post-Reorganization)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Directory Structure](#directory-structure)
3. [Results Overview](#results-overview)
4. [Appendix Materials](#appendix-materials)
5. [Radar Charts](#radar-charts)
6. [Configuration](#configuration)
7. [Script Usage](#script-usage)
8. [File Inventory](#file-inventory)
9. [Paper Integration](#paper-integration)
10. [Troubleshooting](#troubleshooting)

---

## 1. Quick Start

### Generate All Materials

```bash
cd /Users/songhune/Workspace/korean_eda/notebook/experiments/exp7
bash run_exp7.sh
```

This generates:
- ✅ Appendix A: Task examples (5 figures + 5 tables)
- ✅ Appendix B: Statistical analysis (4 figures)
- ✅ Detailed analysis (4 figures + 4 tables)
- ✅ Radar charts (11 figures + 1 table)

### Key Result Files

**Top performers table:**
```
/Users/songhune/Workspace/korean_eda/results/tables/performance/radar_performance_summary.csv
```

**All figures:**
```
/Users/songhune/Workspace/korean_eda/results/figures/
```

**Main configuration:**
```
/Users/songhune/Workspace/korean_eda/notebook/experiments/config.yaml
```

---

## 2. Directory Structure

```
korean_eda/
├── benchmark/
│   └── kls_bench/              # Benchmark data (CSV + JSON)
│
├── results/                    # ⭐ ALL OUTPUTS HERE
│   ├── raw_evaluation/        # Raw model evaluation results
│   ├── aggregated/            # Aggregated analysis
│   ├── figures/               # All publication figures
│   │   ├── appendix_a/       # Task examples & distributions
│   │   ├── appendix_b/       # Per-class & error analysis
│   │   ├── detailed/         # Detailed statistics
│   │   ├── radar/            # Performance radar charts
│   │   └── legacy/           # Legacy visualizations
│   ├── tables/               # All CSV tables
│   │   ├── examples/         # Task example data
│   │   ├── statistics/       # Statistical summaries
│   │   └── performance/      # Performance metrics
│   └── data_processing/      # Data processing outputs
│
└── notebook/
    └── experiments/
        ├── config.yaml        # Main configuration
        ├── exp5/             # Evaluation scripts
        ├── exp6/             # Aggregation scripts
        ├── exp7/             # Appendix generation
        │   ├── exp7_generate_appendix.py
        │   ├── exp7_detailed_analysis.py
        │   ├── exp7_radar_charts.py
        │   └── run_exp7.sh
        └── utils/            # Utility functions
```

**Total Files:** 82 organized files

---

## 3. Results Overview

### Performance Summary

| Rank | Model | Average | Classification | Retrieval | Punctuation | NLI | Translation |
|------|-------|---------|----------------|-----------|-------------|-----|-------------|
| 1 | GPT-3.5-turbo | **0.540** | 0.00 | 0.80 | **0.97** | **0.70** | 0.23 |
| 2 | GPT-4-turbo | **0.517** | 0.00 | **0.90** | 0.81 | 0.60 | **0.27** |
| 3 | Claude-3-opus | 0.404 | 0.00 | **0.90** | 0.65 | 0.30 | 0.17 |
| 4 | Qwen2.5-7B | 0.394 | 0.00 | **1.00** | 0.87 | 0.00 | 0.10 |
| 5 | Llama-3.1-8B | 0.337 | **0.20** | 0.40 | 0.92 | 0.00 | 0.16 |
| 6 | Claude-3.5-sonnet | 0.306 | **0.10** | **0.90** | 0.39 | 0.00 | 0.14 |
| 7 | EXAONE-3.0 | 0.291 | 0.00 | **0.90** | 0.36 | 0.00 | **0.20** |

### Task Difficulty

**Ranked by Average Performance (Higher = Easier):**

1. **Retrieval** (0.81) - Source text identification
2. **Punctuation** (0.70) - Punctuation restoration
3. **NLI** (0.23) - Natural language inference
4. **Translation** (0.18) - Cross-lingual translation
5. **Classification** (0.04) - Genre classification ⚠️ Hardest

### Model Type Comparison

| Type | Average | Best Task | Worst Task |
|------|---------|-----------|------------|
| **API Models** | 0.42 | Retrieval (0.88) | Classification (0.02) |
| **Open-Source** | 0.34 | Punctuation (0.72) | Classification (0.07) |

**Key Insight:** API models show +0.32 advantage in NLI tasks.

---

## 4. Appendix Materials

### Appendix A: Task Examples

**Purpose:** Provide representative examples from each benchmark task.

#### A.1 Classification Task
- **Figure:** `figures/appendix_a/genre_distribution.png`
- **Table:** `tables/examples/classification_examples.csv`
- **Content:** Distribution of 21 literary genres (賦, 詩, 疑, 義, 策, etc.)
- **Example Genres:**
  - 賦 (Bu): Rhyme-prose
  - 詩 (Si): Regulated poetry
  - 疑 (Eui): Essays on doubtful points
  - 義 (Ui): Argumentative essays
  - 策 (Chaek): Policy proposals

#### A.2 Retrieval Task
- **Figure:** `figures/appendix_a/book_distribution.png`
- **Table:** `tables/examples/retrieval_examples.csv`
- **Content:** Distribution across Four Books (論語, 孟子, 大學, 中庸)
- **Total Examples:** 1,209

#### A.3 Punctuation Task
- **Figure:** `figures/appendix_a/language_distribution.png`
- **Table:** `tables/examples/punctuation_examples.csv`
- **Content:** Before/After punctuation restoration examples
- **Languages:** Korean, Literary Sinitic
- **Total Examples:** 2,000

#### A.4 NLI Task
- **Figure:** `figures/appendix_a/nli_label_distribution.png`
- **Table:** `tables/examples/nli_examples.csv`
- **Content:** Entailment/Neutral/Contradiction examples
- **Total Examples:** 1,854

#### A.5 Translation Task
- **Figure:** `figures/appendix_a/translation_pairs.png`
- **Table:** `tables/examples/translation_examples.csv`
- **Content:** Literary Sinitic→Korean, Korean→English
- **Total Examples:** 2,000

### Appendix B: Detailed Statistics

#### B.1 Per-Class Performance
- **Classification:** `figures/appendix_b/appendix_b1_classification_per_genre.png`
  - Performance breakdown across 21 genres
- **Retrieval:** `figures/appendix_b/appendix_b1_retrieval_per_book.png`
  - Performance breakdown by source book

#### B.2 Error Analysis
- **Error Patterns:** `figures/appendix_b/appendix_b2_error_patterns.png`
  - Common error types: Format Error, Wrong Label, Partial Match, Hallucination, No Response
- **Error Rates:** `figures/appendix_b/appendix_b2_error_rate_heatmap.png`
  - Task-wise error rate heatmap across models

---

## 5. Radar Charts

### Available Radar Charts

#### 5.1 Overview Charts

**All Models Comparison**
- File: `figures/radar/radar_all_models.png`
- Shows: All 7 models overlaid
- Use: Overview of performance landscape

**Model Type Comparison**
- File: `figures/radar/radar_model_type_comparison.png`
- Shows: API vs Open-Source average
- Use: Highlight architectural differences

**Top 5 Models**
- File: `figures/radar/radar_top5_models.png`
- Shows: Best performers with scores
- Use: Main paper figure

**Small Multiples Grid**
- File: `figures/radar/radar_small_multiples_grid.png`
- Shows: All models in compact grid
- Use: Supplementary materials

#### 5.2 Individual Model Cards (7 files)

Each model has a dedicated card with performance annotations:
- `radar_individual_gpt-3.5-turbo.png`
- `radar_individual_gpt-4-turbo.png`
- `radar_individual_claude-3-5-sonnet-20241022.png`
- `radar_individual_claude-3-opus-20240229.png`
- `radar_individual_Qwen_Qwen2.5-7B-Instruct.png`
- `radar_individual_meta-llama_Llama-3.1-8B-Instruct.png`
- `radar_individual_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct.png`

**Color Scheme:**
- 🟢 Green (0.6-1.0): High performance
- 🟡 Yellow (0.4-0.6): Medium performance
- 🔴 Red (0.0-0.4): Low performance

#### 5.3 Performance Summary Table

**File:** `tables/performance/radar_performance_summary.csv`

Complete performance breakdown with average scores and per-task metrics.

### Reading Radar Charts

**Interpretation Guide:**

1. **Larger area = Better overall performance**
2. **Shape patterns:**
   - ⭐ Star: Excellent in one task, weak in others
   - ⬡ Pentagon: Balanced performance
   - ⊙ Compressed: Universally low

3. **Key patterns:**
   - **Classification gap:** All models struggle (<0.2)
   - **Retrieval cluster:** Most models excel (0.8-1.0)
   - **API advantage:** Only API models solve NLI
   - **Punctuation variance:** Widest spread (0.36-0.97)

---

## 6. Configuration

### Main Configuration File

**Location:** `notebook/experiments/config.yaml`

```yaml
# Benchmark paths
benchmark:
  full: "../../benchmark/kls_bench/kls_bench_full.json"
  classification: "../../benchmark/kls_bench/kls_bench_classification.json"
  retrieval: "../../benchmark/kls_bench/kls_bench_retrieval.json"
  punctuation: "../../benchmark/kls_bench/kls_bench_punctuation.json"
  nli: "../../benchmark/kls_bench/kls_bench_nli.json"
  translation: "../../benchmark/kls_bench/kls_bench_translation.json"

# Output directories (UPDATED)
output:
  base: "../../results/raw_evaluation"
  fewshot: "../../results/fewshot"
  aggregated: "../../results/aggregated"
  figures: "../../results/figures"
  tables: "../../results/tables"

# Models
models:
  api:
    openai: [gpt-4-turbo, gpt-3.5-turbo]
    anthropic: [claude-3-5-sonnet-20241022, claude-3-opus-20240229]
  opensource:
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    - LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

# Task information
tasks:
  classification: {total_items: 808, metric: "Accuracy"}
  retrieval: {total_items: 1209, metric: "Accuracy"}
  punctuation: {total_items: 2000, metric: "F1 Score"}
  nli: {total_items: 1854, metric: "Accuracy"}
  translation: {total_items: 2000, metric: "BLEU Score"}
```

### Font Configuration

**Location:** `notebook/experiments/utils/font_fix.py`

**Strategy:** English-first with Korean fallback

```python
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'AppleGothic']
```

**macOS fonts:** AppleGothic, Apple SD Gothic Neo
**Windows fonts:** Malgun Gothic, Gulim
**Linux fonts:** NanumGothic, Noto Sans CJK KR

---

## 7. Script Usage

### Evaluation Pipeline

```bash
# 1. Run evaluation (exp5)
cd notebook/experiments/exp5
python3 exp5_benchmark_evaluation.py --mode full

# 2. Aggregate results (exp6)
cd ../exp6
python3 exp6_result_aggregation.py \
    --results-dir ../../results/raw_evaluation \
    --output-dir ../../results/aggregated

# 3. Generate appendix (exp7)
cd ../exp7
bash run_exp7.sh
```

### Individual Generation

**Appendix A & B:**
```bash
python3 exp7_generate_appendix.py
# Output: figures/appendix_a/, figures/appendix_b/, tables/examples/
```

**Detailed Analysis:**
```bash
python3 exp7_detailed_analysis.py
# Output: figures/detailed/, tables/statistics/
```

**Radar Charts:**
```bash
python3 exp7_radar_charts.py
# Output: figures/radar/, tables/performance/
```

### Custom Options

**Specify output directory:**
```bash
python3 exp7_radar_charts.py --output-dir /custom/path
```

**Specify results CSV:**
```bash
python3 exp7_radar_charts.py \
    --results-csv /path/to/aggregated_summary.csv
```

---

## 8. File Inventory

### Raw Evaluation Results (16 files)

Latest results per model:
```
results/raw_evaluation/
├── results_gpt-4-turbo_20251023_163232.json
├── results_gpt-3.5-turbo_20251023_163408.json
├── results_claude-3-5-sonnet-20241022_20251023_163823.json
├── results_claude-3-opus-20240229_20251023_164331.json
├── results_Qwen_Qwen2.5-7B-Instruct_20251023_164625.json
├── results_meta-llama_Llama-3.1-8B-Instruct_20251023_164456.json
├── results_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct_20251023_164942.json
└── summary_*.csv (corresponding summaries)
```

### Figures (37 files)

- **Appendix A:** 5 PNG files
- **Appendix B:** 4 PNG files
- **Detailed:** 4 PNG files
- **Radar:** 11 PNG files
- **Legacy:** 8 PNG files (data processing visualizations)

### Tables (13 files)

- **Examples:** 5 CSV files (task examples)
- **Statistics:** 4 CSV files (detailed stats)
- **Performance:** 4 CSV files (performance metrics)

### Aggregated (7 files)

- `aggregated_summary.csv` - All results combined
- `aggregated_pivot.csv` - Model × Task matrix
- `model_average_performance.csv` - Average scores
- 4 PNG files - Legacy aggregation charts

---

## 9. Paper Integration

### Recommended Main Paper Figures

**Figure 1: Model Performance Overview**
```latex
\includegraphics[width=0.8\textwidth]{figures/radar/radar_top5_models.png}
```
Caption: Performance comparison of top 5 models across all KLSBench tasks.

**Figure 2: Task Difficulty Analysis**
```latex
\includegraphics[width=\textwidth]{figures/detailed/detailed_task_difficulty_analysis.png}
```
Caption: Task difficulty comparison showing average performance with error bars.

**Figure 3: Model Type Comparison**
```latex
\includegraphics[width=0.7\textwidth]{figures/radar/radar_model_type_comparison.png}
```
Caption: API models vs Open-source models performance comparison.

### Appendix Organization

**Appendix A: Task Examples**
- Include all 5 distribution figures from `figures/appendix_a/`
- Include corresponding example tables from `tables/examples/`

**Appendix B: Detailed Statistics**
- Include 4 analysis figures from `figures/appendix_b/`
- Include statistical tables from `tables/statistics/`

**Appendix C: Individual Model Profiles (Optional)**
- Include 7 individual radar charts from `figures/radar/radar_individual_*.png`

### LaTeX Templates

**Single Figure:**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/radar/radar_top5_models.png}
    \caption{Top 5 model performance across all tasks.}
    \label{fig:radar_top5}
\end{figure}
```

**Side-by-Side:**
```latex
\begin{figure*}[t]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/radar/radar_model_type_comparison.png}
        \caption{Model type comparison}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/detailed/detailed_task_difficulty_analysis.png}
        \caption{Task difficulty}
    \end{subfigure}
    \caption{Performance analysis}
\end{figure*}
```

**Table from CSV:**
```latex
\begin{table}[h]
\centering
\caption{Model Performance Summary}
\label{tab:performance}
\csvautotabular{tables/performance/radar_performance_summary.csv}
\end{table}
```

### Citation Example

```
As shown in Figure 1, GPT-3.5-turbo achieves the highest average score
(0.540), particularly excelling in punctuation restoration (0.97) and
NLI (0.70). However, classification remains the most challenging task,
with all models achieving near-zero accuracy, highlighting the difficulty
of distinguishing subtle genre differences in classical Korean literature.
```

---

## 10. Troubleshooting

### Common Issues

#### Issue 1: Font Rendering Problems

**Problem:** Korean/Chinese characters show as boxes

**Solution:**
```python
# The scripts automatically handle this
from utils.font_fix import setup_korean_fonts_robust
setup_korean_fonts_robust()
```

All figures are pre-rendered with proper fonts, so no action needed for paper submission.

#### Issue 2: Path Not Found Errors

**Problem:** `FileNotFoundError` when running scripts

**Solution:**
```bash
# Verify you're in the correct directory
cd /Users/songhune/Workspace/korean_eda/notebook/experiments/exp7

# Or run the reorganization script
cd ..
python3 reorganize_all.py
```

#### Issue 3: Missing Dependencies

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
pip install pandas numpy matplotlib seaborn pyyaml --break-system-packages
```

#### Issue 4: Empty CSV Files

**Problem:** Some example CSVs are empty (e.g., translation_examples.csv is 4 bytes)

**Solution:**
```bash
# Regenerate with updated selection criteria
python3 exp7_generate_appendix.py
```

Check the benchmark data to ensure matching selection criteria.

#### Issue 5: Outdated Results

**Problem:** Figures don't match latest evaluation results

**Solution:**
```bash
# Re-aggregate results
python3 exp6/exp6_result_aggregation.py

# Regenerate all figures
cd exp7 && bash run_exp7.sh
```

### Getting Help

**Check Documentation:**
1. This file: `results/COMPLETE_GUIDE.md`
2. Config: `notebook/experiments/config.yaml`
3. Script headers: Each Python file has detailed docstrings

**Verify Setup:**
```bash
# Run verification
cd notebook/experiments
python3 reorganize_all.py
```

**Common Commands:**
```bash
# List all generated figures
ls -lh results/figures/*/*.png

# Count files by category
find results -type f -name "*.png" | wc -l
find results -type f -name "*.csv" | wc -l

# Check latest results
ls -lt results/raw_evaluation/*.json | head -1
```

---

## Appendix: Change Log

### Version 2.0 (2025-10-31) - Major Reorganization

**Changes:**
- ✅ Consolidated 3 scattered directories into unified `results/` structure
- ✅ Organized 82 files by purpose (figures, tables, raw data)
- ✅ Updated all script paths to new structure
- ✅ Removed duplicate evaluation results (kept latest per model)
- ✅ Created comprehensive documentation

**Migration:**
- `/benchmark/results/` → `/results/raw_evaluation/`
- `/notebook/experiments/graphs/` → `/results/figures/`
- Mixed `/results/` → Organized by category

**Scripts Updated:**
- `config.yaml` - All output paths
- `exp6/*.py` - Aggregation scripts
- `exp7/*.py` - Appendix generation scripts
- `exp7/run_exp7.sh` - Main execution script

### Version 1.0 (2025-10-20) - Initial Release

- Initial benchmark evaluation
- Basic result aggregation
- Preliminary visualizations

---

## Quick Reference

### Most Important Files

| Purpose | File |
|---------|------|
| **Configuration** | `notebook/experiments/config.yaml` |
| **Top Models** | `tables/performance/radar_performance_summary.csv` |
| **Best Figure** | `figures/radar/radar_top5_models.png` |
| **Complete Results** | `aggregated/aggregated_summary.csv` |
| **Main Script** | `exp7/run_exp7.sh` |

### Most Important Commands

```bash
# Generate everything
cd notebook/experiments/exp7 && bash run_exp7.sh

# Check structure
python3 notebook/experiments/reorganize_all.py

# View results
ls -lR results/figures/
```

### Contact

For issues or questions:
- Configuration: Check `config.yaml`
- Scripts: See script docstrings
- Structure: Run `reorganize_all.py`

---

**End of Complete Guide**

*This guide consolidates all documentation from README.md, APPENDIX_GUIDE.md, RADAR_CHARTS_GUIDE.md, GENERATED_FILES.md, and REORGANIZATION_PLAN.md into a single comprehensive reference.*
