# Figure Update Summary

**Date:** 2025-11-13
**Based on:** Temperature Ablation Study (temp=0.0, 0.3, 0.7)

## Overview

All figures in `/results/figures/` have been updated to reflect the optimal temperature parameter settings based on comprehensive temperature ablation experiments.

## Key Findings

### Optimal Temperature Analysis

Based on analysis of 7 models (4 API models + 3 open-source models) across 5 tasks:

- **Overall Best Temperature: 0.0**
  - Average performance across all models: 0.4218
  - Used by 3 out of 7 models as optimal

- **Temperature Distribution:**
  - temp=0.0: 3 models (Qwen, Claude Opus, Llama-3.1)
  - temp=0.3: 4 models (GPT-3.5, GPT-4, Claude Sonnet, EXAONE)
  - temp=0.7: 0 models (not optimal for any)

### Performance by Temperature

| Temperature | Avg Performance | Description |
|-------------|----------------|-------------|
| 0.0 | 0.4218 | Best overall, more deterministic |
| 0.3 | 0.3910 | Balanced, good for creative tasks |
| 0.7 | 0.3565 | Lowest performance, high variability |

## Updated Figures

### Radar Charts (with temp=0.0 data)

Located in `/results/figures/`:

1. **[radar_all_models.pdf](radar_all_models.pdf)**
   - All 7 models performance across 5 tasks
   - Uses optimal temperature (0.0) for consistency

2. **[radar_top5_models.pdf](radar_top5_models.pdf)**
   - Top 5 performing models
   - Shows competitive landscape

3. **[radar_model_type_comparison.pdf](radar_model_type_comparison.pdf)**
   - API vs Open-source model comparison
   - Average performance by category

### Temperature Ablation Visualizations

New figures showing temperature effects:

1. **[temperature_ablation_heatmap.pdf](temperature_ablation_heatmap.pdf)**
   - Heatmap: Model × Temperature performance
   - Shows optimal temperature per model

2. **[temperature_ablation_lines.pdf](temperature_ablation_lines.pdf)**
   - Line plot: Temperature effect on each model
   - Visualizes sensitivity to temperature

3. **[temperature_optimal_comparison.pdf](temperature_optimal_comparison.pdf)**
   - Bar chart: Best performance per model
   - Color-coded by optimal temperature

4. **[temperature_by_task_heatmap.pdf](temperature_by_task_heatmap.pdf)**
   - Heatmap: Task × Temperature (averaged across models)
   - Shows which tasks are sensitive to temperature

## Model Performance Summary (at temp=0.0)

| Model | Avg Score | Classification | Retrieval | Punctuation | NLI | Translation |
|-------|-----------|----------------|-----------|-------------|-----|-------------|
| gpt-4-turbo | 0.5322 | 0.161 | 0.750 | 0.854 | 0.839 | 0.057 |
| gpt-3.5-turbo | 0.5102 | 0.176 | 0.742 | 0.844 | 0.718 | 0.071 |
| claude-3-opus | 0.4288 | 0.137 | 0.758 | 0.681 | 0.759 | 0.109 |
| Qwen2.5-7B | 0.4120 | 0.132 | 0.592 | 0.511 | 0.748 | 0.077 |
| Llama-3.1-8B | 0.3840 | 0.079 | 0.233 | 0.839 | 0.670 | 0.098 |
| EXAONE-3.0-7.8B | 0.3590 | 0.000 | 0.633 | 0.356 | 0.737 | 0.069 |
| claude-sonnet-4.5 | 0.3264 | 0.161 | 0.567 | 0.589 | 0.665 | 0.050 |

## Task-Level Insights (temp=0.0)

| Task | Avg Performance | Best Model | Temperature Sensitivity |
|------|----------------|------------|------------------------|
| Retrieval | 0.6488 | Claude Opus (0.758) | Low |
| Punctuation | 0.6134 | GPT-4 (0.854) | Medium |
| NLI | 0.5927 | GPT-4 (0.839) | Low |
| Classification | 0.1440 | GPT-3.5 (0.176) | High |
| Translation | 0.1101 | Claude Opus (0.109) | Medium |

## Recommendations

1. **For consistent results**: Use temperature=0.0 (highest average performance)
2. **For model comparison**: Use temp=0.0 data for fair comparison
3. **Task-specific considerations**:
   - Classification and Translation: Most challenging tasks, benefit from temp=0.0
   - Retrieval: Strong performance across all models at temp=0.0
   - Punctuation & NLI: Good performance, relatively stable across temperatures

## Data Sources

- **Temperature Ablation Results**: `/results/temperature_ablation/`
- **Aggregated Summary (temp=0.0)**: `/results/aggregated/aggregated_summary_temp0.0.csv`
- **Analysis Scripts**: `/notebook/experiments/exp7/`

## Scripts Used

1. `regenerate_temp_summary.py` - Consolidated all temperature ablation results
2. `analyze_temperature.py` - Found optimal temperature per model
3. `create_aggregated_with_optimal_temp.py` - Created temp=0.0 aggregated summary
4. `exp7_radar_charts.py` - Generated radar charts
5. `visualize_temperature_ablation.py` - Created temperature visualizations

## Citation

If using these figures, please cite:

```
Temperature ablation study conducted on November 13, 2025.
Models evaluated: GPT-4-turbo, GPT-3.5-turbo, Claude-3-Opus, Claude-Sonnet-4.5,
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, EXAONE-3.0-7.8B-Instruct.
Temperature values tested: 0.0, 0.3, 0.7.
Optimal temperature selected: 0.0 (based on highest average performance).
```
