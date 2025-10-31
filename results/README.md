# Results Directory

All KLSBench evaluation results, figures, and tables.

## Structure

```
results/
├── raw_evaluation/     # Raw evaluation results (16 files)
├── aggregated/         # Aggregated analysis (7 files)
├── figures/           # All publication figures (37 files)
│   ├── appendix_a/    # Task examples
│   ├── appendix_b/    # Statistics
│   ├── detailed/      # Detailed analysis
│   ├── radar/         # Performance radar charts
│   └── legacy/        # Legacy visualizations
├── tables/            # All CSV tables (13 files)
│   ├── examples/      # Task examples
│   ├── statistics/    # Stats summaries
│   └── performance/   # Performance metrics
└── data_processing/   # Data processing outputs (14 files)
```

**Total:** 82 organized files

## Top Results

| Rank | Model | Average |
|------|-------|---------|
| 1 | GPT-3.5-turbo | 0.540 |
| 2 | GPT-4-turbo | 0.517 |
| 3 | Claude-3-opus | 0.404 |

## Quick Access

**Best figure for paper:**
```
figures/radar/radar_top5_models.png
```

**Complete results:**
```
aggregated/aggregated_summary.csv
tables/performance/radar_performance_summary.csv
```

## Complete Documentation

**📘 See: `COMPLETE_GUIDE.md`**

This comprehensive guide includes:
- Full directory structure
- All figure descriptions  
- Script usage instructions
- LaTeX integration
- Troubleshooting

## Regenerate

```bash
cd /Users/songhune/Workspace/korean_eda/notebook/experiments/exp7
bash run_exp7.sh
```
