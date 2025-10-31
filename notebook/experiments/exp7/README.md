# Experiment 7: KLSBench Appendix Generation

Generate publication-ready appendix materials for the KLSBench paper.

## Quick Start

```bash
# Generate all materials
bash run_exp7.sh
```

## What Gets Generated

**Appendix A: Task Examples** (5 figures + 5 tables)
- Genre distribution, source books, language splits, etc.

**Appendix B: Statistics** (4 figures)
- Per-class performance, error analysis

**Detailed Analysis** (4 figures + 4 tables)
- Task difficulty, model consistency

**Radar Charts** (11 figures + 1 table)
- All models, top 5, individual profiles

## Individual Scripts

```bash
# Appendix A & B
python3 exp7_generate_appendix.py

# Detailed statistics
python3 exp7_detailed_analysis.py

# Radar performance charts
python3 exp7_radar_charts.py
```

## Output Location

All outputs go to: `/Users/songhune/Workspace/korean_eda/results/`

## Complete Documentation

**See:** `/Users/songhune/Workspace/korean_eda/results/COMPLETE_GUIDE.md`

This comprehensive guide includes everything you need.
