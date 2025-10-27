# Experiment 6: Few-shot Evaluation - Implementation Summary

## Status: ✅ COMPLETE - Ready to Execute

**Created**: 2025-10-27
**Estimated Runtime**: 1-2 days
**Purpose**: Address zero-shot evaluation limitation by testing few-shot learning

---

## What Was Created

### 1. Main Evaluation Script
**File**: `exp6_fewshot_evaluation.py` (407 lines)

**Features**:
- `FewShotEvaluator` class with complete evaluation pipeline
- Balanced exemplar selection across labels
- Task-specific prompt formatting (Classification + NLI)
- Integration with exp5 model wrappers
- Automatic results saving (JSON + CSV)

**Key Methods**:
- `select_exemplars()` - Balanced sampling across labels
- `format_classification_fewshot()` - Classification prompts with examples
- `format_nli_fewshot()` - NLI prompts with examples
- `evaluate_model_fewshot()` - Run evaluation with metrics
- `run_fewshot_experiment()` - Complete experiment pipeline

### 2. Automated Runner
**File**: `exp6_run_fewshot.sh` (executable)

**Features**:
- Automated execution for 3 open-source models (Llama, Qwen, EXAONE)
- Optional API model support (Claude 3.5, Claude 3 Opus - commented out)
- Comprehensive logging with timestamps
- Automatic result aggregation
- Progress tracking and error handling

**Usage**:
```bash
./exp6_run_fewshot.sh
```

### 3. Analysis Script
**File**: `exp6_analyze_improvements.py` (379 lines)

**Features**:
- Comparison with zero-shot baseline (exp5)
- Improvement delta calculation (Δ1-shot, Δ3-shot)
- Statistical summaries
- Three visualization types:
  - Bar charts: 0-shot vs 1-shot vs 3-shot comparison
  - Heatmaps: Improvement deltas by model/task
  - Learning curves: Performance progression

**Usage**:
```bash
python exp6_analyze_improvements.py
```

### 4. Documentation
- **EXP6_README.md**: Complete experiment documentation (300+ lines)
  - Research questions and rationale
  - Target models and experimental design
  - Usage instructions and troubleshooting
  - Expected outputs and interpretation guide
  - Integration with paper

- **EXP6_INTEGRATION_NOTES.md**: Paper integration guide
  - LaTeX text suggestions for Section V
  - Figure/table recommendations
  - Possible findings and interpretations
  - Decision points and timeline estimates

- **EXP6_SUMMARY.md**: This file

---

## Target Models

### Priority Tier 1 (Open-Source - Default)
✅ **Llama 3.1 8B** - NLI 0%, Classification 20%
✅ **Qwen 2.5 7B** - Both 0%
✅ **EXAONE 3.0 7.8B** - Both 0%

### Priority Tier 2 (API - Optional)
⏳ **Claude 3.5 Sonnet** - NLI 0%, Classification 10%
⏳ **Claude 3 Opus** - NLI 30%, Classification 0%

---

## Experimental Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tasks | Classification + NLI | Worst zero-shot performance |
| Shot numbers | 1, 3 | Standard few-shot configs |
| Sample size | 50 per task | Fast evaluation, statistically valid |
| Exemplar selection | Balanced across labels | Fair representation |
| Test/exemplar split | No overlap | Prevent data leakage |
| Random seed | 42 | Reproducibility |

---

## Expected Outputs

### Directory Structure
```
results/fewshot/
├── fewshot_meta-llama_Llama-3.1-8B-Instruct_20251027_HHMMSS.json
├── summary_meta-llama_Llama-3.1-8B-Instruct_20251027_HHMMSS.csv
├── fewshot_Qwen_Qwen2.5-7B-Instruct_20251027_HHMMSS.json
├── summary_Qwen_Qwen2.5-7B-Instruct_20251027_HHMMSS.csv
├── fewshot_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct_20251027_HHMMSS.json
├── summary_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct_20251027_HHMMSS.csv
├── exp6_fewshot_summary_combined.csv
├── exp6_run_20251027_HHMMSS.log
└── analysis/
    ├── fewshot_vs_zeroshot_comparison.csv
    ├── fewshot_vs_zeroshot_comparison.json
    ├── fewshot_comparison.png
    ├── improvement_heatmap.png
    └── learning_curves.png
```

### Key Files
1. **Per-model JSON**: Detailed predictions with exemplars
2. **Per-model CSV**: Summary metrics (accuracy, F1, sample counts)
3. **Combined CSV**: All models in single table
4. **Comparison CSV**: Delta calculations (improvement over zero-shot)
5. **Visualizations**: 3 PNG files for paper integration
6. **Log file**: Complete execution trace

---

## How to Execute

### Step 1: Run Experiments (1-2 days)

```bash
cd /home/work/songhune/korean_R-CoA/notebook/experiments

# Run open-source models (default)
./exp6_run_fewshot.sh

# Monitor progress
tail -f ../../results/fewshot/exp6_run_*.log
```

### Step 2: Analyze Results (2 hours)

```bash
# Generate comparison with zero-shot baseline
python exp6_analyze_improvements.py

# Check analysis outputs
ls -lh ../../results/fewshot/analysis/
```

### Step 3: Review Results

```bash
# View combined summary
cat ../../results/fewshot/exp6_fewshot_summary_combined.csv

# View improvement comparison
cat ../../results/fewshot/analysis/fewshot_vs_zeroshot_comparison.csv
```

### Step 4: Integrate into Paper

See `EXP6_INTEGRATION_NOTES.md` for LaTeX text and figure recommendations.

---

## Expected Timeline

### Fast Track (Open-Source Only)
| Phase | Time | Notes |
|-------|------|-------|
| Llama 3.1 8B | 2-3h | H100 local inference |
| Qwen 2.5 7B | 2-3h | H100 local inference |
| EXAONE 3.0 7.8B | 2-3h | H100 local inference |
| Analysis | 2h | Visualization + comparison |
| **Total** | **8-11h** | Can run overnight |

### Full Evaluation (With API Models)
| Phase | Time | Notes |
|-------|------|-------|
| Open-source (above) | 8-11h | Parallel possible |
| Claude 3.5 Sonnet | 3-4h | API rate limited |
| Claude 3 Opus | 3-4h | API rate limited |
| Analysis | 3h | More comprehensive |
| **Total** | **14-19h** | ~2 days sequential |

---

## Integration with Paper (Section V)

### Recommended Addition

Add new subsection **V-F: Few-shot Learning Analysis** after V-E (Model Comparison):

```latex
\subsection{Few-shot Learning Analysis}

To investigate whether zero-shot failures stem from task comprehension
or fundamental capability limitations, we conduct few-shot evaluation
(1-shot and 3-shot) for models that achieved near-zero accuracy.

\textbf{Motivation.} [Explain 0% results in Classification/NLI]

\textbf{Setup.} [Describe experimental configuration]

\textbf{Results.} [INSERT FINDINGS]

\textbf{Interpretation.} [DISCUSS IMPLICATIONS]
```

### Figures to Include

**Option 1 (Recommended)**: Single combined figure
- `fewshot_comparison.png` - Side-by-side bar charts
- Caption: "Classification and NLI performance comparison across 0-shot, 1-shot, and 3-shot configurations"

**Option 2 (Comprehensive)**: Two figures
- `fewshot_comparison.png` - Bar charts
- `learning_curves.png` - Line plots showing progression

**Option 3 (Minimal)**: Table only
- Numerical comparison table with Δ improvements

---

## Success Criteria

### Minimum Viable
- ✅ Implementation complete
- ⏳ Execution for 3 open-source models
- ⏳ Analysis with visualizations
- ⏳ At least one figure for paper

### Complete
- ✅ Implementation complete
- ⏳ All 5 target models evaluated
- ⏳ Comprehensive analysis with statistical tests
- ⏳ Integration into paper (Section V + Discussion)

---

## Verification Checklist

- ✅ Python scripts compile without errors
- ✅ Shell script is executable (chmod +x)
- ✅ Imports from exp5 verified (model wrappers available)
- ✅ Documentation complete (README + integration notes)
- ✅ Output directories configured correctly
- ⏳ Benchmark file exists (`k_classic_bench_full.json`)
- ⏳ Zero-shot results available for comparison (`aggregated_results.json`)
- ⏳ GPU available for open-source models (H100)

---

## Potential Issues and Solutions

### Issue 1: Missing Benchmark File
**Error**: `FileNotFoundError: benchmark file not found`
**Solution**: Verify path in script (line 371) or pass via `--benchmark` arg

### Issue 2: Missing Zero-shot Baseline
**Error**: `aggregated_results.json not found`
**Solution**: Run exp5 first or manually create from individual results

### Issue 3: CUDA OOM
**Error**: `RuntimeError: CUDA out of memory`
**Solution**: Reduce batch size or evaluate models sequentially

### Issue 4: API Rate Limits (Claude)
**Error**: `429 Too Many Requests`
**Solution**: Increase sleep time (line 233: `time.sleep(0.5)` → `time.sleep(1.0)`)

---

## Next Steps

1. **Decide on scope**: Open-source only (faster) vs All models (complete)
2. **Execute experiments**: Run `./exp6_run_fewshot.sh`
3. **Monitor progress**: Check log files and GPU usage
4. **Analyze results**: Run `exp6_analyze_improvements.py`
5. **Review findings**: Interpret improvements and select key insights
6. **Integrate into paper**: Add subsection to Section V with figure
7. **Update Discussion**: Mention implications in Section VI

---

## Key Questions for User

1. **Priority**: Should this be executed immediately or after other paper sections?
2. **Scope**: Open-source only (1 day) or include API models (2 days)?
3. **Sample size**: 50 instances (faster) or full dataset (more robust)?
4. **Integration**: Full subsection in Results or brief mention in Discussion?

---

## Files Created

```
korean_R-CoA/notebook/experiments/
├── exp6_fewshot_evaluation.py      # Main evaluation (407 lines)
├── exp6_run_fewshot.sh             # Automated runner (executable)
├── exp6_analyze_improvements.py    # Analysis script (379 lines)
├── EXP6_README.md                  # Complete documentation
└── EXP6_SUMMARY.md                 # This file

IEEE_for_journals_template_with_bibtex_example_files_included/
└── EXP6_INTEGRATION_NOTES.md       # Paper integration guide
```

---

## Summary

✅ **Experiment 6 is fully implemented and ready to execute.**

The complete pipeline includes:
- Few-shot evaluation for 5 target models
- Automated execution and logging
- Analysis with zero-shot comparison
- Visualization for paper integration
- Comprehensive documentation

**Estimated time**: 1-2 days for execution + analysis
**Expected outcome**: Quantify whether 0% results stem from task ambiguity or capability gaps
**Paper impact**: Addresses potential reviewer concern about zero-shot-only evaluation

---

**To begin execution**: `./exp6_run_fewshot.sh`
**For questions**: See `EXP6_README.md`
**For paper integration**: See `EXP6_INTEGRATION_NOTES.md`
