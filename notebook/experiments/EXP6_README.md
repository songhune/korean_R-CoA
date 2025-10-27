# Experiment 6: Few-shot Evaluation

## Overview

This experiment addresses the zero-shot evaluation limitation identified in JC2Bench baseline results. Many models achieved 0% accuracy in zero-shot settings, particularly on:
- **Classification**: 5 out of 7 models scored 0%
- **NLI**: 4 out of 7 models scored 0%

Few-shot evaluation can distinguish between:
1. **Task comprehension failures**: Models don't understand the task format (fixable with examples)
2. **Fundamental capability limitations**: Models lack the required knowledge/reasoning (not fixable with examples)

## Research Questions

1. Does few-shot learning improve performance for models that failed in zero-shot?
2. How many shots are needed for meaningful improvement (1-shot vs 3-shot)?
3. Which models benefit most from few-shot examples?
4. Does improvement differ between Classification and NLI tasks?

## Target Models

### Priority 1: API Models
- **Claude 3.5 Sonnet**: NLI 0%, Classification 10%
- **Claude 3 Opus**: NLI 30%, Classification 0%

### Priority 2: Open-Source Models
- **Llama 3.1 8B**: NLI 0%, Classification 20% (best open-source on classification)
- **Qwen 2.5 7B**: Both 0%
- **EXAONE 3.0 7.8B**: Both 0%

## Experimental Design

### Tasks
- **Classification**: 21-way literary style classification
- **NLI**: 3-way entailment classification

### Configurations
- **0-shot** (baseline): No examples
- **1-shot**: Single example per label (balanced)
- **3-shot**: Three examples (balanced across labels)

### Sample Size
- 50 instances per task (reduced from full dataset for faster evaluation)
- Exemplars selected from training set (first 100 instances)
- Test instances exclude exemplars to prevent data leakage

### Evaluation Metrics
- Accuracy (primary)
- Macro F1 score
- Per-label precision/recall (for error analysis)

## Expected Timeline

### Fast Track (1 day)
- Open-source models only (Llama, Qwen, EXAONE)
- Local inference on H100
- ~6-8 hours total (50 samples × 2 tasks × 2 shots × 3 models)

### Full Evaluation (2 days)
- All 5 target models (including API models)
- Requires API keys for Claude models
- ~12-16 hours total

## File Structure

```
exp6_fewshot_evaluation.py    # Main evaluation script
exp6_run_fewshot.sh            # Automated runner
exp6_analyze_improvements.py   # Analysis and visualization
EXP6_README.md                 # This file
```

## Usage

### Quick Start (Open-Source Models)

```bash
cd /home/work/songhune/korean_R-CoA/notebook/experiments
./exp6_run_fewshot.sh
```

This will automatically:
1. Run few-shot evaluation for Llama, Qwen, and EXAONE
2. Generate results in `../../results/fewshot/`
3. Create summary CSVs and aggregated results
4. Log all outputs to timestamped log file

### API Models (Optional)

To include Claude models, uncomment and configure in `exp6_run_fewshot.sh`:

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Uncomment the Claude sections in exp6_run_fewshot.sh
```

### Manual Execution

For individual model evaluation:

```bash
# Example: Llama 3.1 8B
python exp6_fewshot_evaluation.py \
    --benchmark ../../benchmark/k_classic_bench/k_classic_bench_full.json \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --model-type opensource \
    --shots 1 3 \
    --tasks classification nli \
    --max-samples 50 \
    --output ../../results/fewshot
```

### Analysis

After experiments complete, analyze improvements:

```bash
python exp6_analyze_improvements.py
```

This generates:
- Comparison tables (0-shot vs 1-shot vs 3-shot)
- Improvement heatmaps
- Learning curves
- Statistical summaries

## Expected Outputs

### 1. Per-Model Results
- `fewshot_{model}_{timestamp}.json`: Detailed predictions
- `summary_{model}_{timestamp}.csv`: Aggregated metrics

### 2. Combined Analysis
- `exp6_fewshot_summary_combined.csv`: All models combined
- `fewshot_vs_zeroshot_comparison.csv`: Improvement deltas
- `fewshot_comparison.png`: Side-by-side bar charts
- `improvement_heatmap.png`: Heatmap of improvements
- `learning_curves.png`: 0→1→3 shot progression

### 3. Log Files
- `exp6_run_{timestamp}.log`: Execution log with all outputs

## Interpreting Results

### Scenario 1: Significant Improvement (>20% accuracy gain)
**Interpretation**: Task comprehension failure in zero-shot
**Implication**: Model has capability but needs task format clarification
**Paper claim**: "Few-shot examples resolve task ambiguity"

### Scenario 2: Modest Improvement (5-20% gain)
**Interpretation**: Partial benefit from examples
**Implication**: Mix of task comprehension and knowledge gaps
**Paper claim**: "Limited improvement suggests domain knowledge requirements"

### Scenario 3: Minimal Improvement (<5% gain)
**Interpretation**: Fundamental capability limitation
**Implication**: Model lacks required knowledge/reasoning
**Paper claim**: "Few-shot learning insufficient to overcome domain expertise gaps"

### Scenario 4: No Improvement (0% → 0%)
**Interpretation**: Severe capability gap or output format issues
**Implication**: Requires investigation of model responses
**Paper claim**: "Complete failure suggests architectural or training limitations"

## Integration with Paper

### Key Findings to Highlight

1. **Zero-shot vs Few-shot Gap**
   - Quantify average improvement across models
   - Identify which task benefits more (Classification vs NLI)

2. **Model-Specific Patterns**
   - Do larger API models benefit more/less than smaller open-source?
   - Is improvement correlated with zero-shot performance?

3. **Shot Number Sensitivity**
   - Diminishing returns from 1-shot to 3-shot?
   - Task-dependent optimal shot number?

### Potential Paper Additions

**Results Section (V)**:
```latex
\subsubsection{Few-shot Learning Analysis}

To investigate whether zero-shot failures stem from task comprehension
or fundamental capability limitations, we conduct few-shot evaluation
(1-shot and 3-shot) on Classification and NLI tasks for models that
achieved 0\% zero-shot accuracy...

[Table: Few-shot improvements]
[Figure: Learning curves]
```

**Discussion Section (VI)**:
```latex
Our few-shot analysis reveals that [INSERT FINDING]. Models that
achieved 0\% in zero-shot showed [INSERT IMPROVEMENT PATTERN],
suggesting that [INSERT INTERPRETATION]. This indicates that
classical Chinese text understanding requires [INSERT IMPLICATION].
```

## Success Criteria

### Minimum Viable Results
- ✅ Complete evaluation for 3 open-source models
- ✅ 1-shot and 3-shot results for Classification and NLI
- ✅ Comparison with zero-shot baseline
- ✅ At least one visualization showing improvements

### Complete Results
- ✅ All 5 target models evaluated
- ✅ Statistical significance testing (paired t-test)
- ✅ Error analysis with qualitative examples
- ✅ Per-label breakdown for Classification

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use smaller models first

### Issue: Rate Limiting (API Models)
**Solution**: Increase sleep time in `exp6_fewshot_evaluation.py` line 233

### Issue: Different Results on Re-run
**Solution**: Verify random seed consistency (line 77, seed=42)

### Issue: Missing Zero-shot Baseline
**Solution**: Run exp5 first or manually create aggregated_results.json

## References

- Exp5 (Baseline): `exp5_benchmark_evaluation.py`
- Zero-shot results: `../../results/aggregated/aggregated_results.json`
- Benchmark data: `../../benchmark/k_classic_bench/k_classic_bench_full.json`

## Timeline Estimate

| Task | Time | Notes |
|------|------|-------|
| Llama 3.1 8B | 2-3h | Local inference, H100 |
| Qwen 2.5 7B | 2-3h | Local inference, H100 |
| EXAONE 3.0 7.8B | 2-3h | Local inference, H100 |
| Claude 3.5 Sonnet | 3-4h | API, rate limited |
| Claude 3 Opus | 3-4h | API, rate limited |
| **Total (Open-source)** | **6-9h** | Parallelizable |
| **Total (All models)** | **12-18h** | Sequential (rate limits) |

Add 2-4 hours for analysis and visualization.

## Questions for Advisor

1. Should we evaluate 5-shot as well, or is 1-shot + 3-shot sufficient?
2. Priority on API models (Claude) vs open-source only?
3. Should we include few-shot for Retrieval/Punctuation (currently high-performing)?
4. Desired sample size: 50 (faster) or full dataset (more robust)?

---

**Status**: Ready to run
**Created**: 2025-10-27
**Last Updated**: 2025-10-27
