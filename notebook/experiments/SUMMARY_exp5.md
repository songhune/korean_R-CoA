# Experiment 5: KLSBench Evaluation Framework - Summary

## Objective Achieved

**Built a complete framework for evaluating KLSBench benchmark with various LLMs**

---

## Recent Updates (2025-10-30)

### Major Changes

1. **Unified Evaluation Runner** [run_evaluation.sh](run_evaluation.sh)
   - Integrated zero-shot and few-shot evaluations
   - YAML-based configuration management
   - 4 modes: test, sample, full, fewshot

2. **Configuration System** [config.yaml](config.yaml)
   - Centralized path management
   - Model enable/disable settings
   - Task metadata

3. **Unicode Normalization Fix** [fix_classification_labels.py](fix_classification_labels.py)
   - Fixed duplicate labels (21 → 19 classes)
   - Normalized Unicode Compatibility Ideographs
   -論 (U+F941 → U+8AD6): 2 items
   - 禮義 (U+F9B6 → U+79AE): 6 items

4. **Rebranding**
   - K-ClassicBench → KLSBench (Korean Literary Style Benchmark)
   - Removed all emojis from code
   - Professional [BRACKET] format logging
   - English-first documentation

---

## File Structure

### Core Files
- [exp5_benchmark_evaluation.py](exp5_benchmark_evaluation.py) - Main evaluation framework
- [run_evaluation.sh](run_evaluation.sh) - Unified runner script
- [config.yaml](config.yaml) - Configuration file
- [config_loader.py](config_loader.py) - Config utility

### Jupyter Notebooks
- `5번실험.ipynb` - Interactive evaluation and visualization

### Documentation
- [README_exp5.md](README_exp5.md) - Detailed usage guide
- [README_UNIFIED_RUNNER.md](README_UNIFIED_RUNNER.md) - Unified runner guide
- [SUMMARY_exp5.md](SUMMARY_exp5.md) - This file

### Setup
- [setup_exp5.sh](setup_exp5.sh) - Environment setup
- [requirements_evaluation.txt](requirements_evaluation.txt) - Python dependencies

### Deprecated (moved to [deprecated/](deprecated/))
- `run_all_evaluations.sh` → Use [run_evaluation.sh](run_evaluation.sh)
- `exp6_run_fewshot.sh` → Use `run_evaluation.sh fewshot`

---

## Framework Architecture

```
KLSBench Evaluation Framework
│
├── Configuration (config.yaml)
│   ├── Benchmark paths
│   ├── Output directories
│   ├── Model settings
│   └── Task metadata
│
├── Data Loading
│   └── kls_bench_full.json (7,871 items)
│       └── 19 classification labels (after normalization)
│
├── Model Wrappers
│   ├── API Models
│   │   ├── OpenAIWrapper (GPT-4, GPT-3.5)
│   │   └── AnthropicWrapper (Claude)
│   │
│   ├── Open Source Models
│   │   └── HuggingFaceWrapper (Llama, Qwen, EXAONE)
│   │
│   └── Supervised Models
│       ├── TongGuWrapper (SCUT-DLVCLab/TongGu-7B-Instruct)
│       └── GwenBertWrapper (encoder, limited functionality)
│
├── Task Evaluation
│   ├── Classification (19 classes) → Accuracy, F1
│   ├── Retrieval → Accuracy
│   ├── Punctuation → F1, ROUGE
│   ├── NLI → Accuracy, F1
│   └── Translation → BLEU, ROUGE
│
└── Results
    ├── JSON (detailed results + predictions)
    └── CSV (summary)
```

---

## Quick Start

### 1. Environment Setup

```bash
cd /Users/songhune/Workspace/korean_eda/notebook/experiments

# Install dependencies
pip install -r requirements_evaluation.txt

# Or use setup script
./setup_exp5.sh
```

### 2. Configure API Keys (if using API models)

```bash
# Create .env file
echo "export OPENAI_API_KEY='your-key'" > .env
echo "export ANTHROPIC_API_KEY='your-key'" >> .env
```

### 3. Run Evaluation

**Unified Runner (Recommended)**
```bash
# Test mode (10 samples per task)
./run_evaluation.sh test

# Sample mode (30% of data)
./run_evaluation.sh sample 0.3

# Full evaluation (7,871 items)
./run_evaluation.sh full

# Few-shot evaluation
./run_evaluation.sh fewshot "1 3"
```

**Direct Script (Advanced)**
```bash
# Single model evaluation
python exp5_benchmark_evaluation.py \
    --config config.yaml \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --sample-ratio 0.3
```

**Configuration Management**
```bash
# View configuration
python config_loader.py --summary

# List models
python config_loader.py --models all

# Calculate sample size
python config_loader.py --sample-size 0.3
```

---

## Evaluation Modes

| Mode | Samples | Time | Cost (GPT-4) | Use Case |
|:-----|--------:|-----:|-------------:|:---------|
| **test** | 10/task (50 total) | 5-10 min | <$1 | Quick testing |
| **sample 0.1** | 787 (10%) | 20 min | $2 | Rapid experiments |
| **sample 0.3** | 2,361 (30%) | 1 hour | $6-7 | Balanced evaluation ⭐ |
| **sample 0.5** | 3,936 (50%) | 2 hours | $10-12 | Detailed analysis |
| **full** | 7,871 (100%) | 3-5 hours | $19-20 | Final results |
| **fewshot** | 50/task | 30 min | $3-5 | Few-shot learning |

---

## Evaluation Tasks

| Task | Input | Output | Metrics | Classes |
|:-----|:------|:-------|:--------|:--------|
| **Classification** | Classical Chinese text | Literary style | Accuracy, F1 | 19 labels |
| **Retrieval** | Sentence | Source (Four Books) | Accuracy | - |
| **Punctuation** | Unpunctuated text | Punctuated text | F1, ROUGE-L | - |
| **NLI** | Premise + Hypothesis | entailment/contradiction/neutral | Accuracy, F1 | 3 labels |
| **Translation** | Source text | Target translation | BLEU, ROUGE | - |

### Classification Labels (19 classes)

After Unicode normalization:

**Balanced classes (95 items each):**
- 賦 (Fu), 詩 (Shi), 疑 (Yi - questions), 義 (Yi - meanings), 策 (Ce), 表 (Biao)

**Other classes:**
- 論 (53), 銘 (53), 箋 (49), 頌 (24), 禮義 (13), 箴 (12), 易義 (9), 詩義 (7), 書義 (6), 詔 (5), 制 (3), 講 (2), 擬 (2)

**Note**: High class imbalance (47.5x ratio). Consider stratified sampling or class weighting.

---

## Supported Models

### API Models ✅
- GPT-4 Turbo (`gpt-4-turbo`)
- GPT-3.5 Turbo (`gpt-3.5-turbo`)
- Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)
- Claude 3 Opus (`claude-3-opus-20240229`)

### Open Source Models ✅
- Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- Qwen 2.5 7B (`Qwen/Qwen2.5-7B-Instruct`)
- EXAONE 3.0 7.8B (`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`)

### Supervised Models 🔧
- TongGu 7B (`SCUT-DLVCLab/TongGu-7B-Instruct`) ✅
- GwenBert (`ethanyt/guwenbert-base`) - Encoder only, limited functionality

**Enable/Disable**: Edit [config.yaml](config.yaml) `models` section

---

## Results Format

### Output Structure

```
../../benchmark/results/
├── results_gpt-4-turbo_20241030_150000.json      # Detailed results
├── summary_gpt-4-turbo_20241030_150000.csv       # Summary
├── fewshot/
│   ├── fewshot_gpt-4-turbo_*.json
│   └── summary_gpt-4-turbo_*.csv
└── aggregated/
    ├── aggregated_summary.csv
    ├── heatmap_performance.png
    └── radar_chart.png
```

### CSV Summary Example

| model | task | accuracy | f1 | bleu | rouge1_f1 | rougeL_f1 |
|:---|:---|---:|---:|---:|---:|---:|
| gpt-4-turbo | classification | 0.850 | 0.840 | - | - | - |
| gpt-4-turbo | retrieval | 0.920 | - | - | - | - |
| gpt-4-turbo | punctuation | - | 0.780 | - | 0.820 | 0.790 |
| gpt-4-turbo | nli | 0.780 | 0.770 | - | - | - |
| gpt-4-turbo | translation | - | - | 0.650 | 0.710 | 0.680 |

---

## Customization

### Add New Model

```python
# In exp5_benchmark_evaluation.py

class MyModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str):
        # Load model
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Inference implementation
        return prediction
```

Then add to [config.yaml](config.yaml):

```yaml
models:
  custom:
    - name: "my-org/my-model"
      enabled: true
```

### Add New Task

```python
def evaluate_my_task(self, predictions: List[str], ground_truths: List[str]) -> Dict:
    # Evaluation logic
    return {
        'my_metric': score,
        'num_samples': len(predictions)
    }
```

---

## Important Notes

### GPU Memory
- 7B models: 16GB+ VRAM
- 70B models: 40GB+ VRAM (use 8-bit quantization)

### API Costs (Estimated)
- **Test mode** (50 samples): <$1
- **Sample 30%** (2,361 items): $6-7 (GPT-4)
- **Full evaluation** (7,871 items): $19-20 (GPT-4)

Always test with `--max-samples 10` first!

### Execution Time
- API models (full): 2-3 hours
- Open source 7B (full): 4-5 hours
- Open source 70B (full): 10-15 hours

### Statistical Considerations

**Sample mode recommended settings:**
- 10%: Quick experiments (±3% error)
- 30%: Balanced evaluation (±2% error) ⭐ **Recommended**
- 50%: Detailed analysis (±1.5% error)

See [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) for details.

---

## Comparison with C3Bench

### Similarities
- 5 core tasks
- Multiple model types
- Quantitative metrics

### KLSBench Differentiators

1. **Korean Classical Literature Focus**
   - Gwageo (civil service exam) data
   - Four Books (四書) corpus

2. **Punctuation Restoration Task**
   - Unpunctuated text → Punctuated text
   - Tests classical Chinese processing

3. **Multilingual Translation**
   - Classical Chinese ↔ Korean ↔ English

4. **Unicode Normalization**
   - Fixed CJK compatibility ideographs
   - 19 properly normalized classification labels

---

## Recent Fixes and Improvements

### Unicode Normalization (2025-10-30)
- **Issue**: Classification had 21 labels due to Unicode variants
- **Fixed**: Normalized to 19 canonical labels
- **Tool**: [fix_classification_labels.py](fix_classification_labels.py)
- **Details**: See commit `e6dce245`

### Unified Runner (2025-10-30)
- **Created**: [run_evaluation.sh](run_evaluation.sh)
- **Deprecated**: `run_all_evaluations.sh`, `exp6_run_fewshot.sh`
- **Benefits**: Single entry point, YAML config, 4 modes
- **Guide**: [README_UNIFIED_RUNNER.md](README_UNIFIED_RUNNER.md)

### Rebranding (2025-10-30)
- **Name**: K-ClassicBench → KLSBench
- **Code**: Removed emojis, professional logging
- **Docs**: English-first with Classical Chinese context
- **Details**: See commit `fe4bf74a`

---

## Future Improvements

1. **Few-shot Learning** ✅ (Completed in EXP6)
   - 0-shot, 1-shot, 3-shot, 5-shot comparison
   - Example selection strategies

2. **Chain-of-Thought**
   - Step-by-step reasoning evaluation
   - Explanation generation

3. **More Models**
   - Gemini
   - Mistral
   - Korean-specific models

4. **Benchmark Expansion**
   - More tasks
   - Difficulty levels
   - Domain-specific subsets

---

## Troubleshooting

### Config Not Found
```bash
[ERROR] Configuration file not found: config.yaml
```
**Solution**: Ensure [config.yaml](config.yaml) exists in `notebook/experiments/`

### API Key Not Set
```bash
[WARNING] OPENAI_API_KEY not set
```
**Solution**: Create `.env` file or export variable:
```bash
export OPENAI_API_KEY='your-key'
```

### Unicode Issues
```bash
[WARNING] Found compatibility ideographs
```
**Solution**: Run normalization:
```bash
python fix_classification_labels.py --verify
```

### GPU Out of Memory
**Solution**: Use smaller models or 8-bit quantization:
```python
model = AutoModel.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

---

## References

- **C3Bench Paper**: Classical Chinese Understanding Benchmark
- **Benchmark Details**: [../../benchmark/kls_bench/README.md](../../benchmark/kls_bench/README.md)
- **Data Sources**:
  - Gwageo (과거시험) examination data
  - Four Books (四書) classical texts

---

## Checklist

### Environment Setup
- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements_evaluation.txt`)
- [ ] API keys configured (if using API models)
- [ ] GPU available (if using open source models)

### Evaluation
- [ ] Test mode completed (`./run_evaluation.sh test`)
- [ ] Configuration verified (`python config_loader.py --summary`)
- [ ] API costs estimated
- [ ] Full evaluation executed
- [ ] Results verified

### Analysis
- [ ] CSV summaries reviewed
- [ ] Visualizations generated (Jupyter notebook)
- [ ] Error cases analyzed
- [ ] Insights documented

---

**Created**: 2024-10-21
**Last Updated**: 2025-10-30
**Version**: 2.0
**Status**: Complete ✅

---

**Related Documents**:
- [README_exp5.md](README_exp5.md) - Detailed usage guide
- [README_UNIFIED_RUNNER.md](README_UNIFIED_RUNNER.md) - Unified runner documentation
- [EXP6_README.md](EXP6_README.md) - Few-shot evaluation (Experiment 6)
- [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) - Sampling strategy guide
- [config.yaml](config.yaml) - Configuration file reference
