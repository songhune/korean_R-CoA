# KLSBench Experiments

This directory contains evaluation experiments for **KLSBench** (Korean Literary Style Benchmark), a comprehensive benchmark for evaluating large language models on Korean classical literature understanding.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_evaluation.txt

# 2. Configure API keys (optional)
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'

# 3. Run evaluation
./run_evaluation.sh test
```

## Directory Structure

```
experiments/
├── README.md                           # This file
├── config.yaml                         # Central configuration
├── run_evaluation.sh                   # Unified evaluation runner ⭐
│
├── Evaluation Scripts
│   ├── exp5_benchmark_evaluation.py   # Zero-shot evaluation framework
│   ├── exp6_fewshot_evaluation.py     # Few-shot evaluation framework
│   ├── exp6_analyze_improvements.py   # Few-shot analysis
│   └── exp6_result_aggregation.py     # Result aggregation
│
├── Utilities
│   ├── config_loader.py               # Configuration loader
│   ├── fix_classification_labels.py   # Unicode normalization tool
│   └── kls_bench_generator.py         # Benchmark generator
│
├── Setup
│   ├── setup_exp5.sh                  # Environment setup
│   └── requirements_evaluation.txt    # Python dependencies
│
├── Documentation
│   ├── README_exp5.md                 # Experiment 5 guide (Korean)
│   ├── README_UNIFIED_RUNNER.md       # Unified runner guide (English)
│   ├── SUMMARY_exp5.md                # Experiment 5 summary
│   ├── EXP6_README.md                 # Experiment 6 guide
│   ├── EXP6_SUMMARY.md                # Experiment 6 summary
│   └── SAMPLING_GUIDE.md              # Sampling strategy guide
│
├── Notebooks
│   ├── 1번실험.ipynb                   # Experiment 1
│   ├── 2번실험.ipynb                   # Experiment 2
│   ├── 3번실험.ipynb                   # Experiment 3
│   ├── 4번실험.ipynb                   # Experiment 4
│   ├── 5번실험.ipynb                   # Experiment 5 ⭐
│   └── kls_bench_summary.ipynb        # Benchmark summary
│
└── deprecated/                         # Deprecated scripts
    ├── run_all_evaluations.sh
    ├── exp6_run_fewshot.sh
    └── README.md
```

## Experiments Overview

### Experiment 5: Zero-Shot Evaluation ⭐
**Main evaluation framework for KLSBench**

- **Script**: [exp5_benchmark_evaluation.py](exp5_benchmark_evaluation.py)
- **Runner**: [run_evaluation.sh](run_evaluation.sh)
- **Config**: [config.yaml](config.yaml)
- **Documentation**: [README_exp5.md](README_exp5.md), [SUMMARY_exp5.md](SUMMARY_exp5.md)

**Features:**
- 5 evaluation tasks (classification, retrieval, punctuation, NLI, translation)
- Support for API models (GPT-4, Claude) and open-source models (Llama, Qwen, EXAONE)
- Configurable sampling (test/sample/full modes)
- YAML-based configuration
- Professional logging and results

**Quick Run:**
```bash
./run_evaluation.sh test              # Test mode (10 samples)
./run_evaluation.sh sample 0.3        # Sample 30% of data
./run_evaluation.sh full              # Full evaluation
```

### Experiment 6: Few-Shot Learning
**Evaluating few-shot learning performance**

- **Script**: [exp6_fewshot_evaluation.py](exp6_fewshot_evaluation.py)
- **Runner**: `./run_evaluation.sh fewshot`
- **Documentation**: [EXP6_README.md](EXP6_README.md), [EXP6_SUMMARY.md](EXP6_SUMMARY.md)

**Features:**
- 1-shot, 3-shot, 5-shot evaluation
- Dynamic example selection
- Performance comparison analysis
- Focused on classification and NLI tasks

**Quick Run:**
```bash
./run_evaluation.sh fewshot "1 3"     # 1-shot and 3-shot
./run_evaluation.sh fewshot "1 3 5"   # All shot configurations
```

### Earlier Experiments
- **Experiment 1-4**: Initial development and data preparation
- **Notebooks**: Interactive analysis and visualization

## Usage Modes

### 1. Test Mode (Quick Testing)
```bash
./run_evaluation.sh test
```
- **Samples**: 10 per task (50 total)
- **Time**: 5-10 minutes
- **Cost**: <$1
- **Purpose**: Quick validation

### 2. Sample Mode (Recommended)
```bash
./run_evaluation.sh sample 0.3
```
- **Samples**: 30% of data (2,361 items)
- **Time**: ~1 hour
- **Cost**: $6-7 (GPT-4)
- **Purpose**: Balanced evaluation with statistical validity

### 3. Full Mode (Final Results)
```bash
./run_evaluation.sh full
```
- **Samples**: 7,871 items (100%)
- **Time**: 3-5 hours
- **Cost**: $19-20 (GPT-4)
- **Purpose**: Complete evaluation for publication

### 4. Few-Shot Mode
```bash
./run_evaluation.sh fewshot "1 3 5"
```
- **Samples**: 50 per task (limited for speed)
- **Time**: 30 minutes
- **Cost**: $3-5
- **Purpose**: Evaluate few-shot learning capability

## Configuration

All settings are managed via [config.yaml](config.yaml):

```yaml
# Benchmark paths
benchmark:
  full: "../../benchmark/kls_bench/kls_bench_full.json"

# Output directories
output:
  base: "../../benchmark/results"
  fewshot: "../../benchmark/results/fewshot"

# Models (enable/disable)
models:
  api:
    openai:
      - name: "gpt-4-turbo"
        enabled: true
```

**View configuration:**
```bash
python config_loader.py --summary
python config_loader.py --models all
```

## Supported Models

### API Models ✅
- GPT-4 Turbo, GPT-3.5 Turbo
- Claude 3.5 Sonnet, Claude 3 Opus

### Open Source Models ✅
- Llama 3.1 8B
- Qwen 2.5 7B
- EXAONE 3.0 7.8B

### Supervised Models 🔧
- TongGu 7B ✅
- GwenBert (limited functionality)

## Evaluation Tasks

| Task | Description | Items | Metric |
|:-----|:------------|------:|:-------|
| **Classification** | Classify literary style | 808 | Accuracy, F1 |
| **Retrieval** | Identify source text | 1,209 | Accuracy |
| **Punctuation** | Restore punctuation | 2,000 | F1, ROUGE |
| **NLI** | Natural language inference | 1,854 | Accuracy, F1 |
| **Translation** | Translate between languages | 2,000 | BLEU, ROUGE |

**Total**: 7,871 items across 5 tasks

### Classification Labels
After Unicode normalization (2025-10-30):
- **19 unique labels** (reduced from 21)
- Fixed Unicode Compatibility Ideographs
- Balanced classes: 賦, 詩, 疑, 義, 策, 表 (95 items each)
- See [fix_classification_labels.py](fix_classification_labels.py) for details

## Results

Results are saved to `../../benchmark/results/`:

```
results/
├── results_gpt-4-turbo_*.json         # Detailed results
├── summary_gpt-4-turbo_*.csv          # Summary CSV
├── fewshot/
│   ├── fewshot_*_*.json
│   └── summary_*_*.csv
└── aggregated/
    ├── aggregated_summary.csv
    ├── heatmap_performance.png
    └── radar_chart.png
```

## Recent Updates (2025-10-30)

### ✨ Unified Evaluation Runner
- Single entry point for all evaluation types
- YAML-based configuration
- 4 modes: test, sample, full, fewshot

### 🔧 Unicode Normalization Fix
- Fixed duplicate classification labels (21 → 19)
- Normalized CJK compatibility ideographs
- Tool: [fix_classification_labels.py](fix_classification_labels.py)

### 📝 Rebranding
- K-ClassicBench → **KLSBench**
- English-first documentation
- Professional logging format

### 📂 Code Cleanup
- Moved deprecated scripts to [deprecated/](deprecated/)
- Updated all documentation
- Centralized configuration

## Cost Estimates

| Mode | Samples | GPT-4 Cost | Claude Cost |
|:-----|--------:|-----------:|------------:|
| Test | 50 | <$1 | <$1 |
| Sample 30% | 2,361 | $6-7 | $4-5 |
| Full | 7,871 | $19-20 | $12-15 |
| Few-shot | 250 | $3-5 | $2-3 |

**Always test with small samples first!**

## Environment Requirements

### Python Dependencies
```bash
pip install -r requirements_evaluation.txt
```

Key packages:
- PyYAML (configuration)
- transformers, torch (models)
- openai, anthropic (API clients)
- scikit-learn, rouge-score, nltk (metrics)

### Hardware
- **CPU**: Any modern CPU
- **RAM**: 16GB+ recommended
- **GPU**: Optional for open-source models
  - 7B models: 16GB+ VRAM
  - 70B models: 40GB+ VRAM (8-bit quantization)

## Troubleshooting

### Configuration Issues
```bash
# View current configuration
python config_loader.py --summary

# Verify benchmark paths
ls ../../benchmark/kls_bench/kls_bench_full.json
```

### API Key Issues
```bash
# Check if keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Set keys
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
```

### Unicode Issues
```bash
# Verify label normalization
python fix_classification_labels.py --verify
```

### GPU Memory Issues
Use 8-bit quantization for large models:
```python
model = AutoModel.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

## Development

### Adding New Models
1. Edit [config.yaml](config.yaml) to add model configuration
2. Implement model wrapper in evaluation script (if needed)
3. Test with `./run_evaluation.sh test`

### Adding New Tasks
1. Add task data to benchmark JSON
2. Implement evaluation logic in `exp5_benchmark_evaluation.py`
3. Update task metadata in [config.yaml](config.yaml)

### Running Tests
```bash
# Quick test all models
./run_evaluation.sh test

# Test specific model
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --max-samples 5
```

## Documentation

- [README_UNIFIED_RUNNER.md](README_UNIFIED_RUNNER.md) - Unified runner detailed guide
- [README_exp5.md](README_exp5.md) - Experiment 5 guide (Korean)
- [SUMMARY_exp5.md](SUMMARY_exp5.md) - Experiment 5 summary
- [EXP6_README.md](EXP6_README.md) - Few-shot evaluation guide
- [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) - Sampling strategy guide
- [config.yaml](config.yaml) - Configuration reference

## References

- **Benchmark**: [../../benchmark/kls_bench/README.md](../../benchmark/kls_bench/README.md)
- **C3Bench Paper**: Classical Chinese Understanding Benchmark
- **Data Sources**: Gwageo examination data, Four Books (四書)

## Citation

If you use KLSBench in your research, please cite:

```bibtex
@misc{klsbench2024,
  title={KLSBench: Korean Literary Style Benchmark},
  author={Your Name},
  year={2024},
  note={Korean classical literature understanding benchmark}
}
```

## Support

For issues or questions:
- Check [README_UNIFIED_RUNNER.md](README_UNIFIED_RUNNER.md)
- Review experiment summaries
- Check configuration with `python config_loader.py --summary`

---

**Last Updated**: 2025-10-30
**Version**: 2.0
**Status**: Active Development ✅

**Key Files**:
- Runner: [run_evaluation.sh](run_evaluation.sh)
- Config: [config.yaml](config.yaml)
- Zero-shot: [exp5_benchmark_evaluation.py](exp5_benchmark_evaluation.py)
- Few-shot: [exp6_fewshot_evaluation.py](exp6_fewshot_evaluation.py)
