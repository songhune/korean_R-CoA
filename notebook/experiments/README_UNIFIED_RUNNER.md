# KLSBench Unified Evaluation Runner

This directory contains the unified evaluation framework for KLSBench with centralized configuration management.

## Overview

The evaluation system has been unified into a single runner script with YAML-based configuration:

- **Configuration**: [config.yaml](config.yaml) - Centralized configuration for paths, models, and settings
- **Runner Script**: [run_evaluation.sh](run_evaluation.sh) - Unified script for all evaluation types
- **Config Loader**: [config_loader.py](config_loader.py) - Python utility for reading configuration

## Quick Start

### 1. Basic Usage

```bash
# Test mode (10 samples per task)
./run_evaluation.sh test

# Sample mode (30% of data)
./run_evaluation.sh sample 0.3

# Full evaluation (all 7,871 items)
./run_evaluation.sh full

# Few-shot evaluation (1-shot and 3-shot)
./run_evaluation.sh fewshot "1 3"
```

### 2. Configuration

Edit [config.yaml](config.yaml) to customize:

```yaml
# Benchmark paths
benchmark:
  full: "../../benchmark/kls_bench/kls_bench_full.json"

# Output directories
output:
  base: "../../benchmark/results"
  fewshot: "../../benchmark/results/fewshot"

# Enable/disable models
models:
  api:
    openai:
      - name: "gpt-4-turbo"
        enabled: true
```

### 3. View Configuration

```bash
# Print configuration summary
python config_loader.py --summary

# Get benchmark path
python config_loader.py --benchmark full

# List enabled models
python config_loader.py --models api

# Calculate sample size
python config_loader.py --sample-size 0.3
```

## File Structure

```
notebook/experiments/
├── config.yaml                      # Central configuration file
├── config_loader.py                 # Configuration loader utility
├── run_evaluation.sh                # Unified evaluation runner
│
├── exp5_benchmark_evaluation.py     # Zero-shot evaluation script
├── exp6_fewshot_evaluation.py       # Few-shot evaluation script
│
├── run_all_evaluations.sh          # [DEPRECATED] Use run_evaluation.sh
└── exp6_run_fewshot.sh             # [DEPRECATED] Use run_evaluation.sh
```

## Evaluation Modes

### Test Mode
- **Purpose**: Quick testing with minimal samples
- **Samples**: 10 per task (50 total)
- **Time**: ~5-10 minutes
- **Cost**: <$1

```bash
./run_evaluation.sh test
```

### Sample Mode
- **Purpose**: Cost-effective evaluation with statistical validity
- **Samples**: Configurable ratio (10%, 30%, 50%)
- **Recommended**: 30% (2,361 items, ±2% error)
- **Time**: ~1 hour
- **Cost**: $6-7 (GPT-4)

```bash
./run_evaluation.sh sample 0.3    # 30% sampling
./run_evaluation.sh sample 0.1    # 10% sampling
./run_evaluation.sh sample 0.5    # 50% sampling
```

### Full Mode
- **Purpose**: Complete evaluation for final results
- **Samples**: 7,871 items (100%)
- **Time**: 3-5 hours
- **Cost**: $19-20 (GPT-4)

```bash
./run_evaluation.sh full
```

### Few-Shot Mode
- **Purpose**: Improve zero-shot performance with examples
- **Samples**: 50 per task (limited for faster evaluation)
- **Tasks**: Classification and NLI only
- **Shots**: 1-shot, 3-shot, 5-shot

```bash
./run_evaluation.sh fewshot "1 3"      # 1-shot and 3-shot
./run_evaluation.sh fewshot "1 3 5"    # 1-shot, 3-shot, and 5-shot
```

## Configuration Reference

### Benchmark Paths

```yaml
benchmark:
  full: "../../benchmark/kls_bench/kls_bench_full.json"
  classification: "../../benchmark/kls_bench/kls_bench_classification.json"
  # ... other task-specific benchmarks
```

### Model Configuration

```yaml
models:
  api:
    openai:
      - name: "gpt-4-turbo"
        enabled: true
      - name: "gpt-3.5-turbo"
        enabled: true
    anthropic:
      - name: "claude-3-5-sonnet-20241022"
        enabled: true
      - name: "claude-3-opus-20240229"
        enabled: true

  opensource:
    - name: "meta-llama/Llama-3.1-8B-Instruct"
      enabled: true
    - name: "Qwen/Qwen2.5-7B-Instruct"
      enabled: true

  supervised:
    - name: "SCUT-DLVCLab/TongGu-7B-Instruct"
      enabled: true
```

### Task Information

```yaml
tasks:
  classification:
    total_items: 808
    description: "Classify literary style (Fu/Shi/Yi/I)"
    metric: "Accuracy"

  retrieval:
    total_items: 1209
    description: "Identify source from Four Books"
    metric: "Accuracy"

  # ... other tasks
```

## Environment Setup

### 1. API Keys

Create a `.env` file in this directory:

```bash
# OpenAI API Key
export OPENAI_API_KEY='sk-...'

# Anthropic API Key
export ANTHROPIC_API_KEY='sk-ant-...'
```

### 2. Python Dependencies

```bash
pip install pyyaml transformers torch anthropic openai sklearn rouge-score nltk tqdm
```

## Advanced Usage

### Custom Configuration File

```bash
# Use custom config
python exp5_benchmark_evaluation.py \
    --config /path/to/custom_config.yaml \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY
```

### Programmatic Access

```python
from config_loader import Config

# Load configuration
config = Config()

# Get benchmark path
benchmark_path = config.get_benchmark_path('classification')

# Get enabled models
api_models = config.get_api_models('openai')
opensource_models = config.get_opensource_models()

# Calculate sample size
sample_size = config.calculate_sample_size(0.3, 'nli')

# Print summary
config.print_summary()
```

### Selective Model Evaluation

Edit [config.yaml](config.yaml) to disable models:

```yaml
models:
  api:
    openai:
      - name: "gpt-4-turbo"
        enabled: true
      - name: "gpt-3.5-turbo"
        enabled: false  # Skip this model
```

## Migration Guide

### From Old Scripts

**Old way:**
```bash
./run_all_evaluations.sh test
./exp6_run_fewshot.sh
```

**New way:**
```bash
./run_evaluation.sh test
./run_evaluation.sh fewshot "1 3"
```

### Configuration Migration

Previously hardcoded paths are now in [config.yaml](config.yaml):

```yaml
# Old: Hardcoded in scripts
BENCHMARK_PATH="/path/to/benchmark.json"
OUTPUT_DIR="/path/to/results"

# New: Centralized in config.yaml
benchmark:
  full: "../../benchmark/kls_bench/kls_bench_full.json"
output:
  base: "../../benchmark/results"
```

## Results

### Zero-Shot Results
```
benchmark/results/
├── results_gpt-4-turbo_*.json
├── summary_gpt-4-turbo_*.csv
└── ...
```

### Few-Shot Results
```
benchmark/results/fewshot/
├── fewshot_gpt-4-turbo_*.json
├── summary_gpt-4-turbo_*.csv
└── ...
```

### Aggregated Results
```
benchmark/results/aggregated/
├── aggregated_summary.csv
├── aggregated_pivot.csv
├── heatmap_performance.png
└── ...
```

## Troubleshooting

### Config Not Found

```bash
[ERROR] Configuration file not found: config.yaml
```

**Solution**: Make sure [config.yaml](config.yaml) exists in `notebook/experiments/`

### API Key Not Set

```bash
[WARNING] OPENAI_API_KEY not set
```

**Solution**: Create `.env` file or export environment variable:
```bash
export OPENAI_API_KEY='your-key-here'
```

### Import Error

```bash
[WARNING] config_loader not available
```

**Solution**: Install PyYAML:
```bash
pip install pyyaml
```

## Best Practices

1. **Test First**: Always run `test` mode before full evaluation
2. **Use Sampling**: Use 30% sampling for cost-effective evaluation
3. **Version Control**: Keep [config.yaml](config.yaml) in version control
4. **Environment Variables**: Never commit API keys, use `.env` file
5. **Monitor Costs**: Check API usage before running full evaluation

## Support

For issues or questions:
- Check configuration: `python config_loader.py --summary`
- Review logs in `benchmark/results/`
- See experiment documentation in `EXP5_README.md` and `EXP6_README.md`
