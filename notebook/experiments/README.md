# KLSBench Experiments

Evaluation experiments for **KLSBench** (Korean Literary Style Benchmark) - a comprehensive benchmark for evaluating LLMs on Korean classical literature understanding.

## Quick Start

```bash
# 1. Install dependencies (REQUIRED - do this first!)
pip install -r requirements_evaluation.txt

# 2. Configure API keys
# Option A: Export directly
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'

# Option B: Create .env file (recommended)
cat > .env << EOF
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
EOF

# 3. Run evaluation
./run_evaluation.sh test              # Test mode (10 samples)
./run_evaluation.sh sample 0.3        # Sample 30%
./run_evaluation.sh full              # Full evaluation
./run_evaluation.sh fewshot "1 3"     # Few-shot (1-shot, 3-shot)
```

**Important**:

- Install dependencies BEFORE running evaluations
- Few-shot mode works: `./run_evaluation.sh fewshot "1 3 5"` for 1-shot, 3-shot, and 5-shot

## Directory Structure

```
experiments/
├── README.md                    # This file
├── config.yaml                  # Configuration
├── run_evaluation.sh           # Unified evaluation runner
├── requirements_evaluation.txt  # Python dependencies
│
├── exp5/                        # Experiment 5: Zero-shot evaluation
│   ├── exp5_benchmark_evaluation.py
│   ├── setup_exp5.sh
│   └── 5번실험.ipynb
│
├── exp6/                        # Experiment 6: Few-shot learning
│   ├── exp6_fewshot_evaluation.py
│   ├── exp6_analyze_improvements.py
│   └── exp6_result_aggregation.py
│
├── utils/                       # Utilities
│   ├── config_loader.py         # Config management
│   ├── fix_classification_labels.py
│   ├── kls_bench_generator.py
│   ├── font_fix.py
│   ├── rename_kclassicbench.sh
│   └── run_translation.py
│
└── [1-4]번실험.ipynb            # Earlier experiments
```

## Experiments

### Experiment 5: Zero-Shot Evaluation

**Location**: [exp5/](exp5/)

Complete evaluation framework for KLSBench with zero-shot learning.

**Key Features**:
- 5 evaluation tasks (classification, retrieval, punctuation, NLI, translation)
- Support for API models (GPT-4, Claude) and open-source models (Llama, Qwen, EXAONE)
- Configurable sampling modes (test/sample/full)
- 19 classification labels (Unicode normalized)

**Run**:
```bash
# Test mode
./run_evaluation.sh test

# Sample mode (recommended: 30%)
./run_evaluation.sh sample 0.3

# Full evaluation
./run_evaluation.sh full
```

**Files**:
- [exp5/exp5_benchmark_evaluation.py](exp5/exp5_benchmark_evaluation.py) - Main evaluation framework
- [exp5/setup_exp5.sh](exp5/setup_exp5.sh) - Environment setup
- [exp5/5번실험.ipynb](exp5/5번실험.ipynb) - Interactive analysis

### Experiment 6: Few-Shot Learning

**Location**: [exp6/](exp6/)

Evaluating few-shot learning performance with dynamic example selection.

**Key Features**:
- 1-shot, 3-shot, 5-shot evaluation
- Focused on classification and NLI tasks
- Performance improvement analysis
- Result aggregation and visualization

**Run**:
```bash
# Few-shot evaluation
./run_evaluation.sh fewshot "1 3 5"

# Analyze improvements
cd exp6 && python exp6_analyze_improvements.py

# Aggregate results
python exp6_result_aggregation.py
```

**Files**:
- [exp6/exp6_fewshot_evaluation.py](exp6/exp6_fewshot_evaluation.py) - Few-shot evaluation
- [exp6/exp6_analyze_improvements.py](exp6/exp6_analyze_improvements.py) - Analysis
- [exp6/exp6_result_aggregation.py](exp6/exp6_result_aggregation.py) - Aggregation

## Configuration

All settings managed via [config.yaml](config.yaml):

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

**Utilities**:
```bash
# View configuration
python utils/config_loader.py --summary

# List models
python utils/config_loader.py --models all

# Calculate sample size
python utils/config_loader.py --sample-size 0.3
```

## Evaluation Modes

| Mode | Samples | Time | Cost (GPT-4) | Use Case |
|:-----|--------:|-----:|-------------:|:---------|
| **test** | 50 | 5-10 min | <$1 | Quick testing |
| **sample 0.3** | 2,361 | 1 hour | $6-7 | Balanced evaluation ⭐ |
| **full** | 7,871 | 3-5 hours | $19-20 | Final results |
| **fewshot** | 250 | 30 min | $3-5 | Few-shot learning |

## Evaluation Tasks

| Task | Description | Items | Metric |
|:-----|:------------|------:|:-------|
| **Classification** | Literary style classification | 808 | Accuracy, F1 |
| **Retrieval** | Source text identification | 1,209 | Accuracy |
| **Punctuation** | Punctuation restoration | 2,000 | F1, ROUGE |
| **NLI** | Natural language inference | 1,854 | Accuracy, F1 |
| **Translation** | Multilingual translation | 2,000 | BLEU, ROUGE |

**Total**: 7,871 items across 5 tasks

### Classification Labels (19 classes)

After Unicode normalization:
- **Balanced** (95 each): 賦, 詩, 疑, 義, 策, 表
- **Other**: 論(53), 銘(53), 箋(49), 頌(24), 禮義(13), 箴(12), 易義(9), 詩義(7), 書義(6), 詔(5), 制(3), 講(2), 擬(2)

## Supported Models

### API Models ✅
- GPT-4 Turbo, GPT-3.5 Turbo
- Claude 3.5 Sonnet, Claude 3 Opus

### Open Source Models ✅
- Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- Qwen 2.5 7B (`Qwen/Qwen2.5-7B-Instruct`)
- EXAONE 3.0 7.8B (`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`)

### Supervised Models 🔧
- TongGu 7B (`SCUT-DLVCLab/TongGu-7B-Instruct`) ✅
- GwenBert (`ethanyt/guwenbert-base`) - Limited functionality

## Results

Results saved to `../../benchmark/results/`:

```
results/
├── results_gpt-4-turbo_*.json
├── summary_gpt-4-turbo_*.csv
├── fewshot/
│   ├── fewshot_*_*.json
│   └── summary_*_*.csv
└── aggregated/
    ├── aggregated_summary.csv
    └── *.png (visualizations)
```

## Utilities

### Configuration Management
```bash
python utils/config_loader.py --summary
python utils/config_loader.py --models all
```

### Unicode Normalization
```bash
python utils/fix_classification_labels.py --verify
```

### Benchmark Generation
```bash
python utils/kls_bench_generator.py
```

### Font Fix
```bash
python utils/font_fix.py <input_file> <output_file>
```

## Cost Estimates

| Mode | Samples | GPT-4 | Claude |
|:-----|--------:|------:|-------:|
| Test | 50 | <$1 | <$1 |
| Sample 30% | 2,361 | $6-7 | $4-5 |
| Full | 7,871 | $19-20 | $12-15 |
| Few-shot | 250 | $3-5 | $2-3 |

## Environment Requirements

### Python Dependencies
```bash
pip install -r requirements_evaluation.txt
```

Key packages:
- PyYAML, transformers, torch
- openai, anthropic
- scikit-learn, rouge-score, nltk

### Hardware
- **RAM**: 16GB+ recommended
- **GPU**: Optional (16GB+ VRAM for 7B models)

## Recent Changes (2025-10-30)

### Directory Reorganization
- Created `exp5/`, `exp6/`, `utils/` directories
- Consolidated all font_fix scripts into one
- Removed all legacy README files
- Simplified directory structure

### Unicode Normalization
- Fixed duplicate classification labels (21 → 19)
- Normalized CJK compatibility ideographs

### Unified Runner
- Single entry point for all evaluations
- YAML-based configuration
- 4 modes: test, sample, full, fewshot

### Rebranding
- K-ClassicBench → KLSBench
- Professional logging format
- English-first documentation

## Troubleshooting

### Configuration
```bash
# Check configuration
python utils/config_loader.py --summary

# Verify paths
ls ../../benchmark/kls_bench/kls_bench_full.json
```

### API Keys
```bash
# Set keys
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'

# Or create .env file
echo "export OPENAI_API_KEY='your-key'" > .env
```

### Unicode Issues
```bash
python utils/fix_classification_labels.py --verify
```

### GPU Memory
Use 8-bit quantization:
```python
model = AutoModel.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

## Development

### Adding New Models
1. Edit [config.yaml](config.yaml)
2. Implement wrapper in evaluation script (if needed)
3. Test with `./run_evaluation.sh test`

### Adding New Tasks
1. Add task data to benchmark JSON
2. Implement evaluation logic
3. Update task metadata in [config.yaml](config.yaml)

### Running Tests
```bash
# Quick test
./run_evaluation.sh test

# Specific model
python exp5/exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --max-samples 5
```

## File Organization

### Scripts
- `run_evaluation.sh` - Main evaluation runner
- `exp5/setup_exp5.sh` - Environment setup

### Configuration
- `config.yaml` - Central configuration
- `requirements_evaluation.txt` - Python dependencies
- `.env` - API keys (create manually)

### Experiments
- `exp5/` - Zero-shot evaluation
- `exp6/` - Few-shot learning
- `[1-4]번실험.ipynb` - Earlier experiments

### Utilities
- `utils/config_loader.py` - Config management
- `utils/fix_classification_labels.py` - Unicode normalization
- `utils/kls_bench_generator.py` - Benchmark generation
- `utils/font_fix.py` - Font normalization
- `utils/run_translation.py` - Translation utilities

## References

- **Benchmark**: [../../benchmark/kls_bench/README.md](../../benchmark/kls_bench/README.md)
- **C3Bench**: Classical Chinese Understanding Benchmark
- **Data Sources**: Gwageo examination data, Four Books (四書)

## Citation

```bibtex
@misc{klsbench2024,
  title={KLSBench: Korean Literary Style Benchmark},
  author={Your Name},
  year={2024},
  note={Korean classical literature understanding benchmark}
}
```

---

**Last Updated**: 2025-10-30
**Version**: 3.0
**Status**: Active ✅
