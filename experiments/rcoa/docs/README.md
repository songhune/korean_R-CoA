# R-CoA: Relational Chain-of-Anchor

**Proof-of-Concept Implementation for Cross-lingual Classical Chinese Understanding**

## 📖 Overview

R-CoA (Relational Chain-of-Anchor) is a framework for cross-lingual alignment and relational reasoning in classical Chinese literature. This PoC implements the **Anchor Head** component with InfoNCE loss for cross-lingual semantic alignment.

### Key Features

- **Cross-lingual Alignment**: Classical Chinese ↔ Modern Chinese ↔ Korean ↔ English
- **Efficient Fine-tuning**: XLM-RoBERTa + LoRA adapters
- **Contrastive Learning**: InfoNCE loss for semantic similarity
- **Scalable**: Works with limited multilingual data

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Anchor Head                          │
│                                                         │
│  Input Text (Classical Chinese)                        │
│       │                                                 │
│       ▼                                                 │
│  XLM-RoBERTa Encoder (+ LoRA)                          │
│       │                                                 │
│       ▼                                                 │
│  Pooling (Mean/CLS/Max)                                │
│       │                                                 │
│       ▼                                                 │
│  Projection Head (Hidden → 256d)                       │
│       │                                                 │
│       ▼                                                 │
│  Embeddings ──────────┐                                │
│                       │                                 │
│  Positive (Modern Chinese/Korean/English)              │
│       │               │                                 │
│       ▼               │                                 │
│  [Same pipeline]      │                                 │
│       │               │                                 │
│       ▼               ▼                                 │
│  Embeddings ──► InfoNCE Loss                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 📊 Data

### Sources

1. **ACCN-INS** (`/home/work/songhune/ACCN-INS.json`)
   - TongGu model's training data
   - Classical Chinese ↔ Modern Chinese pairs
   - ~48M lines

2. **Combined Translations** (`combined_ACCN-INS_chunks.jsonl`)
   - Subset with Korean/English translations
   - ~18k samples

3. **KLSBench** (Korean Literary Style Benchmark)
   - Classification, NLI, Retrieval, Punctuation, Translation tasks
   - 7,871 samples

### Preprocessing

```bash
python data_preprocessing.py
```

Output:
- `data/train_pairs.jsonl`: Training pairs
- `data/val_pairs.jsonl`: Validation pairs
- `data/statistics.json`: Dataset statistics

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n rcoa python=3.10
conda activate rcoa
pip install -r requirements.txt
```

### 2. Run PoC Pipeline

```bash
# Run all steps (preprocessing + training + evaluation)
./run_poc.sh all

# Or run individual steps
./run_poc.sh 1  # Data preprocessing
./run_poc.sh 2  # Quick test (1 epoch)
./run_poc.sh 3  # Full training (10 epochs)
./run_poc.sh 4  # Evaluation
```

### 3. Manual Training

```bash
python anchor_train.py \
    --train-data data/train_pairs.jsonl \
    --val-data data/val_pairs.jsonl \
    --batch-size 32 \
    --epochs 10 \
    --lr 2e-5 \
    --lora-r 8 \
    --lora-alpha 16 \
    --projection-dim 256 \
    --temperature 0.07 \
    --output-dir checkpoints/anchor_head \
    --device cuda
```

### 4. Evaluation

```bash
python anchor_evaluate.py \
    --checkpoint checkpoints/anchor_head/best_model.pt \
    --test-data data/val_pairs.jsonl \
    --max-samples 1000 \
    --retrieval-k 10 \
    --output-dir results \
    --device cuda
```

## 📈 Evaluation Metrics

### 1. Retrieval Accuracy
- **Recall@K**: Proportion of correct positive pairs retrieved in top-K
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct pairs

### 2. Semantic Similarity
- **Cosine Similarity**: Average similarity between anchor-positive pairs
- Higher similarity indicates better alignment

## 📁 Project Structure

```
rcoa/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── plan.md                         # Graduation plan (4 weeks)
├── rcoa_concept.md                 # Concept document
│
├── data_preprocessing.py           # Data pipeline
├── anchor_head_model.py            # Model architecture
├── anchor_train.py                 # Training script
├── anchor_evaluate.py              # Evaluation script
├── run_poc.sh                      # PoC pipeline script
│
├── data/                           # Generated data
│   ├── train_pairs.jsonl
│   ├── val_pairs.jsonl
│   └── statistics.json
│
├── checkpoints/                    # Model checkpoints
│   └── anchor_head/
│       ├── best_model.pt
│       └── checkpoint_epoch*.pt
│
└── results/                        # Evaluation results
    └── evaluation_results.json
```

## 🔬 Expected Results

### Week 1 Goals (Current PoC)

- ✅ Data preprocessing pipeline
- ✅ Anchor Head implementation (XLM-R + LoRA + InfoNCE)
- ✅ Training pipeline
- ✅ Evaluation metrics (Recall@K, MRR, Cosine Similarity)

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@10 | > 0.7 | Top-10 retrieval accuracy |
| MRR | > 0.5 | Mean reciprocal rank |
| Cosine Sim | > 0.6 | Average similarity |

### Baseline Comparison

Compare against:
- Random embeddings (baseline)
- Frozen XLM-R (no training)
- Full fine-tuning XLM-R (without LoRA)

## 🛠️ Hyperparameters

### Model Architecture
- **Encoder**: XLM-RoBERTa-base (270M params)
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Projection dim**: 256
- **Pooling**: Mean pooling

### Training
- **Batch size**: 32
- **Learning rate**: 2e-5
- **Epochs**: 10
- **Warmup ratio**: 0.1
- **Weight decay**: 0.01
- **InfoNCE temperature**: 0.07

## 🔮 Future Work (Week 2-4)

### Week 2: Performance Analysis
- [ ] t-SNE/UMAP visualization of embeddings
- [ ] Cross-lingual alignment quality analysis
- [ ] Ablation studies (LoRA rank, projection dim, temperature)
- [ ] Baseline comparisons

### Week 3: Chain Head
- [ ] Knowledge Graph construction from classical texts
- [ ] TransE implementation for relational reasoning
- [ ] Chain Loss for multi-hop consistency
- [ ] Integration with Anchor Head

### Week 4: Paper & Presentation
- [ ] Results aggregation and analysis
- [ ] Visualization and figures
- [ ] Presentation slides (Marp)
- [ ] Paper draft

## 📚 References

### Papers
- **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **XLM-R**: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (2020)

### Datasets
- **TongGu**: Classical-Modern Chinese translation model
- **KLSBench**: Korean Literary Style Benchmark (this work)

## 📝 Citation

```bibtex
@misc{rcoa2024,
  title={R-CoA: Relational Chain-of-Anchor for Cross-lingual Classical Chinese Understanding},
  author={Your Name},
  year={2024},
  note={Graduation Thesis, Ajou University}
}
```

## 📧 Contact

- Author: [Your Name]
- Email: songhune@ajou.ac.kr
- Institution: Ajou University

## 📄 License

This project is for academic research purposes.

---

**Status**: ✅ Week 1 Complete (Anchor Head PoC)
**Next**: Week 2 - Performance Analysis & Visualization
**Graduation Defense**: 2025.12.12
