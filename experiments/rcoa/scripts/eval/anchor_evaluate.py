"""
R-CoA Anchor Head Evaluation
Evaluate cross-lingual alignment quality
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import XLMRobertaTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.anchor_head_model import AnchorHead


class AnchorEvaluator:
    """Evaluator for Anchor Head"""

    def __init__(self, model: AnchorHead, tokenizer: XLMRobertaTokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            # Encode
            batch_embeddings = self.model.encode(
                encodings['input_ids'].to(self.device),
                encodings['attention_mask'].to(self.device)
            )

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def evaluate_retrieval_accuracy(self,
                                    anchor_texts: List[str],
                                    positive_texts: List[str],
                                    k: int = 10) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy

        For each anchor, retrieve top-k most similar positives
        and compute Recall@k
        """
        print(f"\n[EVAL] Retrieval Accuracy (Recall@{k})")

        # Encode
        print("  Encoding anchors...")
        anchor_embeddings = self.encode_texts(anchor_texts)

        print("  Encoding positives...")
        positive_embeddings = self.encode_texts(positive_texts)

        # Compute similarity matrix
        print("  Computing similarities...")
        similarities = cosine_similarity(anchor_embeddings, positive_embeddings)

        # For each anchor, check if correct positive is in top-k
        recalls = []
        for i in range(len(anchor_texts)):
            # Get top-k indices
            top_k_indices = np.argsort(similarities[i])[-k:][::-1]

            # Check if correct index (i) is in top-k
            recalls.append(1.0 if i in top_k_indices else 0.0)

        recall_at_k = np.mean(recalls)

        # Also compute MRR (Mean Reciprocal Rank)
        mrrs = []
        for i in range(len(anchor_texts)):
            # Get rank of correct positive
            ranks = np.argsort(similarities[i])[::-1]
            rank = np.where(ranks == i)[0][0] + 1
            mrrs.append(1.0 / rank)

        mrr = np.mean(mrrs)

        print(f"  Recall@{k}: {recall_at_k:.4f}")
        print(f"  MRR: {mrr:.4f}")

        return {
            f'recall@{k}': recall_at_k,
            'mrr': mrr
        }

    def evaluate_semantic_similarity(self,
                                     anchor_texts: List[str],
                                     positive_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate semantic similarity correlation

        Compute cosine similarity between anchor-positive pairs
        """
        print(f"\n[EVAL] Semantic Similarity")

        # Encode
        print("  Encoding...")
        anchor_embeddings = self.encode_texts(anchor_texts)
        positive_embeddings = self.encode_texts(positive_texts)

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(anchor_texts)):
            sim = np.dot(anchor_embeddings[i], positive_embeddings[i]) / (
                np.linalg.norm(anchor_embeddings[i]) * np.linalg.norm(positive_embeddings[i])
            )
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        print(f"  Avg Cosine Similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")

        return {
            'avg_cosine_similarity': avg_similarity,
            'std_cosine_similarity': std_similarity
        }


def load_test_data(data_path: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load test data"""
    anchor_texts = []
    positive_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            item = json.loads(line.strip())
            anchor_texts.append(item['classical_chinese'])
            positive_texts.append(item['modern_chinese'])

    return anchor_texts, positive_texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate R-CoA Anchor Head")

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='xlm-roberta-base',
                       help='HuggingFace model name')

    # Data
    parser.add_argument('--test-data', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/data/splits/val_pairs.jsonl',
                       help='Path to test data')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of samples to evaluate')

    # Eval
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--retrieval-k', type=int, default=10,
                       help='K for Recall@K')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("="*70)
    print("R-CoA Anchor Head Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load checkpoint
    print("\n[LOAD] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get model args from checkpoint
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        lora_r = model_args.get('lora_r', 8)
        lora_alpha = model_args.get('lora_alpha', 16)
        projection_dim = model_args.get('projection_dim', 256)
        pooling = model_args.get('pooling', 'mean')
    else:
        # Use defaults
        lora_r = 8
        lora_alpha = 16
        projection_dim = 256
        pooling = 'mean'

    # Initialize model
    print("\n[INIT] Initializing model...")
    model = AnchorHead(
        model_name=args.model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        projection_dim=projection_dim,
        pooling=pooling
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)

    # Initialize evaluator
    evaluator = AnchorEvaluator(model, tokenizer, args.device)

    # Load test data
    print(f"\n[LOAD] Loading test data...")
    anchor_texts, positive_texts = load_test_data(args.test_data, args.max_samples)
    print(f"  Test samples: {len(anchor_texts):,}")

    # Evaluate
    results = {}

    # 1. Retrieval accuracy
    retrieval_metrics = evaluator.evaluate_retrieval_accuracy(
        anchor_texts, positive_texts, k=args.retrieval_k
    )
    results.update(retrieval_metrics)

    # 2. Semantic similarity
    similarity_metrics = evaluator.evaluate_semantic_similarity(
        anchor_texts, positive_texts
    )
    results.update(similarity_metrics)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    results_serializable = {k: float(v) if hasattr(v, 'item') else v
                           for k, v in results.items()}

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("[COMPLETE] Evaluation finished")
    print(f"{'='*70}")
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
