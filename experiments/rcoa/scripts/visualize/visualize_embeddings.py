"""
Week 2: Embedding Visualization
t-SNE, UMAP visualization for cross-lingual embeddings
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from typing import List, Dict
from transformers import XLMRobertaTokenizer
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.anchor_head_model import AnchorHead

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Try to use Korean font
try:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
except:
    print("[WARNING] Korean font not found, using default")


class EmbeddingVisualizer:
    """Visualize cross-lingual embeddings"""

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
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            batch_embeddings = self.model.encode(
                encodings['input_ids'].to(self.device),
                encodings['attention_mask'].to(self.device)
            )

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def plot_tsne(self,
                  embeddings_dict: Dict[str, np.ndarray],
                  output_path: str,
                  perplexity: int = 30,
                  n_iter: int = 1000):
        """
        Plot t-SNE visualization

        Args:
            embeddings_dict: {'language': embeddings_array}
            output_path: Output file path
        """
        print(f"\n[TSNE] Running t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

        # Combine all embeddings
        all_embeddings = []
        labels = []
        colors = []

        color_map = {
            'classical_chinese': '#FF6B6B',
            'modern_chinese': '#4ECDC4',
            'korean': '#45B7D1',
            'english': '#96CEB4'
        }

        for lang, embs in embeddings_dict.items():
            all_embeddings.append(embs)
            labels.extend([lang] * len(embs))
            colors.extend([color_map.get(lang, '#95A5A6')] * len(embs))

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Plot
        plt.figure(figsize=(12, 8))

        for lang in embeddings_dict.keys():
            mask = np.array(labels) == lang
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color_map.get(lang, '#95A5A6'),
                label=lang.replace('_', ' ').title(),
                alpha=0.6,
                s=20
            )

        plt.legend(loc='best', fontsize=12)
        plt.title('Cross-lingual Embedding Space (t-SNE)', fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SAVE] t-SNE plot saved: {output_path}")

    def plot_similarity_heatmap(self,
                               anchor_embeddings: np.ndarray,
                               positive_embeddings: np.ndarray,
                               output_path: str,
                               sample_size: int = 100):
        """Plot similarity heatmap"""
        print(f"\n[HEATMAP] Creating similarity heatmap (top {sample_size} samples)...")

        # Sample
        if len(anchor_embeddings) > sample_size:
            indices = np.random.choice(len(anchor_embeddings), sample_size, replace=False)
            anchor_embeddings = anchor_embeddings[indices]
            positive_embeddings = positive_embeddings[indices]

        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(anchor_embeddings, positive_embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Cross-lingual Similarity Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Modern Chinese', fontsize=12)
        plt.ylabel('Classical Chinese', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SAVE] Heatmap saved: {output_path}")


def load_data(data_path: str, max_samples: int = 500) -> Dict[str, List[str]]:
    """Load data for visualization"""
    data_dict = {
        'classical_chinese': [],
        'modern_chinese': [],
        'korean': [],
        'english': []
    }

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            item = json.loads(line.strip())
            data_dict['classical_chinese'].append(item['classical_chinese'])
            data_dict['modern_chinese'].append(item['modern_chinese'])

            if item.get('korean'):
                data_dict['korean'].append(item['korean'])
            if item.get('english'):
                data_dict['english'].append(item['english'])

    # Balance dataset
    min_len = min(len(v) for v in data_dict.values() if len(v) > 0)
    for key in data_dict:
        if len(data_dict[key]) > min_len:
            data_dict[key] = data_dict[key][:min_len]

    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Visualize R-CoA embeddings")

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/data/splits/val_pairs.jsonl')
    parser.add_argument('--output-dir', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/results/figures')
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--model-name', type=str, default='xlm-roberta-base')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("R-CoA Embedding Visualization")
    print("="*70)

    # Load checkpoint
    print("\n[LOAD] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model_args = checkpoint.get('args', {})
    model = AnchorHead(
        model_name=args.model_name,
        lora_r=model_args.get('lora_r', 8),
        lora_alpha=model_args.get('lora_alpha', 16),
        projection_dim=model_args.get('projection_dim', 256),
        pooling=model_args.get('pooling', 'mean')
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)

    # Initialize visualizer
    visualizer = EmbeddingVisualizer(model, tokenizer, args.device)

    # Load data
    print(f"\n[LOAD] Loading data (max {args.max_samples} samples)...")
    data_dict = load_data(args.data, args.max_samples)

    for lang, texts in data_dict.items():
        print(f"  {lang}: {len(texts)} samples")

    # Encode all languages
    print("\n[ENCODE] Encoding texts...")
    embeddings_dict = {}
    for lang, texts in data_dict.items():
        if len(texts) > 0:
            print(f"  Encoding {lang}...")
            embeddings_dict[lang] = visualizer.encode_texts(texts)

    # 1. t-SNE visualization
    visualizer.plot_tsne(
        embeddings_dict,
        output_path=str(output_dir / 'tsne_cross_lingual.png'),
        perplexity=30,
        n_iter=1000
    )

    # 2. Similarity heatmap
    visualizer.plot_similarity_heatmap(
        embeddings_dict['classical_chinese'],
        embeddings_dict['modern_chinese'],
        output_path=str(output_dir / 'similarity_heatmap.png'),
        sample_size=100
    )

    print(f"\n{'='*70}")
    print("[COMPLETE] Visualization finished")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
