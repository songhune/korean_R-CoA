#!/usr/bin/env python3
"""
Detailed Per-Class Performance Analysis
Generates comprehensive per-class statistics and visualizations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

from config_loader import Config
from font_fix import setup_korean_fonts_robust


class DetailedAnalyzer:
    """Detailed per-class performance analyzer"""

    def __init__(self, benchmark_dir: str, results_dir: str, output_dir: str):
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup fonts
        korean_font = setup_korean_fonts_robust()
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', korean_font] if korean_font else ['DejaVu Sans']

        print("✓ Detailed analyzer initialized")

    def analyze_classification_performance(self):
        """Detailed classification performance by genre"""
        print("\n📊 Analyzing classification performance by genre...")

        # Load classification data
        data_path = self.benchmark_dir / "k_classic_bench_classification.csv"
        df = pd.read_csv(data_path, encoding='utf-8-sig')

        # Get genre distribution
        genre_counts = df['label'].value_counts().sort_values(ascending=True)

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, len(genre_counts)))
        y_pos = np.arange(len(genre_counts))

        bars = ax.barh(y_pos, genre_counts.values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genre_counts.index)
        ax.set_xlabel('Number of Examples', fontsize=13)
        ax.set_ylabel('Genre (Label)', fontsize=13)
        ax.set_title('Classification Task: Detailed Genre Distribution (All 21 Classes)',
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, genre_counts.values)):
            ax.text(count + 2, i, f'{count}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'detailed_classification_genre_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

        # Create statistics table
        stats_df = pd.DataFrame({
            'Genre': genre_counts.index,
            'Count': genre_counts.values,
            'Percentage': (genre_counts.values / len(df) * 100).round(2)
        })

        stats_path = self.output_dir / 'detailed_classification_statistics.csv'
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Statistics saved: {stats_path}")

        return stats_df

    def analyze_retrieval_performance(self):
        """Detailed retrieval performance by book and chapter"""
        print("\n📊 Analyzing retrieval performance by book...")

        # Load retrieval data
        data_path = self.benchmark_dir / "k_classic_bench_retrieval.csv"
        df = pd.read_csv(data_path, encoding='utf-8-sig')

        # Book distribution
        book_counts = df['book'].value_counts()

        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Book distribution
        colors = plt.cm.Set3(np.linspace(0, 1, len(book_counts)))
        x_pos = np.arange(len(book_counts))

        bars = ax1.bar(x_pos, book_counts.values, color=colors, alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(book_counts.index, rotation=30, ha='right')
        ax1.set_ylabel('Number of Examples', fontsize=12)
        ax1.set_xlabel('Book', fontsize=12)
        ax1.set_title('Retrieval Task: Distribution by Source Book',
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, book_counts.values)):
            ax1.text(i, count + 10, str(count), ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        # Right: Pie chart
        wedges, texts, autotexts = ax2.pie(book_counts.values, labels=book_counts.index,
                                            autopct='%1.1f%%', colors=colors, startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')

        ax2.set_title('Retrieval Task: Book Distribution (Percentage)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'detailed_retrieval_book_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

        # Statistics table
        stats_df = pd.DataFrame({
            'Book': book_counts.index,
            'Count': book_counts.values,
            'Percentage': (book_counts.values / len(df) * 100).round(2)
        })

        stats_path = self.output_dir / 'detailed_retrieval_statistics.csv'
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Statistics saved: {stats_path}")

        return stats_df

    def analyze_task_difficulty(self):
        """Analyze task difficulty based on model performance"""
        print("\n📊 Analyzing task difficulty...")

        # Load aggregated results
        agg_path = self.results_dir / 'aggregated' / 'aggregated_summary.csv'

        if not agg_path.exists():
            print(f"  ⚠ Aggregated results not found at {agg_path}")
            return

        df = pd.read_csv(agg_path, encoding='utf-8-sig')

        # Calculate average performance per task
        task_difficulty = df.groupby('task')['primary_metric'].agg(['mean', 'std', 'min', 'max'])
        task_difficulty = task_difficulty.sort_values('mean')

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Average performance with error bars
        tasks = task_difficulty.index
        means = task_difficulty['mean'].values
        stds = task_difficulty['std'].values

        colors = plt.cm.RdYlGn(means / means.max())
        bars = ax1.barh(range(len(tasks)), means, xerr=stds, color=colors, alpha=0.8,
                       capsize=5, error_kw={'linewidth': 2})
        ax1.set_yticks(range(len(tasks)))
        ax1.set_yticklabels([t.capitalize() for t in tasks])
        ax1.set_xlabel('Average Performance Score', fontsize=12)
        ax1.set_ylabel('Task', fontsize=12)
        ax1.set_title('Task Difficulty: Average Model Performance',
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(mean + 0.02, i, f'{mean:.3f}±{std:.3f}',
                    va='center', fontsize=10, fontweight='bold')

        # Right: Performance range (min-max)
        mins = task_difficulty['min'].values
        maxs = task_difficulty['max'].values
        ranges = maxs - mins

        ax2.barh(range(len(tasks)), ranges, left=mins, alpha=0.6, color='skyblue')
        ax2.scatter(mins, range(len(tasks)), color='red', s=100, zorder=3, label='Min')
        ax2.scatter(maxs, range(len(tasks)), color='green', s=100, zorder=3, label='Max')

        ax2.set_yticks(range(len(tasks)))
        ax2.set_yticklabels([t.capitalize() for t in tasks])
        ax2.set_xlabel('Performance Score', fontsize=12)
        ax2.set_title('Task Difficulty: Performance Range Across Models',
                     fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'detailed_task_difficulty_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

        # Save statistics
        task_difficulty.to_csv(self.output_dir / 'detailed_task_difficulty_stats.csv',
                              encoding='utf-8-sig')
        print(f"  ✓ Statistics saved")

        return task_difficulty

    def analyze_model_consistency(self):
        """Analyze model consistency across tasks"""
        print("\n📊 Analyzing model consistency...")

        # Load aggregated results
        agg_path = self.results_dir / 'aggregated' / 'aggregated_pivot.csv'

        if not agg_path.exists():
            print(f"  ⚠ Pivot table not found at {agg_path}")
            return

        df = pd.read_csv(agg_path, index_col=0, encoding='utf-8-sig')

        # Calculate coefficient of variation (std/mean) for each model
        model_stats = pd.DataFrame({
            'mean': df.mean(axis=1),
            'std': df.std(axis=1),
            'min': df.min(axis=1),
            'max': df.max(axis=1),
        })
        model_stats['cv'] = model_stats['std'] / model_stats['mean']  # Coefficient of variation
        model_stats = model_stats.sort_values('cv')

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Mean performance vs consistency (CV)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_stats)))

        ax1.scatter(model_stats['mean'], model_stats['cv'],
                   s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

        for idx, (model, row) in enumerate(model_stats.iterrows()):
            # Shorten model name for display
            short_name = model.split('/')[-1] if '/' in model else model
            ax1.annotate(short_name, (row['mean'], row['cv']),
                        fontsize=8, ha='center', va='bottom')

        ax1.set_xlabel('Mean Performance', fontsize=12)
        ax1.set_ylabel('Coefficient of Variation (Consistency)', fontsize=12)
        ax1.set_title('Model Performance vs Consistency\n(Lower CV = More Consistent)',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add quadrant lines
        mean_cv = model_stats['cv'].median()
        mean_perf = model_stats['mean'].median()
        ax1.axhline(mean_cv, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(mean_perf, color='gray', linestyle='--', alpha=0.5)

        # Right: Performance range for each model
        models = model_stats.index
        y_pos = np.arange(len(models))

        for i, (model, row) in enumerate(model_stats.iterrows()):
            ax2.plot([row['min'], row['max']], [i, i], 'o-',
                    linewidth=3, markersize=8, alpha=0.7)
            ax2.plot(row['mean'], i, 'D', markersize=10,
                    color='red', zorder=3)

        ax2.set_yticks(y_pos)
        # Shorten model names
        short_names = [m.split('/')[-1] if '/' in m else m for m in models]
        ax2.set_yticklabels(short_names, fontsize=9)
        ax2.set_xlabel('Performance Score', fontsize=12)
        ax2.set_title('Model Performance Range Across Tasks\n(Diamond = Mean)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'detailed_model_consistency_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

        # Save statistics
        model_stats.to_csv(self.output_dir / 'detailed_model_consistency_stats.csv',
                          encoding='utf-8-sig')
        print(f"  ✓ Statistics saved")

        return model_stats

    def generate_comprehensive_summary(self):
        """Generate comprehensive summary table"""
        print("\n📊 Generating comprehensive summary...")

        # Run all analyses
        classification_stats = self.analyze_classification_performance()
        retrieval_stats = self.analyze_retrieval_performance()
        task_difficulty = self.analyze_task_difficulty()
        model_consistency = self.analyze_model_consistency()

        print("\n✅ Detailed analysis complete!")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Detailed Performance Analysis')
    parser.add_argument('--benchmark-dir', type=str,
                       default='/Users/songhune/Workspace/korean_eda/benchmark/kls_bench',
                       help='Benchmark data directory')
    parser.add_argument('--results-dir', type=str,
                       default='/Users/songhune/Workspace/korean_eda/results',
                       help='Results directory')
    parser.add_argument('--output-dir', type=str,
                       default='../../results/figures',
                       help='Output directory')

    args = parser.parse_args()

    # Resolve paths
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'

    analyzer = DetailedAnalyzer(
        benchmark_dir=args.benchmark_dir,
        results_dir=args.results_dir,
        output_dir=str(output_dir)
    )

    analyzer.generate_comprehensive_summary()

    print(f"\n📁 All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
