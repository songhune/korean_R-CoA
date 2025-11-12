#!/usr/bin/env python3
"""
Temperature Ablation Analysis Script
Analyzes results across different temperature settings
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

try:
    from font_fix import setup_korean_fonts_robust
    setup_korean_fonts_robust()
except ImportError:
    print("[WARNING] font_fix not available, Korean fonts may not display correctly")

def load_results(results_dir):
    """Load all result files"""
    results_dir = Path(results_dir)
    all_results = []

    for json_file in results_dir.glob("results_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = data['model_name']
        temperature = data.get('temperature', 0.0)

        for task_name, task_data in data['tasks'].items():
            metrics = task_data['metrics']

            row = {
                'model': model_name,
                'temperature': temperature,
                'task': task_name,
                **metrics
            }
            all_results.append(row)

    return pd.DataFrame(all_results)

def create_visualizations(df, output_dir):
    """Create visualization plots"""
    output_dir = Path(output_dir)

    # Main metric per task
    metric_map = {
        'classification': 'accuracy',
        'retrieval': 'accuracy',
        'punctuation': 'char_f1',  # Fixed: use char_f1 instead of rougeL_f1
        'nli': 'accuracy',
        'translation': 'bleu'
    }

    models = df['model'].unique()

    for model in models:
        model_df = df[df['model'] == model]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Temperature Ablation: {model}', fontsize=16, fontweight='bold')

        for idx, (task, metric) in enumerate(metric_map.items()):
            ax = axes[idx // 3, idx % 3]
            task_df = model_df[model_df['task'] == task]

            if not task_df.empty and metric in task_df.columns:
                # Sort by temperature for proper line plot
                task_df = task_df.sort_values('temperature')
                ax.plot(task_df['temperature'], task_df[metric], marker='o', linewidth=2, markersize=8)
                ax.set_xlabel('Temperature')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(task.upper())
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0)

        # Remove unused subplot
        if len(metric_map) < 6:
            axes[1, 2].axis('off')

        plt.tight_layout()
        safe_model_name = model.replace('/', '_')
        plt.savefig(output_dir / f'temperature_ablation_{safe_model_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Visualizations saved to {output_dir}")

def generate_summary_table(df, output_dir):
    """Generate summary CSV"""
    output_dir = Path(output_dir)

    # Pivot table: model x temperature x task
    summary_path = output_dir / 'temperature_ablation_summary.csv'
    df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    print(f"✓ Summary saved to {summary_path}")

    # Print best temperature per task per model
    print("\n" + "="*70)
    print("Best Temperature per Task per Model")
    print("="*70)

    metric_map = {
        'classification': 'accuracy',
        'retrieval': 'accuracy',
        'punctuation': 'char_f1',  # Fixed: use char_f1 instead of rougeL_f1
        'nli': 'accuracy',
        'translation': 'bleu'
    }

    for model in df['model'].unique():
        print(f"\n{model}:")
        model_df = df[df['model'] == model]

        for task, metric in metric_map.items():
            task_df = model_df[model_df['task'] == task]
            if not task_df.empty and metric in task_df.columns:
                best_row = task_df.loc[task_df[metric].idxmax()]
                print(f"  {task:15s}: temp={best_row['temperature']:.1f}, {metric}={best_row[metric]:.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_temperature_ablation.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    print(f"Loading results from {results_dir}...")
    df = load_results(results_dir)

    if df.empty:
        print("No results found!")
        sys.exit(1)

    print(f"✓ Loaded {len(df)} result rows")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Temperatures: {sorted(df['temperature'].unique().tolist())}")
    print(f"  Tasks: {df['task'].unique().tolist()}")

    print("\nGenerating visualizations...")
    create_visualizations(df, results_dir)

    print("\nGenerating summary table...")
    generate_summary_table(df, results_dir)

    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
