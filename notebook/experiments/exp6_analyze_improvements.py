#!/usr/bin/env python3
"""
Experiment 6: Few-shot vs Zero-shot Analysis
===========================================

Compares few-shot results (exp6) with zero-shot baseline (exp5) to quantify
the improvement from few-shot learning.

Generates:
1. Comparison table (zero-shot vs 1-shot vs 3-shot)
2. Improvement delta visualization
3. Statistical analysis of improvements
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


class FewShotAnalyzer:
    """Analyze few-shot improvements over zero-shot baseline"""

    def __init__(
        self,
        zeroshot_results_path: str = "../../results/aggregated",
        fewshot_results_path: str = "../../results/fewshot"
    ):
        self.zeroshot_path = Path(zeroshot_results_path)
        self.fewshot_path = Path(fewshot_results_path)
        self.output_path = self.fewshot_path / "analysis"
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_zeroshot_results(self) -> pd.DataFrame:
        """Load zero-shot results from exp5"""
        print("Loading zero-shot results...")

        # Try to load from aggregated results
        agg_file = self.zeroshot_path / "aggregated_results.json"
        if agg_file.exists():
            with open(agg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            records = []
            for model, tasks in data.items():
                if model == 'metadata':
                    continue
                for task, metrics in tasks.items():
                    record = {
                        'model': model,
                        'task': task,
                        'n_shots': 0,  # Zero-shot
                    }
                    if task == 'classification' or task == 'nli':
                        record['accuracy'] = metrics.get('accuracy', 0.0)
                        record['f1'] = metrics.get('f1', 0.0)
                    elif task == 'retrieval':
                        record['accuracy'] = metrics.get('accuracy', 0.0)
                    elif task == 'punctuation':
                        record['f1'] = metrics.get('f1', 0.0)
                    elif task == 'translation':
                        record['bleu'] = metrics.get('bleu', 0.0)

                    records.append(record)

            return pd.DataFrame(records)
        else:
            print(f"Warning: {agg_file} not found. Please run exp5 first.")
            return pd.DataFrame()

    def load_fewshot_results(self) -> pd.DataFrame:
        """Load few-shot results from exp6"""
        print("Loading few-shot results...")

        # Load combined summary if exists
        combined_file = self.fewshot_path / "exp6_fewshot_summary_combined.csv"
        if combined_file.exists():
            return pd.read_csv(combined_file)

        # Otherwise, collect individual summaries
        all_summaries = []
        for csv_file in self.fewshot_path.glob('summary_*.csv'):
            df = pd.read_csv(csv_file)
            all_summaries.append(df)

        if all_summaries:
            return pd.concat(all_summaries, ignore_index=True)
        else:
            print(f"Warning: No few-shot results found in {self.fewshot_path}")
            return pd.DataFrame()

    def compare_results(self) -> pd.DataFrame:
        """Compare zero-shot and few-shot results"""
        print("\nComparing results...")

        zeroshot_df = self.load_zeroshot_results()
        fewshot_df = self.load_fewshot_results()

        if zeroshot_df.empty or fewshot_df.empty:
            print("Error: Missing results. Cannot proceed with comparison.")
            return pd.DataFrame()

        # Filter zero-shot to only classification and NLI (exp6 focus)
        zeroshot_df = zeroshot_df[zeroshot_df['task'].isin(['classification', 'nli'])]

        # Combine dataframes
        combined = pd.concat([zeroshot_df, fewshot_df], ignore_index=True)

        # Calculate improvements
        improvements = []
        for model in combined['model'].unique():
            for task in ['classification', 'nli']:
                model_task = combined[(combined['model'] == model) & (combined['task'] == task)]

                if model_task.empty:
                    continue

                zero_shot = model_task[model_task['n_shots'] == 0]
                one_shot = model_task[model_task['n_shots'] == 1]
                three_shot = model_task[model_task['n_shots'] == 3]

                record = {
                    'model': model,
                    'task': task,
                    'zero_shot_acc': zero_shot['accuracy'].values[0] if len(zero_shot) > 0 else None,
                    'one_shot_acc': one_shot['accuracy'].values[0] if len(one_shot) > 0 else None,
                    'three_shot_acc': three_shot['accuracy'].values[0] if len(three_shot) > 0 else None,
                }

                # Calculate deltas
                if record['zero_shot_acc'] is not None and record['one_shot_acc'] is not None:
                    record['delta_1shot'] = record['one_shot_acc'] - record['zero_shot_acc']
                if record['zero_shot_acc'] is not None and record['three_shot_acc'] is not None:
                    record['delta_3shot'] = record['three_shot_acc'] - record['zero_shot_acc']

                improvements.append(record)

        return pd.DataFrame(improvements)

    def generate_comparison_table(self, improvements_df: pd.DataFrame):
        """Generate formatted comparison table"""
        print("\n" + "="*80)
        print("Few-shot Learning Improvements")
        print("="*80)

        for task in ['classification', 'nli']:
            print(f"\n{task.upper()}")
            print("-"*80)

            task_data = improvements_df[improvements_df['task'] == task].copy()

            if task_data.empty:
                print(f"No results for {task}")
                continue

            # Format for display
            display_df = task_data[[
                'model', 'zero_shot_acc', 'one_shot_acc', 'three_shot_acc',
                'delta_1shot', 'delta_3shot'
            ]].copy()

            display_df.columns = [
                'Model', '0-shot', '1-shot', '3-shot', 'Δ1-shot', 'Δ3-shot'
            ]

            # Format percentages
            for col in ['0-shot', '1-shot', '3-shot']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            for col in ['Δ1-shot', 'Δ3-shot']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"+{x*100:.1f}%" if pd.notna(x) and x > 0 else
                              f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )

            print(display_df.to_string(index=False))

        # Summary statistics
        print("\n" + "="*80)
        print("Summary Statistics")
        print("="*80)

        for task in ['classification', 'nli']:
            task_data = improvements_df[improvements_df['task'] == task]
            if task_data.empty:
                continue

            print(f"\n{task.upper()}:")
            print(f"  Mean 0-shot: {task_data['zero_shot_acc'].mean()*100:.1f}%")
            print(f"  Mean 1-shot: {task_data['one_shot_acc'].mean()*100:.1f}%")
            print(f"  Mean 3-shot: {task_data['three_shot_acc'].mean()*100:.1f}%")
            print(f"  Mean improvement (1-shot): {task_data['delta_1shot'].mean()*100:.1f}%")
            print(f"  Mean improvement (3-shot): {task_data['delta_3shot'].mean()*100:.1f}%")

            # Count models that improved
            improved_1 = (task_data['delta_1shot'] > 0).sum()
            improved_3 = (task_data['delta_3shot'] > 0).sum()
            total = len(task_data)

            print(f"  Models improved (1-shot): {improved_1}/{total}")
            print(f"  Models improved (3-shot): {improved_3}/{total}")

    def plot_improvements(self, improvements_df: pd.DataFrame):
        """Generate improvement visualizations"""
        print("\nGenerating visualizations...")

        # 1. Bar chart: Zero-shot vs Few-shot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, task in enumerate(['classification', 'nli']):
            task_data = improvements_df[improvements_df['task'] == task]

            if task_data.empty:
                continue

            ax = axes[idx]

            # Prepare data for grouped bar chart
            models = task_data['model'].tolist()
            x = np.arange(len(models))
            width = 0.25

            zero_shot = task_data['zero_shot_acc'].tolist()
            one_shot = task_data['one_shot_acc'].fillna(0).tolist()
            three_shot = task_data['three_shot_acc'].fillna(0).tolist()

            ax.bar(x - width, zero_shot, width, label='0-shot', alpha=0.8)
            ax.bar(x, one_shot, width, label='1-shot', alpha=0.8)
            ax.bar(x + width, three_shot, width, label='3-shot', alpha=0.8)

            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{task.upper()} Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.0)

        plt.tight_layout()
        output_file = self.output_path / "fewshot_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

        # 2. Heatmap: Improvement deltas
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, delta_col in enumerate(['delta_1shot', 'delta_3shot']):
            ax = axes[idx]
            shot_num = '1' if delta_col == 'delta_1shot' else '3'

            # Pivot for heatmap
            pivot_data = improvements_df.pivot(
                index='model',
                columns='task',
                values=delta_col
            )

            # Plot heatmap
            sns.heatmap(
                pivot_data * 100,  # Convert to percentage
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'Improvement (%)'},
                ax=ax
            )

            ax.set_title(f'{shot_num}-shot Improvement over Zero-shot')
            ax.set_xlabel('Task')
            ax.set_ylabel('Model')

        plt.tight_layout()
        output_file = self.output_path / "improvement_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

        # 3. Learning curve: 0-shot → 1-shot → 3-shot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, task in enumerate(['classification', 'nli']):
            task_data = improvements_df[improvements_df['task'] == task]

            if task_data.empty:
                continue

            ax = axes[idx]

            for _, row in task_data.iterrows():
                model = row['model']
                shots = [0, 1, 3]
                accs = [
                    row['zero_shot_acc'],
                    row['one_shot_acc'],
                    row['three_shot_acc']
                ]

                # Only plot if we have all data points
                if all(pd.notna(accs)):
                    ax.plot(shots, accs, marker='o', label=model, linewidth=2)

            ax.set_xlabel('Number of Shots')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{task.upper()} Learning Curve')
            ax.set_xticks([0, 1, 3])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)

        plt.tight_layout()
        output_file = self.output_path / "learning_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def save_results(self, improvements_df: pd.DataFrame):
        """Save comparison results"""
        output_file = self.output_path / "fewshot_vs_zeroshot_comparison.csv"
        improvements_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Also save as JSON for easier processing
        output_json = self.output_path / "fewshot_vs_zeroshot_comparison.json"
        improvements_df.to_json(output_json, orient='records', indent=2)
        print(f"Results saved to: {output_json}")

    def run_analysis(self):
        """Run complete analysis"""
        print("="*80)
        print("Experiment 6: Few-shot vs Zero-shot Analysis")
        print("="*80)

        # Load and compare results
        improvements_df = self.compare_results()

        if improvements_df.empty:
            print("\nError: No data to analyze. Please run exp6_run_fewshot.sh first.")
            return

        # Generate outputs
        self.generate_comparison_table(improvements_df)
        self.plot_improvements(improvements_df)
        self.save_results(improvements_df)

        print("\n" + "="*80)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_path}")
        print("="*80)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze few-shot improvements over zero-shot baseline'
    )
    parser.add_argument(
        '--zeroshot-results',
        type=str,
        default='../../results/aggregated',
        help='Path to zero-shot results directory'
    )
    parser.add_argument(
        '--fewshot-results',
        type=str,
        default='../../results/fewshot',
        help='Path to few-shot results directory'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = FewShotAnalyzer(args.zeroshot_results, args.fewshot_results)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
