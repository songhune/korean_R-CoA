#!/usr/bin/env python3
"""
Visualize temperature ablation results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from font_fix import setup_korean_fonts_robust

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMP_ABLATION_DIR = PROJECT_ROOT / "results" / "temperature_ablation"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TEMP_SUMMARY_CSV = TEMP_ABLATION_DIR / "temperature_ablation_summary_complete.csv"

def configure_publication_style():
    """Configure Matplotlib for consistent A4-friendly PDF output."""
    selected_font = setup_korean_fonts_robust()
    if not selected_font:
        selected_font = 'AppleGothic'
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams.update({
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
    })
    return selected_font

def calculate_average_performance(df):
    """Calculate average performance per model and temperature"""

    def get_primary_metric(row):
        task = row['task']
        if task == 'classification':
            return row['f1'] if pd.notna(row['f1']) else 0
        elif task == 'retrieval':
            return row['accuracy'] if pd.notna(row['accuracy']) else 0
        elif task == 'punctuation':
            return row['char_f1'] if pd.notna(row['char_f1']) else 0
        elif task == 'nli':
            return row['f1'] if pd.notna(row['f1']) else 0
        elif task == 'translation':
            return row['rougeL_f1'] if pd.notna(row['rougeL_f1']) else 0
        else:
            return 0

    df['primary_metric'] = df.apply(get_primary_metric, axis=1)

    # Group by model and temperature, calculate average
    avg_df = df.groupby(['model', 'temperature'])['primary_metric'].mean().reset_index()
    avg_df.columns = ['model', 'temperature', 'avg_performance']

    return avg_df

def create_temperature_heatmap(df_avg):
    """Create heatmap showing performance across temperatures"""
    print("\n📊 Creating temperature ablation heatmap...")

    # Pivot for heatmap
    pivot = df_avg.pivot(index='model', columns='temperature', values='avg_performance')

    # Shorten model names
    pivot.index = [name.split('/')[-1] if '/' in name else name for name in pivot.index]

    # Sort by average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    pivot = pivot.drop('avg', axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1.0, cbar_kws={'label': 'Average Performance'},
                ax=ax, linewidths=0.5)

    ax.set_title('Temperature Ablation: Model Performance Heatmap',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Temperature', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'temperature_ablation_heatmap.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")

def create_temperature_line_plot(df_avg):
    """Create line plot showing temperature effects per model"""
    print("\n📊 Creating temperature ablation line plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for model in df_avg['model'].unique():
        model_data = df_avg[df_avg['model'] == model].sort_values('temperature')
        short_name = model.split('/')[-1] if '/' in model else model

        ax.plot(model_data['temperature'], model_data['avg_performance'],
               'o-', linewidth=2, markersize=8, label=short_name)

    ax.set_xlabel('Temperature', fontsize=13)
    ax.set_ylabel('Average Performance', fontsize=13)
    ax.set_title('Temperature Ablation: Effect on Model Performance',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0.0, 0.3, 0.7])
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'temperature_ablation_lines.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")

def create_temperature_bar_plot(df_avg):
    """Create bar plot comparing optimal temperatures"""
    print("\n📊 Creating temperature optimal comparison bar plot...")

    # Find optimal temperature per model
    optimal_temps = []
    for model in df_avg['model'].unique():
        model_data = df_avg[df_avg['model'] == model]
        best_idx = model_data['avg_performance'].idxmax()
        best_row = model_data.loc[best_idx]

        optimal_temps.append({
            'model': model,
            'optimal_temp': best_row['temperature'],
            'best_performance': best_row['avg_performance']
        })

    opt_df = pd.DataFrame(optimal_temps)
    opt_df['short_name'] = opt_df['model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    opt_df = opt_df.sort_values('best_performance', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by optimal temperature
    colors = opt_df['optimal_temp'].map({0.0: '#4ECDC4', 0.3: '#FF6B6B', 0.7: '#FFA07A'})

    bars = ax.bar(range(len(opt_df)), opt_df['best_performance'], color=colors, edgecolor='black', linewidth=1)

    # Add temperature labels on bars
    for i, (idx, row) in enumerate(opt_df.iterrows()):
        ax.text(i, row['best_performance'] + 0.02, f"T={row['optimal_temp']:.1f}",
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(opt_df)))
    ax.set_xticklabels(opt_df['short_name'], rotation=45, ha='right')
    ax.set_ylabel('Best Performance', fontsize=13)
    ax.set_title('Optimal Temperature per Model',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', edgecolor='black', label='T=0.0'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='T=0.3'),
        Patch(facecolor='#FFA07A', edgecolor='black', label='T=0.7')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'temperature_optimal_comparison.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")

def create_task_temperature_heatmap(df):
    """Create heatmap showing temperature effect per task"""
    print("\n📊 Creating per-task temperature heatmap...")

    def get_primary_metric(row):
        task = row['task']
        if task == 'classification':
            return row['f1'] if pd.notna(row['f1']) else 0
        elif task == 'retrieval':
            return row['accuracy'] if pd.notna(row['accuracy']) else 0
        elif task == 'punctuation':
            return row['char_f1'] if pd.notna(row['char_f1']) else 0
        elif task == 'nli':
            return row['f1'] if pd.notna(row['f1']) else 0
        elif task == 'translation':
            return row['rougeL_f1'] if pd.notna(row['rougeL_f1']) else 0
        else:
            return 0

    df['primary_metric'] = df.apply(get_primary_metric, axis=1)

    # Average across all models per task and temperature
    task_avg = df.groupby(['task', 'temperature'])['primary_metric'].mean().reset_index()
    pivot = task_avg.pivot(index='task', columns='temperature', values='primary_metric')

    # Reorder tasks
    task_order = ['classification', 'retrieval', 'punctuation', 'nli', 'translation']
    pivot = pivot.reindex(task_order)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1.0, cbar_kws={'label': 'Average Performance'},
                ax=ax, linewidths=0.5)

    ax.set_title('Temperature Effect by Task (Averaged Across Models)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Temperature', fontsize=13)
    ax.set_ylabel('Task', fontsize=13)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'temperature_by_task_heatmap.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")

def main():
    print("\n" + "="*70)
    print("Temperature Ablation Visualization")
    print("="*70)

    # Setup fonts
    configure_publication_style()

    # Load data
    print("\n📊 Loading temperature ablation data...")
    df = pd.read_csv(TEMP_SUMMARY_CSV, encoding='utf-8-sig')
    print(f"  ✓ Loaded {len(df)} rows")

    # Calculate average performance
    print("\n📊 Calculating average performance...")
    df_avg = calculate_average_performance(df)
    print(f"  ✓ Calculated for {len(df_avg)} model-temperature combinations")

    # Create visualizations
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    create_temperature_heatmap(df_avg)
    create_temperature_line_plot(df_avg)
    create_temperature_bar_plot(df_avg)
    create_task_temperature_heatmap(df)

    print("\n" + "="*70)
    print("✅ Temperature ablation visualizations completed!")
    print(f"📁 All outputs saved to: {FIGURES_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
