#!/usr/bin/env python3
"""
Create aggregated summary using optimal temperature (0.0)
This will be used to generate updated figures
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMP_ABLATION_DIR = PROJECT_ROOT / "results" / "temperature_ablation"
AGGREGATED_DIR = PROJECT_ROOT / "results" / "aggregated"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TEMP_SUMMARY_CSV = TEMP_ABLATION_DIR / "temperature_ablation_summary_complete.csv"
OPTIMAL_TEMP = 0.0  # Based on analysis: best overall performance

def main():
    print("\n" + "="*70)
    print("Creating Aggregated Summary with Optimal Temperature")
    print("="*70)

    # Load temperature ablation data
    print(f"\n📊 Loading temperature ablation data...")
    df = pd.read_csv(TEMP_SUMMARY_CSV, encoding='utf-8-sig')
    print(f"  ✓ Loaded {len(df)} rows")

    # Filter for optimal temperature only
    print(f"\n📊 Filtering for temperature = {OPTIMAL_TEMP}...")
    df_optimal = df[df['temperature'] == OPTIMAL_TEMP].copy()
    print(f"  ✓ Filtered to {len(df_optimal)} rows")

    # Create primary_metric column based on task
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

    df_optimal['primary_metric'] = df_optimal.apply(get_primary_metric, axis=1)

    # Add model_type based on model name
    def get_model_type(model_name):
        if model_name.startswith('gpt-') or model_name.startswith('claude-'):
            return 'api'
        else:
            return 'opensource'

    df_optimal['model_type'] = df_optimal['model'].apply(get_model_type)

    # Prepare output DataFrame in aggregated format
    output_columns = [
        'model',
        'model_type',
        'task',
        'num_samples',
        'primary_metric',
        'accuracy',
        'f1',
        'char_f1',
        'rougeL_f1',
        'bleu'
    ]

    df_output = df_optimal[output_columns].copy()

    # Sort by model and task
    df_output = df_output.sort_values(['model', 'task'])

    # Save to aggregated directory
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGGREGATED_DIR / "aggregated_summary_temp0.0.csv"
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ Saved aggregated summary to: {output_path}")
    print(f"   Total rows: {len(df_output)}")
    print(f"   Models: {df_output['model'].nunique()}")
    print(f"   Tasks: {df_output['task'].nunique()}")

    # Show summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Temperature: {OPTIMAL_TEMP}")
    print(f"\n  Models included:")
    for model in sorted(df_output['model'].unique()):
        model_type = df_output[df_output['model'] == model]['model_type'].iloc[0]
        avg_metric = df_output[df_output['model'] == model]['primary_metric'].mean()
        short_name = model.split('/')[-1] if '/' in model else model
        print(f"    - {short_name} ({model_type}): avg = {avg_metric:.4f}")

    # Calculate per-task averages
    print(f"\n  Average performance by task:")
    for task in ['classification', 'retrieval', 'punctuation', 'nli', 'translation']:
        task_data = df_output[df_output['task'] == task]
        avg_perf = task_data['primary_metric'].mean()
        print(f"    - {task}: {avg_perf:.4f}")

    # Overall average
    overall_avg = df_output['primary_metric'].mean()
    print(f"\n  Overall average: {overall_avg:.4f}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Use this aggregated summary to generate updated radar charts:")
    print(f"   python3 notebook/experiments/exp7/exp7_radar_charts.py \\")
    print(f"     --results-csv {output_path} \\")
    print(f"     --output-dir {FIGURES_DIR / 'radar'}")
    print("\n2. Generate other figures using exp7 scripts with temperature data")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
