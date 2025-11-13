#!/usr/bin/env python3
"""
Analyze temperature ablation results to find optimal temperature per model
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMP_ABLATION_DIR = PROJECT_ROOT / "results" / "temperature_ablation"
TEMP_SUMMARY_CSV = TEMP_ABLATION_DIR / "temperature_ablation_summary_complete.csv"

def load_temperature_data():
    """Load temperature ablation summary"""
    df = pd.read_csv(TEMP_SUMMARY_CSV, encoding='utf-8-sig')
    return df

def calculate_average_performance(df):
    """Calculate average performance per model and temperature"""

    # Group by model and temperature
    grouped = df.groupby(['model', 'temperature'])

    results = []
    for (model, temp), group in grouped:
        # Calculate average across all tasks (normalize each metric to 0-1 range)
        task_scores = {}

        for _, row in group.iterrows():
            task = row['task']

            # Use appropriate metric for each task
            if task == 'classification':
                score = row['f1'] if pd.notna(row['f1']) else 0
            elif task == 'retrieval':
                score = row['accuracy'] if pd.notna(row['accuracy']) else 0
            elif task == 'punctuation':
                score = row['char_f1'] if pd.notna(row['char_f1']) else 0
            elif task == 'nli':
                score = row['f1'] if pd.notna(row['f1']) else 0
            elif task == 'translation':
                score = row['rougeL_f1'] if pd.notna(row['rougeL_f1']) else 0
            else:
                score = 0

            task_scores[task] = score

        avg_score = np.mean(list(task_scores.values())) if task_scores else 0

        results.append({
            'model': model,
            'temperature': temp,
            'avg_performance': avg_score,
            'num_tasks': len(task_scores),
            **{f'{task}_score': task_scores.get(task, 0) for task in ['classification', 'retrieval', 'punctuation', 'nli', 'translation']}
        })

    return pd.DataFrame(results)

def find_optimal_temperature(avg_df):
    """Find optimal temperature for each model"""

    optimal_temps = []

    for model in avg_df['model'].unique():
        model_data = avg_df[avg_df['model'] == model]

        # Find temperature with highest average performance
        best_idx = model_data['avg_performance'].idxmax()
        best_row = model_data.loc[best_idx]

        optimal_temps.append({
            'model': model,
            'optimal_temperature': best_row['temperature'],
            'best_avg_performance': best_row['avg_performance'],
            'all_temps': dict(zip(model_data['temperature'], model_data['avg_performance']))
        })

    return pd.DataFrame(optimal_temps)

def main():
    print("\n" + "="*70)
    print("Temperature Ablation Analysis")
    print("="*70)

    # Load data
    print("\n📊 Loading temperature ablation data...")
    df = load_temperature_data()
    print(f"  ✓ Loaded {len(df)} rows")

    # Calculate average performance
    print("\n📊 Calculating average performance per model/temperature...")
    avg_df = calculate_average_performance(df)
    print(f"  ✓ Calculated for {len(avg_df)} model-temperature combinations")

    # Find optimal temperature
    print("\n📊 Finding optimal temperature for each model...")
    optimal_df = find_optimal_temperature(avg_df)

    print("\n" + "="*70)
    print("Optimal Temperature Results")
    print("="*70)

    for _, row in optimal_df.iterrows():
        model_name = row['model'].split('/')[-1] if '/' in row['model'] else row['model']
        print(f"\n{model_name}:")
        print(f"  Optimal Temperature: {row['optimal_temperature']}")
        print(f"  Best Avg Performance: {row['best_avg_performance']:.4f}")
        print(f"  All temperatures:")
        for temp, perf in sorted(row['all_temps'].items()):
            marker = " ⭐" if temp == row['optimal_temperature'] else ""
            print(f"    temp={temp}: {perf:.4f}{marker}")

    # Save results
    output_path = TEMP_ABLATION_DIR / "optimal_temperatures.csv"
    optimal_df[['model', 'optimal_temperature', 'best_avg_performance']].to_csv(
        output_path, index=False, encoding='utf-8-sig'
    )
    print(f"\n✅ Results saved to: {output_path}")

    # Overall recommendation
    print("\n" + "="*70)
    print("Overall Recommendation")
    print("="*70)

    temp_counts = optimal_df['optimal_temperature'].value_counts()
    print(f"\nTemperature distribution across all models:")
    for temp, count in sorted(temp_counts.items()):
        print(f"  temp={temp}: {count} models")

    most_common_temp = temp_counts.idxmax()
    print(f"\n💡 Most common optimal temperature: {most_common_temp}")
    print(f"   ({temp_counts[most_common_temp]}/{len(optimal_df)} models)")

    # Calculate average performance by temperature
    print(f"\n📊 Average performance across all models by temperature:")
    for temp in sorted(avg_df['temperature'].unique()):
        temp_data = avg_df[avg_df['temperature'] == temp]
        avg_perf = temp_data['avg_performance'].mean()
        print(f"  temp={temp}: {avg_perf:.4f}")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
