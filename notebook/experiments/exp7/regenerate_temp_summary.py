#!/usr/bin/env python3
"""
Regenerate temperature ablation summary including all models (API + opensource)
"""

import json
import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMP_ABLATION_DIR = PROJECT_ROOT / "results" / "temperature_ablation"

def parse_result_filename(filename):
    """Parse model name and temperature from filename"""
    # Pattern: results_{model}_temp{temp}_{timestamp}.json
    # e.g., results_claude-sonnet-4-5-20250929_temp0.0_20251112_201431.json
    # e.g., results_gpt-4-turbo_temp0.3_20251112_164558.json
    # e.g., results_meta-llama_Llama-3.1-8B-Instruct_temp0.0_20251113_003623.json

    match = re.search(r'results_(.+)_temp(\d+\.\d+)_\d+_\d+\.json', filename)
    if match:
        model_part = match.group(1)
        temp = float(match.group(2))

        # Handle different model name formats
        if model_part.startswith('claude-') or model_part.startswith('gpt-'):
            model = model_part
        elif '_' in model_part:
            # For models like "meta-llama_Llama-3.1-8B-Instruct"
            model = model_part.replace('_', '/', 1)
        else:
            model = model_part

        return model, temp

    return None, None

def load_result_json(json_path):
    """Load and parse a result JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_name = data.get('model_name', data.get('model', ''))
    tasks = data.get('tasks', {})

    rows = []
    for task, task_data in tasks.items():
        metrics = task_data.get('metrics', {})

        row = {
            'model': model_name,
            'task': task,
            'num_samples': metrics.get('num_samples', 0),
            'accuracy': metrics.get('accuracy'),
            'f1': metrics.get('f1'),
            'char_f1': metrics.get('char_f1'),
            'rougeL_f1': metrics.get('rougeL_f1'),
            'bleu': metrics.get('bleu'),
        }
        rows.append(row)

    return rows

def main():
    print("\n" + "="*70)
    print("Regenerating Temperature Ablation Summary")
    print("="*70)

    # Find all result JSON files
    result_files = list(TEMP_ABLATION_DIR.glob("results_*.json"))
    print(f"\n📊 Found {len(result_files)} result files")

    all_rows = []

    for json_path in sorted(result_files):
        model, temp = parse_result_filename(json_path.name)

        if model is None or temp is None:
            print(f"  ⚠️  Could not parse: {json_path.name}")
            continue

        # Load the JSON and extract metrics
        rows = load_result_json(json_path)

        # Add temperature to each row
        for row in rows:
            row['temperature'] = temp
            # Override model name if needed (from filename)
            if not row['model']:
                row['model'] = model

        all_rows.extend(rows)
        print(f"  ✓ Processed: {json_path.name} (model={model}, temp={temp})")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Reorder columns
    column_order = ['model', 'temperature', 'task', 'num_samples', 'accuracy', 'f1', 'char_f1', 'rougeL_f1', 'bleu']
    df = df[column_order]

    # Sort by model and temperature
    df = df.sort_values(['model', 'temperature', 'task'])

    # Save
    output_path = TEMP_ABLATION_DIR / "temperature_ablation_summary_complete.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ Saved complete summary to: {output_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Models: {df['model'].nunique()}")
    print(f"   Temperatures: {sorted(df['temperature'].unique())}")

    # Show model breakdown
    print("\n📊 Models included:")
    for model in sorted(df['model'].unique()):
        temps = sorted(df[df['model'] == model]['temperature'].unique())
        print(f"  - {model}: temps={temps}")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
