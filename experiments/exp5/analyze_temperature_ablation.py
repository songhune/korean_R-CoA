#!/usr/bin/env python3
"""
Temperature Ablation Analysis
온도별 성능 분석 스크립트
"""

import json
import pandas as pd
import sys
from pathlib import Path

def analyze_temperature_ablation(results_dir: str):
    """온도 ablation 결과 분석"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    # JSON 파일 수집
    json_files = list(results_path.glob("results_*.json"))

    if not json_files:
        print(f"[INFO] No result files found in {results_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Temperature Ablation Analysis")
    print(f"{'='*80}\n")
    print(f"Found {len(json_files)} result files\n")

    # 데이터 수집
    data = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                result = json.load(f)

                model_name = result.get('model_name', 'unknown')
                temperature = result.get('temperature', 0.0)

                if 'tasks' in result:
                    for task_name, task_data in result['tasks'].items():
                        metrics = task_data.get('metrics', {})

                        row = {
                            'model': model_name,
                            'temperature': temperature,
                            'task': task_name,
                            'num_samples': metrics.get('num_samples', 0)
                        }

                        # 태스크별 메트릭 추가
                        if 'accuracy' in metrics:
                            row['accuracy'] = metrics['accuracy']
                        if 'f1' in metrics:
                            row['f1'] = metrics['f1']
                        if 'char_f1' in metrics:
                            row['char_f1'] = metrics['char_f1']
                        if 'bleu' in metrics:
                            row['bleu'] = metrics['bleu']
                        if 'rougeL_f1' in metrics:
                            row['rougeL_f1'] = metrics['rougeL_f1']

                        data.append(row)
        except Exception as e:
            print(f"[WARNING] Failed to process {json_file.name}: {e}")

    if not data:
        print("[INFO] No data to analyze")
        return

    # DataFrame 생성
    df = pd.DataFrame(data)

    # 요약 출력
    print("="*80)
    print("Results Summary")
    print("="*80)
    print(f"\nModels: {df['model'].unique().tolist()}")
    print(f"Temperatures: {sorted(df['temperature'].unique().tolist())}")
    print(f"Tasks: {df['task'].unique().tolist()}")

    # 모델별 온도별 요약
    print("\n" + "="*80)
    print("Performance by Model and Temperature")
    print("="*80)

    for model in df['model'].unique():
        print(f"\n{'='*80}")
        print(f"Model: {model}")
        print(f"{'='*80}")

        model_df = df[df['model'] == model]

        for temp in sorted(model_df['temperature'].unique()):
            temp_df = model_df[model_df['temperature'] == temp]
            print(f"\n  Temperature: {temp}")

            for task in temp_df['task'].unique():
                task_row = temp_df[temp_df['task'] == task].iloc[0]
                print(f"    {task}:")

                if 'accuracy' in task_row and pd.notna(task_row['accuracy']):
                    print(f"      accuracy: {task_row['accuracy']:.4f}")
                if 'f1' in task_row and pd.notna(task_row['f1']):
                    print(f"      f1: {task_row['f1']:.4f}")
                if 'char_f1' in task_row and pd.notna(task_row['char_f1']):
                    print(f"      char_f1: {task_row['char_f1']:.4f}")
                if 'bleu' in task_row and pd.notna(task_row['bleu']):
                    print(f"      bleu: {task_row['bleu']:.4f}")

    # CSV 저장
    output_csv = results_path / "temperature_ablation_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"[SAVE] Summary saved to: {output_csv}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_temperature_ablation.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    analyze_temperature_ablation(results_dir)
