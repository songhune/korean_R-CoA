#!/usr/bin/env python3
"""
모든 평가 결과를 하나의 거대 CSV로 통합
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def load_temperature_ablation_results(results_dir):
    """Temperature ablation 결과 로드"""
    results_dir = Path(results_dir)
    all_results = []

    for result_file in results_dir.glob("results_*_temp*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = data['model_name']
            temperature = data.get('temperature', 0.0)
            timestamp = result_file.stem.split('_')[-2] + '_' + result_file.stem.split('_')[-1]

            # 각 태스크별 결과 추출
            for task_name, task_data in data.get('tasks', {}).items():
                row = {
                    'source': 'temperature_ablation',
                    'model_name': model_name,
                    'temperature': temperature,
                    'task': task_name,
                    'timestamp': timestamp,
                    'num_samples': task_data.get('num_samples', 0),
                }

                # 메트릭 추가
                for metric_name, metric_value in task_data.get('metrics', {}).items():
                    row[f'metric_{metric_name}'] = metric_value

                all_results.append(row)

        except Exception as e:
            print(f"Error loading {result_file}: {e}", file=sys.stderr)

    return all_results

def load_full_predictions_results(predictions_dir):
    """Full predictions 결과 로드"""
    predictions_dir = Path(predictions_dir)
    all_results = []

    for pred_file in predictions_dir.glob("full_predictions_*.json"):
        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = data['model_name']
            temperature = data.get('temperature', 0.0)
            task_name = data.get('task', 'unknown')
            num_samples = data.get('num_samples', len(data.get('predictions', [])))

            row = {
                'source': 'full_predictions',
                'model_name': model_name,
                'temperature': temperature,
                'task': task_name,
                'timestamp': pred_file.stem.split('_temp')[1].replace('.json', ''),
                'num_samples': num_samples,
                'has_full_predictions': True,
                'predictions_count': len(data.get('predictions', []))
            }

            all_results.append(row)

        except Exception as e:
            print(f"Error loading {pred_file}: {e}", file=sys.stderr)

    return all_results

def load_confusion_matrix_results(cm_dir):
    """Confusion matrix 결과 로드"""
    cm_dir = Path(cm_dir)
    all_results = []

    # 개별 모델 리포트
    for report_file in cm_dir.glob("confusion_matrix_*_report.txt"):
        try:
            model_name = report_file.stem.replace('confusion_matrix_', '').replace('_report', '')

            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 정확도 추출 (간단한 파싱)
            lines = content.split('\n')
            for line in lines:
                if 'accuracy' in line.lower() or 'macro avg' in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            accuracy = float(parts[-2])
                            row = {
                                'source': 'confusion_matrix',
                                'model_name': model_name,
                                'temperature': 0.0,
                                'task': 'classification',
                                'timestamp': 'latest',
                                'metric_accuracy': accuracy,
                            }
                            all_results.append(row)
                            break
                        except:
                            pass

        except Exception as e:
            print(f"Error loading {report_file}: {e}", file=sys.stderr)

    return all_results

def consolidate_all_results(output_path):
    """모든 결과 통합"""
    print("=" * 70)
    print("CONSOLIDATING ALL RESULTS")
    print("=" * 70)

    all_data = []

    # 1. Temperature ablation 결과
    print("\n[1/3] Loading temperature ablation results...")
    temp_results = load_temperature_ablation_results('../../results/temperature_ablation')
    print(f"  Found {len(temp_results)} records")
    all_data.extend(temp_results)

    # 2. Full predictions 결과
    print("\n[2/3] Loading full predictions results...")
    pred_results = load_full_predictions_results('../../results/full_predictions')
    print(f"  Found {len(pred_results)} records")
    all_data.extend(pred_results)

    # 3. Confusion matrix 결과
    print("\n[3/3] Loading confusion matrix results...")
    cm_results = load_confusion_matrix_results('../../results/confusion_matrices_full')
    print(f"  Found {len(cm_results)} records")
    all_data.extend(cm_results)

    # DataFrame 생성
    print("\n[CONSOLIDATE] Creating unified dataframe...")
    df = pd.DataFrame(all_data)

    # 정렬
    df = df.sort_values(['model_name', 'task', 'temperature', 'source'])

    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n[SAVE] Consolidated results saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Unique models: {df['model_name'].nunique()}")
    print(f"  Unique tasks: {df['task'].nunique()}")
    print(f"  Temperature values: {sorted(df['temperature'].unique())}")

    # 요약 통계
    print("\n[SUMMARY] Records per source:")
    print(df['source'].value_counts())

    print("\n[SUMMARY] Records per model:")
    print(df['model_name'].value_counts())

    print("\n[SUMMARY] Records per task:")
    print(df['task'].value_counts())

    return df

if __name__ == '__main__':
    output_file = '../../results/aggregated/consolidated_all_results.csv'
    df = consolidate_all_results(output_file)

    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
