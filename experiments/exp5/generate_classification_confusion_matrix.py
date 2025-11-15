"""
Classification Confusion Matrix Generator for KLSBench
과문육체 (賦, 詩, 疑, 義, 策, 表) 분류에 대한 Confusion Matrix 생성

모든 모델의 classification 결과에 대해 confusion matrix를 생성하고 시각화합니다.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import re
import unicodedata
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# 한글 폰트 설정 (matplotlib)
# Try to use system fonts that support CJK characters
import matplotlib.font_manager as fm

# Try to find a suitable font for CJK characters
fonts = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic',
         'Arial Unicode MS', 'DejaVu Sans']
available_fonts = [f.name for f in fm.fontManager.ttflist]

font_found = False
for font in fonts:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        font_found = True
        break

if not font_found:
    # Fallback: use sans-serif and suppress warnings
    plt.rcParams['font.family'] = 'sans-serif'
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

plt.rcParams['axes.unicode_minus'] = False


class ConfusionMatrixGenerator:
    """Classification task의 confusion matrix 생성"""

    def __init__(self,
                 benchmark_path: str,
                 results_dir: str,
                 output_dir: str,
                 temperature: float = 0.0,
                 use_full_predictions: bool = False,
                 full_predictions_dir: str = None):
        """
        Args:
            benchmark_path: 벤치마크 JSON 파일 경로
            results_dir: 결과 파일들이 저장된 디렉토리
            output_dir: Confusion matrix 이미지를 저장할 디렉토리
            temperature: 사용할 temperature 값 (결과 파일 필터링용)
            use_full_predictions: 전체 predictions 파일을 사용할지 여부
            full_predictions_dir: 전체 predictions 파일이 저장된 디렉토리
        """
        self.benchmark_path = Path(benchmark_path)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.use_full_predictions = use_full_predictions
        self.full_predictions_dir = Path(full_predictions_dir) if full_predictions_dir else None

        # 과문육체 레이블 (정규화된 형태)
        self.target_labels = ['賦', '詩', '疑', '義', '策', '表']

        # 한글-한자 레이블 매핑 (EXAONE 모델용)
        self.korean_to_hanja_mapping = {
            '부': '賦',
            '시': '詩',
            '의': '義',
            '책': '策',
            '표': '表',
            '의문': '疑',
        }

        # 레이블의 영어 표기 (논문용)
        self.label_display_names = {
            '賦': '賦\n(Fu)',
            '詩': '詩\n(Si)',
            '疑': '疑\n(Eui)',
            '義': '義\n(Ui)',
            '策': '策\n(Chaek)',
            '表': '表\n(Pyo)'
        }

        # Display용 레이블 리스트
        self.display_labels = [self.label_display_names[label] for label in self.target_labels]

        # 모델명 매핑 (파일명 -> 표시명)
        self.model_name_mapping = {
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gpt-3.5-turbo': 'GPT-3.5 Turbo',
            'claude-3-5-sonnet': 'Claude 3.5 Sonnet',
            'claude-3-opus': 'Claude 3 Opus',
            'meta-llama_Llama-3.1-8B-Instruct': 'Llama 3.1 8B Instruct',
            'Qwen_Qwen2.5-7B-Instruct': 'Qwen 2.5 7B Instruct',
            'LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct': 'EXAONE 3.0 7.8B Instruct'
        }

        # Load benchmark
        self.load_benchmark()

    def load_benchmark(self):
        """벤치마크 데이터 로드 (ground truth 가져오기)"""
        print(f"[LOAD] Benchmark: {self.benchmark_path}")
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)

        # Classification task의 ground truth 추출
        if 'tasks' in benchmark and 'classification' in benchmark['tasks']:
            classification_data = benchmark['tasks']['classification']['data']
            self.ground_truths = [self.normalize_label(item['label'])
                                 for item in classification_data]
            print(f"  Loaded {len(self.ground_truths)} ground truth labels")
        else:
            # 개별 classification JSON 파일인 경우
            classification_data = benchmark['data']
            self.ground_truths = [self.normalize_label(item['label'])
                                 for item in classification_data]
            print(f"  Loaded {len(self.ground_truths)} ground truth labels")

        # 레이블 분포 확인
        from collections import Counter
        label_counts = Counter(self.ground_truths)
        print("\n  Label distribution:")
        for label in self.target_labels:
            count = label_counts.get(label, 0)
            print(f"    {label}: {count} ({count/len(self.ground_truths)*100:.1f}%)")

    def normalize_label(self, text: str) -> str:
        """레이블 정규화 (평가 코드와 동일한 로직)"""
        # 괄호 내용 제거
        text = re.sub(r'\([^)]*\)', '', text)
        text = text.strip()

        # 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)

        # 간체자 -> 번체자 변환
        simplified_to_traditional = {
            '赋': '賦',
            '论': '論',
            '诗': '詩',
            '书': '書',
            '传': '傳',
            '记': '記',
        }
        for simp, trad in simplified_to_traditional.items():
            text = text.replace(simp, trad)

        return text

    def extract_label_from_exaone_prediction(self, pred_text: str) -> str:
        """
        EXAONE 모델의 문장 형태 예측에서 레이블 추출
        예: "賦(부)\n\n이 텍스트는..." -> "賦"
            "시(시)\n\n이 텍스트는..." -> "詩"
        """
        if not pred_text or not isinstance(pred_text, str):
            return ""

        # 첫 줄 추출
        first_line = pred_text.split('\n')[0].strip()

        # 공백 전 부분 추출 (첫 토큰)
        parts = first_line.split()
        if not parts:
            return ""

        first_token = parts[0].strip()

        # 패턴 1: 한자(한글) 형식 또는 한글(한글) 형식
        if '(' in first_token:
            label_candidate = first_token.split('(')[0].strip()

            # 한자 확인 (유니코드 범위: 4E00-9FFF)
            han_chars = ''.join(c for c in label_candidate if '\u4e00' <= c <= '\u9fff')
            if han_chars:
                return han_chars

            # 한글인 경우 매핑
            if label_candidate in self.korean_to_hanja_mapping:
                return self.korean_to_hanja_mapping[label_candidate]

        # 패턴 2: 한글만 있는 경우 (괄호 없음)
        if first_token in self.korean_to_hanja_mapping:
            return self.korean_to_hanja_mapping[first_token]

        # 패턴 3: 한자만 있는 경우
        han_chars = ''.join(c for c in first_token if '\u4e00' <= c <= '\u9fff')
        if han_chars:
            # 1자인 경우 그대로 반환
            if len(han_chars) == 1:
                return han_chars
            # 2자 이상인 경우 첫 글자만 (과문육체는 모두 1자)
            return han_chars[0]

        return ""

    def load_result_file(self, result_path: Path) -> Tuple[str, List[str]]:
        """결과 파일 로드 및 예측값 추출"""
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        model_name = result['model_name']
        is_exaone = 'EXAONE' in model_name

        # full_predictions 파일인 경우
        if 'predictions' in result and 'task' in result:
            predictions = result['predictions']
            print(f"  Loaded full predictions: {len(predictions)} samples")

        # 기존 results 파일인 경우
        elif 'classification' in result.get('tasks', {}):
            predictions = result['tasks']['classification']['predictions']

            # 만약 전체 predictions가 없고 샘플만 있는 경우
            if len(predictions) < len(self.ground_truths):
                print(f"  [WARNING] {model_name}: Only {len(predictions)} predictions available")
                print(f"            Expected {len(self.ground_truths)} predictions")
                print(f"            Using available predictions for analysis")
        else:
            raise ValueError(f"No classification predictions found in {result_path}")

        # EXAONE 모델인 경우 특별한 파싱
        if is_exaone:
            print(f"  [INFO] Applying EXAONE-specific label extraction")
            predictions_extracted = [self.extract_label_from_exaone_prediction(p) for p in predictions]
            # 빈 문자열은 원본 텍스트로 대체 (정규화에 맡김)
            predictions = [extracted if extracted else orig
                          for extracted, orig in zip(predictions_extracted, predictions)]

        # 정규화
        predictions_normalized = [self.normalize_label(p) for p in predictions]

        return model_name, predictions_normalized

    def filter_target_labels(self, predictions: List[str], ground_truths: List[str]) -> Tuple[List[str], List[str]]:
        """과문육체 레이블만 필터링"""
        filtered_preds = []
        filtered_truths = []

        for pred, truth in zip(predictions, ground_truths):
            if truth in self.target_labels:
                filtered_preds.append(pred)
                filtered_truths.append(truth)

        return filtered_preds, filtered_truths

    def generate_confusion_matrix(self, model_name: str, predictions: List[str],
                                  ground_truths: List[str], save_path: Path):
        """Confusion matrix 생성 및 시각화"""
        # 과문육체 레이블만 필터링
        predictions, ground_truths = self.filter_target_labels(predictions, ground_truths)

        if len(predictions) == 0:
            print(f"  [ERROR] No valid predictions for {model_name}")
            return

        print(f"\n[GENERATE] Confusion matrix for {model_name}")
        print(f"  Samples: {len(predictions)}")

        # Confusion matrix 계산
        cm = confusion_matrix(ground_truths, predictions, labels=self.target_labels)

        # 정규화된 confusion matrix (행 합이 1이 되도록)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # NaN을 0으로 변환

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 1. 절대값 confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.display_labels,
                   yticklabels=self.display_labels,
                   cbar_kws={'label': 'Count'},
                   ax=axes[0])
        axes[0].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=13, fontweight='bold')
        axes[0].set_title(f'{model_name}\nConfusion Matrix (Counts)', fontsize=15, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=10)

        # 2. 정규화된 confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.display_labels,
                   yticklabels=self.display_labels,
                   cbar_kws={'label': 'Proportion'},
                   vmin=0, vmax=1,
                   ax=axes[1])
        axes[1].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=13, fontweight='bold')
        axes[1].set_title(f'{model_name}\nConfusion Matrix (Normalized)', fontsize=15, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")

        # Classification report 저장
        report = classification_report(ground_truths, predictions,
                                      labels=self.target_labels,
                                      target_names=self.target_labels,
                                      zero_division=0)

        report_path = save_path.parent / f"{save_path.stem}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)

        print(f"  Report saved: {report_path}")

        return cm, cm_normalized

    def find_result_files(self) -> List[Path]:
        """Temperature에 맞는 결과 파일들 찾기"""
        if self.use_full_predictions and self.full_predictions_dir:
            # 전체 predictions 파일 사용
            pattern = f"full_predictions_*temp{self.temperature:.1f}.json"
            result_files = list(self.full_predictions_dir.glob(pattern))
            search_dir = self.full_predictions_dir
            file_type = "full predictions"
        else:
            # 기존 결과 파일 사용 (처음 10개만)
            pattern = f"*temp{self.temperature:.1f}_*.json"
            result_files = list(self.results_dir.glob(pattern))
            # summary 파일 제외
            result_files = [f for f in result_files if not f.name.startswith('summary_')]
            search_dir = self.results_dir
            file_type = "result"

        print(f"\n[SEARCH] Looking for {file_type} files with temperature={self.temperature}")
        print(f"  Directory: {search_dir}")
        print(f"  Pattern: {pattern}")
        print(f"  Found {len(result_files)} files")

        return sorted(result_files)

    def generate_all_matrices(self):
        """모든 모델에 대해 confusion matrix 생성"""
        result_files = self.find_result_files()

        if len(result_files) == 0:
            print("[ERROR] No result files found!")
            return

        print("\n" + "="*70)
        print("GENERATING CONFUSION MATRICES")
        print("="*70)

        all_results = {}

        for result_path in result_files:
            try:
                # 결과 파일 로드
                model_name, predictions = self.load_result_file(result_path)

                # Ground truth 길이에 맞추기
                predictions = predictions[:len(self.ground_truths)]

                # 파일명에서 간단한 모델명 추출
                filename = result_path.stem
                simple_name = filename.replace('results_', '').split('_temp')[0]

                # Display name 매핑
                display_name = self.model_name_mapping.get(simple_name, model_name)

                # Confusion matrix 생성
                save_path = self.output_dir / f"confusion_matrix_{simple_name}.png"
                cm, cm_normalized = self.generate_confusion_matrix(
                    display_name, predictions, self.ground_truths, save_path
                )

                all_results[display_name] = {
                    'confusion_matrix': cm,
                    'confusion_matrix_normalized': cm_normalized,
                    'model_file': simple_name
                }

            except Exception as e:
                print(f"\n[ERROR] Failed to process {result_path.name}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        # 종합 비교 레포트 생성
        self.generate_comparison_report(all_results)

        # 평균 confusion matrix 생성
        if len(all_results) > 0:
            print("\n[GENERATE] Average Confusion Matrix across all models...")
            self.generate_average_confusion_matrix(all_results)

        print("\n" + "="*70)
        print("[COMPLETE] All confusion matrices generated")
        print(f"Output directory: {self.output_dir}")
        print("="*70)

    def generate_comparison_report(self, all_results: Dict):
        """모델 간 비교 레포트 생성"""
        report_path = self.output_dir / "comparison_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION CONFUSION MATRIX COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Target Labels: {', '.join(self.target_labels)}\n")
            f.write(f"Temperature: {self.temperature}\n")
            f.write(f"Number of models: {len(all_results)}\n\n")

            # 각 모델의 per-class accuracy 계산
            f.write("Per-Class Accuracy (Recall):\n")
            f.write("-"*70 + "\n")

            # 헤더
            f.write(f"{'Model':<30} ")
            for label in self.target_labels:
                f.write(f"{label:>8} ")
            f.write("\n")
            f.write("-"*70 + "\n")

            # 각 모델의 결과
            for model_name, results in all_results.items():
                cm = results['confusion_matrix']

                # Per-class recall (diagonal / row sum)
                per_class_recall = []
                for i, label in enumerate(self.target_labels):
                    row_sum = cm[i].sum()
                    if row_sum > 0:
                        recall = cm[i, i] / row_sum
                    else:
                        recall = 0.0
                    per_class_recall.append(recall)

                # 출력
                f.write(f"{model_name:<30} ")
                for recall in per_class_recall:
                    f.write(f"{recall:>7.1%} ")
                f.write("\n")

        print(f"\n[SAVE] Comparison report: {report_path}")

    def generate_average_confusion_matrix(self, all_results: Dict):
        """모든 모델의 평균 confusion matrix 생성"""
        if len(all_results) == 0:
            print("  [ERROR] No results to average")
            return

        # 모든 confusion matrix 수집
        all_cms = []
        all_cms_normalized = []

        for model_name, results in all_results.items():
            all_cms.append(results['confusion_matrix'])
            all_cms_normalized.append(results['confusion_matrix_normalized'])

        # 평균 계산
        avg_cm = np.mean(all_cms, axis=0)
        avg_cm_normalized = np.mean(all_cms_normalized, axis=0)

        print(f"  Averaging {len(all_results)} models")

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 1. 평균 절대값 confusion matrix
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.display_labels,
                   yticklabels=self.display_labels,
                   cbar_kws={'label': 'Average Count'},
                   ax=axes[0])
        axes[0].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=13, fontweight='bold')
        axes[0].set_title(f'Average Across All Models (n={len(all_results)})\nConfusion Matrix (Average Counts)',
                         fontsize=15, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=10)

        # 2. 평균 정규화된 confusion matrix
        sns.heatmap(avg_cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.display_labels,
                   yticklabels=self.display_labels,
                   cbar_kws={'label': 'Average Proportion'},
                   vmin=0, vmax=1,
                   ax=axes[1])
        axes[1].set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=13, fontweight='bold')
        axes[1].set_title(f'Average Across All Models (n={len(all_results)})\nConfusion Matrix (Average Proportions)',
                         fontsize=15, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix_AVERAGE_all_models.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")

        # 평균 성능 통계 저장
        avg_report_path = self.output_dir / "confusion_matrix_AVERAGE_all_models_stats.txt"
        with open(avg_report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("AVERAGE CONFUSION MATRIX STATISTICS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Number of models averaged: {len(all_results)}\n")
            f.write(f"Models: {', '.join(all_results.keys())}\n\n")

            f.write("Average Counts (Confusion Matrix):\n")
            f.write("-"*70 + "\n")
            f.write(f"{'True \\ Pred':<15} ")
            for label in self.target_labels:
                f.write(f"{label:>10} ")
            f.write("\n" + "-"*70 + "\n")

            for i, true_label in enumerate(self.target_labels):
                f.write(f"{true_label:<15} ")
                for j in range(len(self.target_labels)):
                    f.write(f"{avg_cm[i,j]:>10.1f} ")
                f.write("\n")

            f.write("\n\n")
            f.write("Average Proportions (Normalized by Row):\n")
            f.write("-"*70 + "\n")
            f.write(f"{'True \\ Pred':<15} ")
            for label in self.target_labels:
                f.write(f"{label:>10} ")
            f.write("\n" + "-"*70 + "\n")

            for i, true_label in enumerate(self.target_labels):
                f.write(f"{true_label:<15} ")
                for j in range(len(self.target_labels)):
                    f.write(f"{avg_cm_normalized[i,j]:>10.3f} ")
                f.write("\n")

            # Per-class average recall
            f.write("\n\n")
            f.write("Average Per-Class Recall:\n")
            f.write("-"*70 + "\n")
            for i, label in enumerate(self.target_labels):
                avg_recall = avg_cm_normalized[i, i]
                f.write(f"  {label}: {avg_recall:.3f} ({avg_recall*100:.1f}%)\n")

            # Overall average accuracy
            overall_avg_acc = np.trace(avg_cm_normalized) / len(self.target_labels)
            f.write(f"\nOverall Average Accuracy: {overall_avg_acc:.3f} ({overall_avg_acc*100:.1f}%)\n")

        print(f"  Stats saved: {avg_report_path}")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate confusion matrices for classification task'
    )
    parser.add_argument('--benchmark', type=str,
                       default='../../benchmark/kls_bench_classification.json',
                       help='Path to benchmark JSON file')
    parser.add_argument('--results-dir', type=str,
                       default='../../results/temperature_ablation',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str,
                       default='../../results/confusion_matrices_full',
                       help='Output directory for confusion matrices')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature value to filter results (default: 0.0)')
    parser.add_argument('--use-full-predictions', action='store_true',
                       help='Use full predictions files instead of result files')
    parser.add_argument('--full-predictions-dir', type=str,
                       default='../../results/full_predictions',
                       help='Directory containing full predictions files')

    args = parser.parse_args()

    # Generator 초기화 및 실행
    generator = ConfusionMatrixGenerator(
        benchmark_path=args.benchmark,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        temperature=args.temperature,
        use_full_predictions=args.use_full_predictions,
        full_predictions_dir=args.full_predictions_dir
    )

    generator.generate_all_matrices()


if __name__ == "__main__":
    main()
