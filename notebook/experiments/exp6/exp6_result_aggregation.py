"""
Experiment 6: K-ClassicBench Result Aggregation and Visualization
벤치마크 평가 결과를 통합하고 시각화하는 스크립트

기능:
1. 모든 평가 결과 JSON/CSV 파일을 로드
2. 최신 결과만 선택 (같은 모델의 여러 실행 중)
3. 결과를 통합 테이블로 정리
4. 시각화 생성:
   - 태스크별 성능 비교 (히트맵)
   - 모델별 종합 성능 (바 차트)
   - 레이더 차트
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import argparse
import sys

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

try:
    from font_fix import setup_korean_fonts_robust
except ImportError:
    setup_korean_fonts_robust = None

A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "raw_evaluation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "aggregated"


def abbreviate_model_name(model_name: str) -> str:
    """
    모델 이름을 축약하여 그래프에서 보기 좋게 만듦

    예시:
    - meta-llama/Llama-3.1-8B-Instruct -> Llama-3.1-8B
    - Qwen/Qwen2.5-7B-Instruct -> Qwen2.5-7B
    - claude-3-5-sonnet-20241022 -> Claude-3.5-Sonnet
    """
    # OpenAI 모델
    if 'gpt-4o' in model_name.lower():
        if 'mini' in model_name.lower():
            return 'GPT-4o-mini'
        return 'GPT-4o'
    if 'gpt-4-turbo' in model_name.lower():
        return 'GPT-4-Turbo'
    if 'gpt-3.5-turbo' in model_name.lower():
        return 'GPT-3.5-Turbo'

    # Anthropic 모델
    if 'claude-3-5-sonnet' in model_name.lower() or 'claude-3.5-sonnet' in model_name.lower():
        return 'Claude-3.5-Sonnet'
    if 'claude-3-opus' in model_name.lower():
        return 'Claude-3-Opus'
    if 'claude-3-sonnet' in model_name.lower():
        return 'Claude-3-Sonnet'
    if 'claude-3-haiku' in model_name.lower():
        return 'Claude-3-Haiku'

    # Meta Llama 모델
    if 'llama' in model_name.lower():
        if '3.3' in model_name:
            size = '70B' if '70b' in model_name.lower() else '8B'
            return f'Llama-3.3-{size}'
        if '3.1' in model_name:
            size = '70B' if '70b' in model_name.lower() else '8B'
            return f'Llama-3.1-{size}'
        if '3.0' in model_name or 'llama-3-' in model_name.lower():
            size = '70B' if '70b' in model_name.lower() else '8B'
            return f'Llama-3-{size}'

    # Qwen 모델
    if 'qwen' in model_name.lower():
        if '2.5' in model_name:
            size = '72B' if '72b' in model_name.lower() else '32B' if '32b' in model_name.lower() else '7B'
            return f'Qwen2.5-{size}'
        if '2' in model_name:
            size = '72B' if '72b' in model_name.lower() else '7B'
            return f'Qwen2-{size}'

    # Google Gemini 모델
    if 'gemini' in model_name.lower():
        if '1.5-pro' in model_name.lower():
            return 'Gemini-1.5-Pro'
        if '1.5-flash' in model_name.lower():
            return 'Gemini-1.5-Flash'
        if '1.0-pro' in model_name.lower():
            return 'Gemini-1.0-Pro'

    # 기타 모델 - 경로 제거
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    # -Instruct, -Chat 등 접미사 제거
    for suffix in ['-Instruct', '-Chat', '-instruct', '-chat']:
        model_name = model_name.replace(suffix, '')

    return model_name


def configure_matplotlib():
    """Ensure consistent typography and sizing for publication-ready figures."""
    if setup_korean_fonts_robust:
        setup_korean_fonts_robust()
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams.update({
        'figure.figsize': (A4_WIDTH_INCH, A4_HEIGHT_INCH * 0.6),
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })


class ResultAggregator:
    """벤치마크 결과 통합 클래스"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None

    def load_all_results(self):
        """모든 결과 파일 로드"""
        print("📂 결과 파일 로딩 중...")

        # JSON 파일 찾기
        json_files = list(self.results_dir.glob("results_*.json"))
        print(f"  ✓ 총 {len(json_files)}개 결과 파일 발견")

        # 모델별로 그룹화 (최신 결과만 선택)
        model_results = defaultdict(list)

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    model_name = result['model_name']
                    # 타임스탬프 추출 (파일명에서)
                    timestamp = json_file.stem.split('_')[-2:]
                    timestamp_str = '_'.join(timestamp)
                    result['timestamp'] = timestamp_str
                    result['file_path'] = str(json_file)
                    model_results[model_name].append(result)
            except Exception as e:
                print(f"    파일 로드 실패: {json_file.name} - {e}")

        # 각 모델의 최신 결과만 선택
        print(f"\n 모델별 최신 결과 선택:")
        for model_name, results in model_results.items():
            # 타임스탬프 기준 정렬
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            latest = results[0]
            self.results.append(latest)
            print(f"  ✓ {model_name}: {latest['timestamp']}")

        print(f"\n 총 {len(self.results)}개 모델 결과 로드 완료")

    def create_summary_table(self) -> pd.DataFrame:
        """결과를 요약 테이블로 변환"""
        print("\n 요약 테이블 생성 중...")

        rows = []
        for result in self.results:
            model_name = result['model_name']
            model_name_short = abbreviate_model_name(model_name)
            model_type = result.get('model_type', 'unknown')

            # 각 태스크의 주요 메트릭 추출
            for task_name, task_result in result['tasks'].items():
                metrics = task_result['metrics']

                row = {
                    'model': model_name_short,
                    'model_full': model_name,
                    'model_type': model_type,
                    'task': task_name,
                }

                # 태스크별 주요 메트릭 추가
                if task_name == 'classification':
                    row['accuracy'] = metrics.get('accuracy', 0)
                    row['f1'] = metrics.get('f1', 0)
                    row['primary_metric'] = metrics.get('accuracy', 0)

                elif task_name == 'retrieval':
                    row['accuracy'] = metrics.get('accuracy', 0)
                    row['primary_metric'] = metrics.get('accuracy', 0)

                elif task_name == 'punctuation':
                    row['char_f1'] = metrics.get('char_f1', 0)
                    row['rougeL_f1'] = metrics.get('rougeL_f1', 0)
                    row['primary_metric'] = metrics.get('char_f1', 0)

                elif task_name == 'nli':
                    row['accuracy'] = metrics.get('accuracy', 0)
                    row['f1'] = metrics.get('f1', 0)
                    row['primary_metric'] = metrics.get('accuracy', 0)

                elif task_name == 'translation':
                    row['bleu'] = metrics.get('bleu', 0)
                    row['rougeL_f1'] = metrics.get('rougeL_f1', 0)
                    row['primary_metric'] = metrics.get('bleu', 0)

                rows.append(row)

        self.df = pd.DataFrame(rows)
        print(f"  ✓ {len(self.df)}개 행 생성 완료")

        return self.df

    def save_aggregated_results(self, output_dir: str):
        """통합 결과 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 전체 요약 테이블
        summary_path = output_dir / "aggregated_summary.csv"
        self.df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n 요약 테이블 저장: {summary_path}")

        # 2. 피벗 테이블 (모델 × 태스크)
        pivot = self.df.pivot_table(
            index='model',
            columns='task',
            values='primary_metric',
            aggfunc='first'
        )
        pivot_path = output_dir / "aggregated_pivot.csv"
        pivot.to_csv(pivot_path, encoding='utf-8-sig')
        print(f" 피벗 테이블 저장: {pivot_path}")

        # 3. 모델별 평균 성능
        model_avg = self.df.groupby('model')['primary_metric'].mean().sort_values(ascending=False)
        model_avg_path = output_dir / "model_average_performance.csv"
        model_avg.to_csv(model_avg_path, header=['average_score'], encoding='utf-8-sig')
        print(f" 모델 평균 성능: {model_avg_path}")

        return pivot

    def visualize_results(self, output_dir: str):
        """결과 시각화"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n 시각화 생성 중...")

        configure_matplotlib()

        # 1. 히트맵: 모델 × 태스크 성능
        self._create_heatmap(output_dir)

        # 2. 바 차트: 모델별 평균 성능
        self._create_bar_chart(output_dir)

        # 3. 태스크별 성능 비교 (그룹 바 차트)
        self._create_grouped_bar_chart(output_dir)

        # 4. 레이더 차트
        self._create_radar_chart(output_dir)

        print(f" 시각화 완료: {output_dir}")

    def _create_heatmap(self, output_dir: Path):
        """히트맵 생성"""
        pivot = self.df.pivot_table(
            index='model',
            columns='task',
            values='primary_metric',
            aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(A4_WIDTH_INCH, A4_HEIGHT_INCH * 0.7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Score'},
            annot_kws={'fontsize': 12},
            ax=ax
        )
        ax.set_title('K-ClassicBench: Model Performance Heatmap', fontsize=18, fontweight='bold')
        ax.set_xlabel('Task')
        ax.set_ylabel('Model')
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        fig.tight_layout()

        heatmap_path = output_dir / 'heatmap_performance.pdf'
        fig.savefig(heatmap_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 히트맵: {heatmap_path}")

    def _create_bar_chart(self, output_dir: Path):
        """바 차트: 모델별 평균 성능"""
        model_avg = self.df.groupby('model')['primary_metric'].mean().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(A4_WIDTH_INCH, A4_HEIGHT_INCH * 0.65))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_avg)))
        model_avg.plot(kind='barh', color=colors, ax=ax)
        ax.set_title('K-ClassicBench: Average Model Performance', fontsize=18, fontweight='bold')
        ax.set_xlabel('Average Score')
        ax.set_ylabel('Model')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()

        bar_path = output_dir / 'bar_average_performance.pdf'
        fig.savefig(bar_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 바 차트: {bar_path}")

    def _create_grouped_bar_chart(self, output_dir: Path):
        """그룹 바 차트: 태스크별 모델 성능"""
        pivot = self.df.pivot_table(
            index='model',
            columns='task',
            values='primary_metric',
            aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(A4_WIDTH_INCH * 1.05, A4_HEIGHT_INCH * 0.75))
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('K-ClassicBench: Task-wise Model Performance', fontsize=18, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()

        grouped_bar_path = output_dir / 'grouped_bar_taskwise.pdf'
        fig.savefig(grouped_bar_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 그룹 바 차트: {grouped_bar_path}")

    def _create_radar_chart(self, output_dir: Path):
        """레이더 차트: 모델별 태스크 성능"""
        pivot = self.df.pivot_table(
            index='model',
            columns='task',
            values='primary_metric',
            aggfunc='first'
        ).fillna(0)

        # 태스크 수
        categories = list(pivot.columns)
        N = len(categories)

        # 각도 계산
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 원을 닫기 위해

        fig, ax = plt.subplots(figsize=(A4_WIDTH_INCH * 0.9, A4_HEIGHT_INCH * 0.9), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot)))

        for idx, (model, row) in enumerate(pivot.iterrows()):
            values = row.tolist()
            values += values[:1]  # 원을 닫기 위해

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=14)
        ax.set_ylim(0, 1)
        ax.set_title('K-ClassicBench: Radar Chart - Model Performance',
                     size=18, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.grid(True)

        fig.tight_layout()
        radar_path = output_dir / 'radar_chart.pdf'
        fig.savefig(radar_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 레이더 차트: {radar_path}")

    def print_summary_statistics(self):
        """요약 통계 출력"""
        print("\n" + "=" * 70)
        print(" K-ClassicBench 평가 결과 요약")
        print("=" * 70)

        # 1. 모델별 평균 성능
        print("\n🏆 모델별 평균 성능:")
        model_avg = self.df.groupby('model')['primary_metric'].mean().sort_values(ascending=False)
        for rank, (model, score) in enumerate(model_avg.items(), 1):
            print(f"  {rank}. {model:50s} {score:.4f}")

        # 2. 태스크별 최고 성능 모델
        print("\n 태스크별 최고 성능:")
        for task in self.df['task'].unique():
            task_df = self.df[self.df['task'] == task]
            best_row = task_df.loc[task_df['primary_metric'].idxmax()]
            print(f"  - {task:15s}: {best_row['model']:40s} ({best_row['primary_metric']:.4f})")

        # 3. 모델 타입별 평균 성능
        print("\n 모델 타입별 평균 성능:")
        type_avg = self.df.groupby('model_type')['primary_metric'].mean().sort_values(ascending=False)
        for model_type, score in type_avg.items():
            print(f"  - {model_type:15s}: {score:.4f}")

        print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='K-ClassicBench Result Aggregation')
    parser.add_argument('--results-dir', type=str,
                       default=str(DEFAULT_RESULTS_DIR),
                       help='결과 파일 디렉토리')
    parser.add_argument('--output-dir', type=str,
                       default=str(DEFAULT_OUTPUT_DIR),
                       help='통합 결과 저장 디렉토리')

    args = parser.parse_args()

    # 결과 통합
    aggregator = ResultAggregator(args.results_dir)
    aggregator.load_all_results()
    aggregator.create_summary_table()

    # 결과 저장
    aggregator.save_aggregated_results(args.output_dir)

    # 시각화
    aggregator.visualize_results(args.output_dir)

    # 통계 출력
    aggregator.print_summary_statistics()

    print("\n 결과 통합 및 시각화 완료!")


if __name__ == "__main__":
    main()
