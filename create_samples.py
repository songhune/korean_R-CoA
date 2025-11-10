"""
KLSBench 데이터 샘플링 스크립트
웹페이지에 표시할 샘플 데이터를 생성합니다.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# 재현 가능한 결과를 위한 시드 설정
random.seed(42)

def load_json(file_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """JSON 파일을 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sample_data(data: Dict[str, Any], n_samples: int = 10) -> Dict[str, Any]:
    """
    데이터셋에서 n개의 샘플을 무작위로 추출합니다.

    Args:
        data: 원본 데이터셋 (task, description, size, metric, data 포함)
        n_samples: 추출할 샘플 수

    Returns:
        샘플링된 데이터셋
    """
    sampled = data.copy()

    # 데이터가 충분하지 않으면 전체 데이터 사용
    actual_samples = min(n_samples, len(data['data']))
    sampled['data'] = random.sample(data['data'], actual_samples)
    sampled['original_size'] = data['size']
    sampled['sample_size'] = actual_samples

    return sampled

def create_summary_stats(benchmark_dir: Path) -> Dict[str, Any]:
    """전체 벤치마크 통계를 생성합니다."""
    tasks = ['classification', 'retrieval', 'nli', 'translation', 'punctuation']
    summary = {
        'total_instances': 0,
        'tasks': {}
    }

    for task in tasks:
        task_file = benchmark_dir / f'kls_bench_{task}.json'
        if task_file.exists():
            data = load_json(str(task_file))
            summary['tasks'][task] = {
                'description': data.get('description', ''),
                'size': data.get('size', 0),
                'metric': data.get('metric', '')
            }
            summary['total_instances'] += data.get('size', 0)

    return summary

def main():
    """메인 샘플링 함수"""
    # 경로 설정
    base_dir = Path(__file__).parent
    benchmark_dir = base_dir / 'benchmark' / 'kls_bench'
    output_dir = base_dir / 'docs' / 'samples'

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 태스크별로 샘플링
    tasks = {
        'classification': 10,
        'retrieval': 15,
        'nli': 15,
        'translation': 15,
        'punctuation': 10
    }

    print("🔄 KLSBench 데이터 샘플링 시작...\n")

    for task, n_samples in tasks.items():
        input_file = benchmark_dir / f'kls_bench_{task}.json'
        output_file = output_dir / f'sample_{task}.json'

        if input_file.exists():
            print(f"📊 {task.upper()}: {n_samples}개 샘플 추출 중...")
            data = load_json(str(input_file))
            sampled_data = sample_data(data, n_samples)
            save_json(sampled_data, str(output_file))
            print(f"   ✅ 저장 완료: {output_file}")
        else:
            print(f"   ⚠️  파일을 찾을 수 없습니다: {input_file}")

    # 요약 통계 생성
    print(f"\n📈 벤치마크 요약 통계 생성 중...")
    summary = create_summary_stats(benchmark_dir)
    summary_file = output_dir / 'summary.json'
    save_json(summary, str(summary_file))
    print(f"   ✅ 저장 완료: {summary_file}")

    print(f"\n✨ 샘플링 완료!")
    print(f"   총 인스턴스: {summary['total_instances']}")
    print(f"   태스크 수: {len(summary['tasks'])}")
    print(f"   출력 디렉토리: {output_dir}")

if __name__ == '__main__':
    main()
