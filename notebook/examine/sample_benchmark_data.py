#!/usr/bin/env python3
"""
KLSBench 전문가 평가를 위한 랜덤 샘플링 스크립트

벤치마크 데이터에서 각 태스크별로 랜덤 샘플을 추출하여 JSON 또는 Markdown 파일로 저장합니다.
재현성을 위해 랜덤 시드를 사용합니다.

사용법:
    python3 sample_benchmark_data.py --seed 42 --samples 10 --output expert_evaluation_sample.json
    python3 sample_benchmark_data.py --seed 42 --samples 10 --format markdown --output sample.md
    python3 sample_benchmark_data.py --seed 42 --samples-per-task classification=5,retrieval=10 --output sample.json
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any


# 벤치마크 데이터 경로 설정
BENCHMARK_DIR = Path(__file__).parent.parent.parent / "benchmark" / "kls_bench"

TASK_FILES = {
    "classification": BENCHMARK_DIR / "kls_bench_classification.json",
    "retrieval": BENCHMARK_DIR / "kls_bench_retrieval.json",
    "punctuation": BENCHMARK_DIR / "kls_bench_punctuation.json",
    "nli": BENCHMARK_DIR / "kls_bench_nli.json",
    "translation": BENCHMARK_DIR / "kls_bench_translation.json",
}


def load_task_data(task_name: str) -> List[Dict[str, Any]]:
    """
    특정 태스크의 벤치마크 데이터 로드

    Args:
        task_name: 태스크 이름 (classification, retrieval, etc.)

    Returns:
        데이터 항목 리스트
    """
    file_path = TASK_FILES.get(task_name)
    if not file_path or not file_path.exists():
        print(f"Warning: {task_name} 파일을 찾을 수 없습니다: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('data', [])


def sample_data(
    samples_per_task: Dict[str, int],
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    각 태스크별로 랜덤 샘플링 수행

    Args:
        samples_per_task: 각 태스크별 샘플 개수 딕셔너리
        seed: 랜덤 시드 (재현성을 위해)

    Returns:
        태스크별 샘플링된 데이터 딕셔너리
    """
    random.seed(seed)
    sampled_data = {}

    for task_name, sample_count in samples_per_task.items():
        print(f"\n{task_name} 샘플링 중...")

        # 데이터 로드
        all_data = load_task_data(task_name)

        if not all_data:
            print(f"  - 데이터 없음, 스킵")
            continue

        # 샘플 개수 조정 (요청한 개수가 전체 데이터보다 많으면 전체 사용)
        actual_count = min(sample_count, len(all_data))
        if actual_count < sample_count:
            print(f"  - 경고: 요청 {sample_count}개, 실제 {actual_count}개만 샘플링 가능")

        # 랜덤 샘플링
        sampled = random.sample(all_data, actual_count)
        sampled_data[task_name] = sampled

        print(f"  - 전체 {len(all_data)}개 중 {actual_count}개 샘플링 완료")

    return sampled_data


def save_sampled_data_json(
    sampled_data: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    seed: int
):
    """
    샘플링된 데이터를 JSON 파일로 저장

    Args:
        sampled_data: 샘플링된 데이터
        output_path: 출력 파일 경로
        seed: 사용된 랜덤 시드
    """
    # 메타데이터 추가
    output_data = {
        "metadata": {
            "seed": seed,
            "total_samples": sum(len(items) for items in sampled_data.values()),
            "tasks": {
                task: len(items) for task, items in sampled_data.items()
            }
        },
        "data": sampled_data
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 샘플 데이터 저장 완료: {output_path}")
    print(f"  - 시드: {seed}")
    print(f"  - 총 샘플 수: {output_data['metadata']['total_samples']}개")
    print(f"  - 태스크별 샘플:")
    for task, count in output_data['metadata']['tasks'].items():
        print(f"    * {task}: {count}개")


def save_sampled_data_markdown(
    sampled_data: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    seed: int
):
    """
    샘플링된 데이터를 Markdown 파일로 저장

    Args:
        sampled_data: 샘플링된 데이터
        output_path: 출력 파일 경로
        seed: 사용된 랜덤 시드
    """
    lines = []

    # 헤더
    lines.append("# KLSBench 전문가 평가 샘플 데이터")
    lines.append("")
    lines.append(f"- **랜덤 시드**: {seed}")
    lines.append(f"- **총 샘플 수**: {sum(len(items) for items in sampled_data.values())}개")
    lines.append("")
    lines.append("## 태스크별 샘플 개수")
    lines.append("")
    for task, items in sampled_data.items():
        lines.append(f"- **{task}**: {len(items)}개")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 태스크별 데이터
    for task_name, items in sampled_data.items():
        lines.append(f"## {task_name.upper()}")
        lines.append("")

        if task_name == "classification":
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('id', 'N/A')}")
                lines.append("")
                lines.append(f"**원문**: {item.get('input', '')}")
                lines.append("")
                lines.append(f"**라벨**: {item.get('label', '')}")
                lines.append("")
                lines.append("**평가**:")
                lines.append("- [ ] 정확함")
                lines.append("- [ ] 부정확함")
                lines.append("")
                lines.append("**올바른 라벨** (부정확한 경우):")
                lines.append("")
                lines.append("**난이도** (1-5):")
                lines.append("")
                lines.append("**의견**:")
                lines.append("")
                lines.append("---")
                lines.append("")

        elif task_name == "retrieval":
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('id', 'N/A')}")
                lines.append("")
                lines.append(f"**Query**: {item.get('input', '')}")
                lines.append("")
                lines.append(f"**출처**: {item.get('answer', '')}")
                lines.append("")
                lines.append("**평가**:")
                lines.append("- [ ] 관련 있음")
                lines.append("- [ ] 관련 없음")
                lines.append("")
                lines.append("**관련 없는 이유** (관련 없는 경우):")
                lines.append("")
                lines.append("**난이도** (1-5):")
                lines.append("")
                lines.append("**의견**:")
                lines.append("")
                lines.append("---")
                lines.append("")

        elif task_name == "punctuation":
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('id', 'N/A')}")
                lines.append("")
                lines.append(f"**원문 (구두점 없음)**: {item.get('input', '')}")
                lines.append("")
                lines.append(f"**정답 (구두점 있음)**: {item.get('answer', '')}")
                lines.append("")
                if item.get('metadata', {}).get('korean_translation'):
                    lines.append(f"**번역**: {item['metadata']['korean_translation']}")
                    lines.append("")
                lines.append("**평가**:")
                lines.append("- [ ] 정확함")
                lines.append("- [ ] 부정확함")
                lines.append("")
                lines.append("**수정 제안** (부정확한 경우):")
                lines.append("")
                lines.append("**오류 유형**:")
                lines.append("")
                lines.append("**난이도** (1-5):")
                lines.append("")
                lines.append("**의견**:")
                lines.append("")
                lines.append("---")
                lines.append("")

        elif task_name == "nli":
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('id', 'N/A')}")
                lines.append("")
                lines.append(f"**Premise**: {item.get('premise', '')}")
                lines.append("")
                lines.append(f"**Hypothesis**: {item.get('hypothesis', '')}")
                lines.append("")
                lines.append(f"**라벨**: {item.get('label', '')}")
                lines.append("")
                lines.append("**평가**:")
                lines.append("- [ ] 정확함")
                lines.append("- [ ] 부정확함")
                lines.append("")
                lines.append("**올바른 라벨** (부정확한 경우):")
                lines.append("")
                lines.append("**판단 근거**:")
                lines.append("")
                lines.append("**난이도** (1-5):")
                lines.append("")
                lines.append("**의견**:")
                lines.append("")
                lines.append("---")
                lines.append("")

        elif task_name == "translation":
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('id', 'N/A')}")
                lines.append("")
                lines.append(f"**원문 ({item.get('source_lang', '')})**:")
                lines.append("")
                lines.append(item.get('source_text', ''))
                lines.append("")
                lines.append(f"**번역 ({item.get('target_lang', '')})**:")
                lines.append("")
                lines.append(item.get('target_text', ''))
                lines.append("")
                lines.append("**정확성 평가**:")
                lines.append("- [ ] 정확함")
                lines.append("- [ ] 부정확함")
                lines.append("")
                lines.append("**수정 제안** (부정확한 경우):")
                lines.append("")
                lines.append("**오역 유형**:")
                lines.append("")
                lines.append("**자연스러움** (1-5):")
                lines.append("")
                lines.append("**난이도** (1-5):")
                lines.append("")
                lines.append("**의견**:")
                lines.append("")
                lines.append("---")
                lines.append("")

    # 전반적 평가
    lines.append("## 전반적 평가")
    lines.append("")
    for task_name in sampled_data.keys():
        lines.append(f"### {task_name.upper()}")
        lines.append("")
        lines.append("**전반적 품질**:")
        lines.append("")
        lines.append("**개선 필요 사항**:")
        lines.append("")

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ 샘플 데이터 저장 완료: {output_path}")
    print(f"  - 형식: Markdown")
    print(f"  - 시드: {seed}")
    print(f"  - 총 샘플 수: {sum(len(items) for items in sampled_data.values())}개")
    print(f"  - 태스크별 샘플:")
    for task, items in sampled_data.items():
        print(f"    * {task}: {len(items)}개")


def parse_samples_per_task(samples_str: str) -> Dict[str, int]:
    """
    태스크별 샘플 개수 문자열 파싱

    예: "classification=5,retrieval=10,nli=3"

    Args:
        samples_str: 태스크별 샘플 개수 문자열

    Returns:
        태스크별 샘플 개수 딕셔너리
    """
    result = {}
    for pair in samples_str.split(','):
        task, count = pair.strip().split('=')
        result[task.strip()] = int(count.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description='KLSBench 전문가 평가를 위한 랜덤 샘플링',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모든 태스크에서 각 10개씩 샘플링 (JSON 형식)
  python3 sample_benchmark_data.py --seed 42 --samples 10

  # Markdown 형식으로 출력
  python3 sample_benchmark_data.py --seed 42 --samples 10 --format markdown --output sample.md

  # 태스크별로 다른 개수 샘플링
  python3 sample_benchmark_data.py --seed 42 --samples-per-task "classification=5,retrieval=10,nli=3"

  # 출력 파일명 지정
  python3 sample_benchmark_data.py --seed 123 --samples 20 --output my_sample.json
        """
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드 (재현성을 위해, 기본값: 42)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        help='모든 태스크에 대해 동일한 샘플 개수 (예: 10)'
    )

    parser.add_argument(
        '--samples-per-task',
        type=str,
        help='태스크별 샘플 개수 (예: "classification=5,retrieval=10,nli=3")'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='expert_evaluation_sample.json',
        help='출력 파일명 (기본값: expert_evaluation_sample.json)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'markdown', 'md'],
        default='json',
        help='출력 형식 (json 또는 markdown/md, 기본값: json)'
    )

    args = parser.parse_args()

    # 샘플 개수 파라미터 검증
    if not args.samples and not args.samples_per_task:
        parser.error("--samples 또는 --samples-per-task 중 하나는 반드시 지정해야 합니다.")

    if args.samples and args.samples_per_task:
        parser.error("--samples와 --samples-per-task는 동시에 사용할 수 없습니다.")

    # 샘플 개수 설정
    if args.samples:
        samples_per_task = {task: args.samples for task in TASK_FILES.keys()}
    else:
        samples_per_task = parse_samples_per_task(args.samples_per_task)
        # 유효한 태스크 이름인지 검증
        invalid_tasks = set(samples_per_task.keys()) - set(TASK_FILES.keys())
        if invalid_tasks:
            parser.error(f"유효하지 않은 태스크 이름: {invalid_tasks}\n"
                        f"가능한 태스크: {list(TASK_FILES.keys())}")

    print("=" * 60)
    print("KLSBench 전문가 평가 샘플링")
    print("=" * 60)
    print(f"랜덤 시드: {args.seed}")
    print(f"출력 파일: {args.output}")

    # 샘플링 수행
    sampled_data = sample_data(samples_per_task, seed=args.seed)

    # 결과 저장 (형식에 따라)
    output_path = Path(args.output)
    output_format = args.format.lower()

    if output_format in ['markdown', 'md']:
        # 출력 파일 확장자가 .md가 아니면 변경
        if not str(output_path).endswith('.md'):
            output_path = output_path.with_suffix('.md')
        save_sampled_data_markdown(sampled_data, output_path, args.seed)
    else:
        # 출력 파일 확장자가 .json이 아니면 변경
        if not str(output_path).endswith('.json'):
            output_path = output_path.with_suffix('.json')
        save_sampled_data_json(sampled_data, output_path, args.seed)

    print("\n" + "=" * 60)
    print("샘플링 완료!")
    print("=" * 60)
    print(f"\n다음 단계: 이 샘플 데이터를 사용하여 평가를 진행하세요.")


if __name__ == '__main__':
    main()
