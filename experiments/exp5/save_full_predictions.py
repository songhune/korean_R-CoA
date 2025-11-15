"""
Save Full Predictions for Confusion Matrix Generation

기존 평가 스크립트를 수정하여 classification task의 전체 predictions를 저장합니다.
이 스크립트는 confusion matrix 생성을 위해 필요한 전체 예측 결과를 저장합니다.
"""

import json
import sys
from pathlib import Path

# exp5_benchmark_evaluation 모듈을 import하기 위해 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from exp5_benchmark_evaluation import (
    KLSBenchEvaluator,
    OpenAIWrapper,
    AnthropicWrapper,
    HuggingFaceWrapper
)


def save_full_predictions_for_model(
    evaluator: KLSBenchEvaluator,
    model,
    model_name: str,
    output_dir: Path,
    temperature: float
):
    """
    단일 모델에 대해 classification task의 전체 predictions를 저장

    Args:
        evaluator: KLSBenchEvaluator 인스턴스
        model: 모델 래퍼 인스턴스
        model_name: 모델 이름
        output_dir: 출력 디렉토리
        temperature: 온도 값
    """
    print(f"\n{'='*70}")
    print(f"[SAVE PREDICTIONS] Model: {model_name}")
    print(f"[TEMPERATURE] {temperature}")
    print(f"{'='*70}\n")

    # Classification task만 처리
    task_name = 'classification'
    task_data = evaluator.benchmark['tasks'][task_name]

    print(f"[{task_name.upper()}] Processing {task_data['size']} samples...")

    predictions = []
    ground_truths = []

    from tqdm import tqdm
    import time

    for idx, item in enumerate(tqdm(task_data['data'], desc=f"  Generating predictions")):
        # 프롬프트 생성
        system_prompt, user_prompt = evaluator.format_prompt(task_name, item)

        # Ground truth 저장
        ground_truths.append(item['label'])

        # 모델 추론
        try:
            prediction = model.generate(system_prompt, user_prompt)
            if not prediction or prediction.strip() == "":
                print(f"  [WARNING] Empty prediction for item {len(predictions)+1}")
            predictions.append(prediction)

        except Exception as e:
            print(f"  [ERROR] Model generation error at item {idx}: {e}")
            predictions.append("")

        # API 호출 제한 대응
        if evaluator.model_type == 'api':
            time.sleep(1.0)

    # 파일명에서 슬래시를 언더스코어로 변경
    safe_model_name = model_name.replace('/', '_')

    # 전체 predictions를 JSON으로 저장
    output_data = {
        'model_name': model_name,
        'model_type': evaluator.model_type,
        'temperature': temperature,
        'task': task_name,
        'num_samples': len(predictions),
        'predictions': predictions,
        'ground_truths': ground_truths
    }

    output_path = output_dir / f"full_predictions_{safe_model_name}_temp{temperature:.1f}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] Full predictions saved to: {output_path}")
    print(f"  Total predictions: {len(predictions)}")

    return output_path


def main():
    """메인 실행 함수"""
    import argparse
    import os
    from pathlib import Path

    # Load .env file
    try:
        from dotenv import load_dotenv
        # .env 파일 경로 (프로젝트 루트)
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[INFO] Loaded .env file from: {env_path}")
        else:
            print(f"[WARNING] .env file not found at: {env_path}")
    except ImportError:
        print("[WARNING] python-dotenv not installed. Install with: pip install python-dotenv")

    parser = argparse.ArgumentParser(
        description='Save full classification predictions for confusion matrix'
    )
    parser.add_argument('--benchmark', type=str,
                       default='../../benchmark/kls_bench_full.json',
                       help='Benchmark JSON file path')
    parser.add_argument('--output-dir', type=str,
                       default='../../results/full_predictions',
                       help='Output directory for full predictions')
    parser.add_argument('--model-type', type=str,
                       choices=['api', 'opensource'],
                       required=True,
                       help='Model type (api or opensource)')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name (e.g., gpt-4-turbo, meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (for API models)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')

    args = parser.parse_args()

    # Output directory 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluator 초기화 (샘플링 없이 전체 데이터 사용)
    evaluator = KLSBenchEvaluator(
        benchmark_path=args.benchmark,
        output_dir=str(output_dir),  # 사용되지 않지만 필수 인자
        model_type=args.model_type,
        max_samples_per_task=args.max_samples,
        temperature=args.temperature,
        save_samples=False  # 샘플 출력은 저장하지 않음
    )

    # 모델 초기화
    if args.model_type == 'api':
        # API key 가져오기 (우선순위: 인자 > 환경 변수)
        if 'gpt' in args.model_name.lower():
            api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found. Set it in .env file or pass via --api-key")

            print(f"[INFO] Using OpenAI API key: {api_key[:10]}...")
            model = OpenAIWrapper(
                model_name=args.model_name,
                api_key=api_key,
                temperature=args.temperature
            )
        elif 'claude' in args.model_name.lower():
            api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found. Set it in .env file or pass via --api-key")

            print(f"[INFO] Using Anthropic API key: {api_key[:10]}...")
            model = AnthropicWrapper(
                model_name=args.model_name,
                api_key=api_key,
                temperature=args.temperature
            )
        else:
            raise ValueError(f"Unknown API model: {args.model_name}")

    elif args.model_type == 'opensource':
        model = HuggingFaceWrapper(
            model_name=args.model_name,
            temperature=args.temperature
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Full predictions 저장
    save_full_predictions_for_model(
        evaluator=evaluator,
        model=model,
        model_name=args.model_name,
        output_dir=output_dir,
        temperature=args.temperature
    )

    print("\n" + "="*70)
    print("[COMPLETE] Full predictions saved")
    print("="*70)


if __name__ == "__main__":
    main()
