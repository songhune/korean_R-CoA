"""
Save Classification Predictions Only
kls_bench_classification.json만 사용하여 빠르게 predictions 저장
"""

import os
import sys
import importlib.util

# CRITICAL: Monkey-patch importlib.util.find_spec to hide flash_attn
_original_find_spec = importlib.util.find_spec

def patched_find_spec(name, package=None):
    """Patched find_spec that returns None for flash_attn packages"""
    if name and ('flash_attn' in name or name.startswith('flash_attn')):
        return None
    return _original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

# Disable flash attention via environment
os.environ['TRANSFORMERS_NO_FLASH_ATTENTION'] = '1'
os.environ['DISABLE_FLASH_ATTENTION'] = '1'

import json
from pathlib import Path
from tqdm import tqdm
import time

# exp5_benchmark_evaluation 모듈 import
sys.path.insert(0, str(Path(__file__).parent))

from exp5_benchmark_evaluation import (
    OpenAIWrapper,
    AnthropicWrapper,
    HuggingFaceWrapper
)


def load_classification_benchmark(benchmark_path: str):
    """Classification 벤치마크 로드"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[LOAD] Classification Benchmark: {benchmark_path}")
    print(f"  Task: {data['task']}")
    print(f"  Samples: {data['size']}")

    return data


def format_prompt(data_item):
    """프롬프트 생성"""
    system_prompt = "당신은 한국 고전 문헌 전문가입니다. 주어진 한문 텍스트의 문체를 정확하게 분류하세요."

    user_prompt = f"""다음 한문 텍스트의 문체를 분류하세요.

가능한 문체: 賦(부), 詩(시), 疑(의), 義(의), 策(책), 論(논), 表(표), 箋(전), 講(강), 頌(송), 箴(잠), 詔(조), 銘(명), 詩義, 禮義, 易義, 書義, 制(제), 擬(의)

텍스트: {data_item['input']}

문체 (한 단어로만 답하세요):"""

    return system_prompt, user_prompt


def save_classification_predictions(
    model,
    model_name: str,
    benchmark_data: dict,
    output_dir: Path,
    temperature: float,
    model_type: str
):
    """Classification predictions 저장"""
    print(f"\n{'='*70}")
    print(f"[SAVE PREDICTIONS] Model: {model_name}")
    print(f"[TEMPERATURE] {temperature}")
    print(f"{'='*70}\n")

    predictions = []
    ground_truths = []

    data_items = benchmark_data['data']

    print(f"[CLASSIFICATION] Processing {len(data_items)} samples...")

    for item in tqdm(data_items, desc="  Generating predictions"):
        # 프롬프트 생성
        system_prompt, user_prompt = format_prompt(item)

        # Ground truth
        ground_truths.append(item['label'])

        # 모델 추론
        try:
            prediction = model.generate(system_prompt, user_prompt)
            predictions.append(prediction if prediction else "")
        except Exception as e:
            print(f"  [ERROR] {e}")
            predictions.append("")

        # API rate limiting
        if model_type == 'api':
            time.sleep(1.0)

    # 저장
    safe_model_name = model_name.replace('/', '_')
    output_data = {
        'model_name': model_name,
        'model_type': model_type,
        'temperature': temperature,
        'task': 'classification',
        'num_samples': len(predictions),
        'predictions': predictions,
        'ground_truths': ground_truths
    }

    output_path = output_dir / f"full_predictions_{safe_model_name}_temp{temperature:.1f}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] Predictions saved: {output_path}")
    print(f"  Total: {len(predictions)} samples")

    return output_path


def main():
    """메인 실행"""
    import argparse
    import os
    from dotenv import load_dotenv

    # Load .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Loaded .env file from: {env_path}")

    parser = argparse.ArgumentParser(
        description='Save classification predictions (808 samples only)'
    )
    parser.add_argument('--benchmark', type=str,
                       default='../../benchmark/kls_bench_classification.json',
                       help='Classification benchmark JSON')
    parser.add_argument('--output-dir', type=str,
                       default='../../results/full_predictions',
                       help='Output directory')
    parser.add_argument('--model-type', type=str,
                       choices=['api', 'opensource'],
                       required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    safe_model_name = args.model_name.replace('/', '_')
    output_file = output_dir / f'full_predictions_{safe_model_name}_temp{args.temperature}.json'
    if output_file.exists():
        print(f"\n{'='*70}")
        print(f"[SKIP] Predictions already exist: {output_file.name}")
        print(f"{'='*70}\n")
        return

    # Load benchmark
    benchmark_data = load_classification_benchmark(args.benchmark)

    # Initialize model
    if args.model_type == 'api':
        if 'gpt' in args.model_name.lower():
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")

            print(f"[INFO] Using OpenAI API key: {api_key[:10]}...")
            model = OpenAIWrapper(
                model_name=args.model_name,
                api_key=api_key,
                temperature=args.temperature
            )
        elif 'claude' in args.model_name.lower():
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")

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

    # Generate predictions
    save_classification_predictions(
        model=model,
        model_name=args.model_name,
        benchmark_data=benchmark_data,
        output_dir=output_dir,
        temperature=args.temperature,
        model_type=args.model_type
    )

    print("\n" + "="*70)
    print("[COMPLETE] Classification predictions saved")
    print("="*70)


if __name__ == "__main__":
    main()
