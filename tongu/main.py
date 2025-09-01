"""메인 실행 스크립트"""

import asyncio
import json
from pathlib import Path

from config import TranslationConfig
from translator import LargeScaleTranslator
from cost_tracker import estimate_translation_cost


async def process_accn_ins_dataset():
    """ACCN-INS 데이터셋 처리 함수"""
    
    config = TranslationConfig(
        api_provider="ollama",  # Ollama 사용으로 변경
        model="llama3.1",  # 또는 "qwen2", "mixtral" 등
        batch_size=10,  # 로컬 모델이므로 배치 사이즈 조정
        max_concurrent=2,  # 로컬 처리이므로 동시성 낮춤
        delay_between_batches=0.5,
        chunk_size=1000,
        checkpoint_interval=100,
        budget_limit=0.0  # 무료이므로 예산 제한 없음
    )
    
    translator = LargeScaleTranslator(config)
    
    # ACCN-INS 데이터셋 처리
    await translator.process_large_dataset(
        input_file="/home/work/songhune/sample.json",  # 원본 ACCN-INS 파일
        output_file="accn_ins_multilingual.jsonl"  # 한글/영어 번역 추가된 파일
    )


async def test_with_sample():
    """작은 샘플로 먼저 테스트"""
    
    # 샘플 데이터 생성
    sample_data = [
        {
            "task": "Classical Chinese to Modern Chinese",
            "data": {
                "instruction": "请将迁骑都尉、光禄大夫、侍中。宿卫谨敕，爵位益尊，翻译为现代汉语。",
                "input": "",
                "output": "又升任骑都尉光禄大夫侍中。王莽在宫中值宿警卫，谨慎认真，地位越是尊贵，",
                "history": []
            }
        },
        {
            "task": "Classical Chinese to Modern Chinese", 
            "data": {
                "instruction": "岁年丰穰，九十月禾黍登场。为春酒瓮浮新酿，",
                "input": "",
                "output": "庄稼丰收，九月十月禾稼登场。制成春酒飘浓香，",
                "history": []
            }
        }
    ]
    
    # 샘플 파일 저장
    with open("accn_sample.jsonl", "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 테스트 실행
    config = TranslationConfig(
        api_provider="anthropic",
        batch_size=2,
        max_concurrent=1,
        delay_between_batches=1.0,
        budget_limit=5.0  # 테스트용 낮은 예산
    )
    
    translator = LargeScaleTranslator(config)
    
    await translator.process_large_dataset(
        input_file="accn_sample.jsonl",
        output_file="accn_sample_translated.jsonl"
    )


async def process_custom_dataset(input_file: str, output_file: str):
    """사용자 지정 데이터셋 처리"""
    
    # 파일 존재 확인
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # 비용 추정
    print("=== 비용 추정 중 ===")
    estimated_cost = estimate_translation_cost(input_file, "anthropic")
    
    if estimated_cost > 50.0:
        response = input(f"예상 비용이 ${estimated_cost:.2f}입니다. 계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("처리를 중단합니다.")
            return
    
    # 설정
    config = TranslationConfig(
        api_provider="anthropic",
        model="claude-3-haiku-20240307",
        batch_size=20,
        max_concurrent=5,
        delay_between_batches=1.0,
        chunk_size=5000,
        checkpoint_interval=250,
        budget_limit=estimated_cost * 1.2  # 20% 여유분
    )
    
    translator = LargeScaleTranslator(config)
    
    await translator.process_large_dataset(
        input_file=input_file,
        output_file=output_file
    )


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) == 1:
        print("KEadapter - 대용량 고전 중국어 번역기")
        print()
        print("사용법:")
        print("  python main.py sample                    # 샘플 테스트")
        print("  python main.py accn                      # ACCN-INS 데이터셋 처리")
        print("  python main.py process <input> <output>  # 사용자 파일 처리")
        print("  python main.py estimate <file>           # 비용 추정")
        print()
        return
    
    command = sys.argv[1]
    
    if command == "sample":
        print("=== 샘플 테스트 실행 ===")
        asyncio.run(test_with_sample())
    
    elif command == "accn":
        print("=== ACCN-INS 데이터셋 처리 ===")
        asyncio.run(process_accn_ins_dataset())
    
    elif command == "process":
        if len(sys.argv) != 4:
            print("사용법: python main.py process <input_file> <output_file>")
            return
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        
        print(f"=== 파일 처리: {input_file} -> {output_file} ===")
        asyncio.run(process_custom_dataset(input_file, output_file))
    
    elif command == "estimate":
        if len(sys.argv) != 3:
            print("사용법: python main.py estimate <input_file>")
            return
        
        input_file = sys.argv[2]
        estimate_translation_cost(input_file, "anthropic")
    
    else:
        print(f"알 수 없는 명령어: {command}")
        print("python main.py 를 실행하여 사용법을 확인하세요.")


if __name__ == "__main__":
    main()