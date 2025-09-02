"""메인 실행 스크립트"""

import asyncio
import json
from pathlib import Path

from config import TranslationConfig
from translator import LargeScaleTranslator
from cost_tracker import estimate_translation_cost
import aiohttp


async def process_accn_ins_dataset():
    """ACCN-INS 데이터셋 처리 함수"""
    
    config = TranslationConfig(
        api_provider="ollama",
        model="winkefinger/alma-13b:Q4_K_M",  # 영어용 기본 모델
        korean_model="jinbora/deepseek-r1-Bllossom:70b",  # 한글 전용 고품질 모델
        english_model="winkefinger/alma-13b:Q4_K_M",  # 영어 전용 모델
        batch_size=1,      # 배치 사이즈 1로 설정 (100% 성공률)
        max_concurrent=1,  # 로컬 처리이므로 동시성 1로 설정
        delay_between_batches=0.5,  # 개별 처리이므로 약간의 딜레이
        chunk_size=50,     # 청크 사이즈도 줄임
        checkpoint_interval=25,
        budget_limit=0.0,  # 무료
        ollama_base_url="http://localhost:11434"  # 기본 Ollama URL
    )
    
    translator = LargeScaleTranslator(config)
    
    # ACCN-INS 데이터셋 처리
    await translator.process_large_dataset(
        input_file="/home/work/songhune/ACCN-INS.json",  # 원본 ACCN-INS 파일
        output_file="accn_ins_multilingual.jsonl"  # 한글/영어 번역 추가된 파일
    )


async def test_ollama_connection():
    """Ollama 연결 테스트"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    models = await response.json()
                    print("✅ Ollama 서버 연결 성공!")
                    print("🤖 사용 가능한 모델:")
                    for model in models.get('models', []):
                        print(f"   - {model['name']}")
                    return True
                else:
                    print("❌ Ollama 서버에 연결할 수 없습니다.")
                    return False
    except Exception as e:
        print(f"❌ Ollama 연결 오류: {e}")
        print("\n🔧 해결 방법:")
        print("1. Ollama가 설치되어 있는지 확인:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("\n2. Ollama 서버 시작:")
        print("   ollama serve")
        print("\n3. 필요한 모델 다운로드:")
        print("   ollama pull jinbora/deepseek-r1-Bllossom:70b  # 한글 번역용")
        print("   ollama pull winkefinger/alma-13b:Q4_K_M      # 영어 번역용")
        return False


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
        api_provider="ollama",
        model="winkefinger/alma-13b:Q4_K_M",
        korean_model="jinbora/deepseek-r1-Bllossom:70b",
        english_model="winkefinger/alma-13b:Q4_K_M",
        batch_size=1,
        max_concurrent=1,
        delay_between_batches=1.0,
        budget_limit=0.0,  # 무료 모델
        ollama_base_url="http://localhost:11434"
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
    
    # Ollama로 처리 (무료)
    print("=== Ollama로 처리합니다 (무료) ===")
    
    # 설정
    config = TranslationConfig(
        api_provider="ollama",
        model="winkefinger/alma-13b:Q4_K_M",
        korean_model="jinbora/deepseek-r1-Bllossom:70b",
        english_model="winkefinger/alma-13b:Q4_K_M",
        batch_size=1,
        max_concurrent=1,
        delay_between_batches=0.5,
        chunk_size=50,
        checkpoint_interval=25,
        budget_limit=0.0,
        ollama_base_url="http://localhost:11434"
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
        print("  python main.py test                      # Ollama 연결 테스트")
        print("  python main.py sample                    # 샘플 테스트")
        print("  python main.py accn                      # ACCN-INS 데이터셋 처리")
        print("  python main.py process <input> <output>  # 사용자 파일 처리")
        print("  python main.py estimate <file>           # 비용 추정")
        print()
        return
    
    command = sys.argv[1]
    
    if command == "test":
        print("=== Ollama 연결 테스트 ===")
        result = asyncio.run(test_ollama_connection())
        if result:
            print("\n✅ 준비 완료! 'python main.py accn'으로 번역을 시작하세요.")
    
    elif command == "sample":
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
        print("Ollama 사용 시 번역 비용: $0.00 (무료)")
    
    else:
        print(f"알 수 없는 명령어: {command}")
        print("python main.py 를 실행하여 사용법을 확인하세요.")


if __name__ == "__main__":
    main()