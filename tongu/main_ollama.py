"""Ollama 사용을 위한 메인 실행 스크립트"""

import asyncio
import json
from pathlib import Path

from config import TranslationConfig
from translator import LargeScaleTranslator
from cost_tracker import estimate_translation_cost


async def process_with_ollama():
    """Ollama를 사용한 데이터셋 처리"""
    
    config = TranslationConfig(
        api_provider="ollama",
        model="winkefinger/alma-13b:Q4_K_M",  # 사용 가능한 모델: llama3, llama3.1, qwen2, mixtral 등
        batch_size=5,      # 로컬 모델이므로 작은 배치 사이즈
        max_concurrent=1,  # 로컬 처리이므로 동시성 1로 설정
        delay_between_batches=0.2,
        chunk_size=100,
        checkpoint_interval=50,
        budget_limit=0.0,  # 무료
        ollama_base_url="http://localhost:11434"  # 기본 Ollama URL
    )
    
    translator = LargeScaleTranslator(config)
    
    # ACCN-INS 데이터셋 처리
    await translator.process_large_dataset(
        input_file="/home/work/songhune/ACCN-INS.json",
        output_file="sample_ollama_translated.jsonl"
    )


async def test_ollama_connection():
    """Ollama 연결 테스트"""
    import aiohttp
    
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
        print("\n3. 모델 다운로드 (예시):")
        print("   ollama pull llama3.1")
        print("   ollama pull qwen2")
        return False


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) == 1:
        print("KEadapter - Ollama 버전 (무료 로컬 번역기)")
        print()
        print("사용법:")
        print("  python main_ollama.py test      # Ollama 연결 테스트")
        print("  python main_ollama.py translate # 번역 실행")
        print()
        return
    
    command = sys.argv[1]
    
    if command == "test":
        print("=== Ollama 연결 테스트 ===")
        result = asyncio.run(test_ollama_connection())
        if result:
            print("\n✅ 준비 완료! 'python main_ollama.py translate'로 번역을 시작하세요.")
    
    elif command == "translate":
        print("=== Ollama 번역 시작 ===")
        asyncio.run(process_with_ollama())
    
    else:
        print(f"알 수 없는 명령어: {command}")
        print("python main_ollama.py 를 실행하여 사용법을 확인하세요.")


if __name__ == "__main__":
    main()