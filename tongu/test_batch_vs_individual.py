#!/usr/bin/env python3
"""배치 vs 개별 번역 비교 테스트"""

import asyncio
import aiohttp
from config import TranslationConfig
from api_clients import APIClientFactory

async def test_batch_vs_individual():
    """배치 번역과 개별 번역 비교"""
    
    config = TranslationConfig(
        api_provider="ollama",
        model="winkefinger/alma-13b:Q4_K_M",
        ollama_base_url="http://localhost:11434"
    )
    
    # 테스트 텍스트 (실패했던 것들)
    test_texts = [
        "光武看见王常很欢喜，慰劳他说：王廷尉很辛苦。。",
        "四是五味败坏口舌，使得味觉丧失；五是好恶迷乱心弦，",
        "临别前，对这件事非常遗憾。高告诉他："
    ]
    
    async with aiohttp.ClientSession() as session:
        client = APIClientFactory.create_client(config, session)
        
        print("=== 배치 vs 개별 번역 비교 ===\n")
        
        # 1. 배치 번역 테스트
        print("배치 번역 (5개 텍스트 동시):")
        try:
            batch_texts = test_texts + ["测试文本1", "测试文本2"]  # 5개로 확장
            batch_result = await client.translate_batch(batch_texts, 'korean')
            
            if isinstance(batch_result, tuple):
                batch_translations, _ = batch_result
            else:
                batch_translations = batch_result
                
            for i, (text, translation) in enumerate(zip(batch_texts, batch_translations)):
                status = "✅" if not translation.startswith("[") else "❌"
                print(f"   {i+1}. {status} {text[:30]}...")
                print(f"      -> {translation[:80]}...")
                
        except Exception as e:
            print(f"   오류: {e}")
        
        print()
        
        # 2. 개별 번역 테스트
        print("개별 번역 (텍스트 하나씩):")
        for i, text in enumerate(test_texts, 1):
            try:
                individual_result = await client.translate_batch([text], 'korean')
                
                if isinstance(individual_result, tuple):
                    individual_translation = individual_result[0][0]
                else:
                    individual_translation = individual_result[0]
                
                status = "✅" if not individual_translation.startswith("[") else "❌"
                print(f"   {i}. {status} {text[:30]}...")
                print(f"      -> {individual_translation[:80]}...")
                
                await asyncio.sleep(0.5)  # 간격 추가
                
            except Exception as e:
                print(f"   {i}. ❌ 오류: {e}")
        
        print()
        
        # 3. 다양한 배치 사이즈 테스트
        print("치 사이즈별 테스트:")
        for batch_size in [1, 2, 3, 5]:
            print(f"\n   배치 사이즈 {batch_size}:")
            try:
                test_batch = test_texts[:batch_size]
                result = await client.translate_batch(test_batch, 'korean')
                
                if isinstance(result, tuple):
                    translations, _ = result
                else:
                    translations = result
                
                success_count = sum(1 for t in translations if not t.startswith("["))
                print(f"      성공률: {success_count}/{batch_size} ({success_count/batch_size*100:.1f}%)")
                
            except Exception as e:
                print(f"      오류: {e}")
                
            await asyncio.sleep(0.3)

if __name__ == "__main__":
    asyncio.run(test_batch_vs_individual())