"""
🐲 Tongu - Korean Translation System (Clean Version)
완전히 정리된 한국어 번역 시스템
"""

import asyncio
import sys
import subprocess
import time
import logging
import json
import aiohttp
import pickle
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# Configuration
# =============================================================================

class TonguConfig:
    """Tongu 설정"""
    
    def __init__(self):
        # Ollama 설정
        self.ollama_base_url = "http://localhost:11434"
        self.korean_model = "jinbora/deepseek-r1-Bllossom:70b"
        self.english_model = "winkefinger/alma-13b:Q4_K_M"
        
        # 번역 설정
        self.batch_size = 5
        self.max_concurrent = 3
        self.delay_between_batches = 2
        self.chunk_size = 100
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 30


# =============================================================================
# Error Handler
# =============================================================================

class ErrorHandler:
    """통합 에러 처리"""
    
    def __init__(self, email: str = "songhune@jou.ac.kr", threshold: int = 5):
        self.email = email
        self.threshold = threshold
        self.error_counts = defaultdict(lambda: deque())
        self.time_window = 300  # 5분
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tongu.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Tongu')
    
    def track_error(self, error_type: str, message: str, **context):
        """에러 추적"""
        current_time = time.time()
        self.error_counts[error_type].append(current_time)
        
        # 오래된 에러 제거
        cutoff_time = current_time - self.time_window
        while (self.error_counts[error_type] and 
               self.error_counts[error_type][0] < cutoff_time):
            self.error_counts[error_type].popleft()
        
        self.logger.error(f"{error_type}: {message} | {context}")
        
        # 알림 체크
        if len(self.error_counts[error_type]) >= self.threshold:
            self._send_alert(error_type, message, len(self.error_counts[error_type]))
    
    def _send_alert(self, error_type: str, message: str, count: int):
        """에러 알림 (콘솔)"""
        alert = f"""
🚨 TONGU ERROR ALERT
==================
Type: {error_type}  
Count: {count} errors in 5 minutes
Message: {message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Email: {self.email}
==================
        """
        print(alert)
        self.logger.error(f"ALERT SENT: {error_type} ({count} errors)")


# =============================================================================
# API Client
# =============================================================================

class OllamaClient:
    """Ollama API 클라이언트"""
    
    def __init__(self, config: TonguConfig, error_handler: ErrorHandler):
        self.config = config
        self.error_handler = error_handler
    
    async def translate_batch(self, texts: List[str], target_lang: str, session: aiohttp.ClientSession) -> List[str]:
        """배치 번역"""
        if not texts:
            return []
        
        # 모델 선택
        if target_lang == "korean":
            model = self.config.korean_model
            prompt_template = """다음 현대 중국어 텍스트를 자연스러운 한국어로 번역하세요:

{texts}

한국어 번역:"""
        else:
            model = self.config.english_model
            prompt_template = """Translate the following Modern Chinese texts to English:

{texts}

English translations:"""
        
        # 프롬프트 생성
        numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
        prompt = prompt_template.format(texts=numbered_texts)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9}
        }
        
        try:
            # 긴 타임아웃 설정
            timeout = aiohttp.ClientTimeout(total=300)
            
            async with session.post(f"{self.config.ollama_base_url}/api/generate", 
                                  json=payload, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get("response", "").strip()
                    
                    if not content:
                        self.error_handler.track_error("Empty Response", f"Model: {model}")
                        return [f"[Error: Empty response]" for _ in texts]
                    
                    # 응답 파싱
                    translations = content.split('\n')
                    cleaned = []
                    
                    for trans in translations:
                        trans = trans.strip()
                        # 번호 제거
                        if trans and trans[0].isdigit() and '.' in trans[:5]:
                            trans = trans.split('.', 1)[1].strip()
                        if trans:
                            cleaned.append(trans)
                    
                    # 길이 맞추기
                    while len(cleaned) < len(texts):
                        cleaned.append("[Error: Missing translation]")
                    
                    return cleaned[:len(texts)]
                else:
                    error_msg = f"API Error {response.status}"
                    self.error_handler.track_error("API Error", error_msg, model=model)
                    return [f"[Error: {error_msg}]" for _ in texts]
                    
        except (aiohttp.ClientConnectorError, ConnectionResetError, BrokenPipeError) as e:
            self.error_handler.track_error("Connection Error", str(e), model=model)
            return [f"[Error: Connection failed]" for _ in texts]
            
        except asyncio.TimeoutError:
            self.error_handler.track_error("Timeout Error", "Request timeout", model=model)
            return [f"[Error: Timeout]" for _ in texts]
            
        except Exception as e:
            self.error_handler.track_error("Unexpected Error", str(e), model=model)
            return [f"[Error: {str(e)}]" for _ in texts]


# =============================================================================
# Main Translator
# =============================================================================

class TonguTranslator:
    """메인 번역기"""
    
    def __init__(self):
        self.config = TonguConfig()
        self.error_handler = ErrorHandler()
        self.client = OllamaClient(self.config, self.error_handler)
        
        # 간단한 캐시
        self.cache = {}
        self.cache_file = "tongu_cache.pkl"
        self.load_cache()
    
    def load_cache(self):
        """캐시 로드"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
        except:
            self.cache = {}
    
    def save_cache(self):
        """캐시 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass
    
    def get_cache_key(self, text: str, target_lang: str) -> str:
        """캐시 키 생성"""
        return f"{target_lang}:{hash(text)}"
    
    async def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        print("✅ Ollama 서버 연결 성공!")
                        print("🤖 사용 가능한 모델:")
                        for model in models.get('models', []):
                            print(f"   - {model['name']}")
                        return True
                    else:
                        print("❌ Ollama 서버 연결 실패")
                        return False
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            self.error_handler.track_error("Connection Test Failed", str(e))
            return False
    
    async def translate_texts(self, texts: List[str], target_lang: str) -> List[str]:
        """텍스트 리스트 번역"""
        if not texts:
            return []
        
        # 캐시 확인
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self.get_cache_key(text, target_lang)
            if cache_key in self.cache:
                cached_results.append((i, self.cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 새로 번역할 텍스트가 있으면 API 호출
        new_translations = []
        if uncached_texts:
            async with aiohttp.ClientSession() as session:
                new_translations = await self.client.translate_batch(
                    uncached_texts, target_lang, session
                )
            
            # 새 번역을 캐시에 저장 (에러가 아닌 경우만)
            for text, translation in zip(uncached_texts, new_translations):
                if not translation.startswith("[Error"):
                    cache_key = self.get_cache_key(text, target_lang)
                    self.cache[cache_key] = translation
        
        # 결과 합치기
        final_results = [''] * len(texts)
        
        # 캐시된 결과 배치
        for i, translation in cached_results:
            final_results[i] = translation
        
        # 새 번역 배치
        for i, translation_idx in enumerate(uncached_indices):
            if i < len(new_translations):
                final_results[translation_idx] = new_translations[i]
        
        return final_results
    
    async def translate_sample(self):
        """샘플 번역"""
        sample_texts = [
            "又升任骑都尉光禄大夫侍中。王莽在宫中值宿警卫，谨慎认真，地位越是尊贵，",
            "庄稼丰收，九月十月禾稼登场。制成春酒飘浓香，"
        ]
        
        print("=== 샘플 번역 시작 ===")
        
        # 한국어 번역
        korean_translations = await self.translate_texts(sample_texts, "korean")
        print("🇰🇷 한국어 번역:")
        for i, (original, korean) in enumerate(zip(sample_texts, korean_translations)):
            print(f"{i+1}. 원문: {original}")
            print(f"   한글: {korean}")
        
        # 영어 번역  
        english_translations = await self.translate_texts(sample_texts, "english")
        print("\n🇺🇸 영어 번역:")
        for i, (original, english) in enumerate(zip(sample_texts, english_translations)):
            print(f"{i+1}. 원문: {original}")
            print(f"   영어: {english}")
        
        self.save_cache()
        print("\n✅ 샘플 번역 완료!")
    
    async def translate_file(self, input_file: str, output_file: str):
        """파일 번역"""
        if not Path(input_file).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
            return False
        
        print(f"📁 파일 번역: {input_file} -> {output_file}")
        
        # 파일 읽기
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            return False
        
        print(f"📊 총 {len(data)}개 항목 처리")
        
        # 배치 처리
        processed_items = []
        
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            
            print(f"🔄 배치 처리 {i//self.config.batch_size + 1}/{(len(data)-1)//self.config.batch_size + 1}")
            
            # 텍스트 추출
            texts = []
            for item in batch:
                if isinstance(item, dict):
                    # ACCN 형식 처리
                    if 'data' in item and 'output' in item['data']:
                        texts.append(item['data']['output'])
                    else:
                        texts.append(str(item))
                else:
                    texts.append(str(item))
            
            # 번역
            korean_translations = await self.translate_texts(texts, "korean")
            await asyncio.sleep(self.config.delay_between_batches)
            
            english_translations = await self.translate_texts(texts, "english") 
            await asyncio.sleep(self.config.delay_between_batches)
            
            # 결과 통합
            for j, (original_item, korean, english) in enumerate(zip(batch, korean_translations, english_translations)):
                enhanced_item = original_item.copy() if isinstance(original_item, dict) else {"original": original_item}
                enhanced_item.update({
                    "korean_translation": korean,
                    "english_translation": english,
                    "translated_at": datetime.now().isoformat()
                })
                processed_items.append(enhanced_item)
        
        # 결과 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            self.save_cache()
            print(f"✅ 번역 완료: {len(processed_items)}개 항목 저장됨")
            return True
            
        except Exception as e:
            print(f"❌ 파일 저장 오류: {e}")
            return False
    
    def check_ollama_status(self) -> bool:
        """Ollama 상태 확인"""
        try:
            result = subprocess.run(
                ['curl', '-s', f'{self.config.ollama_base_url}/api/tags'],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def restart_ollama(self) -> bool:
        """Ollama 재시작"""
        try:
            print("🔄 Ollama 재시작 중...")
            subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True)
            time.sleep(5)
            
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)
            
            return self.check_ollama_status()
        except Exception as e:
            self.error_handler.track_error("Restart Failed", str(e))
            return False


# =============================================================================
# Main Function  
# =============================================================================

async def main():
    """메인 함수"""
    
    if len(sys.argv) < 2:
        print("""
🐲 Tongu - Korean Translation System (Clean)
============================================

사용법:
  python clean_tongu.py test                    # Ollama 연결 테스트
  python clean_tongu.py sample                  # 샘플 번역
  python clean_tongu.py translate <input> <output>  # 파일 번역
  python clean_tongu.py restart <command>       # 자동 재시작과 함께

특징:
  🔧 Broken pipe 문제 해결
  🚨 자동 에러 알림 (songhune@jou.ac.kr)
  💾 번역 캐싱으로 속도 향상
  📊 실시간 에러 모니터링

예시:
  python clean_tongu.py test
  python clean_tongu.py sample
  python clean_tongu.py translate input.jsonl output.jsonl
  python clean_tongu.py restart sample  # 재시작 기능과 함께
        """)
        return
    
    command = sys.argv[1]
    translator = TonguTranslator()
    
    try:
        if command == "test":
            success = await translator.test_connection()
            
        elif command == "sample":
            await translator.translate_sample()
            success = True
            
        elif command == "translate":
            if len(sys.argv) != 4:
                print("사용법: python clean_tongu.py translate <input_file> <output_file>")
                return
            success = await translator.translate_file(sys.argv[2], sys.argv[3])
            
        elif command == "restart":
            # 자동 재시작 모드
            if len(sys.argv) < 3:
                print("사용법: python clean_tongu.py restart <command>")
                return
            
            restart_command = sys.argv[2]
            max_retries = 3
            
            for attempt in range(1, max_retries + 1):
                print(f"📋 시도 {attempt}/{max_retries}")
                
                # Ollama 상태 확인
                if not translator.check_ollama_status():
                    if not translator.restart_ollama():
                        continue
                
                try:
                    if restart_command == "sample":
                        await translator.translate_sample()
                        success = True
                        break
                    elif restart_command == "test":
                        success = await translator.test_connection()
                        break
                    else:
                        print(f"❌ 알 수 없는 재시작 명령: {restart_command}")
                        success = False
                        break
                        
                except Exception as e:
                    print(f"❌ 시도 {attempt} 실패: {e}")
                    translator.error_handler.track_error("Execution Failed", str(e))
                    if attempt < max_retries:
                        time.sleep(30)
                    else:
                        success = False
            
        else:
            print(f"❌ 알 수 없는 명령: {command}")
            success = False
        
        # 결과 출력
        if success:
            print("🎉 완료!")
        else:
            print("💥 실패!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 중단됨")
        sys.exit(130)
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        translator.error_handler.track_error("Unexpected Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())