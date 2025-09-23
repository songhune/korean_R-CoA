"""
Tongu - Korean Translation System (Clean Version)
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
from typing import List


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
        
        # 번역 설정 - 최적화됨
        self.batch_size = 15  # 5 → 15 (3배 향상)
        self.max_concurrent = 8  # 3 → 8 (2.7배 향상)
        self.delay_between_batches = 0.5  # 2 → 0.5초 (4배 향상)
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
    
    def _parse_korean_response(self, content: str) -> List[str]:
        """한국어 모델 응답 파싱 (deepseek-r1 계열)"""
        # <think> 태그가 있으면 제거하고 실제 답변만 추출
        if '<think>' in content:
            # <think>...</think> 이후의 실제 답변 찾기
            parts = content.split('</think>')
            if len(parts) > 1:
                actual_response = parts[-1].strip()
            else:
                # </think>가 없는 경우 <think> 이후 모든 내용
                actual_response = content.split('<think>')[-1].strip()
        else:
            actual_response = content.strip()
        
        # 실제 번역 결과 파싱
        lines = actual_response.split('\n')
        translations = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and not line.startswith('好的') and not line.startswith('我'):
                # 번호가 있는 경우 번호 제거
                if line and line[0].isdigit() and '.' in line[:5]:
                    line = line.split('.', 1)[1].strip()
                if line:
                    translations.append(line)
        
        return translations
    
    def _parse_english_response(self, content: str) -> List[str]:
        """영어 모델 응답 파싱"""
        # 문장 기반으로 분할 시도
        translations = []
        
        # 번호가 있는 경우 번호별로 분할
        if content.count('1.') >= 1 or content.count('2.') >= 1:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith('here') and not line.lower().startswith('i'):
                    translations.append(line)
        else:
            # 단일 블록인 경우 문장으로 분할
            sentences = content.split('. ')
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if sentence:
                    # 마지막 문장이 아니면 마침표 추가
                    if i < len(sentences) - 1 and not sentence.endswith('.'):
                        sentence += '.'
                    translations.append(sentence)
        
        return translations
    
    async def translate_batch(self, texts: List[str], target_lang: str, session: aiohttp.ClientSession) -> List[str]:
        """배치 번역"""
        if not texts:
            return []
        
        # 모델 선택
        if target_lang == "korean":
            model = self.config.korean_model
            prompt_template = """다음 고전 중국어 텍스트를 자연스러운 현대 한국어로 번역하세요. 번역문만 제시하세요:

{texts}

번역:"""
        else:
            model = self.config.english_model
            prompt_template = """Please translate from Chinese to English:

{texts}

English:"""
        
        # 프롬프트 생성
        numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
        prompt = prompt_template.format(texts=numbered_texts)
        
        # 모델별 옵션 설정
        if target_lang == "english":
            options = {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 200,
                "stop": ["\n\n", "Chinese:", "Original:"]
            }
        else:
            options = {"temperature": 0.2, "top_p": 0.9}
            
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
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
                    
                    # 응답 파싱 - 모델별 처리
                    translations = []
                    if target_lang == "korean":
                        # 한국어 모델: <think> 태그 제거 및 실제 번역만 추출
                        translations = self._parse_korean_response(content)
                    else:
                        # 영어 모델: 기존 방식으로 파싱
                        translations = self._parse_english_response(content)
                    
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
        
        # 체크포인트 설정
        self.checkpoint_file = "checkpoints/tongu_checkpoint.json"
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
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
    
    def save_checkpoint(self, input_file: str, output_file: str, current_batch: int, total_batches: int, processed_count: int):
        """체크포인트 저장"""
        checkpoint_data = {
            "input_file": input_file,
            "output_file": output_file,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "processed_count": processed_count,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    def load_checkpoint(self) -> dict:
        """체크포인트 로드"""
        try:
            if Path(self.checkpoint_file).exists():
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
        return None
    
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
    
    async def translate_file(self, input_file: str, output_file: str, resume: bool = False):
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
        
        # 체크포인트 재개 처리
        start_batch = 0
        processed_items = []
        total_batches = (len(data) - 1) // self.config.batch_size + 1
        
        if resume:
            checkpoint = self.load_checkpoint()
            
            # 체크포인트의 input_file 경로 정규화
            checkpoint_input = checkpoint.get('input_file', '') if checkpoint else ''
            if checkpoint_input.startswith('../../'):
                # 상대 경로를 절대 경로로 변환
                checkpoint_input_abs = str(Path(input_file).parent.parent / checkpoint_input[6:])
            else:
                checkpoint_input_abs = checkpoint_input
            
            print(f"🔍 경로 비교:")
            print(f"   체크포인트: {checkpoint_input} → {checkpoint_input_abs}")
            print(f"   현재 파일: {input_file}")
            
            # 경로 비교 (파일 존재 여부 확인 후)
            paths_match = False
            try:
                print(f"🔍 문자열 비교: {checkpoint_input_abs == input_file}")
                if checkpoint_input_abs == input_file:
                    paths_match = True
                    print("✅ 문자열 경로 일치")
                elif Path(checkpoint_input_abs).exists() and Path(input_file).exists():
                    paths_match = Path(checkpoint_input_abs).samefile(Path(input_file))
                    print(f"✅ 파일 시스템 비교: {paths_match}")
                else:
                    print(f"❌ 파일 존재 여부: checkpoint={Path(checkpoint_input_abs).exists()}, current={Path(input_file).exists()}")
            except Exception as e:
                print(f"⚠️ 경로 비교 오류: {e}")
            
            print(f"🎯 최종 매칭 결과: {paths_match}")
            
            if checkpoint and paths_match:
                start_batch = checkpoint.get('current_batch', 0)
                processed_count = checkpoint.get('processed_count', 0)
                
                # 기존 출력 파일에서 이미 처리된 항목들 로드
                if Path(output_file).exists():
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    processed_items.append(json.loads(line.strip()))
                        print(f"🔄 체크포인트에서 재개: 배치 {start_batch}/{total_batches}, 이미 처리된 항목: {len(processed_items)}개")
                    except Exception as e:
                        print(f"⚠️ 기존 출력 파일 로드 실패: {e}, 처음부터 시작합니다")
                        start_batch = 0
                        processed_items = []
            else:
                print("⚠️ 체크포인트 정보가 현재 파일과 맞지 않습니다. 처음부터 시작합니다.")
        
        print(f"📊 총 {len(data)}개 항목 처리 (배치 {start_batch + 1}부터 시작)")
        
        # 배치 처리
        for batch_idx in range(start_batch, total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(data))
            batch = data[start_idx:end_idx]
            
            print(f"🔄 배치 처리 {batch_idx + 1}/{total_batches} (항목 {start_idx + 1}-{end_idx})")
            
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
            batch_results = []
            for original_item, korean, english in zip(batch, korean_translations, english_translations):
                enhanced_item = original_item.copy() if isinstance(original_item, dict) else {"original": original_item}
                enhanced_item.update({
                    "korean_translation": korean,
                    "english_translation": english,
                    "translated_at": datetime.now().isoformat()
                })
                batch_results.append(enhanced_item)
                processed_items.append(enhanced_item)
            
            # 체크포인트 저장 (매 배치마다)
            self.save_checkpoint(input_file, output_file, batch_idx + 1, total_batches, len(processed_items))
            
            # 중간 저장 (매 10배치마다 또는 마지막 배치)
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                try:
                    # 전체 결과를 다시 저장
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for item in processed_items:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"💾 중간 저장 완료: {len(processed_items)}개 항목")
                except Exception as e:
                    print(f"⚠️ 중간 저장 실패: {e}")
                
                self.save_cache()
        
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
    
    
    async def resume_translation(self):
        """체크포인트에서 번역 재개"""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            print("❌ 체크포인트를 찾을 수 없습니다")
            return False
            
        input_path = checkpoint['input_file']
        output_path = checkpoint['output_file']
        current_batch = checkpoint.get('current_batch', 0)
        
        print(f"🔄 체크포인트에서 재개:")
        print(f"   - 입력: {input_path}")
        print(f"   - 출력: {output_path}")
        print(f"   - 현재 배치: {current_batch}/{checkpoint['total_batches']}")
        print(f"   - 처리된 항목: {checkpoint['processed_count']}개")
        print(f"   - 마지막 저장: {checkpoint.get('timestamp', 'Unknown')}")
        
        # 절대 경로로 변환
        if not Path(input_path).is_absolute():
            input_path = str(Path.cwd() / input_path)
        if not Path(output_path).is_absolute():
            output_path = str(Path.cwd() / output_path)
            
        # 원본 파일 경로 정규화 (상대 경로 해결)
        if input_path.startswith("../../"):
            input_path = str(Path.cwd().parent.parent / input_path[6:])
        
        print(f"   - 정규화된 입력: {input_path}")
        
        # 입력이 실제 파일인지 디렉토리인지 확인
        if Path(input_path).is_file():
            # 일반 파일 기반 재개 (대용량 파일)
            print(f"📄 대용량 파일 처리 모드")
            if output_path.endswith("/") or Path(output_path).is_dir():
                # 출력이 디렉토리인 경우 파일명 생성
                output_path = str(Path(output_path) / "translated_ACCN-INS.jsonl")
            print(f"   - 출력 파일: {output_path}")
            return await self.translate_file(input_path, output_path, resume=True)
        elif Path(input_path).is_dir() or "checkpoints" in output_path:
            # 청크 기반 재개
            print(f"🧩 청크 파일 처리 모드")
            chunks_dir = "checkpoints" if "checkpoints" in output_path else input_path
            output_dir = "translated_chunks"  # 번역된 청크 저장 디렉토리
            return await self.translate_chunks(chunks_dir, output_dir, start_chunk=current_batch)
        else:
            print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
            return False
    
    async def translate_chunks(self, chunks_dir: str, output_dir: str, start_chunk: int = 0):
        """청크 파일들을 연속으로 처리"""
        chunks_path = Path(chunks_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 청크 파일 목록 가져오기 (정렬된 순서)
        chunk_files = sorted(chunks_path.glob("ACCN-INS_chunk_*.jsonl"))
        
        if not chunk_files:
            print(f"❌ {chunks_dir}에서 청크 파일을 찾을 수 없습니다")
            return False
        
        total_chunks = len(chunk_files)
        print(f"📁 총 {total_chunks}개 청크 파일 발견")
        
        # 시작 청크 범위 확인
        if start_chunk >= total_chunks:
            print(f"✅ 모든 청크 처리 완료! (요청: {start_chunk}, 총: {total_chunks})")
            return True
        
        print(f"🔄 청크 {start_chunk}부터 처리 시작")
        
        success_count = 0
        
        for i, chunk_file in enumerate(chunk_files[start_chunk:], start_chunk):
            chunk_name = chunk_file.name
            output_file = output_path / f"translated_{chunk_name}"
            
            # 이미 처리된 파일 스킵
            if output_file.exists():
                print(f"⏭️ 스킵: {chunk_name} (이미 처리됨)")
                success_count += 1
                continue
                
            print(f"🔄 처리 중 [{i+1}/{total_chunks}]: {chunk_name}")
            
            try:
                # 개별 청크 파일 번역
                success = await self.translate_file(str(chunk_file), str(output_file))
                
                if success:
                    success_count += 1
                    # 청크 단위 체크포인트 저장
                    self.save_checkpoint(
                        input_file=str(chunks_dir),
                        output_file=str(output_dir), 
                        current_batch=i + 1,
                        total_batches=total_chunks,
                        processed_count=success_count
                    )
                    print(f"✅ 완료: {chunk_name}")
                else:
                    print(f"❌ 실패: {chunk_name}")
                    
            except Exception as e:
                print(f"❌ 에러 [{chunk_name}]: {e}")
                self.error_handler.track_error("Chunk Processing Error", str(e), chunk=chunk_name)
        
        print(f"🎉 청크 처리 완료: {success_count}/{total_chunks}")
        return success_count == total_chunks
    
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
  python main.py test                    # Ollama 연결 테스트
  python main.py sample                  # 샘플 번역
  python main.py translate <input> <output>  # 파일 번역
  python main.py chunks <chunks_dir> <output_dir>  # 청크 파일들 번역
  python main.py resume                  # 체크포인트에서 재개
  python main.py restart <command>       # 자동 재시작과 함께

특징:
  🔧 Broken pipe 문제 해결
  🚨 자동 에러 알림 (songhune@jou.ac.kr)
  💾 번역 캐싱으로 속도 향상
  📊 실시간 에러 모니터링
  🧩 청크 기반 대용량 파일 처리

예시:
  python main.py test
  python main.py sample
  python main.py translate input.jsonl output.jsonl
  python main.py chunks checkpoints translated_chunks
  python main.py resume  # 체크포인트에서 바로 재개
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
                print("사용법: python main.py translate <input_file> <output_file>")
                return
            success = await translator.translate_file(sys.argv[2], sys.argv[3])
            
        elif command == "chunks":
            if len(sys.argv) != 4:
                print("사용법: python main.py chunks <chunks_dir> <output_dir>")
                print("예시: python main.py chunks checkpoints translated_chunks")
                return
            success = await translator.translate_chunks(sys.argv[2], sys.argv[3])
            
        elif command == "resume":
            # 체크포인트에서 단순 재개 (Ollama 재시작 없음)
            success = await translator.resume_translation()
            
        elif command == "restart":
            # 자동 재시작 모드
            if len(sys.argv) < 3:
                print("사용법: python main.py restart <command>")
                print("  sample: 샘플 번역 실행")
                print("  test: 연결 테스트 실행")
                print("  resume: 체크포인트에서 번역 재개")
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
                    elif restart_command == "resume":
                        success = await translator.resume_translation()
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