import json
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import backoff
from tqdm.asyncio import tqdm
import hashlib
import pickle

@dataclass
class TranslationConfig:
    """번역 설정"""
    # API 설정
    api_provider: str = "openai"  # "openai", "anthropic", "google", "deepl"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    
    # 배치 처리 설정
    batch_size: int = 50
    max_concurrent: int = 10
    delay_between_batches: float = 1.0
    
    # 재시도 설정
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # 파일 처리 설정
    chunk_size: int = 10000  # 한 번에 처리할 레코드 수
    checkpoint_interval: int = 1000  # 체크포인트 저장 간격


class LargeScaleTranslator:
    """대용량 데이터셋 번역 처리기"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.session = None
        self.processed_count = 0
        self.failed_items = []
        
        # 비용 추적
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0
        self.budget_limit = 50.0  # $50 예산
        
        # 캐시 설정 (중복 번역 방지)
        self.translation_cache = {}
        self.cache_file = Path("translation_cache.pkl")
        self.load_cache()
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('translation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_cache(self):
        """번역 캐시 로드"""
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.translation_cache = pickle.load(f)
            self.logger.info(f"Loaded {len(self.translation_cache)} cached translations")
    
    def check_budget(self) -> bool:
        """예산 확인"""
        if self.estimated_cost >= self.budget_limit * 0.95:  # 95% 도달시 경고
            self.logger.warning(f"Budget almost exhausted: ${self.estimated_cost:.2f}/${self.budget_limit}")
            return False
        return True
    
    def update_cost_tracking(self, input_tokens: int, output_tokens: int):
        """비용 추적 업데이트"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Claude Haiku 비용 계산
        input_cost = (input_tokens / 1000000) * 0.25
        output_cost = (output_tokens / 1000000) * 1.25
        batch_cost = input_cost + output_cost
        self.estimated_cost += batch_cost
        
        self.logger.info(f"Batch cost: ${batch_cost:.4f}, Total: ${self.estimated_cost:.2f}/${self.budget_limit}")
        
        return batch_cost
    
    def extract_classical_text(self, instruction: str) -> str:
        """instruction에서 순수 한문 텍스트 추출"""
        import re
        
        # 일반적인 번역 요청 패턴들 제거
        patterns_to_remove = [
            r'请将.*?翻译为现代汉语[。，]?',
            r'能否帮我翻译一下[？?]?',
            r'翻译一下这段文言文[：:]?',
            r'解释一下含义[：:]?',
            r'最后，请你再翻译[：:]?',
            r'你能帮我翻译这段古文吗[？?]?',
            r'我还想知道这段古文的翻译[：:]?',
            r'再翻译一下这段古文[：:]?',
            r'，能否帮我翻译一下[？?]?'
        ]
        
        # 패턴 제거
        cleaned_text = instruction
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 추가 정리
        cleaned_text = cleaned_text.strip('：:？?。，')
        cleaned_text = cleaned_text.strip()
        
        # 빈 텍스트면 원본 반환
        if not cleaned_text or len(cleaned_text) < 3:
            return instruction
            
        return cleaned_text
    
    async def create_session(self):
        """HTTP 세션 생성"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def close_session(self):
        """HTTP 세션 종료"""
        if self.session:
            await self.session.close()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def translate_openai(self, texts: List[str], target_lang: str) -> List[str]:
        """OpenAI API를 사용한 번역"""
        lang_map = {"korean": "Korean", "english": "English"}
        target_language = lang_map.get(target_lang, target_lang)
        
        # 배치 번역을 위한 프롬프트
        batch_prompt = f"""
Translate the following Ancient Chinese texts to {target_language}. 
Return ONLY the translations, one per line, maintaining the same order.
Do not include any explanations or numbering.

Texts to translate:
{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}
"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": f"You are a professional translator specializing in Ancient Chinese to {target_language} translation."},
                {"role": "user", "content": batch_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": len(' '.join(texts)) * 3  # 대략적인 토큰 수 추정
        }
        
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                translations = result["choices"][0]["message"]["content"].strip().split('\n')
                
                # 결과 정제 (번호나 여분의 텍스트 제거)
                cleaned_translations = []
                for trans in translations:
                    trans = trans.strip()
                    # 번호 제거 (1., 2. 등)
                    if trans and trans[0].isdigit() and '.' in trans[:5]:
                        trans = trans.split('.', 1)[1].strip()
                    cleaned_translations.append(trans)
                
                return cleaned_translations[:len(texts)]  # 원본과 같은 수만 반환
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    async def translate_anthropic(self, texts: List[str], target_lang: str) -> List[str]:
        """Anthropic Claude API를 사용한 번역"""
        lang_map = {"korean": "Korean", "english": "English"}
        target_language = lang_map.get(target_lang, target_lang)
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # ACCN-INS 데이터 특성에 맞는 프롬프트
        prompt = f"""Translate the following Classical Chinese texts to {target_language}. 
These are from ancient Chinese literature and documents. Provide accurate, natural translations.
Return only the translations, one per line, in the same order:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}"""
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": len(' '.join(texts)) * 4,  # 한문→한글/영어는 더 길어짐
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.2  # 번역은 일관성이 중요
        }
        
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["content"][0]["text"].strip()
                    translations = content.split('\n')
                    
                    # 토큰 사용량 추적
                    usage = result.get("usage", {})
                    input_tokens = usage.get("input_tokens", len(prompt) // 4)
                    output_tokens = usage.get("output_tokens", len(content) // 4)
                    self.update_cost_tracking(input_tokens, output_tokens)
                    
                    # 결과 정제 (번호 제거)
                    cleaned_translations = []
                    for trans in translations:
                        trans = trans.strip()
                        # 번호 제거 (1., 2. 등)
                        if trans and trans[0].isdigit() and '.' in trans[:5]:
                            trans = trans.split('.', 1)[1].strip()
                        if trans:  # 빈 번역 방지
                            cleaned_translations.append(trans)
                    
                    # 원본과 같은 수가 되도록 조정
                    while len(cleaned_translations) < len(texts):
                        cleaned_translations.append("[Translation Error]")
                    
                    return cleaned_translations[:len(texts)]
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API Error {response.status}: {error_text}")
    
    async def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """배치 번역 처리"""
        # 캐시 확인
        cached_translations = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self.get_cache_key(text, target_lang)
            if cache_key in self.translation_cache:
                cached_translations.append((i, self.translation_cache[cache_key]))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # 번역 실행
        if uncached_texts:
            if self.config.api_provider == "openai":
                new_translations = await self.translate_openai(uncached_texts, target_lang)
            elif self.config.api_provider == "anthropic":
                new_translations = await self.translate_anthropic(uncached_texts, target_lang)
            else:
                raise ValueError(f"Unsupported API provider: {self.config.api_provider}")
            
            # 캐시에 저장
            for text, translation in zip(uncached_texts, new_translations):
                cache_key = self.get_cache_key(text, target_lang)
                self.translation_cache[cache_key] = translation
        else:
            new_translations = []
        
        # 결과 재조립
        final_translations = [''] * len(texts)
        
        # 캐시된 번역 배치
        for idx, translation in cached_translations:
            final_translations[idx] = translation
        
        # 새 번역 배치
        for i, translation in enumerate(new_translations):
            original_idx = uncached_indices[i]
            final_translations[original_idx] = translation
        
        return final_translations
    
    async def process_chunk(self, chunk: List[Dict], chunk_id: int) -> List[Dict]:
        """청크 단위 처리"""
        self.logger.info(f"Processing chunk {chunk_id} with {len(chunk)} items")
        
        processed_chunk = []
        
        # 배치 단위로 처리
        for i in range(0, len(chunk), self.config.batch_size):
            # 예산 확인
            if not self.check_budget():
                self.logger.warning("Budget limit reached. Stopping processing.")
                break
                
            batch = chunk[i:i + self.config.batch_size]
            
            try:
                # ACCN-INS 구조에서 텍스트 추출
                chinese_texts = []
                for item in batch:
                    # instruction에서 한문 텍스트 추출
                    instruction = item['data']['instruction']
                    # 번역 요청 부분 제거하고 순수 한문만 추출
                    chinese_text = self.extract_classical_text(instruction)
                    chinese_texts.append(chinese_text)
                
                # 한글 번역
                korean_translations = await self.translate_batch(chinese_texts, 'korean')
                await asyncio.sleep(self.config.delay_between_batches)
                
                # 영어 번역
                english_translations = await self.translate_batch(chinese_texts, 'english')
                await asyncio.sleep(self.config.delay_between_batches)
                
                # 결과 통합 - ACCN-INS 구조 유지
                for j, item in enumerate(batch):
                    enhanced_item = item.copy()
                    enhanced_item['korean_translation'] = korean_translations[j]
                    enhanced_item['english_translation'] = english_translations[j]
                    
                    # 추가 정보
                    enhanced_item['original_classical_text'] = chinese_texts[j]
                    enhanced_item['multilingual_enhanced'] = True
                    
                    processed_chunk.append(enhanced_item)
                
                self.processed_count += len(batch)
                
                # 중간 저장 (체크포인트)
                if self.processed_count % self.config.checkpoint_interval == 0:
                    self.save_cache()
                    self.logger.info(f"Checkpoint: {self.processed_count} items processed")
                
            except Exception as e:
                self.logger.error(f"Error processing batch in chunk {chunk_id}: {e}")
                self.failed_items.extend(batch)
        
        return processed_chunk
    
    async def process_large_dataset(self, input_file: str, output_file: str):
        """대용량 데이터셋 처리"""
        await self.create_session()
        
        try:
            input_path = Path(input_file)
            output_path = Path(output_file)
            
            self.logger.info(f"Starting translation of {input_file}")
            
            # 파일 크기 확인
            file_size = input_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"Input file size: {file_size:.2f} MB")
            
            # 스트리밍 방식으로 파일 처리
            processed_items = []
            chunk_id = 0
            
            async with aiofiles.open(input_file, 'r', encoding='utf-8') as infile:
                current_chunk = []
                
                async for line in infile:
                    try:
                        item = json.loads(line.strip())
                        current_chunk.append(item)
                        
                        # 청크 크기에 도달하면 처리
                        if len(current_chunk) >= self.config.chunk_size:
                            processed_chunk = await self.process_chunk(current_chunk, chunk_id)
                            processed_items.extend(processed_chunk)
                            
                            # 중간 결과 저장
                            await self.save_intermediate_results(processed_items, output_path, chunk_id)
                            processed_items = []  # 메모리 정리
                            
                            current_chunk = []
                            chunk_id += 1
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON line: {e}")
                        continue
                
                # 마지막 청크 처리
                if current_chunk:
                    processed_chunk = await self.process_chunk(current_chunk, chunk_id)
                    processed_items.extend(processed_chunk)
            
            # 최종 결과 저장
            await self.save_final_results(processed_items, output_path)
            
            self.logger.info(f"Translation completed. Total processed: {self.processed_count}")
            self.logger.info(f"Failed items: {len(self.failed_items)}")
            
            # 실패한 항목 저장
            if self.failed_items:
                failed_file = output_path.parent / f"{output_path.stem}_failed.jsonl"
                await self.save_failed_items(failed_file)
        
        finally:
            await self.close_session()
            self.save_cache()
    
    async def save_intermediate_results(self, items: List[Dict], output_path: Path, chunk_id: int):
        """중간 결과 저장"""
        intermediate_file = output_path.parent / f"{output_path.stem}_chunk_{chunk_id}.jsonl"
        
        async with aiofiles.open(intermediate_file, 'w', encoding='utf-8') as f:
            for item in items:
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    async def save_final_results(self, items: List[Dict], output_path: Path):
        """최종 결과 저장"""
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    async def save_failed_items(self, failed_file: Path):
        """실패한 항목 저장"""
        async with aiofiles.open(failed_file, 'w', encoding='utf-8') as f:
            for item in self.failed_items:
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ACCN-INS 데이터셋 특화 사용 예시
async def process_accn_ins_dataset():
    """ACCN-INS 데이터셋 처리 함수"""
    
    config = TranslationConfig(
        api_provider="anthropic",
        api_key="your-claude-api-key-here",
        model="claude-3-haiku-20240307",
        batch_size=30,  # ACCN-INS는 텍스트가 길어서 배치 사이즈 줄임
        max_concurrent=6,
        delay_between_batches=0.8,
        chunk_size=8000,
        checkpoint_interval=500
    )
    
    translator = LargeScaleTranslator(config)
    
    # ACCN-INS 데이터셋 처리
    await translator.process_large_dataset(
        input_file="accn_ins_dataset.jsonl",  # 원본 ACCN-INS 파일
        output_file="accn_ins_multilingual.jsonl"  # 한글/영어 번역 추가된 파일
    )
    
    print(f"\n=== 번역 완료 통계 ===")
    print(f"총 처리 항목: {translator.processed_count:,}")
    print(f"총 비용: ${translator.estimated_cost:.2f}")
    print(f"입력 토큰: {translator.total_input_tokens:,}")
    print(f"출력 토큰: {translator.total_output_tokens:,}")
    print(f"실패 항목: {len(translator.failed_items)}")


# 샘플 테스트 함수
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
        api_key="your-claude-api-key-here",
        model="claude-3-haiku-20240307",
        batch_size=2,
        max_concurrent=1,
        delay_between_batches=1.0
    )
    
    translator = LargeScaleTranslator(config)
    
    await translator.process_large_dataset(
        input_file="accn_sample.jsonl",
        output_file="accn_sample_translated.jsonl"
    )
    
    # 결과 확인
    with open("accn_sample_translated.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            print(f"원문: {translator.extract_classical_text(result['data']['instruction'])}")
            print(f"한글: {result['korean_translation']}")
            print(f"영어: {result['english_translation']}")
            print("---")


if __name__ == "__main__":
    # 먼저 샘플 테스트
    # asyncio.run(test_with_sample())
    
    # 비용 추정
    # estimate_translation_cost("accn_ins_dataset.jsonl", "anthropic")
    
    # 실제 처리
    asyncio.run(process_accn_ins_dataset())


# 비용 추정 함수
def estimate_translation_cost(file_path: str, api_provider: str = "openai"):
    """번역 비용 추정"""
    
    # 대략적인 토큰 수 계산
    total_chars = 0
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get('chinese', item.get('instruction', ''))
                total_chars += len(text)
                line_count += 1
            except:
                continue
    
    # 토큰 수 추정 (중국어: 1 char ≈ 1 token, 영어/한국어 번역: 2x)
    input_tokens = total_chars
    output_tokens = total_chars * 2 * 2  # 한글 + 영어
    
    # 비용 계산 (2024년 기준)
    costs = {
        "openai": {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06}
        },
        "anthropic": {
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
    }
    
    if api_provider == "openai":
        model_cost = costs["openai"]["gpt-3.5-turbo"]
    else:
        model_cost = costs["anthropic"]["claude-3-haiku"]
    
    total_cost = (input_tokens * model_cost["input"] + output_tokens * model_cost["output"]) / 1000
    
    print(f"Dataset Statistics:")
    print(f"- Total lines: {line_count:,}")
    print(f"- Total characters: {total_chars:,}")
    print(f"- Estimated input tokens: {input_tokens:,}")
    print(f"- Estimated output tokens: {output_tokens:,}")
    print(f"- Estimated cost ({api_provider}): ${total_cost:.2f}")
    
    return total_cost


if __name__ == "__main__":
    # 비용 먼저 추정해보기
    # estimate_translation_cost("your_accn_ins_file.jsonl", "openai")
    
    # 실제 번역 실행
    asyncio.run(main())