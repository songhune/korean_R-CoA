"""메인 번역기 클래스"""

import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
import logging

from config import TranslationConfig
from api_clients import APIClientFactory
from cost_tracker import CostTracker
from cache_manager import TranslationCache
from file_handlers import FileHandler, CheckpointManager
from text_processor import ACCNDataProcessor


class LargeScaleTranslator:
    """대용량 데이터셋 번역 처리기"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.session = None
        self.api_client = None
        
        # 컴포넌트 초기화
        self.cost_tracker = CostTracker(config)
        self.cache = TranslationCache()
        self.file_handler = FileHandler()
        self.checkpoint_manager = CheckpointManager()
        self.data_processor = ACCNDataProcessor()
        
        # 처리 상태
        self.processed_count = 0
        self.failed_items = []
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('translation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def create_session(self):
        """HTTP 세션 생성"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self.api_client = APIClientFactory.create_client(self.config, self.session)
    
    async def close_session(self):
        """HTTP 세션 종료"""
        if self.session:
            await self.session.close()
    
    async def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """배치 번역 처리 (캐시 포함)"""
        # 캐시에서 번역 조회
        cached_translations, uncached_indices, uncached_texts = (
            self.cache.get_cached_translations(texts, target_lang)
        )
        
        # 새로 번역할 텍스트가 있으면 API 호출
        new_translations = []
        if uncached_texts:
            if self.config.api_provider == "anthropic":
                new_translations, usage_info = await self.api_client.translate_batch(uncached_texts, target_lang)
                # 비용 추적 업데이트
                self.cost_tracker.update_cost_tracking(
                    usage_info["input_tokens"], 
                    usage_info["output_tokens"]
                )
            else:
                new_translations = await self.api_client.translate_batch(uncached_texts, target_lang)
            
            # 새 번역을 캐시에 저장
            self.cache.store_translations(uncached_texts, new_translations, target_lang)
        
        # 최종 번역 결과 조합
        final_translations = self.cache.merge_translations(
            texts, cached_translations, uncached_indices, new_translations
        )
        
        return final_translations
    
    async def process_chunk(self, chunk: List[Dict], chunk_id: int) -> List[Dict]:
        """청크 단위 처리"""
        self.logger.info(f"Processing chunk {chunk_id} with {len(chunk)} items")
        
        processed_chunk = []
        
        # 배치 단위로 처리
        for i in range(0, len(chunk), self.config.batch_size):
            # 예산 확인
            if not self.cost_tracker.check_budget():
                self.logger.warning("Budget limit reached. Stopping processing.")
                break
            
            batch = chunk[i:i + self.config.batch_size]
            
            try:
                # 유효한 아이템만 필터링
                valid_batch = [item for item in batch if self.data_processor.validate_item_structure(item)]
                
                if not valid_batch:
                    self.logger.warning(f"No valid items in batch {i//self.config.batch_size}")
                    continue
                
                # 텍스트 추출
                chinese_texts = self.data_processor.extract_texts_from_batch(valid_batch)
                
                # 한글 번역
                korean_translations = await self.translate_batch(chinese_texts, 'korean')
                await asyncio.sleep(self.config.delay_between_batches)
                
                # 영어 번역
                english_translations = await self.translate_batch(chinese_texts, 'english')
                await asyncio.sleep(self.config.delay_between_batches)
                
                # 결과 통합
                enhanced_items = self.data_processor.enhance_items_with_translations(
                    valid_batch, chinese_texts, korean_translations, english_translations
                )
                
                processed_chunk.extend(enhanced_items)
                self.processed_count += len(valid_batch)
                
                # 체크포인트 저장
                if self.processed_count % self.config.checkpoint_interval == 0:
                    self.cache.save_cache()
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
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            self.logger.info(f"Starting translation of {input_file}")
            
            # 파일 정보 출력
            file_info = self.file_handler.get_file_info(input_file)
            self.logger.info(f"Input file size: {file_info['size_mb']:.2f} MB")
            
            # 스트리밍 방식으로 파일 처리
            chunk_id = 0
            base_name = output_path.stem
            
            async for chunk in self.file_handler.read_json_or_jsonl_stream(input_file, self.config.chunk_size):
                # 청크 처리
                processed_chunk = await self.process_chunk(chunk, chunk_id)
                
                if processed_chunk:
                    # 중간 결과 체크포인트로 저장
                    await self.checkpoint_manager.save_checkpoint(
                        processed_chunk, chunk_id, base_name
                    )
                
                chunk_id += 1
            
            # 최종 결과 파일 생성
            await self.checkpoint_manager.merge_checkpoints(base_name, str(output_path))
            
            # 실패한 항목 저장
            if self.failed_items:
                failed_file = output_path.parent / f"{output_path.stem}_failed.jsonl"
                await self.file_handler.write_jsonl(self.failed_items, str(failed_file))
                self.logger.info(f"Saved {len(self.failed_items)} failed items to {failed_file}")
            
            # 통계 출력
            self.cost_tracker.print_final_statistics(self.processed_count, len(self.failed_items))
            
            # 체크포인트 정리 (선택사항)
            # self.checkpoint_manager.cleanup_checkpoints(base_name)
            
        finally:
            await self.close_session()
            self.cache.save_cache()
    
    async def process_sample(self, sample_data: List[Dict[str, Any]], output_file: str):
        """샘플 데이터 처리 (테스트용)"""
        await self.create_session()
        
        try:
            processed_items = await self.process_chunk(sample_data, 0)
            await self.file_handler.write_jsonl(processed_items, output_file)
            
            self.logger.info(f"Sample processing completed: {len(processed_items)} items")
            
            # 결과 출력
            for item in processed_items:
                original_text = item.get('original_classical_text', '')
                korean_trans = item.get('korean_translation', '')
                english_trans = item.get('english_translation', '')
                
                print(f"원문: {original_text}")
                print(f"한글: {korean_trans}")
                print(f"영어: {english_trans}")
                print("---")
        
        finally:
            await self.close_session()
            self.cache.save_cache()