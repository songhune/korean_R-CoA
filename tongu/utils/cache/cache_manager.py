"""번역 캐시 관리 모듈"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api.processors.text_processor import ClassicalTextExtractor


class TranslationCache:
    """번역 캐시 관리자"""
    
    def __init__(self, cache_file: str = "translation_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, str] = {}
        self.extractor = ClassicalTextExtractor()
        self.logger = logging.getLogger(__name__)
        
        self.load_cache()
    
    def load_cache(self):
        """캐시 파일에서 번역 캐시 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.cache)} cached translations")
            except (pickle.PickleError, EOFError) as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """번역 캐시를 파일에 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            self.logger.info(f"Saved {len(self.cache)} cached translations")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def get_cache_key(self, text: str, target_lang: str) -> str:
        """캐시 키 생성"""
        return self.extractor.get_cache_key(text, target_lang)
    
    def get_cached_translations(self, texts: List[str], target_lang: str) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """캐시에서 번역 조회
        
        Returns:
            cached_translations: (인덱스, 번역) 튜플 리스트
            uncached_indices: 캐시되지 않은 항목의 인덱스 리스트
            uncached_texts: 캐시되지 않은 텍스트 리스트
        """
        cached_translations = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self.get_cache_key(text, target_lang)
            if cache_key in self.cache:
                cached_translations.append((i, self.cache[cache_key]))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        return cached_translations, uncached_indices, uncached_texts
    
    def store_translations(self, texts: List[str], translations: List[str], target_lang: str):
        """번역 결과를 캐시에 저장 (에러 번역은 캐시하지 않음)"""
        for text, translation in zip(texts, translations):
            # 에러 번역은 캐시하지 않음 (재시도 가능하도록)
            if not (translation.startswith("[Translation Error") or translation == ""):
                cache_key = self.get_cache_key(text, target_lang)
                self.cache[cache_key] = translation
    
    def merge_translations(self, texts: List[str], cached_translations: List[Tuple[int, str]], 
                          uncached_indices: List[int], new_translations: List[str]) -> List[str]:
        """캐시된 번역과 새 번역을 합쳐서 최종 결과 생성"""
        final_translations = ['[Translation Error: Not processed]'] * len(texts)
        
        # 캐시된 번역 배치
        for idx, translation in cached_translations:
            final_translations[idx] = translation
        
        # 새 번역 배치
        for i, translation in enumerate(new_translations):
            if i < len(uncached_indices):
                original_idx = uncached_indices[i]
                final_translations[original_idx] = translation
        
        return final_translations
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        cache_size = len(self.cache)
        cache_file_size = 0
        
        if self.cache_file.exists():
            cache_file_size = self.cache_file.stat().st_size / 1024  # KB
        
        return {
            "cache_entries": cache_size,
            "cache_file_size_kb": cache_file_size,
            "cache_file_path": str(self.cache_file),
            "cache_hit_ratio": 0.0  # 런타임에 계산됨
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.logger.info("Cache cleared")