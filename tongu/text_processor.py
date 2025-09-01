"""텍스트 전처리 및 추출 유틸리티"""

import re
import hashlib
from typing import Dict, Any, List


class ClassicalTextExtractor:
    """고전 중국어 텍스트 추출기"""
    
    def __init__(self):
        # 번역 요청 패턴 (중국어)
        self.chinese_patterns = [
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
        
        # 번역 요청 패턴 (한국어)
        self.korean_patterns = [
            r'다음.*?번역해.*?주세요[.。]?',
            r'이.*?한문.*?번역.*?[.。]?',
            r'.*?현대어로.*?번역.*?[.。]?',
            r'.*?한글로.*?옮겨.*?[.。]?',
            r'다음.*?고전.*?해석.*?[.。]?'
        ]
    
    def extract_classical_text(self, instruction: str) -> str:
        """instruction에서 순수 한문 텍스트 추출"""
        cleaned_text = instruction
        
        # 중국어 패턴 제거
        for pattern in self.chinese_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 한국어 패턴 제거
        for pattern in self.korean_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 추가 정리
        cleaned_text = cleaned_text.strip('：:？?。，')
        cleaned_text = cleaned_text.strip()
        
        # 빈 텍스트면 원본 반환
        if not cleaned_text or len(cleaned_text) < 3:
            return instruction
            
        return cleaned_text
    
    def get_cache_key(self, text: str, target_lang: str) -> str:
        """캐시 키 생성"""
        combined = f"{text}_{target_lang}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


class ACCNDataProcessor:
    """ACCN-INS 데이터셋 전용 처리기"""
    
    def __init__(self):
        self.extractor = ClassicalTextExtractor()
    
    def extract_texts_from_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """배치에서 현대 중국어 텍스트 추출 (output 기반)"""
        modern_chinese_texts = []
        for item in batch:
            # ACCN-INS 구조에서 output(현대 중국어) 추출
            output = item['data']['output']
            # output은 이미 현대 중국어로 번역된 텍스트이므로 그대로 사용
            modern_chinese_texts.append(output.strip())
        return modern_chinese_texts
    
    def enhance_items_with_translations(
        self, 
        batch: List[Dict[str, Any]], 
        modern_chinese_texts: List[str],
        korean_translations: List[str], 
        english_translations: List[str]
    ) -> List[Dict[str, Any]]:
        """번역 결과로 ACCN-INS 아이템 강화"""
        enhanced_items = []
        
        for i, item in enumerate(batch):
            enhanced_item = item.copy()
            enhanced_item['korean_translation'] = korean_translations[i]
            enhanced_item['english_translation'] = english_translations[i]
            enhanced_item['source_modern_chinese'] = modern_chinese_texts[i]
            enhanced_item['original_classical_chinese'] = item['data']['instruction']  # 원본 고전 중국어 보존
            enhanced_item['multilingual_enhanced'] = True
            
            enhanced_items.append(enhanced_item)
        
        return enhanced_items
    
    def validate_item_structure(self, item: Dict[str, Any]) -> bool:
        """ACCN-INS 아이템 구조 검증"""
        try:
            return (
                'data' in item and 
                'instruction' in item['data'] and 
                'output' in item['data'] and
                isinstance(item['data']['instruction'], str) and
                isinstance(item['data']['output'], str) and
                len(item['data']['output'].strip()) > 0  # output이 비어있지 않은지 확인
            )
        except (KeyError, TypeError):
            return False