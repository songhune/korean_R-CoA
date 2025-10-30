import pandas as pd
import json
import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class NLITriple:
    """NLI 삼중체 (premise, hypothesis, label)"""
    premise: str
    hypothesis: str
    label: str  # entailment, contradiction, neutral
    metadata: Dict = None

class KoreanNLIGenerator:
    """한국어 NLI 데이터 생성기"""
    
    def __init__(self, similarity_model: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            similarity_model: 한국어 문장 유사도 계산용 모델
        """
        self.similarity_model = SentenceTransformer(similarity_model)
        
        # 템플릿 정의
        self.entailment_templates = [
            "{premise} 이므로 {conclusion}이다.",
            "{premise} 따라서 {conclusion}이다.", 
            "{premise} 그러므로 {conclusion}이다.",
            "{premise} 이는 곧 {conclusion}을 의미한다.",
            "{premise} 즉, {conclusion}이다."
        ]
        
        self.contradiction_templates = [
            "{premise} 그러나 {negation}이다.",
            "{premise} 하지만 {negation}이다.",
            "{premise} 오히려 {negation}이다.",
            "{premise} 반대로 {negation}이다.",
            "{premise} 그럼에도 {negation}이다."
        ]
        
        self.neutral_templates = [
            "{premise} 또한 {additional}일 수 있다.",
            "{premise} 한편, {additional}이다.",
            "{premise} 동시에 {additional}도 가능하다.",
            "{premise} 그런데 {additional}라고 볼 수도 있다.",
            "{premise} 더불어 {additional}이다."
        ]
        
        # 부정 변환 패턴
        self.negation_patterns = [
            (r'이다', '이 아니다'),
            (r'있다', '없다'),
            (r'한다', '하지 않는다'),
            (r'된다', '되지 않는다'),
            (r'좋다', '나쁘다'),
            (r'크다', '작다'),
            (r'많다', '적다'),
            (r'높다', '낮다')
        ]
    
    def generate_entailment(self, premise: str) -> List[NLITriple]:
        """Entailment 가설 생성"""
        triples = []
        
        # 1. 요약형 entailment (핵심 내용 추출)
        summary = self._extract_key_points(premise)
        if summary:
            template = random.choice(self.entailment_templates)
            hypothesis = template.format(premise=premise[:50] + "...", conclusion=summary)
            triples.append(NLITriple(premise, hypothesis, "entailment"))
        
        # 2. 인과관계 entailment
        causal_conclusion = self._generate_causal_conclusion(premise)
        if causal_conclusion:
            template = random.choice(self.entailment_templates[:3])  # 인과관계 템플릿만
            hypothesis = template.format(premise=premise[:50] + "...", conclusion=causal_conclusion)
            triples.append(NLITriple(premise, hypothesis, "entailment"))
            
        return triples
    
    def generate_contradiction(self, premise: str) -> List[NLITriple]:
        """Contradiction 가설 생성"""
        triples = []
        
        # 1. 직접 부정
        negated = self._negate_statement(premise)
        if negated:
            template = random.choice(self.contradiction_templates)
            hypothesis = template.format(premise=premise[:50] + "...", negation=negated)
            triples.append(NLITriple(premise, hypothesis, "contradiction"))
        
        # 2. 논리적 반대 개념
        opposite = self._generate_opposite_concept(premise)
        if opposite:
            template = random.choice(self.contradiction_templates)
            hypothesis = template.format(premise=premise[:50] + "...", negation=opposite)
            triples.append(NLITriple(premise, hypothesis, "contradiction"))
            
        return triples
    
    def generate_neutral(self, premise: str) -> List[NLITriple]:
        """Neutral 가설 생성"""
        triples = []
        
        # 1. 관련 있지만 함의되지 않는 정보
        related_info = self._generate_related_info(premise)
        if related_info:
            template = random.choice(self.neutral_templates)
            hypothesis = template.format(premise=premise[:50] + "...", additional=related_info)
            triples.append(NLITriple(premise, hypothesis, "neutral"))
        
        # 2. 추측성 정보
        speculative = self._generate_speculative_info(premise)
        if speculative:
            template = random.choice(self.neutral_templates[-2:])  # 추측형 템플릿
            hypothesis = template.format(premise=premise[:50] + "...", additional=speculative)
            triples.append(NLITriple(premise, hypothesis, "neutral"))
            
        return triples
    
    def _extract_key_points(self, text: str) -> str:
        """텍스트에서 핵심 요점 추출"""
        # 간단한 키워드 기반 요약 (실제로는 더 정교한 방법 사용)
        key_terms = re.findall(r'[가-힣]{2,4}', text)
        if len(key_terms) >= 2:
            return f"{key_terms[0]}와 {key_terms[1]}가 중요하다"
        return ""
    
    def _generate_causal_conclusion(self, text: str) -> str:
        """인과관계 결론 생성"""
        # 텍스트에서 원인-결과 관계 추출
        if '때문에' in text or '인해' in text:
            return "결과적으로 변화가 있었다"
        elif '학습' in text or '공부' in text:
            return "지식이 향상되었다"
        elif '예' in text or '의례' in text:
            return "올바른 행동이 중요하다"
        return ""
    
    def _negate_statement(self, text: str) -> str:
        """문장 부정 변환"""
        for pattern, replacement in self.negation_patterns:
            if pattern in text:
                return text.replace(pattern, replacement)
        
        # 기본 부정 패턴
        if text.endswith('다'):
            return text[:-1] + "지 않다"
        return ""
    
    def _generate_opposite_concept(self, text: str) -> str:
        """반대 개념 생성"""
        opposite_pairs = {
            '선': '악', '좋': '나쁘', '크': '작', '많': '적',
            '높': '낮', '빠르': '느리', '어려운': '쉬운',
            '학습': '무지', '예의': '무례', '지혜': '어리석음'
        }
        
        for original, opposite in opposite_pairs.items():
            if original in text:
                return text.replace(original, opposite)
        return ""
    
    def _generate_related_info(self, text: str) -> str:
        """관련 있지만 함의되지 않는 정보 생성"""
        related_templates = [
            "다른 관점에서 볼 수도 있다",
            "추가적인 연구가 필요하다", 
            "역사적 맥락을 고려해야 한다",
            "현대적 해석이 가능하다",
            "문화적 차이가 존재한다"
        ]
        return random.choice(related_templates)
    
    def _generate_speculative_info(self, text: str) -> str:
        """추측성 정보 생성"""
        speculative_templates = [
            "다른 결과가 나올 수도 있었을 것이다",
            "완전히 확실하다고 할 수는 없다",
            "여러 해석이 가능하다",
            "상황에 따라 달라질 수 있다",
            "추가 검증이 필요하다"
        ]
        return random.choice(speculative_templates)
    
    def filter_by_similarity(self, triples: List[NLITriple], threshold: float = 0.98) -> List[NLITriple]:
        """유사도 기반 중복 제거"""
        if not triples:
            return triples
            
        # 문장 임베딩 계산
        premises = [t.premise for t in triples]
        hypotheses = [t.hypothesis for t in triples]
        
        premise_embeddings = self.similarity_model.encode(premises)
        hypothesis_embeddings = self.similarity_model.encode(hypotheses)
        
        filtered = []
        used_indices = set()
        
        for i, triple in enumerate(triples):
            if i in used_indices:
                continue
                
            is_duplicate = False
            for j in range(i + 1, len(triples)):
                if j in used_indices:
                    continue
                    
                # premise와 hypothesis 모두 유사도 확인
                p_sim = cosine_similarity([premise_embeddings[i]], [premise_embeddings[j]])[0][0]
                h_sim = cosine_similarity([hypothesis_embeddings[i]], [hypothesis_embeddings[j]])[0][0]
                
                if p_sim > threshold and h_sim > threshold:
                    used_indices.add(j)
                    is_duplicate = True
            
            if not is_duplicate:
                filtered.append(triple)
                
        return filtered
    
    def generate_nli_dataset(self, premises: List[str], max_per_premise: int = 3) -> List[NLITriple]:
        """전체 NLI 데이터셋 생성"""
        all_triples = []
        
        for premise in premises:
            if not premise or len(premise.strip()) < 10:
                continue
                
            # 각 라벨별로 생성
            entailments = self.generate_entailment(premise)[:max_per_premise//3 + 1]
            contradictions = self.generate_contradiction(premise)[:max_per_premise//3 + 1]
            neutrals = self.generate_neutral(premise)[:max_per_premise//3 + 1]
            
            all_triples.extend(entailments + contradictions + neutrals)
        
        # 유사도 기반 필터링
        filtered_triples = self.filter_by_similarity(all_triples)
        
        return filtered_triples
    
    def save_to_jsonl(self, triples: List[NLITriple], output_path: str):
        """JSONL 형식으로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for triple in triples:
                json_obj = {
                    "premise": triple.premise,
                    "hypothesis": triple.hypothesis, 
                    "label": triple.label,
                    "metadata": triple.metadata or {}
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        print(f" NLI 데이터셋 저장 완료: {output_path} ({len(triples)}개)")