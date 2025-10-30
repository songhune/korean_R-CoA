import pandas as pd
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class STSPair:
    """STS 쌍 (sentence1, sentence2, score)"""
    sentence1: str
    sentence2: str
    score: float  # 0.0 ~ 5.0
    metadata: Dict = None

class KoreanSTSGenerator:
    """한국어 STS 데이터 생성기"""
    
    def __init__(self, similarity_model: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            similarity_model: 한국어 문장 유사도 계산용 모델
        """
        self.similarity_model = SentenceTransformer(similarity_model)
        
        # 유사도 범위별 변환 템플릿
        self.high_similarity_transforms = [
            self._paraphrase_transform,
            self._synonym_transform,
            self._structure_variation_transform
        ]
        
        self.medium_similarity_transforms = [
            self._partial_content_transform,
            self._perspective_change_transform,
            self._abstraction_transform
        ]
        
        self.low_similarity_transforms = [
            self._topic_shift_transform,
            self._negation_transform,
            self._unrelated_content_transform
        ]
    
    def generate_sts_pairs(self, sentences: List[str], target_distribution: Dict[str, int] = None) -> List[STSPair]:
        """STS 쌍 생성"""
        if target_distribution is None:
            target_distribution = {
                "high": 30,    # 4.0-5.0
                "medium": 40,  # 2.0-3.9
                "low": 30      # 0.0-1.9
            }
        
        pairs = []
        
        # 1. 고유사도 쌍 생성 (paraphrase)
        high_pairs = self._generate_high_similarity_pairs(sentences, target_distribution["high"])
        pairs.extend(high_pairs)
        
        # 2. 중간유사도 쌍 생성 (related but different)
        medium_pairs = self._generate_medium_similarity_pairs(sentences, target_distribution["medium"])
        pairs.extend(medium_pairs)
        
        # 3. 저유사도 쌍 생성 (unrelated)
        low_pairs = self._generate_low_similarity_pairs(sentences, target_distribution["low"])
        pairs.extend(low_pairs)
        
        # 4. 실제 유사도로 점수 보정
        pairs = self._calibrate_scores(pairs)
        
        return pairs
    
    def _generate_high_similarity_pairs(self, sentences: List[str], count: int) -> List[STSPair]:
        """고유사도 쌍 생성 (4.0-5.0)"""
        pairs = []
        selected_sentences = random.sample(sentences, min(count, len(sentences)))
        
        for sentence in selected_sentences:
            if len(sentence.strip()) < 10:
                continue
                
            transform = random.choice(self.high_similarity_transforms)
            transformed = transform(sentence)
            
            if transformed and transformed != sentence:
                score = random.uniform(4.0, 5.0)
                pairs.append(STSPair(sentence, transformed, score, 
                                   {"similarity_level": "high", "transform_type": transform.__name__}))
        
        return pairs
    
    def _generate_medium_similarity_pairs(self, sentences: List[str], count: int) -> List[STSPair]:
        """중간유사도 쌍 생성 (2.0-3.9)"""
        pairs = []
        
        # 1. 같은 도메인 내 다른 문장들
        for i, sentence1 in enumerate(sentences):
            if len(pairs) >= count:
                break
                
            # 유사한 주제의 다른 문장 찾기
            similar_sentences = self._find_similar_topic_sentences(sentence1, sentences[i+1:])
            
            for sentence2 in similar_sentences[:2]:  # 최대 2개만
                if len(pairs) >= count:
                    break
                    
                score = random.uniform(2.0, 3.9)
                pairs.append(STSPair(sentence1, sentence2, score,
                                   {"similarity_level": "medium", "transform_type": "similar_topic"}))
        
        # 2. 변환 기반 중간 유사도
        remaining = count - len(pairs)
        if remaining > 0:
            selected_sentences = random.sample(sentences, min(remaining, len(sentences)))
            
            for sentence in selected_sentences:
                transform = random.choice(self.medium_similarity_transforms)
                transformed = transform(sentence)
                
                if transformed and transformed != sentence:
                    score = random.uniform(2.0, 3.9)
                    pairs.append(STSPair(sentence, transformed, score,
                                       {"similarity_level": "medium", "transform_type": transform.__name__}))
        
        return pairs
    
    def _generate_low_similarity_pairs(self, sentences: List[str], count: int) -> List[STSPair]:
        """저유사도 쌍 생성 (0.0-1.9)"""
        pairs = []
        
        # 랜덤 페어링으로 관련 없는 문장 쌍 생성
        sentence_pairs = list(combinations(sentences, 2))
        random.shuffle(sentence_pairs)
        
        for sentence1, sentence2 in sentence_pairs[:count]:
            # 실제로 관련 없는지 확인 (간단한 키워드 기반)
            if not self._are_related(sentence1, sentence2):
                score = random.uniform(0.0, 1.9)
                pairs.append(STSPair(sentence1, sentence2, score,
                                   {"similarity_level": "low", "transform_type": "unrelated"}))
            
            if len(pairs) >= count:
                break
        
        return pairs
    
    def _paraphrase_transform(self, sentence: str) -> str:
        """패러프레이즈 변환"""
        paraphrase_patterns = [
            (r'이다', '이라고 할 수 있다'),
            (r'한다', '을 행한다'),
            (r'있다', '이 존재한다'),
            (r'때문에', '으로 인해'),
            (r'그러므로', '따라서'),
            (r'하지만', '그러나'),
            (r'또한', '더불어'),
            (r'중요하다', '핵심적이다')
        ]
        
        result = sentence
        pattern, replacement = random.choice(paraphrase_patterns)
        if pattern in result:
            result = result.replace(pattern, replacement)
        
        return result
    
    def _synonym_transform(self, sentence: str) -> str:
        """동의어 변환"""
        synonym_dict = {
            '학습': '공부', '지혜': '슬기', '예의': '예절',
            '중요한': '핵심적인', '어려운': '힘든', '좋은': '훌륭한',
            '사람': '인간', '방법': '방식', '생각': '사고',
            '문제': '과제', '결과': '성과', '원인': '이유'
        }
        
        result = sentence
        for original, synonym in synonym_dict.items():
            if original in result:
                result = result.replace(original, synonym)
                break  # 하나만 변경
        
        return result
    
    def _structure_variation_transform(self, sentence: str) -> str:
        """문장 구조 변경"""
        # 간단한 구조 변경 (능동↔피동, 어순 변경 등)
        if '을' in sentence and '한다' in sentence:
            return sentence.replace('을 한다', '이 이루어진다')
        elif '이' in sentence and '이다' in sentence:
            parts = sentence.split('이')
            if len(parts) >= 2:
                return f"{parts[1].strip()}은 {parts[0].strip()}이다"
        
        return sentence
    
    def _partial_content_transform(self, sentence: str) -> str:
        """부분 내용 변경"""
        # 문장의 일부만 유지하고 나머지는 다른 내용으로 변경
        words = sentence.split()
        if len(words) > 3:
            # 앞의 절반은 유지, 뒤의 절반은 일반적인 내용으로 변경
            half = len(words) // 2
            preserved = ' '.join(words[:half])
            return f"{preserved} 다양한 관점에서 이해할 수 있다."
        
        return sentence
    
    def _perspective_change_transform(self, sentence: str) -> str:
        """관점 변경"""
        perspective_additions = [
            " 이는 현대적 관점에서 중요하다.",
            " 역사적으로 볼 때 의미가 있다.",
            " 교육적 측면에서 가치가 있다.",
            " 문화적 맥락에서 이해해야 한다.",
            " 철학적으로 깊이 있게 생각해볼 만하다."
        ]
        
        return sentence.rstrip('.') + random.choice(perspective_additions)
    
    def _abstraction_transform(self, sentence: str) -> str:
        """추상화 변환"""
        abstract_endings = [
            " 이는 일반적인 원리로 확장될 수 있다.",
            " 이러한 개념은 보편적이다.",
            " 이는 근본적인 문제와 관련이 있다.",
            " 이는 본질적인 특성을 보여준다."
        ]
        
        return sentence.rstrip('.') + random.choice(abstract_endings)
    
    def _topic_shift_transform(self, sentence: str) -> str:
        """주제 변경"""
        unrelated_topics = [
            "날씨가 좋아서 기분이 좋다.",
            "새로운 기술이 발전하고 있다.",
            "음식의 맛이 매우 좋았다.",
            "여행을 가고 싶은 마음이 든다.",
            "책을 읽는 것은 즐거운 일이다."
        ]
        
        return random.choice(unrelated_topics)
    
    def _negation_transform(self, sentence: str) -> str:
        """부정 변환"""
        if '이다' in sentence:
            return sentence.replace('이다', '이 아니다')
        elif '한다' in sentence:
            return sentence.replace('한다', '하지 않는다')
        elif '있다' in sentence:
            return sentence.replace('있다', '없다')
        
        return f"{sentence} 이는 사실이 아니다."
    
    def _unrelated_content_transform(self, sentence: str) -> str:
        """완전히 다른 내용"""
        unrelated_sentences = [
            "컴퓨터 프로그래밍은 논리적 사고를 요구한다.",
            "바다의 파도 소리는 마음을 평안하게 한다.",
            "봄꽃이 피는 시기가 빨라지고 있다.",
            "운동은 건강에 매우 중요하다.",
            "영화를 보는 것은 좋은 취미이다."
        ]
        
        return random.choice(unrelated_sentences)
    
    def _find_similar_topic_sentences(self, target: str, candidates: List[str], max_count: int = 3) -> List[str]:
        """유사한 주제의 문장 찾기"""
        # 간단한 키워드 기반 유사성 검사
        target_keywords = set(re.findall(r'[가-힣]{2,}', target))
        
        similar = []
        for candidate in candidates:
            candidate_keywords = set(re.findall(r'[가-힣]{2,}', candidate))
            
            # 공통 키워드 비율 계산
            if target_keywords and candidate_keywords:
                overlap = len(target_keywords & candidate_keywords)
                ratio = overlap / len(target_keywords | candidate_keywords)
                
                if 0.1 < ratio < 0.7:  # 적당한 유사성
                    similar.append(candidate)
                    
                if len(similar) >= max_count:
                    break
        
        return similar
    
    def _are_related(self, sentence1: str, sentence2: str) -> bool:
        """두 문장이 관련 있는지 확인"""
        keywords1 = set(re.findall(r'[가-힣]{2,}', sentence1))
        keywords2 = set(re.findall(r'[가-힣]{2,}', sentence2))
        
        if not keywords1 or not keywords2:
            return False
        
        overlap = len(keywords1 & keywords2)
        ratio = overlap / min(len(keywords1), len(keywords2))
        
        return ratio > 0.3  # 30% 이상 키워드 겹치면 관련 있다고 판단
    
    def _calibrate_scores(self, pairs: List[STSPair]) -> List[STSPair]:
        """실제 임베딩 유사도로 점수 보정"""
        if not pairs:
            return pairs
        
        sentences1 = [pair.sentence1 for pair in pairs]
        sentences2 = [pair.sentence2 for pair in pairs]
        
        embeddings1 = self.similarity_model.encode(sentences1)
        embeddings2 = self.similarity_model.encode(sentences2)
        
        calibrated_pairs = []
        for i, pair in enumerate(pairs):
            # 코사인 유사도 계산 (0~1)
            cos_sim = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
            
            # 0~1을 0~5로 스케일링
            calibrated_score = cos_sim * 5.0
            
            # 원래 점수와 보정된 점수의 가중평균
            final_score = 0.7 * calibrated_score + 0.3 * pair.score
            final_score = np.clip(final_score, 0.0, 5.0)
            
            calibrated_pairs.append(STSPair(
                pair.sentence1, 
                pair.sentence2, 
                round(final_score, 2),
                pair.metadata
            ))
        
        return calibrated_pairs
    
    def save_to_jsonl(self, pairs: List[STSPair], output_path: str):
        """JSONL 형식으로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                json_obj = {
                    "sentence1": pair.sentence1,
                    "sentence2": pair.sentence2,
                    "score": pair.score,
                    "metadata": pair.metadata or {}
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        print(f" STS 데이터셋 저장 완료: {output_path} ({len(pairs)}개)")
        
        # 점수 분포 출력
        scores = [pair.score for pair in pairs]
        print(f"   점수 분포: 평균={np.mean(scores):.2f}, 표준편차={np.std(scores):.2f}")
        print(f"   점수 범위: 최소={min(scores):.2f}, 최대={max(scores):.2f}")