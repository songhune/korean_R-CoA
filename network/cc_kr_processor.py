import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class CCKRPair:
    """고전한문-한국어 쌍"""
    classical_chinese: str
    korean_translation: str
    korean_explanation: str
    source: str
    metadata: Dict = None

class ClassicalChineseKoreanProcessor:
    """기존 CC-KR 데이터를 활용한 NLI/STS 데이터 생성기"""
    
    def __init__(self):
        self.cc_kr_pairs = []
        
        # 고전한문 접속어/논리 표지
        self.cc_logical_markers = {
            "故": "그러므로",
            "是以": "이런 까닭에", 
            "然": "그러나",
            "雖": "비록",
            "而": "그런데",
            "若": "만약",
            "則": "그러면",
            "非": "아니다",
            "不然": "그렇지 않다",
            "且": "또한",
            "亦": "또한",
            "或": "혹은"
        }
        
        # 고전한문 핵심 어휘
        self.classical_vocabulary = {
            "仁": "인", "義": "의", "禮": "예", "智": "지",
            "學": "배움", "道": "도", "德": "덕", "聖": "성인",
            "君子": "군자", "小人": "소인", "孝": "효도",
            "弟": "공경", "忠": "충성", "信": "믿음"
        }
    
    def load_existing_data(self, saseo_jsonl_path: str, sigwon_csv_path: str):
        """기존 CC-KR 데이터 로드"""
        self.cc_kr_pairs = []
        
        # 1. saseo JSONL 데이터 로드 (대화형)
        try:
            with open(saseo_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        cc_kr_pair = self._extract_cc_kr_from_dialog(data)
                        if cc_kr_pair:
                            self.cc_kr_pairs.append(cc_kr_pair)
            print(f" Saseo JSONL 로드: {len([p for p in self.cc_kr_pairs if p.source == 'saseo'])}개")
        except Exception as e:
            print(f" Saseo JSONL 로드 실패: {e}")
        
        # 2. sigwon CSV 데이터 로드 (원문-번역)
        try:
            sigwon_df = pd.read_csv(sigwon_csv_path, encoding='utf-8')
            sigwon_pairs = self._extract_cc_kr_from_sigwon(sigwon_df)
            self.cc_kr_pairs.extend(sigwon_pairs)
            print(f" Sigwon CSV 로드: {len(sigwon_pairs)}개")
        except Exception as e:
            print(f" Sigwon CSV 로드 실패: {e}")
        
        print(f" 총 CC-KR 쌍: {len(self.cc_kr_pairs)}개")
    
    def _extract_cc_kr_from_dialog(self, dialog_data: Dict) -> Optional[CCKRPair]:
        """대화 데이터에서 CC-KR 추출"""
        messages = dialog_data.get("messages", [])
        
        for i, message in enumerate(messages):
            if message.get("role") == "user" and "다음 한문 문장을 해석하고" in message.get("content", ""):
                # 사용자 메시지에서 한문 추출
                content = message.get("content", "")
                cc_text = self._extract_chinese_text(content)
                
                # 다음 assistant 메시지에서 한국어 해석 추출
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_content = messages[i + 1].get("content", "")
                    kr_translation, kr_explanation = self._parse_assistant_response(assistant_content)
                    
                    if cc_text and kr_translation:
                        return CCKRPair(
                            classical_chinese=cc_text,
                            korean_translation=kr_translation,
                            korean_explanation=kr_explanation,
                            source="saseo"
                        )
        return None
    
    def _extract_cc_kr_from_sigwon(self, df: pd.DataFrame) -> List[CCKRPair]:
        """sigwon 데이터에서 CC-KR 추출"""
        pairs = []
        
        for _, row in df.iterrows():
            cc_text = str(row.get('original', '')).strip()
            kr_text = str(row.get('translation', '') or row.get('meaning', '')).strip()
            
            if cc_text and kr_text and len(cc_text) > 5 and len(kr_text) > 5:
                # 한문인지 확인 (한자 비율)
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cc_text))
                if chinese_chars / len(cc_text) > 0.7:  # 70% 이상이 한자
                    pairs.append(CCKRPair(
                        classical_chinese=cc_text,
                        korean_translation=kr_text,
                        korean_explanation="",
                        source="sigwon",
                        metadata={"year": row.get('year'), "author": row.get('writer')}
                    ))
        
        return pairs
    
    def _extract_chinese_text(self, content: str) -> str:
        """텍스트에서 한문 추출"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # 한자 비율이 높은 라인 찾기
            if line and len(line) > 3:
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
                if chinese_chars / len(line) > 0.7:
                    return line
        return ""
    
    def _parse_assistant_response(self, content: str) -> Tuple[str, str]:
        """assistant 응답에서 번역과 해설 분리"""
        # <think> 태그 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        lines = content.split('\n')
        translation = ""
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('"') and not line.startswith("'"):
                if not translation:
                    translation = line
                else:
                    explanation += line + " "
        
        return translation.strip(), explanation.strip()
    
    def generate_cc_kr_nli(self, max_pairs: int = 100) -> List[Dict]:
        """CC-KR 기반 NLI 생성"""
        nli_data = []
        
        for pair in self.cc_kr_pairs[:max_pairs]:
            cc_text = pair.classical_chinese
            kr_text = pair.korean_translation
            
            if not cc_text or not kr_text:
                continue
            
            # 1. Entailment: CC → KR 번역 관계
            nli_data.append({
                "premise": f"고전한문: {cc_text}",
                "hypothesis": f"한국어 번역: {kr_text}",
                "label": "entailment",
                "metadata": {
                    "type": "translation_pair",
                    "source": pair.source,
                    "direction": "cc_to_kr"
                }
            })
            
            # 2. Contradiction: 잘못된 번역
            wrong_translation = self._generate_wrong_translation(kr_text)
            if wrong_translation:
                nli_data.append({
                    "premise": f"고전한문: {cc_text}",
                    "hypothesis": f"한국어 번역: {wrong_translation}",
                    "label": "contradiction",
                    "metadata": {
                        "type": "wrong_translation",
                        "source": pair.source,
                        "direction": "cc_to_kr"
                    }
                })
            
            # 3. Neutral: 관련 있지만 직접적 번역이 아닌 내용
            related_content = self._generate_related_content(cc_text, kr_text)
            if related_content:
                nli_data.append({
                    "premise": f"고전한문: {cc_text}",
                    "hypothesis": f"관련 내용: {related_content}",
                    "label": "neutral",
                    "metadata": {
                        "type": "related_content", 
                        "source": pair.source,
                        "direction": "cc_to_kr"
                    }
                })
        
        return nli_data
    
    def generate_cc_kr_sts(self, max_pairs: int = 100) -> List[Dict]:
        """CC-KR 기반 STS 생성"""
        sts_data = []
        
        # 1. 동일 원문의 다른 번역들 (고유사도)
        cc_groups = {}
        for pair in self.cc_kr_pairs:
            cc = pair.classical_chinese
            if cc not in cc_groups:
                cc_groups[cc] = []
            cc_groups[cc].append(pair)
        
        for cc_text, pairs in cc_groups.items():
            if len(pairs) >= 2:
                for i in range(len(pairs)):
                    for j in range(i + 1, len(pairs)):
                        sts_data.append({
                            "sentence1": pairs[i].korean_translation,
                            "sentence2": pairs[j].korean_translation,
                            "score": random.uniform(4.0, 5.0),  # 같은 원문이므로 고유사도
                            "metadata": {
                                "type": "same_source_different_translation",
                                "classical_source": cc_text
                            }
                        })
                        
                        if len(sts_data) >= max_pairs // 3:
                            break
                    if len(sts_data) >= max_pairs // 3:
                        break
        
        # 2. 비슷한 주제/내용의 CC-KR 쌍들 (중간유사도)
        similar_pairs = self._find_similar_content_pairs(max_pairs // 3)
        sts_data.extend(similar_pairs)
        
        # 3. 완전히 다른 내용 (저유사도)
        different_pairs = self._find_different_content_pairs(max_pairs // 3)
        sts_data.extend(different_pairs)
        
        return sts_data
    
    def _generate_wrong_translation(self, correct_translation: str) -> str:
        """잘못된 번역 생성"""
        wrong_patterns = [
            (r'이다', '이 아니다'),
            (r'한다', '하지 않는다'),
            (r'있다', '없다'),
            (r'좋다', '나쁘다'),
            (r'크다', '작다'),
            (r'중요하다', '중요하지 않다')
        ]
        
        result = correct_translation
        pattern, replacement = random.choice(wrong_patterns)
        if pattern in result:
            result = result.replace(pattern, replacement)
            return result
        
        # 단어 순서 바꾸기
        words = result.split()
        if len(words) >= 3:
            words[0], words[-1] = words[-1], words[0]
            return ' '.join(words)
        
        return ""
    
    def _generate_related_content(self, cc_text: str, kr_text: str) -> str:
        """관련 있는 내용 생성"""
        related_templates = [
            f"{kr_text}와 관련된 다른 교훈도 있다.",
            f"이와 유사한 맥락에서 다른 해석도 가능하다.",
            f"{kr_text}의 배경이 되는 역사적 상황이 있다.",
            f"이러한 가르침은 현대에도 적용될 수 있다.",
            f"{kr_text}와 연관된 다른 고전 문헌도 있다."
        ]
        
        return random.choice(related_templates)
    
    def _find_similar_content_pairs(self, max_pairs: int) -> List[Dict]:
        """유사한 내용의 쌍 찾기"""
        pairs = []
        
        # 공통 키워드 기반 유사성 검사
        for i, pair1 in enumerate(self.cc_kr_pairs):
            if len(pairs) >= max_pairs:
                break
                
            for pair2 in self.cc_kr_pairs[i+1:]:
                if len(pairs) >= max_pairs:
                    break
                    
                similarity = self._calculate_content_similarity(
                    pair1.korean_translation, 
                    pair2.korean_translation
                )
                
                if 0.2 < similarity < 0.8:  # 적당한 유사성
                    pairs.append({
                        "sentence1": pair1.korean_translation,
                        "sentence2": pair2.korean_translation,
                        "score": 2.0 + similarity * 2.0,  # 2.0~4.0 점수
                        "metadata": {
                            "type": "similar_content",
                            "similarity": similarity
                        }
                    })
        
        return pairs
    
    def _find_different_content_pairs(self, max_pairs: int) -> List[Dict]:
        """다른 내용의 쌍 찾기"""
        pairs = []
        random.shuffle(self.cc_kr_pairs)
        
        for i in range(0, len(self.cc_kr_pairs) - 1, 2):
            if len(pairs) >= max_pairs:
                break
                
            pair1 = self.cc_kr_pairs[i]
            pair2 = self.cc_kr_pairs[i + 1]
            
            similarity = self._calculate_content_similarity(
                pair1.korean_translation,
                pair2.korean_translation
            )
            
            if similarity < 0.3:  # 낮은 유사성
                pairs.append({
                    "sentence1": pair1.korean_translation,
                    "sentence2": pair2.korean_translation,
                    "score": similarity * 2.0,  # 0.0~0.6 점수
                    "metadata": {
                        "type": "different_content",
                        "similarity": similarity
                    }
                })
        
        return pairs
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """내용 유사도 계산 (간단한 키워드 기반)"""
        # 중요 키워드 추출
        keywords1 = set(re.findall(r'[가-힣]{2,}', text1))
        keywords2 = set(re.findall(r'[가-힣]{2,}', text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def generate_reverse_translation_nli(self, max_pairs: int = 50) -> List[Dict]:
        """역번역 기반 NLI (KR → CC 방향)"""
        nli_data = []
        
        for pair in self.cc_kr_pairs[:max_pairs]:
            kr_text = pair.korean_translation
            cc_text = pair.classical_chinese
            
            # 1. KR → simulated CC translation
            simulated_cc = self._simulate_kr_to_cc_translation(kr_text)
            if simulated_cc:
                # Entailment: 올바른 역번역
                nli_data.append({
                    "premise": f"한국어: {kr_text}",
                    "hypothesis": f"고전한문 번역: {simulated_cc}",
                    "label": "entailment",
                    "metadata": {
                        "type": "reverse_translation",
                        "direction": "kr_to_cc",
                        "original_cc": cc_text
                    }
                })
        
        return nli_data
    
    def _simulate_kr_to_cc_translation(self, kr_text: str) -> str:
        """한국어를 고전한문 스타일로 변환 시뮬레이션"""
        # 간단한 규칙 기반 변환 (실제로는 더 정교한 모델 필요)
        kr_to_cc_mapping = {
            "그러므로": "故",
            "따라서": "是以", 
            "그러나": "然",
            "만약": "若",
            "또한": "且",
            "인": "仁",
            "의": "義", 
            "예": "禮",
            "배움": "學",
            "도": "道"
        }
        
        result = kr_text
        for kr_word, cc_word in kr_to_cc_mapping.items():
            if kr_word in result:
                result = result.replace(kr_word, cc_word)
        
        # 문장 끝을 고전한문 스타일로
        if result.endswith('다'):
            result = result[:-1] + '也'
        elif result.endswith('이다'):
            result = result[:-2] + '也'
        
        return result if result != kr_text else ""
    
    def save_datasets(self, output_dir: str = "./"):
        """생성된 데이터셋들 저장"""
        # NLI 데이터셋
        nli_data = self.generate_cc_kr_nli(200)
        nli_data.extend(self.generate_reverse_translation_nli(50))
        
        with open(f"{output_dir}/cc_kr_nli.jsonl", 'w', encoding='utf-8') as f:
            for item in nli_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # STS 데이터셋
        sts_data = self.generate_cc_kr_sts(150)
        
        with open(f"{output_dir}/cc_kr_sts.jsonl", 'w', encoding='utf-8') as f:
            for item in sts_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f" CC-KR NLI 데이터셋 저장: {len(nli_data)}개")
        print(f" CC-KR STS 데이터셋 저장: {len(sts_data)}개")