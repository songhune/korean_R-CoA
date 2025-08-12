import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ExamRecord:
    """통합된 시험 레코드"""
    question: str
    answer: str
    category: str
    year: int
    author: str
    grade: Optional[str] = None
    rank: Optional[str] = None

class KwasiLLMConverter:
    """과시 데이터를 LLM용 JSON으로 변환하는 클래스"""
    
    def __init__(self):
        self.gwashi_df = None
        self.munjib_df = None  
        self.sigwon_df = None
        
    def load_data(self, gwashi_path: str, munjib_path: str, sigwon_path: str):
        """CSV 파일들을 로드"""
        try:
            self.gwashi_df = pd.read_csv(gwashi_path, encoding='cp949')
            print(f"gwashi.csv 로드 완료: {len(self.gwashi_df)}행")
        except Exception as e:
            print(f"❌ gwashi.csv 로드 실패: {e}")
            self.gwashi_df = pd.DataFrame()
        
        try:
            self.munjib_df = pd.read_csv(munjib_path, encoding='utf-8')
            print(f"munjib.csv 로드 완료: {len(self.munjib_df)}행")
        except Exception as e:
            print(f"❌ munjib.csv 로드 실패: {e}")
            self.munjib_df = pd.DataFrame()
        
        try:
            self.sigwon_df = pd.read_csv(sigwon_path, encoding='utf-8')
            print(f"sigwon.csv 로드 완료: {len(self.sigwon_df)}행")
        except Exception as e:
            print(f"❌ sigwon.csv 로드 실패: {e}")
            self.sigwon_df = pd.DataFrame()
        
        total_loaded = len(self.gwashi_df) + len(self.munjib_df) + len(self.sigwon_df)
        if total_loaded == 0:
            raise ValueError("모든 파일 로드에 실패했습니다.")
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if pd.isna(text) or text is None:
            return ""
        return str(text).strip().replace('\n', ' ').replace('\r', '')
    
    def extract_literary_form(self, row) -> str:
        """문체 추출 (sortA, category 등에서)"""
        # sortA, category, category2 등에서 문체 정보 추출
        candidates = [row.get('sortA', ''), row.get('category', ''), row.get('category2', '')]
        for candidate in candidates:
            if candidate and len(self.clean_text(candidate)) > 0:
                return self.clean_text(candidate)
        return "미상"
    
    def match_records_by_year_and_type(self) -> List[ExamRecord]:
        """연도와 분류를 기준으로 레코드 매칭"""
        matched_records = []
        
        # 1. gwashi를 기준으로 문제 추출
        for _, gw_row in self.gwashi_df.iterrows():
            question = self.clean_text(gw_row.get('name_question', ''))
            contents = self.clean_text(gw_row.get('contents', ''))
            if not question and not contents:
                continue
                
            # 문제 텍스트 결합
            full_question = f"{question} {contents}".strip()
            
            year = gw_row.get('year', 0)
            category = self.extract_literary_form(gw_row)
            
            # 2. 동일 연도의 munjib 답안 찾기
            matching_answers = self.find_matching_answers(year, category, 'munjib')
            for answer_data in matching_answers:
                record = ExamRecord(
                    question=full_question,
                    answer=answer_data['answer'],
                    category=category,
                    year=int(year) if year else 0,
                    author=answer_data['author']
                )
                matched_records.append(record)
            
            # 3. 동일 연도의 sigwon 답안 찾기  
            matching_sigwons = self.find_matching_answers(year, category, 'sigwon')
            for sigwon_data in matching_sigwons:
                record = ExamRecord(
                    question=full_question,
                    answer=sigwon_data['answer'],
                    category=category,
                    year=int(year) if year else 0,
                    author=sigwon_data['author'],
                    grade=sigwon_data.get('grade'),
                    rank=sigwon_data.get('rank')
                )
                matched_records.append(record)
        
        return matched_records
    
    def find_matching_answers(self, year: float, category: str, source: str) -> List[Dict]:
        """특정 연도/분류에 해당하는 답안 찾기"""
        answers = []
        
        if source == 'munjib' and self.munjib_df is not None:
            # munjib에서 답안 찾기
            for _, row in self.munjib_df.iterrows():
                answer_year = row.get('answer_year', '')
                if self.match_year(year, answer_year):
                    answer = self.clean_text(row.get('answer_contents', ''))
                    author = self.clean_text(row.get('person_korname', '') or row.get('person_fullname', ''))
                    if answer:
                        answers.append({
                            'answer': answer,
                            'author': author or '미상'
                        })
        
        elif source == 'sigwon' and self.sigwon_df is not None:
            # sigwon에서 답안 찾기
            for _, row in self.sigwon_df.iterrows():
                sigwon_year = row.get('year', '')
                if self.match_year(year, sigwon_year):
                    answer = self.clean_text(row.get('original', ''))
                    author = self.clean_text(row.get('writer', ''))
                    grade = self.clean_text(row.get('grade', ''))
                    rank = self.clean_text(row.get('rank', ''))
                    if answer:
                        answers.append({
                            'answer': answer,
                            'author': author or '미상',
                            'grade': grade,
                            'rank': rank
                        })
        
        return answers
    
    def match_year(self, target_year: float, candidate_year: str) -> bool:
        """연도 매칭 (유연한 매칭)"""
        if pd.isna(target_year) or not candidate_year:
            return False
        
        try:
            target = int(target_year)
        except (ValueError, TypeError):
            return False
        
        # 문자열에서 4자리 연도 추출
        year_pattern = r'\b(\d{4})\b'
        matches = re.findall(year_pattern, str(candidate_year))
        
        for match in matches:
            try:
                if abs(int(match) - target) <= 1:  # ±1년 허용
                    return True
            except ValueError:
                continue
        return False
    
    def generate_task_specific_json(self, records: List[ExamRecord]) -> Dict:
        """태스크별 JSON 데이터 생성"""
        
        # 1. Classification 태스크 (문체 분류)
        classification_data = []
        for record in records:
            if record.category != "미상":
                item = {
                    "task": "classification",
                    "input": f"다음 과시 문제와 답안을 보고 문체를 판단하시오.\n문제: {record.question}\n답안: {record.answer[:100]}...",
                    "output": record.category,
                    "metadata": {
                        "year": record.year,
                        "author": record.author
                    }
                }
                classification_data.append(item)
        
        # 2. NLI 태스크 (문제-답안 관계)
        nli_data = []
        for record in records:
            # Entailment 사례 (정답)
            nli_item = {
                "task": "nli",
                "premise": f"{record.year}년 과시 문제: {record.question}",
                "hypothesis": f"답안: {record.answer[:200]}...",
                "label": "entailment",
                "metadata": {
                    "author": record.author,
                    "category": record.category
                }
            }
            nli_data.append(nli_item)
        
        # 3. STS 태스크 (답안 간 유사성)
        sts_data = []
        same_year_records = {}
        for record in records:
            year_key = f"{record.year}_{record.category}"
            if year_key not in same_year_records:
                same_year_records[year_key] = []
            same_year_records[year_key].append(record)
        
        # 동일 연도/분류의 답안들로 STS 생성
        for year_category, group_records in same_year_records.items():
            if len(group_records) >= 2:
                for i in range(min(3, len(group_records))):
                    for j in range(i+1, min(i+3, len(group_records))):
                        sts_item = {
                            "task": "sts",
                            "sentence1": f"{group_records[i].author}의 답안: {group_records[i].answer[:150]}...",
                            "sentence2": f"{group_records[j].author}의 답안: {group_records[j].answer[:150]}...",
                            "score": 4.0,  # 같은 문제에 대한 답안이므로 높은 유사성
                            "metadata": {
                                "year": group_records[i].year,
                                "category": group_records[i].category
                            }
                        }
                        sts_data.append(sts_item)
        
        # 4. 품질 평가 태스크 (성적 예측)
        quality_data = []
        for record in records:
            if record.grade:
                quality_item = {
                    "task": "quality_assessment", 
                    "input": f"다음 과시 답안의 품질을 평가하시오:\n{record.answer}",
                    "output": record.grade,
                    "metadata": {
                        "rank": record.rank,
                        "year": record.year,
                        "category": record.category
                    }
                }
                quality_data.append(quality_item)
        
        return {
            "dataset_info": {
                "total_records": len(records),
                "tasks": ["classification", "nli", "sts", "quality_assessment"],
                "source_files": ["gwashi.csv", "munjib.csv", "sigwon.csv"]
            },
            "classification": classification_data,
            "nli": nli_data, 
            "sts": sts_data,
            "quality_assessment": quality_data
        }
    
    def convert_to_llm_json(self, output_path: str = "kwasi_llm_dataset.json"):
        """전체 변환 프로세스 실행"""
        print("=== 과시 데이터 → LLM JSON 변환 시작 ===")
        
        # 1. 레코드 매칭
        print("1. 레코드 매칭 중...")
        matched_records = self.match_records_by_year_and_type()
        print(f"   매칭된 레코드: {len(matched_records)}개")
        
        # 2. 태스크별 JSON 생성
        print("2. 태스크별 JSON 생성 중...")
        llm_dataset = self.generate_task_specific_json(matched_records)
        
        # 3. 통계 출력
        print("3. 변환 결과:")
        for task, data in llm_dataset.items():
            if isinstance(data, list):
                print(f"   {task}: {len(data)}개")
        
        # 4. JSON 파일 저장
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(llm_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ 변환 완료: {output_path}")
        except Exception as e:
            print(f"❌ JSON 저장 실패: {e}")
            raise
        
        return llm_dataset

if __name__ == "__main__":
    converter = KwasiLLMConverter()
    
    # 데이터 로드
    converter.load_data(
        gwashi_path="./data/gwashi.csv",
        munjib_path="./data/munjib.csv", 
        sigwon_path="./data/sigwon.csv"
    )
    
    # LLM JSON 변환
    result = converter.convert_to_llm_json("kwasi_llm_dataset.json")
    
    # 샘플 출력
    print("\n=== 샘플 데이터 ===")
    if result['classification']:
        print("Classification 샘플:")
        print(json.dumps(result['classification'][0], ensure_ascii=False, indent=2))