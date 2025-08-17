# NLI/STS 데이터 생성 도구

이 프로젝트는 조선시대 과거시험 데이터와 사서(四書) 데이터를 활용하여 자연언어추론(NLI)과 의미유사도(STS) 데이터셋을 생성하는 도구입니다.

## 📁 파일 구조

```
network/
├── kr_nli_generator.py      # 한국어 NLI 데이터 생성기
├── kr_sts_generator.py      # 한국어 STS 데이터 생성기  
├── cc_kr_processor.py       # CC-KR 데이터 처리기
├── run_nli_sts_generation.py # 통합 실행 스크립트
└── README.md               # 이 파일
```

## 🚀 사용법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install pandas numpy scikit-learn sentence-transformers torch transformers
```

### 2. 데이터 생성 실행

```bash
# 모든 데이터 생성 (권장)
python run_nli_sts_generation.py --mode all

# 한국어 NLI/STS만 생성
python run_nli_sts_generation.py --mode kr_only

# CC-KR 기반 데이터만 생성  
python run_nli_sts_generation.py --mode cc_kr_only

# 사용자 지정 경로
python run_nli_sts_generation.py --data_dir ../data --output_dir ./my_output
```

### 3. 결과 확인

생성된 파일들:
- `korean_nli.jsonl`: 한국어 NLI 데이터셋
- `korean_sts.jsonl`: 한국어 STS 데이터셋  
- `cc_kr_nli.jsonl`: CC-KR 기반 NLI 데이터셋
- `cc_kr_sts.jsonl`: CC-KR 기반 STS 데이터셋
- `dataset_summary.json`: 데이터셋 요약 정보
- `requirements_nli_sts.txt`: 필요 패키지 목록

## 📊 데이터셋 형식

### NLI 데이터 형식 (JSONL)
```json
{
  "premise": "전제 문장",
  "hypothesis": "가설 문장", 
  "label": "entailment|contradiction|neutral",
  "metadata": {"type": "생성방식", "source": "데이터소스"}
}
```

### STS 데이터 형식 (JSONL)
```json
{
  "sentence1": "첫 번째 문장",
  "sentence2": "두 번째 문장",
  "score": 3.5,
  "metadata": {"similarity_level": "medium", "transform_type": "paraphrase"}
}
```

## 🔧 알고리즘 개요

이 도구는 원래 제안된 **K→CC 역번역 알고리즘**의 **Plan A (점진적 구현)** 버전입니다.

### 구현된 기능
1. ✅ **구절 단위 정렬**: 기존 데이터 구조 활용
2. ✅ **KR NLI 후보쌍 생성**: 템플릿 기반 E/C/N 생성
3. ✅ **자동 필터링**: 유사도 기반 중복 제거
4. ✅ **CC-KR 데이터 활용**: 기존 병렬 데이터 최대 활용
5. ✅ **STS 보강**: 다양한 유사도 레벨의 쌍 생성

### 미구현 (향후 작업)
- ❌ **역번역 KR→CC**: 고품질 번역 모델 부재
- ❌ **CC 전용 검증**: CC NLI 모델 부재

## 📈 생성 데이터 특징

### NLI 데이터
- **Entailment**: 원문-요약, 원문-인과결론 관계
- **Contradiction**: 부정 변환, 반대 개념 
- **Neutral**: 관련 정보, 추측성 내용

### STS 데이터  
- **고유사도 (4.0-5.0)**: 패러프레이즈, 동의어 변환
- **중간유사도 (2.0-3.9)**: 유사 주제, 관점 변경
- **저유사도 (0.0-1.9)**: 무관한 내용, 부정 변환

## 🎯 활용 방안

1. **모델 학습**: 한국어/고전한문 NLI/STS 모델 훈련
2. **평가 벤치마크**: 고전 문헌 이해 능력 평가
3. **지식 그래프**: 고전-현대 개념 연결
4. **교육 도구**: 고전 문헌 학습 보조

## ⚠️ 주의사항

1. **데이터 품질**: 템플릿 기반 생성으로 일정한 패턴 존재
2. **도메인 특화**: 고전 문헌/과거시험 도메인에 특화됨
3. **스케일 한계**: 현재는 중소 규모 데이터셋 (수백~수천 개)
4. **검증 필요**: 생성된 데이터의 전문가 검토 권장

## 🔮 향후 개선 방향

1. **GPT/Claude API 연동**: 더 자연스러운 텍스트 생성
2. **전문가 검증**: 고전문학 전문가와 협업
3. **CC 번역 모델**: KR→CC 번역 모델 개발/도입
4. **다국어 확장**: 중국어, 일본어 등으로 확장

## 📞 문의

문제 발생 시 이슈를 등록하거나 프로젝트 관리자에게 연락하세요.