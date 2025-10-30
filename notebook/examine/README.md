# KLSBench 전문가 평가 자료

이 폴더는 고려대학교 한문학과 연구원을 대상으로 한 KLSBench 벤치마크 전문가 평가를 위한 자료를 포함합니다.

## 폴더 구조

```
notebook/examine/
├── README.md                      # 이 파일
├── 전문가평가_설문지.md             # 상세 설문지 (참고용)
├── 평가_가이드.md                  # 평가자를 위한 가이드 문서
├── 평가_템플릿_생성.py              # Excel 템플릿 생성 스크립트
└── [생성될 파일]
    ├── expert_evaluation_sample.json   # 샘플링된 평가 데이터
    └── 전문가평가_템플릿.xlsx           # 실제 평가용 Excel 파일
```

## 사용 방법

### 1. 평가 데이터 샘플링

벤치마크에서 10% 샘플을 추출합니다 (총 780개 항목):

```bash
cd notebook/examine

# 방법 1: kls_bench_generator 사용
python3 ../experiments/utils/kls_bench_generator.py \
    --sample 0.1 \
    --output expert_evaluation_sample.json

# 방법 2: Python 코드로 직접 샘플링
python3 -c "
import json
import random

with open('../../benchmark/kls_bench/kls_bench_full.json') as f:
    data = json.load(f)

sampled = {}
for task, items in data.items():
    sample_size = int(len(items) * 0.1)
    sampled[task] = random.sample(items, sample_size)

with open('expert_evaluation_sample.json', 'w', encoding='utf-8') as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f'샘플링 완료: {sum(len(v) for v in sampled.values())}개 항목')
"
```

### 2. Excel 평가 템플릿 생성

샘플링된 데이터를 Excel 형식으로 변환합니다:

```bash
python3 평가_템플릿_생성.py \
    --input expert_evaluation_sample.json \
    --output 전문가평가_템플릿.xlsx
```

**필요한 패키지**:
```bash
pip3 install pandas openpyxl
```

### 3. 평가자에게 전달

다음 파일들을 평가자에게 전달합니다:

1. **필수**:
   - `전문가평가_템플릿.xlsx` - 실제 평가 파일
   - `평가_가이드.md` - 평가 방법 안내

2. **선택** (참고용):
   - `전문가평가_설문지.md` - 상세 설문 항목

### 4. 평가 결과 수집

평가자로부터 작성 완료된 Excel 파일을 받습니다.

## 평가 개요

### 평가 대상
- **전체 데이터**: 7,871개 항목
- **샘플**: 780개 항목 (10%)
- **5개 태스크**:
  - Classification: 80개
  - Retrieval: 120개
  - Punctuation: 200개
  - NLI: 180개
  - Translation: 200개

### 평가 항목
1. **정확성**: 라벨/정답이 올바른가?
2. **난이도**: 1-5점 척도
3. **개선 제안**: 오류 수정 의견
4. **전반적 품질**: 태스크별 종합 평가

### 소요 시간 및 보상
- **예상 시간**: 2-3시간
- **사례금**: 10만원

## Excel 템플릿 구조

생성되는 Excel 파일은 다음과 같은 시트로 구성됩니다:

### 1. Classification 시트
| 번호 | ID | 원문 | 현재_라벨 | 정확성 | 올바른_라벨 | 난이도(1-5) | 의견 |
|-----|----|----|---------|-------|----------|-----------|-----|

### 2. Retrieval 시트
| 번호 | ID | Query | Document | 관련성 | 관련없는_이유 | 난이도(1-5) | 의견 |
|-----|----|----|---------|-------|----------|-----------|-----|

### 3. Punctuation 시트
| 번호 | ID | 원문(구두점_없음) | 정답(구두점_있음) | 정확성 | 수정_제안 | 오류_유형 | 난이도(1-5) | 의견 |
|-----|----|----|---------|-------|----------|---------|-----------|-----|

### 4. NLI 시트
| 번호 | ID | Premise | Hypothesis | 현재_라벨 | 정확성 | 올바른_라벨 | 판단_근거 | 난이도(1-5) | 의견 |
|-----|----|----|---------|-------|----------|---------|-----------|-----|----|

### 5. Translation 시트
| 번호 | ID | 한문_원문 | 제시된_번역 | 정확성 | 수정_제안 | 오역_유형 | 자연스러움 | 난이도(1-5) | 의견 |
|-----|----|----|---------|-------|----------|---------|-----------|-----|----|

### 6. 전반적평가 시트
| 태스크 | 전반적_품질 | 개선_필요_사항 |
|-------|----------|-------------|

## 평가 결과 분석

평가 완료 후 다음과 같이 분석합니다:

### 1. 정량 분석
```python
import pandas as pd

# Excel 읽기
excel_file = '전문가평가_템플릿_완성.xlsx'
df_class = pd.read_excel(excel_file, sheet_name='Classification')

# 정확도 계산
accuracy = (df_class['정확성'] == '정확함').sum() / len(df_class)
print(f'Classification 정확도: {accuracy*100:.2f}%')

# 난이도 평균
avg_difficulty = df_class['난이도(1-5)'].mean()
print(f'평균 난이도: {avg_difficulty:.2f}')
```

### 2. 정성 분석
- 오류 패턴 분석
- 개선 제안 취합
- 태스크별 피드백 정리

### 3. 벤치마크 수정
- 부정확한 항목 수정
- 논란 있는 항목 재검토
- 품질 개선 보고서 작성

## 참고 문서

### 관련 파일
- `../experiments/전문가평가계획.md` - 전체 평가 계획
- `../../benchmark/kls_bench/README.md` - 벤치마크 설명
- `../experiments/README.md` - 실험 전체 구조

### 데이터 위치
```
benchmark/kls_bench/
├── kls_bench_full.json              (전체 7,871개)
├── kls_bench_classification.json    (808개)
├── kls_bench_retrieval.json         (1,209개)
├── kls_bench_punctuation.json       (2,000개)
├── kls_bench_nli.json               (1,854개)
└── kls_bench_translation.json       (2,000개)
```

## 문의

평가 관련 문의사항:
- 이메일: [담당자 이메일]
- 연구실: 고려대학교 [연구실명]

---

**평가에 참여해 주셔서 감사합니다!**
