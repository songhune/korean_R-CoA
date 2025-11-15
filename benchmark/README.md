# KLSBench: Korean Literary Style Benchmark

A Comprehensive Benchmark for Korean Classical Literature Understanding

## Overview

**KLSBench** is a benchmark developed based on C3Bench for evaluating Korean classical literature understanding.
It comprehensively assesses the capabilities of Large Language Models (LLMs) to understand Korean classical Chinese texts and Four Books (四書) data.

- **Version**: 1.0
- **Total Items**: 7,871
- **Tasks**: 5
- **Languages**: Classical Chinese, Korean, English

## Task Structure

| Task | Description | Items | Metrics |
|:---|:---|---:|:---|
| **classification** | Classify literary style (Fu/Shi/Yi/I, 賦/詩/疑/義) of classical texts | 808 | Accuracy |
| **retrieval** | Identify source (Book/Chapter) from Four Books (四書) | 1,209 | Accuracy |
| **punctuation** | Restore punctuation to unpunctuated classical Chinese (白文) | 2,000 | F1 Score |
| **nli** | Determine logical relationship (entailment/contradiction/neutral) | 1,854 | Accuracy |
| **translation** | Translate between Classical Chinese, Korean, and English | 2,000 | BLEU Score |

## Data Statistics

### 1. Classification (분류)

문체별 분포:
- **制**: 3개
- **擬**: 2개
- **易義**: 9개
- **書義**: 6개
- **疑**: 95개
- **禮義**: 7개
- **策**: 95개
- **箋**: 49개
- **箴**: 12개
- **義**: 95개
- **表**: 95개
- **詔**: 5개
- **詩**: 95개
- **詩義**: 7개
- **論**: 51개
- **講**: 2개
- **賦**: 95개
- **銘**: 53개
- **頌**: 24개
- **論**: 2개
- **禮義**: 6개

### 2. Retrieval (검색)

책별 분포:
- ** 論語**: 500개
- ** 孟子**: 500개
- **中庸**: 137개
- ** 大學**: 72개

### 3. Punctuation (구두점)

평균 문장 길이 및 통계는 데이터 로딩 후 확인 가능합니다.

### 4. NLI (자연언어추론)

레이블 분포:
- **contradiction**: 141개
- **entailment**: 1,313개
- **neutral**: 400개

### 5. Translation (번역)

언어 쌍 분포:
- **classical_chinese → korean**: 1,320개
- **korean → english**: 680개

##  사용 방법

### Python에서 로드

```python
import json

# 전체 벤치마크 로드
with open('k_classic_bench_full.json', 'r', encoding='utf-8') as f:
    benchmark = json.load(f)

# 특정 태스크만 로드
with open('k_classic_bench_classification.json', 'r', encoding='utf-8') as f:
    classification_task = json.load(f)

# 데이터 접근
for item in classification_task['data']:
    print(f"Input: {item['input']}")
    print(f"Label: {item['label']}")
```

### Pandas로 분석

```python
import pandas as pd

# CSV로 로드
df = pd.read_csv('k_classic_bench_classification.csv')
print(df.head())
print(df['label'].value_counts())
```

## 📁 파일 구조

```
k_classic_bench/
├── k_classic_bench_full.json          # 전체 벤치마크 (모든 태스크 포함)
├── k_classic_bench_classification.json # 분류 태스크
├── k_classic_bench_retrieval.json     # 검색 태스크
├── k_classic_bench_punctuation.json   # 구두점 태스크
├── k_classic_bench_nli.json           # NLI 태스크
├── k_classic_bench_translation.json   # 번역 태스크
├── k_classic_bench_classification.csv # 분류 태스크 (CSV)
├── k_classic_bench_retrieval.csv      # 검색 태스크 (CSV)
├── k_classic_bench_punctuation.csv    # 구두점 태스크 (CSV)
├── k_classic_bench_nli.csv            # NLI 태스크 (CSV)
├── k_classic_bench_translation.csv    # 번역 태스크 (CSV)
└── README.md                          # 본 문서
```

##  데이터 출처

1. **과거시험 데이터**: 한국 과거시험 문제 및 답안 (문체 분류 포함)
2. **사서(四書) 데이터**: 논어, 맹자, 대학, 중용 등 유교 경전
3. **NLI 예시**: 자연언어추론 템플릿 및 예시

## 📜 라이선스 및 인용

이 벤치마크를 연구에 사용하시는 경우 다음과 같이 인용해 주세요:

```bibtex
@misc{{k_classic_bench_2024,
  title={{K-ClassicBench: Korean Classical Literature Understanding Benchmark}},
  author={{Your Name}},
  year={{2024}},
  note={{Inspired by C3Bench}}
}}
```

## 🔗 참고 자료

- **C3Bench**: [논문 링크]
- **관련 연구**: 고전 한문 자연어처리 연구

##  문의

벤치마크 관련 문의사항은 이메일로 연락 주시기 바랍니다.

---

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
