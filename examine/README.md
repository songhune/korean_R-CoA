# KLSBench 전문가 평가 자료

이 폴더는 고려대학교 한문학과 연구원을 대상으로 한 KLSBench 벤치마크 전문가 평가를 위한 자료를 포함합니다.

## 폴더 구조

```
notebook/examine/
├── README.md                      # 이 파일
├── 전문가평가_설문지.md             # 상세 설문지 (참고용)
├── 평가_가이드.md                  # 평가자를 위한 가이드 문서
├── sample_benchmark_data.py       # 벤치마크 데이터 랜덤 샘플링 스크립트
└── [생성될 파일]
    ├── expert_evaluation_sample.json   # 샘플링된 평가 데이터
    └── 전문가평가_템플릿.xlsx           # 실제 평가용 Excel 파일
```

## 사용 방법

### 1. 평가 데이터 샘플링

벤치마크 데이터에서 랜덤 샘플을 추출합니다. 재현성을 위해 랜덤 시드를 사용합니다.

```bash
cd notebook/examine

# JSON 형식 (프로그래밍 용도)
python3 sample_benchmark_data.py --seed 42 --samples 10 --output expert_evaluation_sample.json

# Markdown 형식 (전문가 평가용 - 추천!)
python3 sample_benchmark_data.py --seed 42 --samples 10 --format markdown --output expert_evaluation_sample.md

# 태스크별로 다른 개수 샘플링
python3 sample_benchmark_data.py --seed 42 --samples-per-task "classification=80,retrieval=120,punctuation=200,nli=180,translation=200" --format markdown --output evaluation.md

# 다른 시드로 샘플링 (다른 랜덤 샘플)
python3 sample_benchmark_data.py --seed 100 --samples 20 --format markdown
```

**주요 파라미터**:
- `--seed`: 랜덤 시드 (재현성을 위해, 기본값: 42)
- `--samples`: 모든 태스크에 동일 개수
- `--samples-per-task`: 태스크별 개수 지정
- `--format`: 출력 형식 (json 또는 markdown/md, 기본값: json)
- `--output`: 출력 파일명 (기본값: expert_evaluation_sample.json)

**재현성**: 동일한 시드를 사용하면 항상 동일한 샘플을 얻을 수 있습니다.

**형식 선택**:
- **JSON**: 프로그래밍으로 분석할 때 사용
- **Markdown**: 비전공자 평가자에게 전달할 때 사용 (체크박스, 입력란 포함)

### 2. 샘플 데이터 확인

**JSON 형식** 예시:
```json
{
  "metadata": {
    "seed": 42,
    "total_samples": 780,
    "tasks": {
      "classification": 80,
      "retrieval": 120,
      ...
    }
  },
  "data": {
    "classification": [...],
    "retrieval": [...],
    ...
  }
}
```

**Markdown 형식** 예시:
```markdown
# KLSBench 전문가 평가 샘플 데이터

- **랜덤 시드**: 42
- **총 샘플 수**: 10개

## CLASSIFICATION

### 1. cls_0655

**원문**: 徐偃王行仁義

**라벨**: 論

**평가**:
- [ ] 정확함
- [ ] 부정확함

**올바른 라벨** (부정확한 경우):

**난이도** (1-5):

**의견**:

---
```

Markdown 형식은 평가자가 직접 체크박스를 선택하고 의견을 작성할 수 있습니다.

### 3. 평가자에게 전달

다음 파일들을 평가자에게 전달합니다:

1. **필수**:
   - `expert_evaluation_sample.md` - 샘플링된 평가 데이터 (Markdown 형식, **추천**)
   - `평가_가이드.md` - 평가 방법 안내

2. **선택** (참고용):
   - `전문가평가_설문지.md` - 상세 설문 항목

**Markdown 형식의 장점**:
- GitHub, Notion, Obsidian 등에서 바로 편집 가능
- 체크박스(`- [ ]`)를 클릭하여 평가 가능
- 프로그래밍 지식 없이도 쉽게 작성 가능
- 버전 관리 시스템(Git)과 호환

### 4. 평가 결과 수집

평가자로부터 작성 완료된 평가 결과를 받습니다.

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

## 샘플 데이터 구조

생성된 JSON 파일은 각 태스크별로 다음 필드를 포함합니다:

### 1. Classification
- `id`: 항목 ID
- `input`: 한문 원문
- `label`: 문체 분류 (賦/詩/疑/義)
- `metadata`: 추가 메타데이터

### 2. Retrieval
- `id`: 항목 ID
- `input`: Query 텍스트
- `answer`: 출처 (책 - 챕터)
- `book`, `chapter`: 서지 정보
- `metadata`: 추가 메타데이터

### 3. Punctuation
- `id`: 항목 ID
- `input`: 구두점 없는 원문
- `answer`: 구두점이 있는 정답
- `source`: 출처
- `metadata`: 한글 번역 등 추가 정보

### 4. NLI
- `id`: 항목 ID
- `premise`: 전제 문장
- `hypothesis`: 가설 문장
- `label`: 관계 (entailment/neutral/contradiction)
- `category`: 관계 유형
- `explanation`: 설명

### 5. Translation
- `id`: 항목 ID
- `source_text`: 원문
- `target_text`: 번역문
- `source_lang`, `target_lang`: 언어 쌍
- `metadata`: 추가 메타데이터

## 평가 결과 분석

평가 완료 후 다음과 같이 분석합니다:

### 1. 정량 분석
```python
import json

# 평가 결과 읽기
with open('expert_evaluation_result.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

# 태스크별 통계 계산
for task_name, items in result['data'].items():
    total = len(items)
    correct = sum(1 for item in items if item.get('is_correct', True))
    accuracy = correct / total if total > 0 else 0

    print(f'{task_name}:')
    print(f'  - 정확도: {accuracy*100:.2f}% ({correct}/{total})')

    # 난이도 분석
    difficulties = [item.get('difficulty', 3) for item in items if 'difficulty' in item]
    if difficulties:
        avg_diff = sum(difficulties) / len(difficulties)
        print(f'  - 평균 난이도: {avg_diff:.2f}')
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
