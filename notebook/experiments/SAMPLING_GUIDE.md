# KLSBench 샘플링 평가 가이드

## 개요

이 가이드는 KLSBench 벤치마크를 다양한 샘플링 비율로 평가하는 방법을 설명합니다.

비용과 시간을 고려하여 전체 데이터셋(7,871개)의 일부를 샘플링하여 평가할 수 있습니다.

---

## 샘플링 모드

### 1. 테스트 모드 (test)
각 태스크당 **10개 샘플**로 빠른 테스트

```bash
./run_all_evaluations.sh test
```

**용도**:
- 코드 동작 확인
- API 연결 테스트
- 빠른 프로토타입

**예상 시간**: 5-10분
**예상 비용**: $0.40 (GPT-4 Turbo 기준)

---

### 2. 샘플링 모드 (sample)
전체 데이터셋의 **일정 비율**을 랜덤 샘플링

```bash
./run_all_evaluations.sh sample [RATIO]
```

**비율 예시**:
- `0.1` (10%): 약 787개 항목
- `0.3` (30%): 약 2,361개 항목 **권장**
- `0.5` (50%): 약 3,936개 항목

#### 권장 비율: 30% (0.3)

**이유**:
1. **통계적 유의성**: 최소 100개 이상 샘플로 의미있는 결과
2. **비용 효율성**: 전체의 1/3 비용으로 평가 가능
3. **신뢰도**: 충분한 샘플 수로 모델 성능 추정 가능

**30% 샘플링 상세**:
| 태스크 | 전체 | 30% 샘플 |
|:---|---:|---:|
| Classification | 808 | 242 |
| Retrieval | 1,209 | 363 |
| Punctuation | 2,000 | 600 |
| NLI | 1,854 | 556 |
| Translation | 2,000 | 600 |
| **총계** | **7,871** | **2,361** |

**예상 시간**: 1-2시간 (API 모델)
**예상 비용** (30% 기준):
- GPT-4 Turbo: **$6-7** (전체 $19의 30%)
- GPT-3.5 Turbo: **$0.6-1** (전체 $2-3의 30%)
- Claude 3.5 Sonnet: **$3-5** (전체 $11-15의 30%)
- Claude 3 Opus: **$8-11** (전체 $28-35의 30%)

---

### 3. 전체 평가 모드 (full)
전체 데이터셋 **7,871개 항목** 평가

```bash
./run_all_evaluations.sh full
```

**용도**: 최종 논문 결과, 공식 벤치마크
**예상 시간**: 3-5시간 (API 모델)
**예상 비용**: $19-35 (모델별 상이)

---

## 사용 예시

### 예시 1: 30% 샘플링으로 GPT-4 Turbo 평가

```bash
# 1. 샘플링 모드로 실행 (30%)
./run_all_evaluations.sh sample 0.3

# 또는 단일 모델만 평가
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --sample-ratio 0.3
```

**결과**:
- 총 샘플: 2,361개 (30%)
- 소요 시간: 약 1시간
- 비용: 약 $6-7

### 예시 2: 50% 샘플링으로 Claude 3.5 Sonnet 평가

```bash
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name claude-3-5-sonnet-20241022 \
    --api-key $ANTHROPIC_API_KEY \
    --sample-ratio 0.5
```

**결과**:
- 총 샘플: 3,936개 (50%)
- 소요 시간: 약 2시간
- 비용: 약 $6-8

### 예시 3: 여러 비율로 실험

```bash
# 10%, 30%, 50%, 100% 순차 실행
for ratio in 0.1 0.3 0.5 1.0; do
    echo "===== 샘플링 비율: $ratio ====="
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name gpt-4-turbo \
        --api-key $OPENAI_API_KEY \
        --sample-ratio $ratio
    echo ""
done
```

---

## 샘플링 비율별 비교

| 비율 | 샘플 수 | 예상 시간 | 예상 비용 (GPT-4) | 통계적 신뢰도 | 권장 용도 |
|:---:|---:|---:|---:|:---:|:---|
| **10%** | 787 | 20분 | $2 | WARNING: 보통 | 빠른 실험 |
| **30%** | 2,361 | 1시간 | $6-7 | 높음 | **권장** |
| **50%** | 3,936 | 2시간 | $10-12 | 매우 높음 | 정밀 평가 |
| **100%** | 7,871 | 3-5시간 | $19-20 | 완전 | 최종 논문 |

---

## 통계적 고려사항

### 샘플 크기 최소 요구사항

각 태스크별로 최소한의 샘플 수가 필요합니다:

| 태스크 | 전체 | 최소 권장 | 10% | 30% | 50% |
|:---|---:|---:|---:|---:|---:|
| Classification | 808 | 100 | 81 | 242 | 404 |
| Retrieval | 1,209 | 100 | 121 | 363 | 605 |
| Punctuation | 2,000 | 150 | 200 | 600 | 1,000 |
| NLI | 1,854 | 100 | 185 | 556 | 927 |
| Translation | 2,000 | 150 | 200 | 600 | 1,000 |

✅ = 통계적으로 충분한 샘플 수

### 신뢰 구간

샘플 크기에 따른 95% 신뢰구간:

- **10% (787개)**: ±3.5% 오차
- **30% (2,361개)**: ±2.0% 오차
- **50% (3,936개)**: ±1.6% 오차
- **100% (7,871개)**: ±1.1% 오차 (기준)

**결론**: 30% 샘플링으로도 ±2% 이내의 신뢰할 수 있는 결과를 얻을 수 있습니다.

---

## 샘플링 방법

### 랜덤 샘플링 (Random Sampling)

현재 구현된 방법입니다:

```python
# 재현성을 위해 seed 고정
np.random.seed(42)
indices = np.random.choice(original_size, sample_size, replace=False)
sampled_data = [data[i] for i in sorted(indices)]
```

**특징**:
- 편향 없는 무작위 선택
- 재현 가능 (seed=42)
- 모든 항목이 선택될 확률 동일

---

## 결과 분석

### 결과 파일

샘플링 비율에 상관없이 동일한 형식으로 저장됩니다:

```
benchmark/results/
├── results_gpt-4-turbo_20251029_143210.json
└── summary_gpt-4-turbo_20251029_143210.csv
```

### 결과 해석

**주의사항**:
1. 샘플 수가 적을수록 변동성이 큽니다
2. 0% 결과는 샘플 수 부족으로 발생할 수 있습니다
3. 30% 이상 샘플링을 권장합니다

**비교 방법**:
- 같은 비율로 여러 모델 평가
- 여러 비율로 같은 모델 평가 (일관성 확인)
- 전체 평가와 샘플링 평가 비교

---

## 비용 최적화 전략

### 전략 1: 단계적 평가

```bash
# 1단계: 10% 샘플로 빠른 스크리닝
./run_all_evaluations.sh sample 0.1

# 2단계: 유망한 모델만 30% 평가
python exp5_benchmark_evaluation.py \
    --model-name gpt-4-turbo \
    --sample-ratio 0.3

# 3단계: 최종 후보만 100% 평가
python exp5_benchmark_evaluation.py \
    --model-name gpt-4-turbo \
    --sample-ratio 1.0
```

### 전략 2: 저비용 모델 우선

```bash
# GPT-3.5로 먼저 전체 평가 ($2-3)
python exp5_benchmark_evaluation.py \
    --model-name gpt-3.5-turbo \
    --sample-ratio 1.0

# GPT-4는 30% 샘플로 ($6-7)
python exp5_benchmark_evaluation.py \
    --model-name gpt-4-turbo \
    --sample-ratio 0.3
```

---

## 빠른 시작

### 권장 워크플로우

```bash
# 1. API 키 설정
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'

# 2. 테스트 모드로 연결 확인 (무료에 가까움)
./run_all_evaluations.sh test

# 3. 30% 샘플링으로 본 평가 ($6-7)
./run_all_evaluations.sh sample 0.3

# 4. 결과 확인
ls -lh ../../benchmark/results/
```

**예상 총 비용**: $6-8
**예상 총 시간**: 1-2시간
**신뢰도**: 95% 신뢰구간 ±2%

---

## 문의

샘플링 관련 질문이나 이슈가 있으면 GitHub Issues에 등록해주세요.

---

**작성일**: 2025-10-29
**버전**: 1.0
**상태**: 완료
