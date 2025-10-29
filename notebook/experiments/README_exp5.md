# 실험 5: KLSBench 벤치마크 평가

C3Bench를 참고하여 개발한 KLSBench로 다양한 LLM을 평가하는 실험입니다.

## 목차

1. [개요](#개요)
2. [벤치마크 구성](#벤치마크-구성)
3. [설치 및 설정](#설치-및-설정)
4. [사용 방법](#사용-방법)
5. [평가 모델](#평가-모델)
6. [결과 분석](#결과-분석)

---

## 개요

**KLSBench**는 한국 고전 문헌 이해 능력을 평가하기 위한 벤치마크입니다.

- **총 항목 수**: 7,871개
- **태스크 수**: 5개
- **언어**: 한문, 한국어, 영문

---

## 벤치마크 구성

| 태스크 | 설명 | 항목 수 | 평가 지표 |
|:---|:---|---:|:---|
| **Classification** | 문체 분류 (賦/詩/疑/義 등) | 808 | Accuracy |
| **Retrieval** | 출처 식별 (論語/孟子/大學/中庸) | 1,209 | Accuracy |
| **Punctuation** | 구두점 복원 (백문 → 구두점본) | 2,000 | F1 Score |
| **NLI** | 자연언어추론 | 1,854 | Accuracy |
| **Translation** | 번역 (한문↔한글↔영문) | 2,000 | BLEU Score |

---

## 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
pandas
numpy
scikit-learn
rouge-score
nltk
tqdm
matplotlib
seaborn

# API 모델용
openai
anthropic

# 오픈소스 모델용
transformers
torch
accelerate
```

### 2. API 키 설정

API 모델을 사용하려면 환경 변수에 API 키를 설정하세요.

```bash
# OpenAI (GPT-4, GPT-3.5)
export OPENAI_API_KEY='your-openai-api-key'

# Anthropic (Claude)
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

---

## 사용 방법

### 방법 1: 커맨드라인 직접 실행

#### API 모델 평가

```bash
# GPT-4 평가 (테스트: 각 태스크당 10개)
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --max-samples 10

# Claude 3.5 Sonnet 평가 (전체)
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name claude-3-5-sonnet-20241022 \
    --api-key $ANTHROPIC_API_KEY
```

#### 오픈소스 모델 평가

```bash
# Llama 3.1 8B 평가
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --max-samples 10

# Qwen 2.5 7B 평가
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct
```

### 방법 2: 배치 스크립트 실행

모든 모델을 순차적으로 평가:

```bash
# 테스트 모드 (각 태스크당 10개 샘플)
./run_all_evaluations.sh test

# 전체 평가 모드
./run_all_evaluations.sh full
```

### 방법 3: Jupyter 노트북

```bash
jupyter notebook 5번실험.ipynb
```

---

## 평가 모델

### 1. 비공개 API 모델

- **GPT-4 Turbo** (`gpt-4-turbo`)
- **GPT-3.5 Turbo** (`gpt-3.5-turbo`)
- **Claude 3.5 Sonnet** (`claude-3-5-sonnet-20241022`)
- **Claude 3 Opus** (`claude-3-opus-20240229`)

### 2. 오픈소스 모델

- **Llama 3.1 8B** (`meta-llama/Llama-3.1-8B-Instruct`)
- **Llama 3.1 70B** (`meta-llama/Llama-3.1-70B-Instruct`)
- **Qwen 2.5 7B** (`Qwen/Qwen2.5-7B-Instruct`)
- **Qwen 2.5 14B** (`Qwen/Qwen2.5-14B-Instruct`)
- **Qwen 2.5 72B** (`Qwen/Qwen2.5-72B-Instruct`)
- **EXAONE 3.0 7.8B** (`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`)

### 3. 지도학습 모델

- **Tongu**: 한문 처리 특화 모델
- **GwenBert**: 고전 문헌 이해 모델

> **주의**: 지도학습 모델은 별도 구현이 필요합니다.

---

## 결과 분석

### 결과 파일

평가 완료 후 다음 파일이 생성됩니다:

```
../../benchmark/results/
├── results_{model_name}_{timestamp}.json  # 전체 결과 (예측값 포함)
└── summary_{model_name}_{timestamp}.csv   # 요약 (메트릭만)
```

### 결과 확인

#### 1. CSV 요약 파일

```python
import pandas as pd

df = pd.read_csv('../../benchmark/results/summary_gpt-4-turbo_20241021_120000.csv')
print(df)
```

| model | task | accuracy | f1 | bleu | ... |
|:---|:---|---:|---:|---:|:---|
| gpt-4-turbo | classification | 0.85 | 0.84 | - | ... |
| gpt-4-turbo | retrieval | 0.92 | - | - | ... |
| gpt-4-turbo | punctuation | - | - | - | ... |
| gpt-4-turbo | nli | 0.78 | 0.77 | - | ... |
| gpt-4-turbo | translation | - | - | 0.65 | ... |

#### 2. 상세 결과 (JSON)

```python
import json

with open('../../benchmark/results/results_gpt-4-turbo_20241021_120000.json', 'r') as f:
    results = json.load(f)

# 모델 정보
print(results['model_name'])
print(results['model_type'])

# 태스크별 결과
for task_name, task_results in results['tasks'].items():
    print(f"\n{task_name}:")
    print(f"  Metrics: {task_results['metrics']}")
    print(f"  Sample predictions: {task_results['predictions'][:3]}")
```

### 시각화

노트북(`5번실험.ipynb`)에서 다음 시각화를 제공합니다:

1. **태스크별 성능 비교**: 막대 그래프
2. **모델별 히트맵**: 태스크 × 모델 성능
3. **종합 점수 랭킹**: 평균 성능 순위

---

## 커맨드라인 옵션

```bash
python exp5_benchmark_evaluation.py [옵션]
```

### 필수 옵션

- `--model-name`: 모델 이름 (예: `gpt-4-turbo`, `meta-llama/Llama-3.1-8B-Instruct`)

### 선택 옵션

| 옵션 | 설명 | 기본값 |
|:---|:---|:---|
| `--benchmark` | 벤치마크 JSON 경로 | `../../benchmark/kls_bench/kls_bench_full.json` |
| `--output` | 결과 저장 디렉토리 | `../../benchmark/results` |
| `--model-type` | 모델 타입 (`api`/`opensource`/`supervised`) | `api` |
| `--api-key` | API 키 (API 모델용) | `None` |
| `--max-samples` | 태스크당 최대 샘플 수 (테스트용) | `None` (전체) |

---

## 문제 해결

### 1. CUDA 메모리 부족

오픈소스 모델 평가 시 GPU 메모리가 부족한 경우:

```python
# 모델 로딩 시 8bit quantization 사용
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. API Rate Limit

API 호출 제한에 걸린 경우:

- `exp5_benchmark_evaluation.py`의 `time.sleep(0.5)` 값을 늘리세요.
- 또는 `--max-samples` 옵션으로 샘플 수를 줄이세요.

### 3. Tokenizer 에러

Chat template이 없는 모델의 경우:

- 코드가 자동으로 fallback (단순 연결)을 사용합니다.
- 또는 수동으로 프롬프트 포맷을 지정하세요.

---

## 향후 계획

1. 기본 벤치마크 구축
2. 평가 프레임워크 개발
3. 🔄 Few-shot learning 지원
4. 더 많은 모델 추가
5. 도메인 특화 Fine-tuning

---

## 참고 자료

- **C3Bench**: [논문 링크]
- **벤치마크 상세**: `../../benchmark/kls_bench/README.md`

---

## 문의

벤치마크 관련 문의사항은 이메일로 연락 주시기 바랍니다.

---

**Generated**: 2024-10-21
