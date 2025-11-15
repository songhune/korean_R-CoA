# Classification Confusion Matrix 생성 가이드

과문육체 (賦, 詩, 疑, 義, 策, 表) 분류 태스크에 대한 Confusion Matrix를 생성합니다.

## 📁 파일 구조

```
exp5/
├── generate_classification_confusion_matrix.py  # Confusion matrix 생성 스크립트
├── save_full_predictions.py                     # 전체 predictions 저장 스크립트
├── run_confusion_matrix_generation.sh           # 통합 실행 스크립트
└── README_confusion_matrix.md                   # 이 파일
```

## 🚀 사용 방법

### 방법 1: 자동 실행 (권장)

```bash
cd /home/work/songhune/korean_R-CoA/experiments/exp5
./run_confusion_matrix_generation.sh
```

스크립트 실행 후 선택 옵션:
- **Option 1**: 기존 결과 파일 사용 (빠름, 처음 10개 샘플만)
- **Option 2**: 전체 predictions 새로 생성 (느림, 전체 샘플, API 키 필요)
- **Option 3**: 기존 full_predictions 파일 사용 (빠름, 전체 샘플)

### 방법 2: 수동 실행

#### Step 1: 전체 Predictions 생성 (필요시)

**API 모델 (GPT-4, Claude 등):**

```bash
# GPT-4 Turbo
python save_full_predictions.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --output-dir ../../results/full_predictions \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --temperature 0.0

# Claude 3.5 Sonnet
python save_full_predictions.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --output-dir ../../results/full_predictions \
    --model-type api \
    --model-name claude-3-5-sonnet-20241022 \
    --api-key $ANTHROPIC_API_KEY \
    --temperature 0.0
```

**오픈소스 모델 (Llama, Qwen, EXAONE 등):**

```bash
# Llama 3.1 8B Instruct
python save_full_predictions.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --output-dir ../../results/full_predictions \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.0

# Qwen 2.5 7B Instruct
python save_full_predictions.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --output-dir ../../results/full_predictions \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0

# EXAONE 3.0 7.8B Instruct
python save_full_predictions.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --output-dir ../../results/full_predictions \
    --model-type opensource \
    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
    --temperature 0.0
```

#### Step 2: Confusion Matrix 생성

**전체 predictions 사용:**

```bash
python generate_classification_confusion_matrix.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --results-dir ../../results/full_predictions \
    --output-dir ../../results/confusion_matrices \
    --temperature 0.0 \
    --use-full-predictions \
    --full-predictions-dir ../../results/full_predictions
```

**기존 결과 파일 사용 (처음 10개만):**

```bash
python generate_classification_confusion_matrix.py \
    --benchmark ../../benchmark/kls_bench_full.json \
    --results-dir ../../results/temperature_ablation \
    --output-dir ../../results/confusion_matrices \
    --temperature 0.0
```

## 📊 출력 결과

### 생성되는 파일들

```
results/confusion_matrices/
├── confusion_matrix_gpt-4-turbo.png                    # Confusion matrix 이미지
├── confusion_matrix_gpt-4-turbo_report.txt            # Classification report
├── confusion_matrix_gpt-3.5-turbo.png
├── confusion_matrix_gpt-3.5-turbo_report.txt
├── confusion_matrix_claude-3-5-sonnet.png
├── confusion_matrix_claude-3-5-sonnet_report.txt
├── confusion_matrix_claude-3-opus.png
├── confusion_matrix_claude-3-opus_report.txt
├── confusion_matrix_meta-llama_Llama-3.1-8B-Instruct.png
├── confusion_matrix_meta-llama_Llama-3.1-8B-Instruct_report.txt
├── confusion_matrix_Qwen_Qwen2.5-7B-Instruct.png
├── confusion_matrix_Qwen_Qwen2.5-7B-Instruct_report.txt
├── confusion_matrix_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct.png
├── confusion_matrix_LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct_report.txt
└── comparison_report.txt                               # 모델 간 비교 리포트
```

### Confusion Matrix 이미지

각 모델에 대해 2개의 confusion matrix가 생성됩니다:
- **좌측**: 절대값 (Count) - 각 셀의 샘플 개수
- **우측**: 정규화 (Proportion) - 각 행의 합이 1이 되도록 정규화

### Classification Report

각 레이블(賦, 詩, 疑, 義, 策, 表)에 대한:
- Precision
- Recall
- F1-Score
- Support (샘플 개수)

### Comparison Report

모든 모델의 per-class accuracy를 비교하는 테이블

## ⚙️ 설정

### API 키 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

또는 환경 변수로 설정:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

### Temperature 설정

기본값은 0.0입니다. 다른 temperature 값을 사용하려면:

```bash
python generate_classification_confusion_matrix.py \
    --temperature 0.3 \
    ...
```

## 📝 주의사항

1. **API 모델 사용 시**:
   - API 키가 필요합니다
   - Rate limiting으로 인해 시간이 오래 걸릴 수 있습니다 (각 요청마다 1초 대기)
   - 전체 808개 샘플 처리에 약 13-15분 소요

2. **오픈소스 모델 사용 시**:
   - GPU가 필요합니다 (CUDA)
   - 모델 다운로드에 시간이 걸릴 수 있습니다
   - 충분한 VRAM이 필요합니다 (~16GB)

3. **기존 결과 파일 사용 시**:
   - 처음 10개 샘플만 사용되므로 제한적입니다
   - 전체 confusion matrix를 위해서는 full_predictions 사용 권장

## 🐛 문제 해결

### ImportError: config_loader

```python
[WARNING] config_loader not available, using default paths
```

이 경고는 무시해도 됩니다. 기본 경로가 사용됩니다.

### API Key 오류

```
[ERROR] OpenAI API Error: Authentication failed
```

`.env` 파일 또는 환경 변수에 올바른 API 키가 설정되어 있는지 확인하세요.

### GPU 메모리 부족

```
RuntimeError: CUDA out of memory
```

- 더 작은 배치 크기 사용
- 다른 프로세스 종료
- 더 작은 모델 사용

## 📚 참고

- 과문육체 레이블: 賦(부), 詩(시), 疑(의), 義(의), 策(책), 表(표)
- 전체 classification 샘플: 808개
- 각 레이블별 샘플 개수: 약 95개 (11.7-11.8%)

## 🔗 관련 파일

- `exp5_benchmark_evaluation.py`: 기본 벤치마크 평가 스크립트
- `analyze_temperature_ablation.py`: Temperature ablation 분석 스크립트
- `run_temperature_ablation.sh`: Temperature ablation 실행 스크립트
