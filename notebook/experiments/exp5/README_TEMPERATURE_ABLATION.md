# Temperature Ablation Study - 실행 가이드

## 개요

Temperature 파라미터가 모델 성능에 미치는 영향을 분석하는 실험입니다.

### 실험 설정
- **모델**: 7개 (GPT-4-turbo, GPT-3.5-turbo, Claude Sonnet 4.5, Claude Opus 3, Llama-3.1-8B, Qwen2.5-7B, EXAONE-3.0-7.8B)
- **Temperature**: 3개 (0.0, 0.3, 0.7)
- **샘플링**: 10% (787개 샘플)
- **총 실험**: 21회 (7 models × 3 temps)

### 예상 소요 시간
- **Mac (API 모델)**: 약 5.5시간
- **H100 GPU (오픈소스)**: 약 1시간
- **병렬 실행 시**: 약 5.5시간

---

## 📋 실행 절차

### Option 1: Mac에서만 실행 (API 모델만)

```bash
# 1. 환경 확인
cd /Users/songhune/Workspace/korean_eda/notebook/experiments/exp5
cat .env  # API 키 확인

# 2. HF_TOKEN 제거 (오픈소스 모델 건너뛰기)
export HF_TOKEN=""

# 3. 실험 실행
./run_temperature_ablation.sh sample

# 결과: GPT-4, GPT-3.5, Claude Sonnet 4.5, Claude Opus만 실행
```

---

### Option 2: H100 GPU에서만 실행 (오픈소스 모델만)

```bash
# 1. 환경 설정
cd /path/to/korean_eda/notebook/experiments/exp5
./setup_exp5_gpu.sh

# 2. HuggingFace 토큰 설정
export HF_TOKEN='hf_your_token_here'

# 3. API 키 제거 (API 모델 건너뛰기)
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

# 4. 실험 실행
./run_temperature_ablation.sh sample

# 결과: Llama-3.1-8B, Qwen2.5-7B, EXAONE-3.0-7.8B만 실행
```

---

### Option 3: 병렬 실행 (권장) ⚡

**Mac (Terminal 1):**
```bash
cd /Users/songhune/Workspace/korean_eda/notebook/experiments/exp5

# HF_TOKEN 제거
export HF_TOKEN=""

# API 모델만 실행
./run_temperature_ablation.sh sample
```

**H100 GPU (Terminal 2):**
```bash
cd /path/to/korean_eda/notebook/experiments/exp5

# 환경 설정 (최초 1회)
./setup_exp5_gpu.sh

# HuggingFace 토큰 설정
export HF_TOKEN='hf_your_token_here'

# API 키 제거
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

# 오픈소스 모델만 실행
./run_temperature_ablation.sh sample
```

---

## 🔍 주요 변경사항

### 1. Temperature 설정
- **변경 전**: 5개 (0.0, 0.1, 0.3, 0.5, 0.7)
- **변경 후**: 3개 (0.0, 0.3, 0.7)
- **이유**: GPT-4 Turbo 분석 결과 temperature 영향이 매우 작음 (변화량 < 0.4%)

### 2. 샘플링 비율
- **변경 전**: 30% (2,361개)
- **변경 후**: 10% (787개)
- **이유**: 시간/비용 절감 (실험 시간 88% 단축)

### 3. Claude 모델명 수정
- **변경 전**: `claude-3-5-sonnet-20241022` (존재하지 않음 → 404 에러)
- **변경 후**:
  - `claude-sonnet-4-5-20250929` (최신)
  - `claude-3-opus-20240229`

---

## 🚨 중요 체크리스트

### Mac 실행 전
- [ ] `.env` 파일에 `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` 존재
- [ ] `HF_TOKEN` 제거 또는 빈 문자열로 설정
- [ ] 기존 실험 결과 백업 (필요시)

### H100 GPU 실행 전
- [ ] CUDA 및 GPU 드라이버 설치 확인
- [ ] Python 3.8+ 환경
- [ ] `HF_TOKEN` 환경변수 설정 (Llama 액세스용)
- [ ] 디스크 공간 최소 50GB 확보 (모델 다운로드용)
- [ ] API 키 제거 (OPENAI_API_KEY, ANTHROPIC_API_KEY)

---

## 📦 모델 자동 다운로드

### HuggingFace 모델 (처음 실행 시)

오픈소스 모델은 처음 실행 시 자동으로 다운로드됩니다:

1. **Llama-3.1-8B** (`meta-llama/Llama-3.1-8B-Instruct`)
   - 크기: ~16GB
   - 액세스: HF_TOKEN 필요 (gated model)
   - 액세스 요청: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

2. **Qwen2.5-7B** (`Qwen/Qwen2.5-7B-Instruct`)
   - 크기: ~15GB
   - 액세스: Public (토큰 불필요)

3. **EXAONE-3.0-7.8B** (`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`)
   - 크기: ~16GB
   - 액세스: Public (토큰 불필요)

**총 다운로드**: ~47GB

### 다운로드 위치
- 기본: `~/.cache/huggingface/hub/`
- 변경: `export HF_HOME=/your/custom/path`

---

## 📊 실행 모니터링

### 진행 상황 확인
```bash
# 결과 파일 확인
ls -lh results/temperature_ablation/

# 실시간 로그
tail -f nohup.out  # background 실행 시
```

### 예상 출력
```
[LOAD] Benchmark: /path/to/kls_bench_full.json
[SAMPLING] Limited to 10% of data
Temperature values to test: 0.0 0.3 0.7

Model: gpt-4-turbo
Temperature: 0.0
========================================
[classification] 100%|████████████| 81/81
[retrieval] 100%|████████████| 121/121
...
```

---

## 🔧 문제 해결

### 1. Claude API 404 에러
```
Error code: 404 - model: claude-3-5-sonnet-20241022
```
**해결**: 스크립트가 이미 수정되었습니다. 최신 버전 사용하세요.

### 2. Llama 액세스 거부
```
Error: You are trying to access a gated repo
```
**해결**:
1. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 접속
2. "Request access" 클릭
3. 승인 후 `export HF_TOKEN='your_token'`

### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**해결**:
- H100 GPU (80GB)에서는 발생하지 않아야 함
- 발생 시: `torch_dtype=torch.float16` 확인
- 또는 batch size 조정

### 4. API Rate Limit
```
RateLimitError: Rate limit exceeded
```
**해결**:
- 잠시 대기 후 재실행
- 스크립트가 자동으로 중단된 지점부터 재개

---

## 📁 결과 파일

실험 완료 후 생성되는 파일:

```
results/temperature_ablation/
├── results_gpt-4-turbo_TIMESTAMP.json       # 원시 결과
├── results_claude-sonnet-4-5_TIMESTAMP.json
├── results_meta-llama_Llama-3.1-8B-Instruct_TIMESTAMP.json
├── ...
├── summary_*.csv                             # 요약 CSV
├── temperature_ablation_summary.csv          # 통합 요약
└── temperature_ablation_*.pdf                # 시각화
```

### 결과 분석
```bash
# 분석 스크립트 실행
python3 analyze_temperature_ablation.py results/temperature_ablation/
```

---

## 📞 지원

문제 발생 시:
1. 로그 파일 확인
2. 환경 변수 재확인 (`env | grep -E "HF|OPENAI|ANTHROPIC"`)
3. GPU 상태 확인 (`nvidia-smi`)

---

**Last Updated**: 2025-11-12
**Version**: 2.0 (10% sampling, 3 temperatures)
