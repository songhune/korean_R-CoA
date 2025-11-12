# GPU 환경 배포 체크리스트

## ✅ 코드 검토 완료 사항

### 1. 파라미터 조정 완료
- [x] Temperature: 5개 → 3개 (0.0, 0.3, 0.7)
- [x] 샘플링: 30% → 10%
- [x] Claude 모델명 수정 (404 에러 해결)

### 2. 자동 다운로드 코드 검토

#### HuggingFaceWrapper 클래스 (exp5_benchmark_evaluation.py:615-687)

**✅ 안전성 확인:**

1. **자동 다운로드 지원**
   ```python
   self.tokenizer = AutoTokenizer.from_pretrained(
       model_name,
       token=token,  # HF_TOKEN 사용
       trust_remote_code=True
   )
   self.model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       device_map="auto",  # H100에 자동 배치
       token=token,
       trust_remote_code=True
   )
   ```

2. **지원 모델**
   - ✅ Llama-3.1-8B: `meta-llama/Llama-3.1-8B-Instruct`
   - ✅ Qwen2.5-7B: `Qwen/Qwen2.5-7B-Instruct`
   - ✅ EXAONE-3.0: `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`

3. **메모리 최적화**
   - float16 사용 (H100에 최적)
   - device_map="auto" (자동 GPU 할당)

4. **에러 처리**
   - Chat template fallback 지원
   - Temperature 0.0 처리 (do_sample=False)

**⚠️ 주의사항:**

1. **Llama 모델 액세스**
   - Gated model이므로 HF_TOKEN 필수
   - 사전에 액세스 요청 필요: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

2. **디스크 공간**
   - 첫 실행 시 모델 자동 다운로드: ~47GB
   - 캐시 위치: `~/.cache/huggingface/hub/`

3. **VRAM 요구사항**
   - 8B 모델 × float16: ~16GB per model
   - H100 80GB: 충분 ✅

## 🚀 배포 프로세스

### Mac (현재 환경)

```bash
# 1. 현재 실행 중단
pkill -f exp5_benchmark_evaluation

# 2. 기존 Claude 결과 삭제 (잘못된 모델명)
cd results/temperature_ablation
rm -f results_claude-3-5-sonnet-20241022*.json
rm -f summary_claude-3-5-sonnet-20241022*.csv
cd -

# 3. HF_TOKEN 제거
export HF_TOKEN=""

# 4. 실험 재시작
./run_temperature_ablation.sh sample
```

### H100 GPU 서버

```bash
# 1. 코드 동기화
cd /path/to/korean_eda
git pull  # 또는 rsync/scp로 전송

# 2. 환경 설정
cd notebook/experiments/exp5
./setup_exp5_gpu.sh

# 3. HuggingFace 토큰 설정
export HF_TOKEN='hf_xxxxxxxxxxxxx'

# 4. API 키 제거 (중요!)
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

# 5. Llama 액세스 확인
python3 -c "
from transformers import AutoTokenizer
import os
token = os.environ.get('HF_TOKEN')
try:
    AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', token=token)
    print('✅ Llama 액세스 가능')
except Exception as e:
    print(f'❌ 에러: {e}')
"

# 6. 실험 실행
./run_temperature_ablation.sh sample
```

## 📊 예상 결과

### 실험 완료 후 생성되는 파일 (21개)

**API 모델 (Mac):**
- gpt-4-turbo: 3개 (temp 0.0, 0.3, 0.7)
- gpt-3.5-turbo: 3개
- claude-sonnet-4-5-20250929: 3개
- claude-3-opus-20240229: 3개
- **소계: 12개**

**오픈소스 (H100):**
- meta-llama/Llama-3.1-8B-Instruct: 3개
- Qwen/Qwen2.5-7B-Instruct: 3개
- LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct: 3개
- **소계: 9개**

**총: 21개 실험**

## 🔍 모니터링

### 진행 상황 확인

```bash
# 생성된 파일 개수
ls results/temperature_ablation/results_*.json | wc -l

# 모델별 완료 개수
cd results/temperature_ablation
for model in gpt-4-turbo gpt-3.5-turbo claude-sonnet-4-5 claude-3-opus Llama Qwen EXAONE; do
    count=$(ls results_*${model}*.json 2>/dev/null | wc -l)
    echo "$model: $count/3"
done
```

### 예상 타임라인

**Mac (API):**
- Start: 현재
- 각 모델: ~1.4시간
- 총 4개 모델: ~5.5시간
- **완료: 오늘 밤 ~11시**

**H100 (GPU):**
- Start: 환경 구성 후
- 각 모델: ~20분
- 총 3개 모델: ~1시간
- **완료: 시작 후 1시간**

## ✅ 최종 확인사항

배포 전 체크:
- [ ] `run_temperature_ablation.sh` 수정 완료 (10%, temp 3개)
- [ ] `setup_exp5_gpu.sh` 생성 완료
- [ ] README 문서 작성 완료
- [ ] GPU 서버 액세스 확인
- [ ] HuggingFace 토큰 발급 및 Llama 액세스 승인
- [ ] 디스크 공간 50GB+ 확보
- [ ] CUDA 환경 확인 (nvidia-smi)

실행 시작:
- [ ] Mac: API 모델 실행 시작
- [ ] H100: 오픈소스 모델 실행 시작
- [ ] 진행 상황 모니터링 설정

## 🐛 알려진 이슈 및 해결

### 이슈 1: Claude 404 에러
- **상태**: ✅ 해결됨
- **원인**: 잘못된 모델명
- **해결**: 모델명 수정 완료

### 이슈 2: 느린 실험 속도
- **상태**: ✅ 해결됨
- **원인**: 30% 샘플링 + 5개 temperature
- **해결**: 10% + 3개 temperature로 조정

### 이슈 3: Llama 액세스
- **상태**: ⚠️ 사전 승인 필요
- **해결**: HuggingFace에서 액세스 요청 후 HF_TOKEN 설정

---

**검토 완료**: 2025-11-12
**배포 준비**: ✅ Ready
