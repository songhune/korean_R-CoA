# 실험 5: K-ClassicBench 평가 프레임워크 완성 요약

## 🎯 목표 달성

✅ **K-ClassicBench 벤치마크를 다양한 LLM으로 평가하는 완전한 프레임워크 구축**

---

## 📦 생성된 파일

### 1. 메인 평가 코드
- **`exp5_benchmark_evaluation.py`** (24KB)
  - 전체 평가 프레임워크 구현
  - API 모델, 오픈소스 모델, 지도학습 모델 지원
  - 5개 태스크 평가 및 메트릭 계산

### 2. Jupyter 노트북
- **`5번실험.ipynb`** (16KB)
  - 인터랙티브 평가 및 분석
  - 시각화 포함
  - 배치 실행 스크립트

### 3. 실행 스크립트
- **`run_all_evaluations.sh`** (3.8KB)
  - 모든 모델 순차 평가
  - 테스트/전체 모드 지원
  - API 키 자동 확인

### 4. 환경 설정
- **`setup_exp5.sh`** (1.5KB)
  - 필수 패키지 자동 설치
  - Python 환경 확인

- **`requirements_exp5.txt`** (506B)
  - 필요한 패키지 목록

### 5. 문서
- **`README_exp5.md`** (6.8KB)
  - 상세 사용 가이드
  - 문제 해결 가이드
  - 예제 코드

---

## 🏗️ 프레임워크 구조

```
K-ClassicBench 평가 프레임워크
│
├── 데이터 로딩
│   └── k_classic_bench_full.json (7,871개 항목)
│
├── 모델 래퍼
│   ├── API 모델
│   │   ├── OpenAIWrapper (GPT-4, GPT-3.5)
│   │   └── AnthropicWrapper (Claude)
│   │
│   ├── 오픈소스 모델
│   │   └── HuggingFaceWrapper (Llama, Qwen, EXAONE)
│   │
│   └── 지도학습 모델
│       ├── TonguWrapper
│       └── GwenBertWrapper
│
├── 태스크별 평가
│   ├── Classification → Accuracy, F1
│   ├── Retrieval → Accuracy
│   ├── Punctuation → F1, ROUGE
│   ├── NLI → Accuracy, F1
│   └── Translation → BLEU, ROUGE
│
└── 결과 저장
    ├── JSON (상세 결과 + 예측값)
    └── CSV (요약)
```

---

## 🚀 사용 방법

### 빠른 시작 (3단계)

#### 1단계: 환경 설정

```bash
cd /Users/songhune/Workspace/korean_eda/notebook/experiments
./setup_exp5.sh
```

#### 2단계: API 키 설정 (API 모델 사용시)

```bash
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
```

#### 3단계: 평가 실행

**옵션 A: 단일 모델 평가**
```bash
# 테스트 (각 태스크 10개)
python exp5_benchmark_evaluation.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --api-key $OPENAI_API_KEY \
    --max-samples 10
```

**옵션 B: 모든 모델 배치 평가**
```bash
# 테스트 모드
./run_all_evaluations.sh test

# 전체 평가
./run_all_evaluations.sh full
```

**옵션 C: Jupyter 노트북**
```bash
jupyter notebook 5번실험.ipynb
```

---

## 📊 평가 태스크

| 태스크 | 입력 | 출력 | 메트릭 |
|:---|:---|:---|:---|
| **Classification** | 한문 텍스트 | 문체 (賦/詩/疑/義) | Accuracy, F1 |
| **Retrieval** | 한문 문장 | 출처 (論語/孟子/大學/中庸) | Accuracy |
| **Punctuation** | 백문 (구두점 없음) | 구두점본 | F1, ROUGE-L |
| **NLI** | 전제 + 가설 | entailment/contradiction/neutral | Accuracy, F1 |
| **Translation** | 원문 (한문/한글/영문) | 번역 | BLEU, ROUGE |

---

## 🤖 지원 모델

### 비공개 API 모델 ✅
- GPT-4 Turbo
- GPT-3.5 Turbo
- Claude 3.5 Sonnet
- Claude 3 Opus

### 오픈소스 모델 ✅
- Llama 3.1 (8B, 70B)
- Qwen 2.5 (7B, 14B, 72B)
- EXAONE 3.0 (7.8B)

### 지도학습 모델 🔧
- GwenBert (구현 필요)
- Tongu (구현 필요)

---

## 📈 결과 형식

### 출력 파일

```
../../benchmark/results/
├── results_gpt-4-turbo_20241021_150000.json
├── summary_gpt-4-turbo_20241021_150000.csv
├── results_claude-3-5-sonnet_20241021_160000.json
└── summary_claude-3-5-sonnet_20241021_160000.csv
```

### CSV 요약 예시

| model | task | accuracy | f1 | bleu | rouge1_f1 | rougeL_f1 |
|:---|:---|---:|---:|---:|---:|---:|
| gpt-4-turbo | classification | 0.850 | 0.840 | - | - | - |
| gpt-4-turbo | retrieval | 0.920 | - | - | - | - |
| gpt-4-turbo | punctuation | - | 0.780 | - | 0.820 | 0.790 |
| gpt-4-turbo | nli | 0.780 | 0.770 | - | - | - |
| gpt-4-turbo | translation | - | - | 0.650 | 0.710 | 0.680 |

---

## 🎨 시각화 (노트북)

1. **태스크별 성능 비교** - 막대 그래프
2. **모델별 히트맵** - 태스크 × 모델 성능
3. **종합 점수 랭킹** - 평균 성능 순위

---

## 🔧 커스터마이징

### 새 모델 추가

```python
# exp5_benchmark_evaluation.py에 추가

class MyModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str):
        # 모델 로드
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # 추론 구현
        return prediction
```

### 새 태스크 추가

```python
def evaluate_my_task(self, predictions: List[str], ground_truths: List[str]) -> Dict:
    # 평가 로직
    return {
        'my_metric': score,
        'num_samples': len(predictions)
    }
```

---

## ⚠️ 주의사항

### GPU 메모리
- Llama 70B, Qwen 72B: **40GB+ 필요**
- 8bit quantization 권장

### API 비용
- GPT-4: 태스크당 약 $X
- Claude: 태스크당 약 $X
- `--max-samples 10`으로 테스트 권장

### 실행 시간
- API 모델 (전체): ~2-3시간
- 오픈소스 7B (전체): ~4-5시간
- 오픈소스 70B (전체): ~10-15시간

---

## 📋 체크리스트

### 환경 설정
- [ ] Python 3.10+ 설치 확인
- [ ] `setup_exp5.sh` 실행
- [ ] API 키 설정 (API 모델 사용시)
- [ ] GPU 사용 가능 확인 (오픈소스 모델)

### 평가 실행
- [ ] 테스트 모드로 먼저 실행 (`--max-samples 10`)
- [ ] API 비용 확인
- [ ] 전체 평가 실행
- [ ] 결과 파일 확인

### 분석
- [ ] CSV 요약 확인
- [ ] 노트북에서 시각화
- [ ] 오류 케이스 분석
- [ ] 인사이트 정리

---

## 🎓 C3Bench와의 비교

### 공통점
- 5개 핵심 태스크
- 다양한 모델 유형 평가
- 정량적 메트릭 기반

### K-ClassicBench 차별점
1. **한국 고전 문헌 특화**
   - 과거시험 데이터
   - 사서(四書) 데이터

2. **구두점 태스크**
   - 백문 → 구두점본 복원
   - 한문 처리 능력 평가

3. **다국어 번역**
   - 한문 ↔ 한글 ↔ 영문

---

## 🚧 향후 개선 계획

1. **Few-shot Learning**
   - 0-shot, 1-shot, 5-shot 비교
   - 예시 선택 전략

2. **Chain-of-Thought**
   - 단계별 추론 평가
   - 설명 생성

3. **더 많은 모델**
   - Gemini
   - Mistral
   - 한국어 특화 모델

4. **벤치마크 확장**
   - 더 많은 태스크
   - 난이도별 분류
   - 도메인별 세분화

---

## 📚 참고 자료

- **C3Bench 논문**: [링크]
- **벤치마크 상세**: `../../benchmark/k_classic_bench/README.md`
- **데이터 출처**:
  - 과거시험 데이터
  - 사서(四書) 원문

---

## 📧 문의

- 이메일: [이메일 주소]
- GitHub: [레포지토리 링크]

---

**작성일**: 2024-10-21
**버전**: 1.0
**상태**: ✅ 완료
