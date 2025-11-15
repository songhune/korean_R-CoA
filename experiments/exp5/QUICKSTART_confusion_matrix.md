# Quick Start: Confusion Matrix 생성

## 🚀 가장 빠른 방법

### 1. 필수 패키지 설치

```bash
cd /home/work/songhune/korean_R-CoA/experiments/exp5
pip install -r requirements_confusion_matrix.txt
```

### 2. 실행

```bash
./run_confusion_matrix_generation.sh
```

실행 후 선택:
- **1번**: 기존 결과 파일 사용 (빠름, 10개 샘플만, 테스트용)
- **2번**: 전체 predictions 새로 생성 (느림, 전체 808개 샘플, 실제 논문용)
- **3번**: 기존 full_predictions 파일 사용 (빠름, 이미 생성된 경우)

---

## 📝 수동 실행 (API 모델)

### GPT-4 Turbo 전체 predictions 생성

```bash
python save_full_predictions.py \
    --model-type api \
    --model-name gpt-4-turbo \
    --temperature 0.0
```

### Claude 3.5 Sonnet 전체 predictions 생성

```bash
python save_full_predictions.py \
    --model-type api \
    --model-name claude-3-5-sonnet-20241022 \
    --temperature 0.0
```

### Confusion Matrix 생성 (전체 predictions 사용)

```bash
python generate_classification_confusion_matrix.py \
    --temperature 0.0 \
    --use-full-predictions \
    --full-predictions-dir ../../results/full_predictions
```

---

## 📊 결과 확인

```bash
# 생성된 파일 확인
ls -l ../../results/confusion_matrices/

# 이미지 파일 보기
open ../../results/confusion_matrices/confusion_matrix_*.png

# 비교 리포트 보기
cat ../../results/confusion_matrices/comparison_report.txt
```

---

## ⚙️ API 키 설정 확인

```bash
# .env 파일 확인
cat ../../.env

# 올바르게 설정되었는지 테스트
python -c "
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path('../../.env'))
print('OPENAI_API_KEY:', os.environ.get('OPENAI_API_KEY', 'NOT SET')[:20] + '...')
print('ANTHROPIC_API_KEY:', os.environ.get('ANTHROPIC_API_KEY', 'NOT SET')[:20] + '...')
"
```

---

## 🐛 문제 해결

### python-dotenv가 없다는 오류

```bash
pip install python-dotenv
```

### API 키가 없다는 오류

`.env` 파일에서 주석(`#`)이 제거되었는지 확인:

```bash
# ../../.env 파일 내용
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
HF_TOKEN=hf_...
```

### matplotlib/seaborn 오류

```bash
pip install --upgrade matplotlib seaborn
```

---

## 📈 예상 실행 시간

- **Method 1** (기존 결과, 10개): ~1분
- **Method 2** (API 전체 생성, 808개):
  - GPT-4: ~15분 (rate limiting)
  - Claude: ~15분 (rate limiting)
  - 오픈소스 (GPU): ~10분
- **Method 3** (기존 full_predictions): ~1분

---

## 📁 출력 파일 구조

```
results/
├── full_predictions/                    # 전체 predictions (Method 2)
│   ├── full_predictions_gpt-4-turbo_temp0.0.json
│   ├── full_predictions_claude-3-5-sonnet_temp0.0.json
│   └── ...
└── confusion_matrices/                  # Confusion matrices
    ├── confusion_matrix_gpt-4-turbo.png
    ├── confusion_matrix_gpt-4-turbo_report.txt
    ├── confusion_matrix_claude-3-5-sonnet.png
    ├── confusion_matrix_claude-3-5-sonnet_report.txt
    └── comparison_report.txt
```

---

## 🎯 과문육체 레이블

실험 대상 6개 레이블:
- **賦** (부): 95개 (11.8%)
- **詩** (시): 95개 (11.8%)
- **疑** (의): 95개 (11.8%)
- **義** (의): 95개 (11.8%)
- **策** (책): 95개 (11.7%)
- **表** (표): 95개 (11.7%)

전체: **570개** (classification task의 일부)

---

## 💡 Tips

1. **테스트할 때**: Method 1 사용 (빠름, 10개만)
2. **논문용 최종 결과**: Method 2 사용 (전체 808개)
3. **이미 생성했다면**: Method 3 사용 (재사용)

더 자세한 내용은 `README_confusion_matrix.md` 참고!
