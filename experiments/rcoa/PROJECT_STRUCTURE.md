# R-CoA Project Structure

ML Pipeline에 맞춰 재구성된 프로젝트 구조입니다.

```
rcoa/
├── 📦 models/                    # 모델 정의
│   ├── __init__.py
│   ├── anchor_head_model.py     # Anchor Head (XLM-R + LoRA + InfoNCE)
│   └── chain_head_model.py      # Chain Head (TransE + Chain Loss)
│
├── 📜 scripts/                   # 실행 스크립트
│   ├── __init__.py
│   │
│   ├── preprocess/              # 데이터 전처리
│   │   ├── __init__.py
│   │   └── data_preprocessing.py
│   │
│   ├── train/                   # 학습 스크립트
│   │   ├── __init__.py
│   │   └── anchor_train.py
│   │
│   ├── eval/                    # 평가 스크립트
│   │   ├── __init__.py
│   │   └── anchor_evaluate.py
│   │
│   ├── visualize/               # 시각화 스크립트
│   │   ├── __init__.py
│   │   └── visualize_embeddings.py
│   │
│   ├── utils/                   # 유틸리티
│   │   └── __init__.py
│   │
│   ├── run_poc.sh               # PoC 실행 스크립트
│   └── run_full_pipeline.sh     # 전체 파이프라인 실행
│
├── 🗂️  data/                     # 데이터
│   ├── raw/                     # 원본 데이터
│   ├── processed/               # 전처리된 데이터
│   └── splits/                  # Train/Val 분할
│       ├── train_pairs.jsonl   # 18,826 pairs
│       ├── val_pairs.jsonl     # 2,091 pairs
│       └── statistics.json     # 데이터 통계
│
├── 💾 checkpoints/              # 모델 체크포인트
│   ├── quick_test/             # Quick test 체크포인트
│   ├── anchor_head/            # Anchor Head 체크포인트
│   └── chain_head/             # Chain Head 체크포인트 (향후)
│
├── 📊 results/                  # 실험 결과
│   ├── metrics/                # 성능 지표
│   ├── logs/                   # 로그 파일
│   │   └── quick_test.log
│   └── figures/                # 시각화 결과
│
├── 📓 notebooks/                # Jupyter notebooks
│
├── 📖 docs/                     # 문서
│   ├── README.md               # 메인 문서
│   ├── QUICKSTART.md           # 빠른 시작 가이드
│   ├── plan.md                 # 4주 계획
│   └── rcoa_concept.md         # R-CoA 컨셉 (Marp 슬라이드)
│
├── ⚙️  configs/                  # 설정 파일
│   ├── __init__.py
│   └── requirements.txt        # Python 패키지 의존성
│
├── 🧪 tests/                    # 테스트 코드
│
└── PROJECT_STRUCTURE.md         # 이 파일
```

---

## 📋 Quick Start

### 1. 환경 설정
```bash
cd /home/work/songhune/korean_R-CoA/experiments/rcoa
pip install -r configs/requirements.txt
```

### 2. 데이터 전처리
```bash
python scripts/preprocess/data_preprocessing.py
```

### 3. 학습
```bash
# Quick test (1 epoch)
bash scripts/run_poc.sh 2

# Full training (10 epochs)
bash scripts/run_poc.sh 3
```

### 4. 평가
```bash
bash scripts/run_poc.sh 4
```

### 5. 시각화
```bash
python scripts/visualize/visualize_embeddings.py \
    --checkpoint checkpoints/anchor_head/best_model.pt \
    --data data/splits/val_pairs.jsonl \
    --output-dir results/figures
```

---

## 📁 주요 디렉토리 설명

### `models/`
모델 아키텍처 정의 파일들이 위치합니다.
- Python 모듈로 import 가능: `from models.anchor_head_model import AnchorHead`

### `scripts/`
모든 실행 가능한 스크립트들이 기능별로 분류되어 있습니다.
- `preprocess/`: 데이터 전처리
- `train/`: 모델 학습
- `eval/`: 모델 평가
- `visualize/`: 결과 시각화
- `utils/`: 공통 유틸리티

### `data/`
- `raw/`: 원본 데이터 (ACCN-INS.json, combined_ACCN-INS_chunks.jsonl 등)
- `processed/`: 전처리된 중간 데이터
- `splits/`: 학습/검증 분할 데이터 (최종 사용 데이터)

### `checkpoints/`
학습된 모델의 체크포인트가 저장됩니다.
- 자동으로 best_model.pt와 epoch별 체크포인트가 저장됨

### `results/`
- `metrics/`: JSON 형태의 평가 지표
- `logs/`: 학습/평가 로그
- `figures/`: t-SNE, heatmap 등 시각화 결과

### `docs/`
프로젝트 관련 모든 문서가 위치합니다.

---

## 🔧 Import 경로

프로젝트 루트를 Python path에 추가하여 사용:
```python
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 모델 import
from models.anchor_head_model import AnchorHead, InfoNCELoss
from models.chain_head_model import ChainHead, TransELoss
```

---

## 📝 파일 경로 규칙

모든 스크립트는 프로젝트 루트에서 실행되는 것을 가정합니다:
```bash
# 예시
cd /home/work/songhune/korean_R-CoA/experiments/rcoa
python scripts/train/anchor_train.py --train-data data/splits/train_pairs.jsonl
```

---

## 🚀 파이프라인 자동화

전체 파이프라인을 한 번에 실행:
```bash
bash scripts/run_full_pipeline.sh
```

단계별 실행:
```bash
bash scripts/run_poc.sh 1  # 데이터 전처리
bash scripts/run_poc.sh 2  # Quick test
bash scripts/run_poc.sh 3  # Full training
bash scripts/run_poc.sh 4  # Evaluation
```

---

## 📦 패키지 구조

각 하위 디렉토리는 `__init__.py`를 포함하여 Python 패키지로 구성되어 있습니다:
- `models`
- `scripts`
- `scripts.train`
- `scripts.eval`
- `scripts.preprocess`
- `scripts.visualize`
- `scripts.utils`
- `configs`

---

## 🔄 변경 사항 요약

기존 구조에서 다음과 같이 변경되었습니다:

1. **모델 파일** → `models/`
2. **스크립트 분류** → `scripts/{train,eval,preprocess,visualize}/`
3. **데이터 구조화** → `data/{raw,processed,splits}/`
4. **결과 분류** → `results/{metrics,logs,figures}/`
5. **문서 정리** → `docs/`
6. **설정 분리** → `configs/`

모든 import 경로와 스크립트가 새로운 구조에 맞게 업데이트되었습니다.
