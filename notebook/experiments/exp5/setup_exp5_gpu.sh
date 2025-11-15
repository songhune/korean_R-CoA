#!/bin/bash

# KLSBench GPU 환경 설정 스크립트 (H100용)
# 오픈소스 모델 (Llama, Qwen, EXAONE) 실행을 위한 환경 구성

echo "========================================"
echo "KLSBench GPU 환경 설정 (H100)"
echo "========================================"
echo ""

# Python 환경 확인
PYTHON_CMD=$(which python3)

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 환경을 찾을 수 없습니다."
    echo "   Python 3.8+ 환경을 활성화하세요."
    exit 1
fi

echo "[OK] Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# CUDA 확인
echo "[CHECK] CUDA 환경 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "[WARNING] nvidia-smi를 찾을 수 없습니다. GPU가 없거나 CUDA가 설치되지 않았을 수 있습니다."
    echo ""
fi

# HuggingFace Token 확인
echo "[CHECK] HuggingFace Token..."
if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "   Llama 모델 사용을 위해서는 필수입니다:"
    echo "   export HF_TOKEN='your_huggingface_token'"
    echo ""
else
    echo "[OK] HF_TOKEN이 설정되어 있습니다."
    echo ""
fi

# 패키지 설치
echo "========================================"
echo "필수 패키지 설치"
echo "========================================"
echo ""

$PYTHON_CMD -m pip install --upgrade pip

# 기본 패키지
echo "[1/6] 기본 패키지..."
$PYTHON_CMD -m pip install pandas numpy tqdm scikit-learn

# 평가 메트릭
echo "[2/6] 평가 메트릭..."
$PYTHON_CMD -m pip install rouge-score nltk

# 시각화
echo "[3/6] 시각화..."
$PYTHON_CMD -m pip install matplotlib seaborn

# PyTorch (CUDA 지원)
echo "[4/6] PyTorch (CUDA)..."
$PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers & Accelerate
echo "[5/6] Transformers & Accelerate..."
$PYTHON_CMD -m pip install transformers>=4.40.0 accelerate>=0.27.0

# Optional: bitsandbytes for quantization
echo "[6/6] bitsandbytes (선택)..."
$PYTHON_CMD -m pip install bitsandbytes || echo "[INFO] bitsandbytes 설치 실패 (선택사항)"

# NLTK 데이터
echo ""
echo "[DOWNLOAD] NLTK 데이터..."
$PYTHON_CMD -c "import nltk; nltk.download('punkt', quiet=True)"

echo ""
echo "========================================"
echo "환경 설정 완료!"
echo "========================================"
echo ""

# 모델 자동 다운로드 테스트
echo "========================================"
echo "모델 다운로드 사전 확인"
echo "========================================"
echo ""

cat << 'EOF' > /tmp/test_model_download.py
import os
import sys

print("Testing model access...")

models = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "EXAONE-3.0": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
}

try:
    from transformers import AutoTokenizer

    for name, model_id in models.items():
        print(f"\n[CHECK] {name} ({model_id})...")
        try:
            # HF_TOKEN 환경변수 사용
            token = os.environ.get("HF_TOKEN", None)

            # Tokenizer만 로드해서 접근 가능 여부 확인
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=token,
                trust_remote_code=True
            )
            print(f"  ✓ {name}: 접근 가능 (모델은 실행 시 자동 다운로드됩니다)")

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"  ✗ {name}: 모델을 찾을 수 없습니다")
            elif "gated" in error_msg.lower() or "access" in error_msg.lower():
                print(f"  ⚠ {name}: 액세스 권한 필요")
                print(f"     https://huggingface.co/{model_id} 에서 액세스 요청")
            else:
                print(f"  ⚠ {name}: {error_msg[:100]}")

    print("\n[INFO] 모델 다운로드는 실행 시 자동으로 진행됩니다.")
    print("[INFO] 처음 실행 시 모델당 15-30GB 다운로드가 발생합니다.")

except ImportError as e:
    print(f"\n[ERROR] transformers 패키지를 찾을 수 없습니다: {e}")
    sys.exit(1)
EOF

$PYTHON_CMD /tmp/test_model_download.py
rm /tmp/test_model_download.py

echo ""
echo "========================================"
echo "다음 단계"
echo "========================================"
echo ""
echo "1. HuggingFace 토큰 설정 (Llama용):"
echo "   export HF_TOKEN='your_token_here'"
echo ""
echo "2. 실험 실행:"
echo "   cd $(dirname $0)"
echo "   ./run_temperature_ablation.sh sample"
echo ""
echo "3. 모델별 예상 다운로드 크기:"
echo "   - Llama-3.1-8B: ~16GB"
echo "   - Qwen2.5-7B: ~15GB"
echo "   - EXAONE-3.0-7.8B: ~16GB"
echo "   총: ~47GB 디스크 공간 필요"
echo ""
