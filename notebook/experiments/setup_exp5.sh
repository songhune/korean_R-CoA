#!/bin/bash

# K-ClassicBench 평가 환경 설정 스크립트

echo "========================================"
echo "K-ClassicBench 평가 환경 설정"
echo "========================================"
echo ""

# Python 환경 확인
PYTHON_CMD=~/.pyenv/versions/3.10.10/envs/llm/bin/python

if [ ! -f "$PYTHON_CMD" ]; then
    echo "⚠️  Python 환경을 찾을 수 없습니다: $PYTHON_CMD"
    echo "   Python 경로를 확인하세요."
    exit 1
fi

echo "✅ Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# 필수 패키지 설치
echo "📦 필수 패키지 설치 중..."
echo ""

$PYTHON_CMD -m pip install --upgrade pip

# 기본 패키지
echo "1/4 기본 패키지..."
$PYTHON_CMD -m pip install pandas numpy tqdm scikit-learn

# 평가 메트릭
echo "2/4 평가 메트릭 패키지..."
$PYTHON_CMD -m pip install rouge-score nltk

# 시각화
echo "3/4 시각화 패키지..."
$PYTHON_CMD -m pip install matplotlib seaborn

# NLTK 데이터 다운로드
echo "4/4 NLTK 데이터..."
$PYTHON_CMD -c "import nltk; nltk.download('punkt')"

echo ""
echo "========================================"
echo "✅ 기본 환경 설정 완료!"
echo "========================================"
echo ""
echo "추가 패키지 설치 (선택사항):"
echo ""
echo "1. API 모델 사용:"
echo "   $PYTHON_CMD -m pip install openai anthropic"
echo ""
echo "2. 오픈소스 모델 사용:"
echo "   $PYTHON_CMD -m pip install transformers torch accelerate"
echo ""
echo "3. Jupyter 노트북:"
echo "   $PYTHON_CMD -m pip install jupyter ipykernel ipywidgets"
echo ""
