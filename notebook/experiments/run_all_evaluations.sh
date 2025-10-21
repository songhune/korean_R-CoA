#!/bin/bash

# K-ClassicBench 전체 모델 평가 스크립트
# 사용법: ./run_all_evaluations.sh [test|full]
# - test: 각 태스크당 10개 샘플로 테스트
# - full: 전체 벤치마크 평가

MODE=${1:-test}  # 기본값: test

if [ "$MODE" = "test" ]; then
    MAX_SAMPLES="--max-samples 10"
    echo "🧪 테스트 모드: 각 태스크당 10개 샘플"
else
    MAX_SAMPLES=""
    echo "🚀 전체 평가 모드"
fi

echo "========================================"
echo "K-ClassicBench 모델 평가 시작"
echo "========================================"
echo ""

# API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY가 설정되지 않았습니다."
    echo "   export OPENAI_API_KEY='your-key'"
else
    echo "✅ OpenAI API Key 확인됨"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY가 설정되지 않았습니다."
    echo "   export ANTHROPIC_API_KEY='your-key'"
else
    echo "✅ Anthropic API Key 확인됨"
fi

echo ""
echo "========================================"

# 1. API 모델 평가
echo ""
echo "📡 1. API 모델 평가"
echo "========================================"

# GPT-4 Turbo
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "🤖 GPT-4 Turbo 평가 중..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name gpt-4-turbo \
        --api-key $OPENAI_API_KEY \
        $MAX_SAMPLES

    echo ""
    echo "🤖 GPT-3.5 Turbo 평가 중..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name gpt-3.5-turbo \
        --api-key $OPENAI_API_KEY \
        $MAX_SAMPLES
fi

# Claude
if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "🤖 Claude 3.5 Sonnet 평가 중..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name claude-3-5-sonnet-20241022 \
        --api-key $ANTHROPIC_API_KEY \
        $MAX_SAMPLES

    echo ""
    echo "🤖 Claude 3 Opus 평가 중..."
    python exp5_benchmark_evaluation.py \
        --model-type api \
        --model-name claude-3-opus-20240229 \
        --api-key $ANTHROPIC_API_KEY \
        $MAX_SAMPLES
fi

# 2. 오픈소스 모델 평가
echo ""
echo "========================================"
echo "🌐 2. 오픈소스 모델 평가"
echo "========================================"

# Llama 3.1 8B
echo ""
echo "🦙 Llama 3.1 8B Instruct 평가 중..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    $MAX_SAMPLES

# Qwen 2.5 7B
echo ""
echo "🤖 Qwen 2.5 7B Instruct 평가 중..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    $MAX_SAMPLES

# EXAONE 3.0 7.8B
echo ""
echo "🤖 EXAONE 3.0 7.8B Instruct 평가 중..."
python exp5_benchmark_evaluation.py \
    --model-type opensource \
    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
    $MAX_SAMPLES

# 3. 지도학습 모델 평가
echo ""
echo "========================================"
echo "🎓 3. 지도학습 모델 평가"
echo "========================================"

# Tongu (구현 필요)
echo ""
echo "⚠️  Tongu 모델 - 구현 필요"
# python exp5_benchmark_evaluation.py \
#     --model-type supervised \
#     --model-name tongu \
#     $MAX_SAMPLES

# GwenBert (구현 필요)
echo ""
echo "⚠️  GwenBert 모델 - 구현 필요"
# python exp5_benchmark_evaluation.py \
#     --model-type supervised \
#     --model-name gwenbert \
#     $MAX_SAMPLES

echo ""
echo "========================================"
echo "✅ 모든 평가 완료!"
echo "========================================"
echo ""
echo "📊 결과 확인:"
echo "   - 결과 디렉토리: ../../benchmark/results/"
echo "   - JSON 파일: results_*_*.json"
echo "   - CSV 요약: summary_*_*.csv"
echo ""
