#!/bin/bash
# Confusion Matrix Generation Workflow
# Classification task의 전체 confusion matrix를 생성합니다.

set -e  # Exit on error

echo "============================================================"
echo "Confusion Matrix Generation for KLSBench Classification"
echo "============================================================"
echo ""

# 디렉토리 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
BENCHMARK_PATH="${PROJECT_ROOT}/benchmark/kls_bench_full.json"
FULL_PREDICTIONS_DIR="${PROJECT_ROOT}/results/full_predictions"
OUTPUT_DIR="${PROJECT_ROOT}/results/confusion_matrices"
TEMPERATURE=0.0

# API 키 로드 (.env 파일이 있는 경우)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "[INFO] Loading API keys from .env file: ${PROJECT_ROOT}/.env"
    set -a  # automatically export all variables
    source "${PROJECT_ROOT}/.env"
    set +a

    # 키가 제대로 로드되었는지 확인
    if [ -n "${OPENAI_API_KEY}" ]; then
        echo "[INFO] OPENAI_API_KEY loaded: ${OPENAI_API_KEY:0:10}..."
    fi
    if [ -n "${ANTHROPIC_API_KEY}" ]; then
        echo "[INFO] ANTHROPIC_API_KEY loaded: ${ANTHROPIC_API_KEY:0:10}..."
    fi
else
    echo "[WARNING] .env file not found at: ${PROJECT_ROOT}/.env"
fi

# 사용자에게 어떤 방법을 사용할지 물어보기
echo "어떤 방법으로 confusion matrix를 생성하시겠습니까?"
echo ""
echo "1) 기존 결과 파일 사용 (빠름, 처음 10개 샘플만)"
echo "2) 전체 predictions 새로 생성 (느림, 전체 샘플)"
echo "3) 기존 full_predictions 파일 사용 (빠름, 전체 샘플, 파일이 있는 경우)"
echo ""
read -p "선택 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "[METHOD 1] 기존 결과 파일 사용 (처음 10개 샘플만)"
        echo "==========================================="

        python "${SCRIPT_DIR}/generate_classification_confusion_matrix.py" \
            --benchmark "${BENCHMARK_PATH}" \
            --results-dir "${PROJECT_ROOT}/results/temperature_ablation" \
            --output-dir "${OUTPUT_DIR}" \
            --temperature ${TEMPERATURE}

        echo ""
        echo "[완료] Confusion matrices 생성 완료"
        echo "출력 디렉토리: ${OUTPUT_DIR}"
        ;;

    2)
        echo ""
        echo "[METHOD 2] 전체 predictions 새로 생성"
        echo "=========================================="
        echo ""
        echo "주의: 이 방법은 시간이 오래 걸립니다 (API 호출 제한)."
        echo ""

        # 생성할 모델 선택
        echo "어떤 모델 타입을 처리하시겠습니까?"
        echo "1) API 모델 (GPT-4, Claude 등)"
        echo "2) 오픈소스 모델 (Llama, Qwen, EXAONE 등)"
        echo "3) 모두"
        echo ""
        read -p "선택 (1/2/3): " model_choice

        # API 모델 처리
        if [ "$model_choice" == "1" ] || [ "$model_choice" == "3" ]; then
            echo ""
            echo "=== API 모델 처리 ==="
            echo ""

            # GPT-4 Turbo
            if [ -n "${OPENAI_API_KEY}" ]; then
                echo "[1/4] GPT-4 Turbo..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type api \
                    --model-name gpt-4-turbo \
                    --temperature ${TEMPERATURE}

                echo "[2/4] GPT-3.5 Turbo..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type api \
                    --model-name gpt-3.5-turbo \
                    --temperature ${TEMPERATURE}
            else
                echo "[SKIP] OPENAI_API_KEY가 설정되지 않았습니다."
                echo "       .env 파일을 확인하세요: ${PROJECT_ROOT}/.env"
            fi

            # Claude
            if [ -n "${ANTHROPIC_API_KEY}" ]; then
                echo "[3/4] Claude 3.5 Sonnet..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type api \
                    --model-name claude-3-5-sonnet-20241022 \
                    --temperature ${TEMPERATURE}

                echo "[4/4] Claude 3 Opus..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type api \
                    --model-name claude-3-opus-20240229 \
                    --temperature ${TEMPERATURE}
            else
                echo "[SKIP] ANTHROPIC_API_KEY가 설정되지 않았습니다."
                echo "       .env 파일을 확인하세요: ${PROJECT_ROOT}/.env"
            fi
        fi

        # 오픈소스 모델 처리
        if [ "$model_choice" == "2" ] || [ "$model_choice" == "3" ]; then
            echo ""
            echo "=== 오픈소스 모델 처리 ==="
            echo ""

            # GPU 사용 가능 여부 확인
            if ! command -v nvidia-smi &> /dev/null; then
                echo "[WARNING] NVIDIA GPU를 찾을 수 없습니다."
                echo "오픈소스 모델 실행에는 GPU가 필요합니다."
                read -p "계속하시겠습니까? (y/n): " continue_choice
                if [ "$continue_choice" != "y" ]; then
                    echo "[SKIP] 오픈소스 모델 건너뛰기"
                    model_choice="skip_opensource"
                fi
            fi

            if [ "$model_choice" != "skip_opensource" ]; then
                echo "[1/3] Llama 3.1 8B Instruct..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type opensource \
                    --model-name meta-llama/Llama-3.1-8B-Instruct \
                    --temperature ${TEMPERATURE}

                echo "[2/3] Qwen 2.5 7B Instruct..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type opensource \
                    --model-name Qwen/Qwen2.5-7B-Instruct \
                    --temperature ${TEMPERATURE}

                echo "[3/3] EXAONE 3.0 7.8B Instruct..."
                python "${SCRIPT_DIR}/save_full_predictions.py" \
                    --benchmark "${BENCHMARK_PATH}" \
                    --output-dir "${FULL_PREDICTIONS_DIR}" \
                    --model-type opensource \
                    --model-name LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
                    --temperature ${TEMPERATURE}
            fi
        fi

        echo ""
        echo "[STEP 2/2] Confusion Matrix 생성..."
        echo "======================================"

        python "${SCRIPT_DIR}/generate_classification_confusion_matrix.py" \
            --benchmark "${BENCHMARK_PATH}" \
            --results-dir "${FULL_PREDICTIONS_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --temperature ${TEMPERATURE} \
            --use-full-predictions \
            --full-predictions-dir "${FULL_PREDICTIONS_DIR}"

        echo ""
        echo "[완료] 전체 워크플로우 완료"
        echo "Full predictions: ${FULL_PREDICTIONS_DIR}"
        echo "Confusion matrices: ${OUTPUT_DIR}"
        ;;

    3)
        echo ""
        echo "[METHOD 3] 기존 full_predictions 파일 사용"
        echo "============================================"

        # 파일 존재 여부 확인
        if [ ! -d "${FULL_PREDICTIONS_DIR}" ] || [ -z "$(ls -A ${FULL_PREDICTIONS_DIR}/full_predictions_*.json 2>/dev/null)" ]; then
            echo "[ERROR] Full predictions 파일을 찾을 수 없습니다."
            echo "디렉토리: ${FULL_PREDICTIONS_DIR}"
            echo ""
            echo "먼저 Method 2를 선택하여 full predictions를 생성하세요."
            exit 1
        fi

        echo "[INFO] 발견된 full predictions 파일:"
        ls -1 "${FULL_PREDICTIONS_DIR}"/full_predictions_*.json
        echo ""

        python "${SCRIPT_DIR}/generate_classification_confusion_matrix.py" \
            --benchmark "${BENCHMARK_PATH}" \
            --results-dir "${FULL_PREDICTIONS_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --temperature ${TEMPERATURE} \
            --use-full-predictions \
            --full-predictions-dir "${FULL_PREDICTIONS_DIR}"

        echo ""
        echo "[완료] Confusion matrices 생성 완료"
        echo "출력 디렉토리: ${OUTPUT_DIR}"
        ;;

    *)
        echo "[ERROR] 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "완료!"
echo "============================================================"
