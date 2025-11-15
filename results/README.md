# Results Directory

모든 평가 결과가 정리된 디렉토리입니다.

## 📁 Directory Structure

```
results/
├── aggregated/                    # 통합된 결과 (주요 사용)
│   ├── consolidated_all_results.csv  # 🌟 모든 결과를 하나로 통합한 거대 CSV
│   ├── aggregated_summary.csv     # 모델별 요약 통계
│   ├── model_average_performance.csv  # 모델 평균 성능
│   └── *.png, *.pdf              # 시각화 그래프들
│
├── confusion_matrices_full/       # Confusion Matrix 결과
│   ├── confusion_matrix_*.png    # 각 모델별 confusion matrix
│   ├── *_report.txt              # 분류 성능 리포트
│   └── confusion_matrix_AVERAGE_all_models.png  # 평균 confusion matrix
│
├── full_predictions/              # 전체 예측 결과
│   └── full_predictions_*.json   # 각 모델의 전체 예측값
│
├── temperature_ablation/          # Temperature 실험 결과
│   ├── results_*_temp*.json      # 각 temperature별 결과
│   └── summary_*_temp*.csv       # Temperature별 요약
│
└── legacy/                        # 구버전 결과 (참고용)
    ├── confusion_matrices/       # 구버전 confusion matrix
    ├── data_processing/          # 데이터 처리 중간 결과
    ├── fewshot/                  # Few-shot 실험 결과
    ├── figures/                  # 구버전 그래프들
    ├── raw_evaluation/           # 원본 평가 결과
    ├── tables/                   # 구버전 테이블들
    └── temperature_ablation_old/ # 구버전 temperature 실험
```

## 🌟 Main Files

### 1. **consolidated_all_results.csv** (가장 중요!)
- **위치**: `aggregated/consolidated_all_results.csv`
- **내용**: 모든 평가 결과를 하나의 CSV로 통합
- **컬럼**:
  - `source`: 결과 출처 (temperature_ablation, full_predictions, confusion_matrix)
  - `model_name`: 모델 이름
  - `temperature`: Temperature 값 (0.0, 0.3, 0.7)
  - `task`: 태스크 이름 (classification, retrieval, punctuation, nli, translation)
  - `timestamp`: 실행 시각
  - `num_samples`: 샘플 수
  - `metric_*`: 각종 평가 지표 (accuracy, precision, recall, f1, bleu, rouge 등)

### 2. **Confusion Matrix 결과**
- **위치**: `confusion_matrices_full/`
- **파일**:
  - `confusion_matrix_*.png`: 시각화된 confusion matrix
  - `*_report.txt`: 상세 분류 성능 리포트
  - `comparison_report.txt`: 모델 간 비교
  - `confusion_matrix_AVERAGE_all_models.png`: 평균 confusion matrix

### 3. **Full Predictions**
- **위치**: `full_predictions/`
- **내용**: 각 모델의 전체 예측값 (808개 샘플)
- **모델**: GPT-4-Turbo, GPT-3.5-Turbo, Claude 3 Opus, Claude 3 Haiku, Qwen 2.5 7B, Llama 3.1 8B, EXAONE 3.0 7.8B

## 📊 Data Summary

### Evaluated Models (7)
1. GPT-4-Turbo
2. GPT-3.5-Turbo
3. Claude 3 Opus
4. Claude 3 Haiku
5. Qwen 2.5 7B Instruct
6. Llama 3.1 8B Instruct
7. EXAONE 3.0 7.8B Instruct

### Tasks (5)
1. **Classification**: 고전 문헌 문체 분류 (과문육체 6개 클래스)
2. **Retrieval**: 문헌 검색 및 매칭
3. **Punctuation**: 구두점 복원
4. **NLI**: 자연어 추론
5. **Translation**: 한문-한글 번역

### Temperature Values
- 0.0 (deterministic)
- 0.3 (balanced)
- 0.7 (creative)

## 📝 Usage

### Python에서 통합 결과 로드하기

```python
import pandas as pd

# 모든 결과 로드
df = pd.read_csv('results/aggregated/consolidated_all_results.csv')

# Classification 태스크만 필터링
classification_df = df[df['task'] == 'classification']

# Temperature 0.0 결과만 필터링
temp0_df = df[df['temperature'] == 0.0]

# 특정 모델의 결과 보기
gpt4_df = df[df['model_name'].str.contains('gpt-4', case=False)]
```

### 평가 지표 확인하기

```python
# Classification 정확도
print(df[df['task'] == 'classification']['metric_accuracy'].describe())

# Translation BLEU 점수
print(df[df['task'] == 'translation']['metric_bleu'].describe())

# 모델별 평균 성능
model_avg = df.groupby('model_name')['metric_accuracy'].mean()
print(model_avg.sort_values(ascending=False))
```

## 🗂️ Legacy Files

`legacy/` 폴더에는 이전 버전의 결과와 중간 처리 파일들이 저장되어 있습니다:
- 데이터 처리 중간 단계
- 구버전 confusion matrix
- Few-shot 실험 결과
- 구버전 그래프 및 테이블

**주의**: Legacy 파일들은 참고용이며, 최신 분석에는 `aggregated/` 폴더의 파일을 사용하세요.

## 📅 Last Updated

- Consolidated results: 2024-11-14
- Temperature ablation: 2024-11-13
- Full predictions: 2024-11-14
- Confusion matrices: 2024-11-14

## 🔗 Related Scripts

- `/experiments/exp5/consolidate_all_results.py`: 결과 통합 스크립트
- `/experiments/exp5/generate_classification_confusion_matrix.py`: Confusion matrix 생성
- `/experiments/exp5/exp5_benchmark_evaluation.py`: 벤치마크 평가

---

**Note**: 10월에 생성된 구버전 파일들은 모두 삭제되었습니다. 최신 결과만 유지됩니다.
