# Tongu - Korean Translation System

author: songhune@ajou.ac.kr

## 주요 특징

- Broken Pipe 문제 완전 해결 - 안정적인 대용량 모델 처리
- 실시간 에러 모니터링 - songhune@jou.ac.kr 자동 알림
- 지능형 캐싱 - 번역 속도 대폭 향상
- 자동 재시작 - GPU 메모리 관리 및 Ollama 서버 자동 복구
- 통합 로깅 - 모든 활동과 에러 추적

## 사용법

  모든 명령어는 프로젝트 루트에서 실행:

  cd /home/work/songhune/korean_R-CoA/experiments/rcoa

  # 전처리
  python scripts/preprocess/data_preprocessing.py

  # 학습
  bash scripts/run_poc.sh 3

  # 평가
  bash scripts/run_poc.sh 4

  # 시각화
  python scripts/visualize/visualize_embeddings.py \
      --checkpoint checkpoints/anchor_head/best_model.pt

### 기본 사용

```bash
# 연결 테스트
python clean_tongu.py test

# 샘플 번역 (빠른 테스트)
python clean_tongu.py sample

# 파일 번역
python clean_tongu.py translate input.jsonl output.jsonl
```

### 자동 재시작 모드 (권장)

```bash
# 안정성 보장과 함께 실행
python clean_tongu.py restart sample
python clean_tongu.py restart test

# 대용량 파일 처리 시 특히 유용
python clean_tongu.py translate large_file.jsonl output.jsonl
```

## 시스템 요구사항

- Ollama 서버: `ollama serve`로 실행 중이어야 함
- 필수 모델들:
  - `jinbora/deepseek-r1-Bllossom:70b` (한국어 번역)
  - `winkefinger/alma-13b:Q4_K_M` (영어 번역)

## 파일 구조

```
tongu/
├── clean_tongu.py          # 메인 실행 파일 (모든 기능 통합)
├── tongu.log              # 실행 로그
├── tongu_cache.pkl        # 번역 캐시
└── README_CLEAN.md        # 이 파일
```

## 에러 처리 및 모니터링

### 자동 추적되는 에러

- Connection Error: Ollama 서버 연결 문제
- Broken Pipe: 대용량 모델 처리 중 연결 끊김
- Timeout Error: 응답 시간 초과
- API Error: 모델 응답 오류

### 알림 시스템

- 5분 내 5회 이상 동일 에러 발생 시 자동 알림
- 콘솔과 로그 파일에 상세 정보 기록
- songhune@jou.ac.kr로 이메일 알림 (설정 시)

## 사용 예시

### 1. 빠른 시작
```bash
# 1. 시스템 확인
python clean_tongu.py test

# 2. 샘플 실행으로 테스트
python clean_tongu.py sample

# 3. 실제 파일 번역
python clean_tongu.py translate my_data.jsonl translated_output.jsonl
```

### 2. 대용량 파일 처리 (안정성 중시)
```bash
# 자동 재시작과 함께 - 네트워크 오류나 서버 문제 시 자동 복구
python clean_tongu.py restart sample

# 진행 상황은 실시간으로 콘솔에 표시됩니다
```

## 시스템 복구

문제 발생 시 자동으로:

1. Ollama 서버 재시작: `ollama serve`
2. GPU 메모리 정리: 고사용량 감지 시 자동 정리
3. 재시도 메커니즘: 최대 3회 자동 재시도
4. 에러 알림: 임계값 초과 시 즉시 알림

## 성능 최적화

- 배치 처리: 5개씩 묶어서 효율적 번역
- 지능형 캐싱: 동일 텍스트 재번역 방지
- 비동기 처리: 한국어/영어 번역 동시 진행
- 메모리 관리: GPU 메모리 모니터링 및 자동 정리

## Release Note
2025.

---

