# 🐲 Tongu - Korean Translation System (Organized)

한국어 번역 시스템이 체계적으로 재구성되었습니다.

## 📁 프로젝트 구조

```
tongu/
├── 🔧 tongu_main.py              # 깔끔한 메인 진입점
├── 📜 main.py                    # 기존 호환 진입점
│
├── 🏗️ core/                     # 핵심 시스템
│   ├── config/
│   │   └── config.py            # 설정 관리
│   └── translation/
│       └── translator.py        # 메인 번역 엔진
│
├── 🔌 api/                      # API 및 처리
│   ├── clients/
│   │   └── api_clients.py       # API 클라이언트들
│   └── processors/
│       └── text_processor.py    # 텍스트 처리
│
├── 🛠️ utils/                   # 유틸리티
│   ├── cache/
│   │   └── cache_manager.py     # 캐시 관리
│   ├── cost/
│   │   └── cost_tracker.py      # 비용 추적
│   └── file_ops/
│       └── file_handlers.py     # 파일 처리
│
├── 📊 monitoring/               # 모니터링 및 오류 처리
│   ├── errors/
│   │   └── error_monitor.py     # 오류 분석
│   └── notifications/
│       └── error_notifier.py    # 알림 시스템
│
├── 🚀 scripts/                 # 실행 스크립트
│   ├── restart/
│   │   ├── auto_restart.sh      # Bash 자동 재시작
│   │   └── restart_optimized.py # Python GPU 최적화 재시작
│   └── maintenance/
│       └── resume_translation.py # 번역 재개
│
└── 🧪 tests/                   # 테스트 파일들
    └── test_batch_vs_individual.py
```

## 🚀 사용법

### 1. 간단한 사용 (권장)

```bash
# 새로운 깔끔한 인터페이스 사용
python tongu_main.py test     # 연결 테스트
python tongu_main.py sample   # 샘플 번역
python tongu_main.py accn     # ACCN 데이터셋 번역
python tongu_main.py translate input.jsonl output.jsonl
```

### 2. 기존 호환성

```bash
# 기존 방식도 계속 작동
python main.py test
python main.py sample
python main.py accn
python main.py process input.jsonl output.jsonl
```

### 3. 자동 재시작과 함께 (안정성)

```bash
# Bash 스크립트 (권장)
./scripts/restart/auto_restart.sh sample
./scripts/restart/auto_restart.sh accn

# Python GPU 최적화 스크립트  
python scripts/restart/restart_optimized.py sample
python scripts/restart/restart_optimized.py accn
```

## 🔧 주요 개선사항

### 1. **체계적인 모듈 구조**
- 기능별로 명확하게 분리된 패키지
- 깔끔한 import 경로
- 재사용 가능한 컴포넌트

### 2. **Broken Pipe 문제 해결**
- 연결 재시도 메커니즘
- 타임아웃 최적화 (5분)
- GPU 메모리 관리

### 3. **에러 모니터링 및 알림**
- 실시간 에러 추적
- songhune@jou.ac.kr 자동 알림
- 임계값 기반 알림 (5회/5분)

### 4. **자동 재시작 시스템**
- 다양한 에러 타입별 대응
- GPU 메모리 정리
- 최대 5회 자동 재시도

## 📦 패키지별 설명

### Core 패키지
- `TranslationConfig`: 번역 설정 관리
- `LargeScaleTranslator`: 메인 번역 엔진

### API 패키지
- `APIClientFactory`: API 클라이언트 팩토리
- `ACCNDataProcessor`: 데이터 처리

### Utils 패키지
- `TranslationCache`: 번역 캐시
- `CostTracker`: 비용 추적
- `FileHandler`: 파일 처리

### Monitoring 패키지
- `ErrorNotifier`: 에러 알림 시스템
- `analyze_translation_errors`: 오류 분석

## 🔄 Migration Guide

기존 코드에서 새 구조로 이동하는 방법:

```python
# 기존
from translator import LargeScaleTranslator
from config import TranslationConfig

# 새로운 방식
from core import LargeScaleTranslator, TranslationConfig
# 또는
from tongu_main import TonguTranslator  # 더 간단함
```

## 🛡️ 에러 처리 및 모니터링

### 자동 추적되는 에러 유형
- Broken Pipe 에러
- Ollama 연결 에러  
- 모델 응답 에러
- 배치 처리 실패

### 알림 기능
- 5회 이상 에러 발생 시 자동 이메일 알림
- 30분 cooldown 기간
- 상세한 에러 컨텍스트 정보

## 🎯 Quick Start

```bash
# 1. 시스템 테스트
python tongu_main.py test

# 2. 샘플 실행
python tongu_main.py sample

# 3. 실제 번역 (자동 재시작)
./scripts/restart/auto_restart.sh accn
```

모든 기존 기능은 그대로 유지되면서 더 체계적이고 안정적인 구조로 개선되었습니다!