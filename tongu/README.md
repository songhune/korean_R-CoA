# KEadapter - 대용량 고전 중국어 번역기

ACCN-INS와 같은 대용량 고전 중국어 데이터셋을 한국어와 영어로 번역하는 비동기 번역 도구입니다.

## 주요 기능

- **비동기 배치 처리**: 높은 성능의 병렬 번역
- **비용 관리**: 실시간 비용 추적 및 예산 제한
- **스마트 캐싱**: 중복 번역 방지로 비용 절약
- **체크포인트 시스템**: 중단 시 재시작 가능
- **다중 API 지원**: OpenAI, Anthropic Claude API
- **상세 로깅**: 처리 과정 모니터링

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export ANTHROPIC_API_KEY="your-claude-api-key"
# 또는
export OPENAI_API_KEY="your-openai-api-key"
```

## 사용법

### 1. 샘플 테스트
```bash
python main.py sample
```

### 2. 비용 추정
```bash
python main.py estimate your_data.jsonl
```

### 3. 실제 번역 처리
```bash
python main.py process input.jsonl output.jsonl
```

### 4. ACCN-INS 데이터셋 처리
```bash
python main.py accn
```

## 모듈 구조

```
tongu/
├── __init__.py          # 패키지 진입점
├── main.py             # 메인 실행 스크립트
├── config.py           # 설정 및 구성 관리
├── translator.py       # 메인 번역기 클래스
├── api_clients.py      # API 클라이언트들
├── cost_tracker.py     # 비용 추적 및 관리
├── cache_manager.py    # 번역 캐시 관리
├── file_handlers.py    # 파일 입출력 처리
├── text_processor.py   # 텍스트 전처리
└── requirements.txt    # 의존성 목록
```

## 데이터 형식

### 입력 (JSONL)
```json
{
  "task": "Classical Chinese to Modern Chinese",
  "data": {
    "instruction": "请将迁骑都尉、光禄大夫、侍中。翻译为现代汉语。",
    "input": "",
    "output": "又升任骑都尉光禄大夫侍中。",
    "history": []
  }
}
```

### 출력 (JSONL)
```json
{
  "task": "Classical Chinese to Modern Chinese",
  "data": {
    "instruction": "请将迁骑都尉、光禄大夫、侍中。翻译为现代汉语。",
    "input": "",
    "output": "又升任骑都尉光禄대부사중。",
    "history": []
  },
  "korean_translation": "기도위, 광록대부, 시중으로 승진하였다.",
  "english_translation": "Promoted to cavalry commander, grand minister, and imperial attendant.",
  "original_classical_text": "迁骑都尉、光禄大夫、侍中。",
  "multilingual_enhanced": true
}
```

## 설정 옵션

```python
config = TranslationConfig(
    api_provider="anthropic",  # "openai" 또는 "anthropic"
    model="claude-3-haiku-20240307",
    batch_size=30,            # 배치 크기
    max_concurrent=6,         # 동시 처리 수
    delay_between_batches=0.8, # 배치 간 지연 시간
    chunk_size=8000,          # 청크 크기
    checkpoint_interval=500,   # 체크포인트 간격
    budget_limit=50.0         # 예산 제한 ($)
)
```

## 비용 정보

### Anthropic Claude (권장)
- **claude-3-haiku**: $0.25/1M input tokens, $1.25/1M output tokens
- 높은 품질의 번역, 비용 효율적

### OpenAI
- **gpt-3.5-turbo**: $1.00/1M input tokens, $2.00/1M output tokens  
- **gpt-4**: $30.00/1M input tokens, $60.00/1M output tokens

## 주의사항

1. **API 키 보안**: 환경변수로 API 키 관리
2. **예산 관리**: 예상 비용을 미리 확인
3. **네트워크 안정성**: 인터넷 연결 상태 확인
4. **파일 백업**: 원본 데이터 백업 권장

## 문제 해결

### 자주 발생하는 오류

1. **API 키 오류**
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. **메모리 부족**
   - `chunk_size`와 `batch_size` 감소

3. **API 제한**
   - `delay_between_batches` 증가

4. **파일 인코딩 오류**
   - UTF-8 인코딩 확인

## 문의

한승현, songhune@ajou.ac.kr