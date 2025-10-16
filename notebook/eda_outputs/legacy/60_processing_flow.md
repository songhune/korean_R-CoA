
# 과거사 데이터 EDA 프로세싱 플로우

```mermaid
flowchart TD
    A[원본 gwashi.csv<br/>3348 rows × 24 cols] --> B{데이터 품질 검사}
    
    B --> C[NaN/빈값 탐지]
    B --> D[더미 데이터 탐지]
    B --> E[의미있는 텍스트 검증]
    
    C --> F[컬럼별 품질 등급<br/>HIGH/MEDIUM/LOW]
    D --> G[Question_XXX, answer_XXX<br/>패턴 제거]
    E --> H[한글/한자 포함<br/>최소 길이 검증]
    
    F --> I[스키마 요약 생성<br/>01_schema_summary.csv]
    G --> J[텍스트 정제<br/>clean_text() 함수]
    H --> K[의미있는 텍스트 필터<br/>is_meaningful_text()]
    
    I --> L[컬럼 역할 분류<br/>02_column_role_guess.csv]
    J --> M[엔티티 테이블 생성]
    K --> M
    
    M --> N[Person 테이블<br/>실제 데이터 기반]
    M --> O[Exam 테이블<br/>시험정보 추출]
    M --> P[Question 테이블<br/>유효한 질문만]
    M --> Q[Answer 테이블<br/>컨텍스트 기반]
    
    N --> R[FK 무결성 검사<br/>20_fk_integrity_report.csv]
    O --> R
    P --> R
    Q --> R
    
    R --> S[Edge 관계 분석<br/>30-33_edge_*.csv]
    
    P --> T[프리미엄 QA 생성<br/>실제 데이터만]
    Q --> T
    
    T --> U[품질 필터링]
    U --> V[길이 검증 >= 10자]
    V --> W[한글/한자 포함 확인]
    W --> X[더미 데이터 완전 제거]
    
    X --> Y[40_qa_premium.jsonl<br/>고품질 QA 데이터]
    X --> Z[41_nli_premium.jsonl<br/>고품질 NLI 데이터]
    
    Y --> AA[최종 품질 리포트<br/>50_quality_report.json]
    Z --> AA
    S --> AA
    
    style A fill:#e1f5fe
    style AA fill:#c8e6c9
    style Y fill:#fff3e0
    style Z fill:#fff3e0
```

## 주요 개선사항

### 1. 완전한 더미 데이터 제거
- `Question_XXX`, `answer_XXX`, `default_answer` 패턴 완전 제거
- 정규표현식 기반 무효값 탐지
- 의미있는 텍스트만 추출

### 2. 텍스트 품질 검증
- 한글/한자 포함 여부 확인
- 최소 길이 요구사항 (10자 이상)
- 실제 내용이 있는지 semantic 검증

### 3. 실제 데이터 기반 엔티티 생성  
- 원본 CSV의 실제 시험 정보 활용
- question-exam 관계를 연도 기반으로 매칭
- person 정보 부재시에만 최소한의 더미 생성

### 4. 프리미엄 품질 데이터셋
- 모든 샘플이 실제 의미있는 내용
- 한국 고전/역사 도메인 특화
- NLI 라벨링의 신뢰성 확보
