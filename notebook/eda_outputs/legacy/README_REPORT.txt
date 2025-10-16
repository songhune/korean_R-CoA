=== 과거사 데이터 EDA 완료 ===
입력 파일: /home/work/songhune/korean_R-CoA/data/gwashi.csv
처리 시간: 2025-09-23T14:19:50.043324

[원본 데이터]
- 총 로우/컬럼: 3348 rows × 24 cols
- 평균 유효 데이터 비율: 83.1%

[정제된 엔티티]
- 유효한 질문: 2447개
- 답변 컨텍스트: 2682개
- 시험 정보: 1692개
- 인물 정보: 1320개

[고품질 데이터셋]
- QA 샘플: 1617개 (더미데이터 0%)
- NLI 샘플: 2304개 (더미데이터 0%)

[품질 등급]
- HIGH: 19개 컬럼
- MEDIUM: 1개 컬럼
- LOW: 4개 컬럼

[출력 파일]
- 00_gwashi_head20.csv (데이터 샘플)
- 01_schema_summary.csv (완전한 품질 분석)
- 02_column_role_guess.csv (컬럼 역할 + 예시)
- 10-13_*_min.csv (정제된 엔티티 테이블)
- 20_fk_integrity_report.csv (참조무결성)
- 30-33_edge_*.csv (관계 분석)
- 40_qa_premium.jsonl (프리미엄 QA)
- 41_nli_premium.jsonl (프리미엄 NLI)
- 50_quality_report.json (종합 품질 리포트)
- 60_processing_flow.md (Mermaid 플로우 차트)