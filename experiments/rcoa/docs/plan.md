# 🎓 R-CoA Graduation Defense Plan  
**기간:** 2025.11.12 ~ 2025.12.12  
**목표:** R-CoA (Relational Chain-of-Anchor) 프레임워크 PoC 완성과 졸업발표

---

## 📆 전체 타임라인

| 주차 | 기간 | 주요 목표 |
|------|------|-----------|
| **Week 1** | 11.12 – 11.18 | 데이터 및 Anchor Head PoC 세팅 |
| **Week 2** | 11.19 – 11.25 | Anchor Head 실험 완성과 시각화 |
| **Week 3** | 11.26 – 12.02 | Chain Head 및 통합 학습 |
| **Week 4** | 12.03 – 12.12 | 발표자료·논문 패키징 및 리허설 |

---

## 🧩 Week 1 – 데이터 및 Anchor Head 구축

**🎯 목표:**  
Tongu ENG-KOR, 사서 데이터 정제 후 XLM-R 기반 Anchor Head 파이프라인 구현.

**📋 작업 항목:**  
- Tongu ENG-KOR 병렬 코퍼스 확정 (~5,000 pairs)  
- 사서 인용 체인 JSONL 형식으로 변환  
- XLM-R base + LoRA 환경 세팅 (`anchor_train.py`)  
- InfoNCE 기반 Anchor Head 학습 및 초기 성능 확인  

**💡 프롬프트 마일스톤:**  
> “XLM-R base + LoRA 구조로 InfoNCE 기반 anchor alignment를 빠르게 검증할 코드 템플릿을 만들어줘.”  
>  
> “사서 인용 데이터를 NLI-style (entailment/neutral/contradiction) pair로 변환하는 규칙을 설계해줘.”

---

## 📊 Week 2 – Anchor Head 성능 검증 및 시각화

**🎯 목표:**  
Anchor Head 성능 평가 및 시각화 자료 확보.

**📋 작업 항목:**  
- STS correlation, retrieval accuracy 측정  
- t-SNE 기반 cross-lingual embedding 시각화  
- baseline (mBERT) 대비 성능 비교  
- Anchor Head ablation 결과 테이블 정리  
- 발표용 아키텍처 인포그래픽 초안 제작  

**💡 프롬프트 마일스톤:**  
> “InfoNCE 학습 후 cross-lingual STS correlation과 retrieval accuracy를 계산하는 평가 코드 작성해줘.”  
>  
> “t-SNE 시각화를 논문용 색상 팔레트로 만들어줘 (언어별 cord 색상 고정).”

---

## 🔗 Week 3 – Chain Head 및 통합 학습

**🎯 목표:**  
TransE 기반 관계 정합(KG Head) 및 다중 홉 체인 학습(Chain Loss) 적용.

**📋 작업 항목:**  
- 사서/과시 데이터 기반 triple `(s, r, o)` 생성  
- TransE + Chain Loss 학습 추가  
- 체인 일관성(Hits@10, MRR) 평가  
- Ablation Study: Anchor only / Chain only / Full  
- 관계 시각화 (Graphviz, PyVis 등)  

**💡 프롬프트 마일스톤:**  
> “사서 본문과 주석 간 ‘인용 관계’를 (s, r, o) triple로 자동 생성하는 파이프라인을 만들어줘.”  
>  
> “TransE + Chain Loss 학습 루프에서 multi-hop chain consistency를 평가할 메트릭을 구현해줘.”

---

## 🧠 Week 4 – 발표자료 및 논문 패키징

**🎯 목표:**  
PoC 결과 정리, 발표 슬라이드 완성, 논문 초안 제출.

**📋 작업 항목:**  
- Marp 슬라이드 및 PDF 발표자료 완성  
- 실험 결과 및 그래프 포함한 `results.md` 작성  
- Colab / Notebook 시연 환경 정리  
- 지도교수 피드백 반영 및 리허설  
- 졸업발표 12.12 제출  

**💡 프롬프트 마일스톤:**  
> “Marp 발표 슬라이드에 넣을 인포그래픽형 아키텍처 그림을 개선해줘 (Figma 스타일).”  
>  
> “논문 Conclusion 섹션을 ‘PoC 성과 + 한계 + 향후 연구’ 구조로 정리해줘.”

---

## 📁 산출물 구조 (제안)