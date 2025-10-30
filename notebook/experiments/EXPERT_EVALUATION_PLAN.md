# KLSBench Expert Evaluation Plan

A comprehensive plan for obtaining validation from classical Chinese literature experts (한문학 전공자).

## Table of Contents
1. [Objectives](#objectives)
2. [Target Experts](#target-experts)
3. [Evaluation Phases](#evaluation-phases)
4. [Materials to Prepare](#materials-to-prepare)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Timeline](#timeline)
7. [Success Metrics](#success-metrics)

---

## Objectives

### Primary Goals
1. **Validate benchmark quality**: Confirm that tasks and data accurately reflect classical Chinese literature understanding
2. **Verify label correctness**: Validate classification labels (19 classes) and task annotations
3. **Assess task difficulty**: Determine if tasks appropriately measure expertise levels
4. **Identify improvements**: Gather feedback for benchmark refinement

### Secondary Goals
1. Establish credibility for academic publication
2. Build network with domain experts
3. Gather expert-level baseline performance data
4. Identify culturally/historically important edge cases

---

## Target Experts

### Profile Requirements

**Primary Target** (5-10 experts):
- PhD in Classical Chinese Literature, Korean Classical Literature, or East Asian Studies
- Active researchers or professors
- Specialization in:
  - Joseon Dynasty literature (조선시대 문학)
  - Four Books (四書: 論語, 孟子, 大學, 中庸)
  - Gwageo examination system (과거시험)
  - Classical Chinese poetry and prose (한시, 산문)

**Secondary Target** (10-20 experts):
- Graduate students (MA/PhD) in relevant fields
- Teachers of classical Chinese (한문 교사)
- Researchers at institutes (한국학중앙연구원, 한국고전번역원)

### Recruitment Strategy

**Academic Institutions**:
- 서울대학교 한문학과, 중어중문학과
- 고려대학교, 연세대학교 한문학과
- 성균관대학교 한문학과 (전통적 강점)
- 한국학중앙연구원
- 한국고전번역원

**Approach**:
1. Email to department chairs requesting participation
2. Present at relevant conferences/seminars
3. Leverage existing academic networks
4. Offer co-authorship on validation paper for key contributors
5. Provide honorarium for extensive evaluation (50-100k KRW per expert)

---

## Evaluation Phases

### Phase 1: Pilot Evaluation (2-3 experts, 2 weeks)

**Objectives**:
- Test evaluation methodology
- Identify major issues early
- Refine evaluation materials

**Tasks**:
- Sample evaluation (50 items per task)
- Interview for qualitative feedback
- Iterative refinement

**Deliverables**:
- Revised evaluation protocol
- Initial feedback report
- Refined benchmark (if needed)

### Phase 2: Full Expert Evaluation (5-10 experts, 1 month)

**Objectives**:
- Comprehensive validation
- Inter-annotator agreement analysis
- Statistical validation

**Tasks**:
- Full task evaluation (stratified sample: 10-30% per task)
- Structured questionnaire
- Difficulty rating
- Error analysis

**Deliverables**:
- Expert validation report
- Inter-annotator agreement scores (Fleiss' kappa)
- Benchmark quality metrics

### Phase 3: Expert Baseline (3-5 experts, 2 weeks)

**Objectives**:
- Establish human expert performance
- Compare LLM vs. human performance
- Identify challenging items

**Tasks**:
- Timed evaluation on test set
- Think-aloud protocol (optional)
- Error pattern analysis

**Deliverables**:
- Human baseline scores
- Human vs. LLM comparison
- Qualitative insights on AI limitations

---

## Materials to Prepare

### 1. Executive Summary (Korean)

**Content** (2-3 pages):
```markdown
# KLSBench: 한국 고전 문헌 이해도 벤치마크

## 개요
- 연구 배경 및 목적
- 벤치마크 구성 (5개 태스크, 7,871 항목)
- 데이터 출처 (과거시험, 사서)
- AI 모델 평가 결과 요약

## 평가 의뢰 사항
- 벤치마크 데이터 품질 검증
- 분류 체계 타당성 평가
- 난이도 적절성 평가
- 개선 제안
```

### 2. Detailed Task Description (Korean)

**For each task**, provide:

**Classification (문체 분류)**:
```markdown
### 태스크 설명
주어진 한문 텍스트의 문체를 19개 카테고리로 분류

### 분류 체계 (19개 라벨)
- 균형 클래스 (95개): 賦, 詩, 疑, 義, 策, 表
- 기타 클래스: 論(53), 銘(53), 箋(49), 頌(24), 禮義(13), 箴(12), 易義(9), 詩義(7), 書義(6), 詔(5), 制(3), 講(2), 擬(2)

### 평가 요청 사항
1. 라벨 체계가 적절한가?
2. 각 샘플의 라벨이 정확한가?
3. 혼동하기 쉬운 카테고리는?
4. 제안 사항
```

**Retrieval (출처 식별)**:
```markdown
### 태스크 설명
사서(論語, 孟子, 大學, 中庸) 문장의 출처 식별

### 평가 요청 사항
1. 출처 정보가 정확한가?
2. 난이도가 적절한가?
3. 오류가 있는 항목은?
```

**Similar for other tasks...**

### 3. Sample Data for Review

**Stratified Sample**:
```python
# Per task sampling strategy
classification: 80 items (10% of 808)
  - Balanced classes: 10 each (60 items)
  - Other classes: 2-5 each (20 items)

retrieval: 120 items (10% of 1,209)
  - By source book: 30 each (論語, 孟子, 大學, 中庸)

punctuation: 200 items (10% of 2,000)
  - By difficulty: Easy(50), Medium(100), Hard(50)

nli: 180 items (10% of 1,854)
  - By label: 60 each (entailment, contradiction, neutral)

translation: 200 items (10% of 2,000)
  - By direction: Classical Chinese→Korean(100), Korean→English(50), etc.
```

### 4. Evaluation Interface

**Option A: Web-based Interface** (Recommended)
```
Features:
- Clean, intuitive UI
- Task-by-task evaluation
- Save/resume capability
- Real-time validation
- Export results

Technology:
- Streamlit or Gradio (Python)
- Simple deployment (Hugging Face Spaces)
```

**Option B: Excel/Google Sheets**
```
Pros:
- Familiar interface
- Easy to distribute
- Offline capability

Cons:
- Manual data processing
- Less user-friendly
```

**Option C: PDF + Response Form**
```
Use for:
- Small pilot group
- Quick feedback
```

### 5. Evaluation Questionnaire

**Per Task**:
```markdown
## 1. 데이터 품질 (5-point Likert scale)
- 데이터의 정확성: 1(매우 부정확) - 5(매우 정확)
- 라벨의 일관성: 1(일관성 없음) - 5(매우 일관적)
- 데이터의 대표성: 1(비대표적) - 5(매우 대표적)

## 2. 난이도 평가
- 전체 난이도: 1(매우 쉬움) - 5(매우 어려움)
- 전공자에게 적절한가?: 예/아니오
- 일반인에게는?: 1(불가능) - 5(가능)

## 3. 오류 지적
항목 ID, 오류 내용, 제안 사항

## 4. 분류 체계 (Classification만 해당)
- 19개 카테고리가 적절한가?: 예/아니오
- 추가/제거/병합할 카테고리: (자유 기술)

## 5. 종합 의견
- 강점
- 약점
- 개선 제안
- 추가 제안
```

### 6. IRB/Ethics Approval

**If collecting personal data**:
- Consent form
- Data protection plan
- IRB approval (if affiliated with institution)

---

## Evaluation Methodology

### Quantitative Metrics

**1. Inter-Annotator Agreement**
```python
# Fleiss' kappa for multiple annotators
# Target: κ > 0.75 (substantial agreement)

from statsmodels.stats.inter_rater import fleiss_kappa

# For each task
kappa_classification = fleiss_kappa(ratings_matrix)
kappa_nli = fleiss_kappa(ratings_matrix)
# ...
```

**2. Label Accuracy Validation**
```python
# Expert consensus vs. current labels
# Threshold: 95% agreement

accuracy = correct_labels / total_labels
if accuracy < 0.95:
    # Identify problematic items
    # Revise benchmark
```

**3. Difficulty Rating**
```python
# Average difficulty score (1-5 scale)
# Expected: 3.0-4.0 (moderate to challenging)

mean_difficulty = np.mean(difficulty_ratings)
std_difficulty = np.std(difficulty_ratings)
```

### Qualitative Analysis

**1. Thematic Coding**
```
Open-ended responses → Code categories:
- Data quality issues
- Label disagreements
- Task design suggestions
- Domain-specific insights
```

**2. Error Pattern Analysis**
```
Common error types:
- Mislabeling patterns
- Ambiguous cases
- Historical context issues
- Translation inconsistencies
```

**3. Expert Interviews**
```
Semi-structured interviews (30-60 min):
- Overall impressions
- Specific concerns
- Suggestions for improvement
- Potential use cases
```

---

## Timeline

### Month 1: Preparation
- Week 1-2: Prepare materials (executive summary, samples, interface)
- Week 3: Recruit pilot experts (2-3)
- Week 4: Pilot evaluation

### Month 2: Pilot & Revision
- Week 1-2: Analyze pilot results
- Week 2-3: Revise benchmark based on feedback
- Week 4: Recruit main evaluation experts (5-10)

### Month 3: Main Evaluation
- Week 1-3: Expert evaluation period
- Week 4: Data collection and initial analysis

### Month 4: Analysis & Reporting
- Week 1-2: Statistical analysis
- Week 2-3: Qualitative analysis
- Week 4: Write validation report

**Total: 4 months**

---

## Success Metrics

### Tier 1: Essential (Must Achieve)
✅ **Inter-annotator agreement**: κ > 0.70 (substantial)
✅ **Label accuracy**: >90% expert consensus
✅ **Expert participation**: 5+ PhD-level experts
✅ **Sample coverage**: >10% per task evaluated

### Tier 2: Desired (Should Achieve)
⭐ **Inter-annotator agreement**: κ > 0.80 (almost perfect)
⭐ **Label accuracy**: >95% expert consensus
⭐ **Expert participation**: 8+ experts
⭐ **Qualitative feedback**: Rich insights for improvement

### Tier 3: Aspirational (Nice to Have)
🎯 **Inter-annotator agreement**: κ > 0.85
🎯 **Expert participation**: 10+ experts
🎯 **Publication**: Co-authored validation paper
🎯 **Baseline**: Human expert performance data

---

## Budget Estimate

### Honorarium (50-100k KRW per expert)
- Pilot (3 experts × 50k): 150,000 KRW
- Main evaluation (8 experts × 100k): 800,000 KRW
- Baseline (3 experts × 150k): 450,000 KRW
- **Subtotal**: 1,400,000 KRW (~$1,000 USD)

### Interface Development
- Web interface (Streamlit/Gradio): 500,000 KRW (or free if DIY)
- **Subtotal**: 500,000 KRW

### Miscellaneous
- Transcription/translation: 300,000 KRW
- Meeting expenses: 200,000 KRW
- **Subtotal**: 500,000 KRW

### **Total: ~2,400,000 KRW (~$1,800 USD)**

---

## Risk Mitigation

### Risk 1: Low Expert Participation
**Mitigation**:
- Start recruitment early
- Offer appropriate compensation
- Leverage institutional connections
- Present at conferences

### Risk 2: Major Quality Issues Found
**Mitigation**:
- Pilot evaluation catches issues early
- Iterative refinement process
- Budget time for revisions

### Risk 3: Low Inter-Annotator Agreement
**Mitigation**:
- Clear annotation guidelines
- Training session before evaluation
- Allow discussion and consensus building
- Refine ambiguous items

### Risk 4: Timeline Delays
**Mitigation**:
- Buffer time in schedule (4 months total)
- Rolling recruitment
- Flexible evaluation deadlines

---

## Deliverables

### Academic Deliverables
1. **Expert Validation Report** (Korean + English)
   - Methodology
   - Quantitative results (IAA, accuracy)
   - Qualitative findings
   - Recommendations

2. **Revised Benchmark** (if needed)
   - Corrected labels
   - Removed/modified problematic items
   - Enhanced documentation

3. **Technical Paper** (for publication)
   - "KLSBench: A Validated Benchmark for Classical Chinese Understanding"
   - Include expert validation as key contribution
   - Target venues: ACL, EMNLP, LREC, or domain-specific journals

### Community Deliverables
1. **Public Dataset** with validation metadata
2. **Expert Evaluation Tool** (open-source)
3. **Best Practices Guide** for benchmark validation

---

## Next Steps (Immediate Actions)

### Week 1-2: Material Preparation
- [ ] Write executive summary (Korean)
- [ ] Prepare detailed task descriptions
- [ ] Create sample dataset (stratified)
- [ ] Draft evaluation questionnaire
- [ ] Design evaluation interface (prototype)

### Week 3: Expert Outreach
- [ ] Identify target experts (list 20+)
- [ ] Draft recruitment email
- [ ] Prepare presentation slides
- [ ] Contact department chairs

### Week 4: Pilot Setup
- [ ] Finalize pilot materials
- [ ] Recruit 2-3 pilot experts
- [ ] Schedule pilot evaluation
- [ ] Prepare consent forms

---

## Contact & Follow-up

### For Experts
**Email template** (Korean):
```
제목: [협조 요청] 한국 고전 문헌 AI 벤치마크 검증 참여

안녕하십니까,

저희는 대규모 언어 모델의 한국 고전 문헌 이해 능력을 평가하기 위한
KLSBench 벤치마크를 개발하였습니다.

본 연구의 학술적 타당성을 확보하기 위해 한문학 전공 교수님들의
전문적 검증을 요청드립니다.

- 평가 대상: 과거시험 및 사서(四書) 기반 5개 태스크
- 소요 시간: 2-3시간 (온라인 평가)
- 사례금: 10만원
- 기여 인정: 논문 감사의 글 또는 공동저자 (기여도에 따라)

관심 있으신 경우, 상세 자료를 보내드리겠습니다.

감사합니다.
```

### For Follow-up
- Weekly progress updates
- Thank you notes after evaluation
- Share results with participants
- Acknowledge contributions in publications

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Planning Phase
