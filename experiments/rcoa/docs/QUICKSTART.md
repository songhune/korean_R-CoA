# R-CoA Quick Start Guide

**졸업 발표 준비용 빠른 실행 가이드**

## 📅 타임라인 (4주)

- **Week 1 (11.12-11.18)**: ✅ Anchor Head 구현 완료
- **Week 2 (11.19-11.25)**: 성능 평가 & 시각화
- **Week 3 (11.26-12.02)**: Chain Head 추가
- **Week 4 (12.03-12.12)**: 발표자료 & 논문

---

## 🚀 즉시 실행 (Week 1)

### 1분 Quick Test
```bash
cd /home/work/songhune/korean_R-CoA/experiments/rcoa
./run_poc.sh 2  # 1 epoch, 빠른 검증 (~5분)
```

### Full Training (추천)
```bash
./run_poc.sh 3  # 10 epochs (~2-3시간)
```

### 평가
```bash
./run_poc.sh 4  # Recall@10, MRR, Cosine Similarity
```

---

## 📊 Week 2: 시각화 & 분석

### 임베딩 시각화
```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/anchor_head/best_model.pt \
    --data data/val_pairs.jsonl \
    --output-dir figures/week2
```

**생성 파일:**
- `figures/week2/tsne_cross_lingual.png` - t-SNE 시각화
- `figures/week2/similarity_heatmap.png` - 유사도 히트맵

### Baseline 비교 (TODO)
- [ ] Frozen XLM-R (no training)
- [ ] Full fine-tuning (without LoRA)
- [ ] Random embeddings

---

## 🔗 Week 3: Chain Head

### KG Triple 데이터 준비 (TODO)
```python
# Saseo 인용 관계 추출
triples = [
    ('논어_1.1', 'cites', '맹자_3.4'),
    ('논어_2.3', 'cites', '대학_1.2'),
    ...
]
```

### Chain Head 학습 (TODO)
```bash
python chain_train.py \
    --anchor-checkpoint checkpoints/anchor_head/best_model.pt \
    --kg-triples data/saseo_triples.json \
    --output-dir checkpoints/chain_head
```

---

## 📈 Week 4: 발표자료

### 생성할 자료
1. **Marp 슬라이드** (rcoa_concept.md 기반)
2. **실험 결과 요약**
   - Recall@10, MRR 테이블
   - t-SNE 시각화
   - Ablation study 그래프
3. **Demo Notebook**
   - 인터랙티브 예제
   - Anchor retrieval 데모

---

## 📁 현재 파일 구조

```
rcoa/
├── QUICKSTART.md           ⭐ 이 파일
├── README.md               📖 전체 문서
├── plan.md                 📅 4주 계획
├── rcoa_concept.md         💡 R-CoA 컨셉 (Marp 슬라이드)
│
├── 🔧 Core Implementation (Week 1 완료)
│   ├── data_preprocessing.py
│   ├── anchor_head_model.py     # Anchor Head + InfoNCE
│   ├── anchor_train.py
│   ├── anchor_evaluate.py
│   └── run_poc.sh
│
├── 📊 Visualization (Week 2)
│   ├── visualize_embeddings.py  # t-SNE, heatmap
│   └── plot_results.py          # (TODO) 성능 그래프
│
├── 🔗 Chain Head (Week 3)
│   ├── chain_head_model.py      # TransE + Chain Loss
│   ├── chain_train.py           # (TODO) 학습 스크립트
│   └── kg_data_prep.py          # (TODO) Triple 추출
│
├── 📑 Automation
│   └── run_full_pipeline.sh     # Week 1-4 자동화
│
├── 📂 Generated (실행 후 생성)
│   ├── data/
│   │   ├── train_pairs.jsonl    # 18,826 pairs
│   │   └── val_pairs.jsonl      # 2,091 pairs
│   ├── checkpoints/
│   │   └── anchor_head/
│   │       ├── best_model.pt
│   │       └── checkpoint_epoch*.pt
│   ├── results/
│   │   └── evaluation_results.json
│   └── figures/
│       └── tsne_cross_lingual.png
```

---

## ✅ Checklist

### Week 1 (완료!)
- [x] 데이터 전처리 (20,917 pairs)
- [x] Anchor Head 구현 (XLM-R + LoRA + InfoNCE)
- [x] 학습 스크립트
- [x] 평가 스크립트
- [x] README & 문서

### Week 2 (진행중)
- [x] t-SNE 시각화 코드
- [ ] Baseline 비교 실험
- [ ] Ablation study (LoRA rank, temperature)
- [ ] 성능 그래프 자동 생성

### Week 3 (준비됨)
- [x] Chain Head 모델 코드 (TransE + Chain Loss)
- [ ] KG triple 데이터 추출
- [ ] Chain Head 학습 스크립트
- [ ] 통합 모델 평가

### Week 4 (TODO)
- [ ] Marp 슬라이드 완성
- [ ] 실험 결과 정리
- [ ] Demo notebook
- [ ] 발표 리허설

---

## 🎯 핵심 실험 결과 목표

| Metric | Target | Baseline |
|--------|--------|----------|
| Recall@10 | > 0.70 | ~0.40 |
| MRR | > 0.50 | ~0.25 |
| Cosine Sim | > 0.60 | ~0.30 |

**Baseline**: Frozen XLM-R (no training)

---

## 🐛 Troubleshooting

### GPU 메모리 부족
```bash
# Batch size 줄이기
python anchor_train.py --batch-size 16  # 기본 32에서
```

### 학습 느림
```bash
# Epoch 수 줄이기 (빠른 테스트)
python anchor_train.py --epochs 3
```

### Dependencies 설치
```bash
pip install -r requirements.txt
```

---

## 📞 다음 단계

1. **지금 바로**: `./run_poc.sh 3` 실행 (Full training)
2. **학습 중**: Week 2-3 코드 테스트
3. **학습 완료 후**:
   - 평가 실행 (`./run_poc.sh 4`)
   - 시각화 생성 (`visualize_embeddings.py`)
4. **다음 주**: Chain Head 데이터 준비 & 학습

---

## 📧 문의

- 코드 이슈: GitHub Issues
- 빠른 질문: songhune@ajou.ac.kr

**화이팅! 🚀**
