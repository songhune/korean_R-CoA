# R-CoA Memory Optimization Guide

H100 GPU에서 대형 모델을 효율적으로 학습하기 위한 메모리 최적화 가이드입니다.

## 🚀 적용된 최적화 기법

### 1. **Automatic Mixed Precision (AMP)**

**BF16 (Brain Float 16) 사용:**
```python
# BF16은 H100에 최적화되어 있음
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(batch)
```

**효과:**
- 메모리 사용량: **~50% 감소**
- 학습 속도: **~2배 향상**
- 수치 안정성: FP16보다 우수 (더 넓은 dynamic range)

### 2. **Gradient Checkpointing**

```python
model.encoder.gradient_checkpointing_enable()
```

**효과:**
- 메모리 사용량: **~40% 감소**
- 학습 속도: ~20% 느려짐 (trade-off)
- Activation을 저장하지 않고 필요 시 재계산

### 3. **Gradient Accumulation**

```bash
--gradient-accumulation-steps 2
```

**효과:**
- 실질적인 batch size: `batch_size × accumulation_steps`
- 메모리: 작은 batch로 실행하면서 큰 batch 효과
- 예: batch_size=64, accumulation=2 → effective batch=128

### 4. **LoRA (Low-Rank Adaptation)**

```python
# 전체 모델이 아닌 소수의 파라미터만 학습
lora_config = LoraConfig(r=32, alpha=64)
```

**효과:**
- 학습 파라미터: **~1% of total**
- 메모리 사용량: **대폭 감소**
- 학습 속도: 빠름

---

## 📊 메모리 사용량 비교

| 모델 | 기본 (FP32) | + AMP | + Checkpointing | + LoRA | 최종 |
|------|------------|-------|-----------------|--------|------|
| **Large (550M)** | ~12GB | ~6GB | ~4GB | ~3GB | **~3GB** |
| **XL (3.5B)** | ~42GB | ~21GB | ~13GB | ~10GB | **~10GB** |
| **XXL (10.7B)** | ~128GB | ~64GB | ~38GB | ~28GB | **~28GB** |

---

## 🎯 최적 설정 (H100 79GB)

### XLM-RoBERTa-Large (550M)
```bash
bash scripts/run_poc_h100.sh large

# 메모리: ~3GB
# Batch Size: 256
# 예상 시간: 2-3시간
```

### XLM-RoBERTa-XL (3.5B) ⭐ 추천
```bash
bash scripts/run_poc_h100.sh xl

# 메모리: ~10GB
# Batch Size: 192
# 예상 시간: 6-8시간
```

### XLM-RoBERTa-XXL (10.7B)
```bash
bash scripts/run_poc_h100.sh xxl

# 메모리: ~28GB
# Batch Size: 128
# 예상 시간: 12-15시간
```

---

## 🔧 커스텀 설정

### Batch Size 조정

```bash
python scripts/train/anchor_train.py \
    --model-name facebook/xlm-roberta-xl \
    --batch-size 128 \              # OOM 발생 시 줄이기
    --gradient-accumulation-steps 4 \ # Effective batch=512
    --use-amp \
    --gradient-checkpointing
```

### 메모리가 부족할 때

1. **Batch size 줄이기**
   ```bash
   --batch-size 64  # 기본 192에서
   ```

2. **Gradient accumulation 늘리기**
   ```bash
   --gradient-accumulation-steps 4  # 기본 2에서
   ```

3. **LoRA rank 줄이기**
   ```bash
   --lora-r 16  # 기본 32에서
   ```

4. **Projection dim 줄이기**
   ```bash
   --projection-dim 512  # 기본 768에서
   ```

### 메모리가 충분할 때

1. **Batch size 늘리기**
   ```bash
   --batch-size 256
   ```

2. **LoRA rank 늘리기**
   ```bash
   --lora-r 64
   --lora-alpha 128
   ```

3. **Projection dim 늘리기**
   ```bash
   --projection-dim 1024
   ```

---

## 🧮 메모리 계산 공식

### 총 메모리 요구량
```
Total = Model + Optimizer + Gradients + Activations + Batch

Model:        params × 2 bytes (BF16)
Optimizer:    params × 8 bytes (AdamW states)
Gradients:    params × 2 bytes (BF16)
Activations:  batch_size × seq_len × hidden × layers × 2
Batch Data:   batch_size × seq_len × 2
```

### 최적화 후
```
Model:        params × 2 × lora_ratio  (~1%)
Optimizer:    params × 8 × lora_ratio
Gradients:    params × 2 × lora_ratio
Activations:  batch_size × seq_len × hidden × sqrt(layers)  (checkpointing)
```

---

## 📈 성능 vs 메모리 Trade-off

| 설정 | 메모리 | 속도 | 성능 |
|------|--------|------|------|
| **Full Precision (FP32)** | 100% | 50% | 100% |
| **+ AMP (BF16)** | 50% | 100% | 99% |
| **+ Gradient Checkpointing** | 30% | 80% | 99% |
| **+ LoRA** | 15% | 90% | 98% |
| **All Combined** ⭐ | **15%** | **90%** | **98%** |

---

## 🐛 Troubleshooting

### OOM 에러 발생 시

```bash
# 1단계: Batch size 절반으로
--batch-size 96  # XL의 경우

# 2단계: Gradient accumulation 2배로
--gradient-accumulation-steps 4

# 3단계: LoRA rank 줄이기
--lora-r 16

# 4단계: Sequence length 줄이기 (데이터 전처리)
max_length = 128  # 기본 256에서
```

### CUDA Out of Memory 메시지

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**해결 방법:**
1. Batch size를 절반으로 줄임
2. `torch.cuda.empty_cache()` 추가
3. Gradient accumulation 사용
4. 다른 프로세스 확인 (`nvidia-smi`)

### 학습이 너무 느릴 때

```bash
# Gradient checkpointing 비활성화
# (메모리가 충분하면)
python scripts/train/anchor_train.py \
    --use-amp \
    # --gradient-checkpointing  제거
```

---

## 📚 참고 자료

- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [BF16 vs FP16](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)

---

## ✅ Quick Checklist

실행 전 확인사항:

- [ ] H100 GPU 사용 가능 확인 (`nvidia-smi`)
- [ ] CUDA 12.0+ 설치 확인
- [ ] PyTorch 2.0+ 설치 확인
- [ ] `--use-amp` 플래그 활성화
- [ ] `--gradient-checkpointing` 플래그 활성화
- [ ] Batch size가 GPU 메모리에 맞는지 확인
- [ ] 적절한 gradient accumulation 설정

**추천 실행 명령:**
```bash
# XL 모델 (3.5B) - 최적 균형
bash scripts/run_poc_h100.sh xl
```

이제 H100의 성능을 최대한 활용할 수 있습니다! 🚀
