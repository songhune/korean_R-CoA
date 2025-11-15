"""
R-CoA Anchor Head Training Script
XLM-RoBERTa + LoRA + InfoNCE
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available, logging disabled")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.anchor_head_model import AnchorHead, RCoAAnchorTrainer, InfoNCELoss


class CrossLingualDataset(Dataset):
    """Dataset for cross-lingual pairs"""

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """Simple collate function"""
    return batch


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEMORY] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0


def train_epoch(model: AnchorHead,
                trainer: RCoAAnchorTrainer,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                epoch: int,
                device: str,
                use_wandb: bool = False,
                scaler=None,
                use_amp: bool = False) -> float:
    """Train for one epoch with optional AMP"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # AMP forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = trainer.train_step(batch)

            # AMP backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            loss = trainer.train_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

        # Log to wandb
        if use_wandb and batch_idx % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/step': epoch * len(dataloader) + batch_idx
            })

    return total_loss / num_batches


def evaluate(model: AnchorHead,
             trainer: RCoAAnchorTrainer,
             dataloader: DataLoader,
             device: str) -> float:
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss = trainer.evaluate_step(batch)
            total_loss += loss
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train R-CoA Anchor Head")

    # Data
    parser.add_argument('--train-data', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/data/splits/train_pairs.jsonl',
                       help='Path to training data')
    parser.add_argument('--val-data', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/data/splits/val_pairs.jsonl',
                       help='Path to validation data')

    # Model
    parser.add_argument('--model-name', type=str, default='xlm-roberta-base',
                       help='HuggingFace model name')
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--projection-dim', type=int, default=256,
                       help='Projection dimension')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'cls', 'max'],
                       help='Pooling strategy')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='InfoNCE temperature')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')

    # Mixed Precision
    parser.add_argument('--use-amp', action='store_true',
                       help='Use Automatic Mixed Precision (BF16)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Logging
    parser.add_argument('--output-dir', type=str,
                       default='/home/work/songhune/korean_R-CoA/experiments/rcoa/checkpoints',
                       help='Output directory')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Save interval (epochs)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='rcoa-anchor-head',
                       help='W&B project name')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[WARNING] wandb requested but not available, disabling...")
            args.use_wandb = False
        else:
            wandb.init(project=args.wandb_project, config=vars(args))

    print("="*70)
    print("R-CoA Anchor Head Training")
    print("="*70)
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Model: {args.model_name}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load datasets
    print("\n[LOAD] Loading datasets...")
    train_dataset = CrossLingualDataset(args.train_data)
    val_dataset = CrossLingualDataset(args.val_data)

    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")

    # Create dataloaders (pin_memory=False for memory efficiency)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Clear cache before model initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[MEMORY] Cleared CUDA cache")

    # Initialize model
    print("\n[INIT] Initializing model...")
    model = AnchorHead(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        projection_dim=args.projection_dim,
        pooling=args.pooling
    )

    # Print memory after model loading
    print_gpu_memory()

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)

    # Initialize trainer
    trainer = RCoAAnchorTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        temperature=args.temperature
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize AMP scaler
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("[AMP] Automatic Mixed Precision enabled (BF16)")

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        try:
            model.encoder.gradient_checkpointing_enable()
            print("[MEMORY] Gradient checkpointing enabled")
        except (AttributeError, NotImplementedError):
            print("[WARNING] Gradient checkpointing not supported for this model")

    print(f"\n[TRAIN] Starting training...")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    if args.use_amp:
        print(f"  Mixed Precision: BF16")
    if args.gradient_checkpointing:
        print(f"  Gradient Checkpointing: Enabled")

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_loss = train_epoch(
            model=model,
            trainer=trainer,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=args.device,
            use_wandb=args.use_wandb,
            scaler=scaler,
            use_amp=args.use_amp
        )

        print(f"\n[TRAIN] Epoch {epoch} - Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(
            model=model,
            trainer=trainer,
            dataloader=val_loader,
            device=args.device
        )

        print(f"[EVAL] Epoch {epoch} - Val Loss: {val_loss:.4f}")

        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/loss': val_loss
            })

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"[SAVE] Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, best_model_path)
            print(f"[SAVE] Best model saved: {best_model_path} (val_loss={val_loss:.4f})")

    print(f"\n{'='*70}")
    print("[COMPLETE] Training finished")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
