"""
R-CoA Anchor Head Implementation
XLM-RoBERTa + LoRA + InfoNCE Loss for cross-lingual alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Tuple, Optional
import numpy as np


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss for cross-lingual alignment

    Given anchor (classical Chinese) and positive (modern Chinese/Korean/English),
    maximize similarity between positive pairs while minimizing similarity
    with negative samples (other items in the batch).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: (batch_size, hidden_dim)
            positive_embeddings: (batch_size, hidden_dim)
            negative_embeddings: (batch_size, num_negatives, hidden_dim) or None
                If None, use in-batch negatives

        Returns:
            loss: scalar tensor
        """
        batch_size = anchor_embeddings.size(0)

        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Positive logits: (batch_size,)
        positive_logits = torch.sum(anchor_embeddings * positive_embeddings, dim=1) / self.temperature

        if negative_embeddings is None:
            # In-batch negatives: use all other samples as negatives
            # Similarity matrix: (batch_size, batch_size)
            similarity_matrix = torch.matmul(
                anchor_embeddings,
                positive_embeddings.t()
            ) / self.temperature

            # Mask diagonal (positive pairs)
            mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
            similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

            # Concatenate positive and negative logits
            logits = torch.cat([
                positive_logits.unsqueeze(1),  # (batch_size, 1)
                similarity_matrix  # (batch_size, batch_size)
            ], dim=1)

            # Labels: positive is always the first one (index 0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        else:
            # Explicit negatives provided
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)

            # Negative logits: (batch_size, num_negatives)
            negative_logits = torch.matmul(
                anchor_embeddings.unsqueeze(1),  # (batch_size, 1, hidden_dim)
                negative_embeddings.transpose(1, 2)  # (batch_size, hidden_dim, num_negatives)
            ).squeeze(1) / self.temperature

            # Concatenate logits
            logits = torch.cat([
                positive_logits.unsqueeze(1),  # (batch_size, 1)
                negative_logits  # (batch_size, num_negatives)
            ], dim=1)

            # Labels: positive is always the first one
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss


class AnchorHead(nn.Module):
    """
    Anchor Head for R-CoA

    Architecture:
    - XLM-RoBERTa base (multilingual encoder)
    - LoRA adapters for efficient fine-tuning
    - Projection head for embedding space
    - InfoNCE loss for alignment
    """

    def __init__(self,
                 model_name: str = "xlm-roberta-base",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 projection_dim: int = 256,
                 pooling: str = "mean"):
        """
        Args:
            model_name: HuggingFace model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            projection_dim: Dimension of projection head output
            pooling: Pooling strategy ('mean', 'cls', 'max')
        """
        super().__init__()

        self.pooling = pooling
        self.projection_dim = projection_dim

        # Load XLM-RoBERTa with memory optimization
        print(f"[INIT] Loading {model_name}...")
        # Load model without torch_dtype to avoid to_empty() issues
        self.encoder = XLMRobertaModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        self.config = self.encoder.config
        print(f"[MEMORY] Model loaded (precision handled by AMP during training)")

        # LoRA configuration
        print(f"[INIT] Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],  # Apply LoRA to attention layers
            bias="none"
        )

        # Apply LoRA
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()

        # Projection head
        hidden_dim = self.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )

        print(f"[INIT] Anchor Head initialized")
        print(f"  Encoder: {model_name}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Pooling: {pooling}")

    def pool_embeddings(self,
                       last_hidden_state: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool token embeddings to sentence embedding

        Args:
            last_hidden_state: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, hidden_dim)
        """
        if self.pooling == "cls":
            # Use [CLS] token
            return last_hidden_state[:, 0, :]

        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def encode(self,
               input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, projection_dim)
        """
        # Encode with XLM-R
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Pool to sentence embedding
        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)

        # Project to embedding space
        embeddings = self.projection(pooled)

        return embeddings

    def forward(self,
                anchor_input_ids: torch.Tensor,
                anchor_attention_mask: torch.Tensor,
                positive_input_ids: torch.Tensor,
                positive_attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training

        Args:
            anchor_input_ids: (batch_size, seq_len)
            anchor_attention_mask: (batch_size, seq_len)
            positive_input_ids: (batch_size, seq_len)
            positive_attention_mask: (batch_size, seq_len)

        Returns:
            Dict with 'anchor_embeddings' and 'positive_embeddings'
        """
        # Encode anchor and positive
        anchor_embeddings = self.encode(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = self.encode(positive_input_ids, positive_attention_mask)

        return {
            'anchor_embeddings': anchor_embeddings,
            'positive_embeddings': positive_embeddings
        }


class RCoAAnchorTrainer:
    """Trainer for Anchor Head"""

    def __init__(self,
                 model: AnchorHead,
                 tokenizer: XLMRobertaTokenizer,
                 device: str = "cuda",
                 temperature: float = 0.07):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = InfoNCELoss(temperature=temperature)

    def prepare_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training

        Args:
            batch: List of dicts with 'classical_chinese', 'modern_chinese', etc.

        Returns:
            Dict of tensors ready for model
        """
        # Extract texts
        anchor_texts = [item['classical_chinese'] for item in batch]
        positive_texts = [item['modern_chinese'] for item in batch]

        # Tokenize
        anchor_encodings = self.tokenizer(
            anchor_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        positive_encodings = self.tokenizer(
            positive_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Move to device
        return {
            'anchor_input_ids': anchor_encodings['input_ids'].to(self.device),
            'anchor_attention_mask': anchor_encodings['attention_mask'].to(self.device),
            'positive_input_ids': positive_encodings['input_ids'].to(self.device),
            'positive_attention_mask': positive_encodings['attention_mask'].to(self.device)
        }

    def train_step(self, batch: List[Dict]) -> float:
        """Single training step"""
        self.model.train()

        # Prepare batch
        inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**inputs)

        # Compute InfoNCE loss
        loss = self.criterion(
            outputs['anchor_embeddings'],
            outputs['positive_embeddings']
        )

        return loss

    @torch.no_grad()
    def evaluate_step(self, batch: List[Dict]) -> float:
        """Single evaluation step"""
        self.model.eval()

        # Prepare batch
        inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**inputs)

        # Compute loss
        loss = self.criterion(
            outputs['anchor_embeddings'],
            outputs['positive_embeddings']
        )

        return loss.item()
