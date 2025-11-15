"""
Week 3: Chain Head Implementation
TransE + Chain Loss for relational reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class TransELoss(nn.Module):
    """
    TransE Loss for Knowledge Graph triples

    L_KG = Σ ||e_s + r - e_o||²

    For triple (subject, relation, object):
    - e_s: subject embedding
    - r: relation embedding
    - e_o: object embedding

    Goal: e_s + r ≈ e_o
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                subject_embeddings: torch.Tensor,
                relation_embeddings: torch.Tensor,
                object_embeddings: torch.Tensor,
                negative_object_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            subject_embeddings: (batch_size, hidden_dim)
            relation_embeddings: (batch_size, hidden_dim)
            object_embeddings: (batch_size, hidden_dim)
            negative_object_embeddings: (batch_size, num_negatives, hidden_dim) or None

        Returns:
            loss: scalar tensor
        """
        # Positive score: ||e_s + r - e_o||²
        positive_score = torch.sum(
            (subject_embeddings + relation_embeddings - object_embeddings) ** 2,
            dim=1
        )

        if negative_object_embeddings is None:
            # No negative sampling, just minimize positive distance
            loss = torch.mean(positive_score)
        else:
            # Negative score: ||e_s + r - e_o'||²
            # (batch_size, num_negatives, hidden_dim)
            expanded_subject = subject_embeddings.unsqueeze(1).expand_as(negative_object_embeddings)
            expanded_relation = relation_embeddings.unsqueeze(1).expand_as(negative_object_embeddings)

            negative_score = torch.sum(
                (expanded_subject + expanded_relation - negative_object_embeddings) ** 2,
                dim=2
            )  # (batch_size, num_negatives)

            # Margin ranking loss: max(0, margin + positive - negative)
            negative_score = torch.mean(negative_score, dim=1)  # (batch_size,)
            loss = torch.mean(F.relu(self.margin + positive_score - negative_score))

        return loss


class ChainLoss(nn.Module):
    """
    Chain Loss for multi-hop reasoning

    L_chain = Σ ||e_start + Σr_k - e_end||²

    For chain: entity_1 --r1--> entity_2 --r2--> ... --rN--> entity_N
    Goal: e_1 + r_1 + r_2 + ... + r_N ≈ e_N
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                start_embeddings: torch.Tensor,
                relation_chain: List[torch.Tensor],
                end_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            start_embeddings: (batch_size, hidden_dim)
            relation_chain: List of (batch_size, hidden_dim) tensors
            end_embeddings: (batch_size, hidden_dim)

        Returns:
            loss: scalar tensor
        """
        # Sum all relations in the chain
        cumulative_relation = sum(relation_chain)  # (batch_size, hidden_dim)

        # Chain consistency: ||e_start + Σr - e_end||²
        chain_score = torch.sum(
            (start_embeddings + cumulative_relation - end_embeddings) ** 2,
            dim=1
        )

        loss = torch.mean(chain_score)
        return loss


class ChainHead(nn.Module):
    """
    Chain Head for R-CoA

    Combines:
    1. Anchor embeddings from Anchor Head
    2. Relation embeddings for TransE
    3. Chain reasoning with multi-hop paths
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 num_relations: int = 10,
                 relation_dim: int = 256):
        """
        Args:
            embedding_dim: Dimension of entity embeddings (from Anchor Head)
            num_relations: Number of relation types
            relation_dim: Dimension of relation embeddings
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.relation_dim = relation_dim

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)

        # Initialize with small values
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

        # Projection if needed
        if embedding_dim != relation_dim:
            self.projection = nn.Linear(embedding_dim, relation_dim)
        else:
            self.projection = None

        print(f"[INIT] Chain Head initialized")
        print(f"  Num relations: {num_relations}")
        print(f"  Relation dim: {relation_dim}")
        print(f"  Entity embedding dim: {embedding_dim}")

    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Get relation embeddings"""
        return self.relation_embeddings(relation_ids)

    def project_entity(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """Project entity embeddings to relation space if needed"""
        if self.projection is not None:
            return self.projection(entity_embeddings)
        return entity_embeddings

    def forward(self,
                subject_embeddings: torch.Tensor,
                relation_ids: torch.Tensor,
                object_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-hop triples

        Args:
            subject_embeddings: (batch_size, embedding_dim)
            relation_ids: (batch_size,) integer relation IDs
            object_embeddings: (batch_size, embedding_dim)

        Returns:
            Dict with projected embeddings and relation embeddings
        """
        # Project entity embeddings
        subject_projected = self.project_entity(subject_embeddings)
        object_projected = self.project_entity(object_embeddings)

        # Get relation embeddings
        relation_embeddings = self.get_relation_embedding(relation_ids)

        return {
            'subject_embeddings': subject_projected,
            'relation_embeddings': relation_embeddings,
            'object_embeddings': object_projected
        }


class IntegratedRCoAModel(nn.Module):
    """
    Integrated R-CoA Model

    Combines:
    - Anchor Head (cross-lingual alignment)
    - Chain Head (relational reasoning)

    Loss:
    L = L_anchor + λ·L_KG + μ·L_chain
    """

    def __init__(self,
                 anchor_head,
                 chain_head: ChainHead,
                 lambda_kg: float = 0.5,
                 mu_chain: float = 0.5):
        super().__init__()

        self.anchor_head = anchor_head
        self.chain_head = chain_head

        self.lambda_kg = lambda_kg
        self.mu_chain = mu_chain

        # Loss functions
        from models.anchor_head_model import InfoNCELoss
        self.anchor_loss_fn = InfoNCELoss()
        self.kg_loss_fn = TransELoss()
        self.chain_loss_fn = ChainLoss()

    def compute_integrated_loss(self,
                               anchor_data: Optional[Dict] = None,
                               kg_data: Optional[Dict] = None,
                               chain_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute integrated loss

        Args:
            anchor_data: Dict with 'anchor_embeddings', 'positive_embeddings'
            kg_data: Dict with 'subject_embeddings', 'relation_embeddings', 'object_embeddings'
            chain_data: Dict with 'start_embeddings', 'relation_chain', 'end_embeddings'

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # 1. Anchor loss (InfoNCE)
        if anchor_data is not None:
            anchor_loss = self.anchor_loss_fn(
                anchor_data['anchor_embeddings'],
                anchor_data['positive_embeddings']
            )
            losses['anchor_loss'] = anchor_loss
            total_loss = total_loss + anchor_loss

        # 2. KG loss (TransE)
        if kg_data is not None:
            kg_loss = self.kg_loss_fn(
                kg_data['subject_embeddings'],
                kg_data['relation_embeddings'],
                kg_data['object_embeddings'],
                kg_data.get('negative_embeddings')
            )
            losses['kg_loss'] = kg_loss
            total_loss = total_loss + self.lambda_kg * kg_loss

        # 3. Chain loss
        if chain_data is not None:
            chain_loss = self.chain_loss_fn(
                chain_data['start_embeddings'],
                chain_data['relation_chain'],
                chain_data['end_embeddings']
            )
            losses['chain_loss'] = chain_loss
            total_loss = total_loss + self.mu_chain * chain_loss

        losses['total_loss'] = total_loss
        return losses

    def forward(self,
                anchor_input_ids: Optional[torch.Tensor] = None,
                anchor_attention_mask: Optional[torch.Tensor] = None,
                positive_input_ids: Optional[torch.Tensor] = None,
                positive_attention_mask: Optional[torch.Tensor] = None,
                subject_embeddings: Optional[torch.Tensor] = None,
                relation_ids: Optional[torch.Tensor] = None,
                object_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Can handle:
        1. Anchor Head inputs (for cross-lingual alignment)
        2. Chain Head inputs (for relational reasoning)
        3. Both simultaneously
        """
        outputs = {}

        # Anchor Head
        if anchor_input_ids is not None:
            anchor_outputs = self.anchor_head(
                anchor_input_ids,
                anchor_attention_mask,
                positive_input_ids,
                positive_attention_mask
            )
            outputs.update(anchor_outputs)

        # Chain Head
        if subject_embeddings is not None and relation_ids is not None:
            chain_outputs = self.chain_head(
                subject_embeddings,
                relation_ids,
                object_embeddings
            )
            outputs['kg_outputs'] = chain_outputs

        return outputs
