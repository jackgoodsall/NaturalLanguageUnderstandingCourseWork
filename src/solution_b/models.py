from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.rnn(packed)
        unpacked, _ = pad_packed_sequence(encoded, batch_first=True, total_length=inputs.size(1))
        return self.dropout(unpacked)


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    max_len = int(max_length or lengths.max().item())
    range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return range_tensor < lengths.unsqueeze(1)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, -1e9)
    return torch.softmax(masked_scores, dim=dim)


def replace_masked(tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    return tensor.masked_fill(~mask, value)


class ESIMModel(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int = 128,
        projection_size: int = 128,
        dropout: float = 0.2,
        trainable_embeddings: bool = False,
    ) -> None:
        super().__init__()
        embedding_dim = embedding_matrix.size(1)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=not trainable_embeddings,
            padding_idx=0,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = Seq2SeqEncoder(embedding_dim, hidden_size, dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 8, projection_size),
            nn.ReLU(),
        )
        self.composition = Seq2SeqEncoder(projection_size, hidden_size, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 8, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(
        self,
        claim_ids: torch.Tensor,
        claim_lengths: torch.Tensor,
        evidence_ids: torch.Tensor,
        evidence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        claim_mask = sequence_mask(claim_lengths, claim_ids.size(1))
        evidence_mask = sequence_mask(evidence_lengths, evidence_ids.size(1))

        claim_emb = self.embedding_dropout(self.embedding(claim_ids))
        evidence_emb = self.embedding_dropout(self.embedding(evidence_ids))

        claim_encoded = self.encoder(claim_emb, claim_lengths)
        evidence_encoded = self.encoder(evidence_emb, evidence_lengths)

        similarity = torch.matmul(claim_encoded, evidence_encoded.transpose(1, 2))
        claim_attention = masked_softmax(similarity, evidence_mask.unsqueeze(1), dim=-1)
        evidence_attention = masked_softmax(similarity.transpose(1, 2), claim_mask.unsqueeze(1), dim=-1)

        claim_aligned = torch.bmm(claim_attention, evidence_encoded)
        evidence_aligned = torch.bmm(evidence_attention, claim_encoded)

        claim_enhanced = torch.cat(
            [
                claim_encoded,
                claim_aligned,
                claim_encoded - claim_aligned,
                claim_encoded * claim_aligned,
            ],
            dim=-1,
        )
        evidence_enhanced = torch.cat(
            [
                evidence_encoded,
                evidence_aligned,
                evidence_encoded - evidence_aligned,
                evidence_encoded * evidence_aligned,
            ],
            dim=-1,
        )

        projected_claim = self.projection(claim_enhanced)
        projected_evidence = self.projection(evidence_enhanced)

        composed_claim = self.composition(projected_claim, claim_lengths)
        composed_evidence = self.composition(projected_evidence, evidence_lengths)

        claim_features = self._pool_sequence(composed_claim, claim_mask)
        evidence_features = self._pool_sequence(composed_evidence, evidence_mask)
        features = torch.cat([claim_features, evidence_features], dim=-1)
        logits = self.classifier(features).squeeze(-1)
        return logits

    @staticmethod
    def _pool_sequence(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        summed = (sequence * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_pool = summed / denom
        max_pool = replace_masked(sequence, mask, -1e9).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)
