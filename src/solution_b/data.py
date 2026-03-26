from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Vocabulary:
    tokens: list[str]
    token_to_id: dict[str, int]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for sequence in token_sequences:
            counts.update(sequence)

        kept_tokens = [
            token for token, count in counts.most_common() if count >= min_freq
        ]
        if max_vocab_size is not None:
            kept_tokens = kept_tokens[:max(0, max_vocab_size - 2)]

        tokens = [PAD_TOKEN, UNK_TOKEN] + kept_tokens
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(tokens=tokens, token_to_id=token_to_id)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk_id) for token in tokens]


def _normalise_columns(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        normalised = column.strip().lower()
        if normalised == "claim":
            mapping[column] = "claim"
        elif normalised == "evidence":
            mapping[column] = "evidence"
        elif normalised == "label":
            mapping[column] = "label"
    return mapping


def load_pair_dataframe(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=_normalise_columns(df.columns))
    required = {"claim", "evidence"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    else:
        df = df.copy()

    df["claim"] = df["claim"].fillna("").astype(str)
    df["evidence"] = df["evidence"].fillna("").astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def add_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["claim_tokens"] = enriched["claim"].map(tokenize_text)
    enriched["evidence_tokens"] = enriched["evidence"].map(tokenize_text)
    return enriched


def build_embedding_matrix(
    vocab: Vocabulary,
    embedding_name: str,
    embedding_dim: int,
    seed: int = 13,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.05, size=(len(vocab.tokens), embedding_dim)).astype(np.float32)
    matrix[vocab.token_to_id[PAD_TOKEN]] = 0.0

    if embedding_name == "random":
        return matrix

    vectors = gensim_api.load(embedding_name)
    if vectors.vector_size != embedding_dim:
        raise ValueError(
            f"Embedding dim mismatch for {embedding_name}: expected {embedding_dim}, got {vectors.vector_size}"
        )

    for token, idx in vocab.token_to_id.items():
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if token in vectors:
            matrix[idx] = vectors[token]
    return matrix


class ClaimEvidenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary):
        self.claim_ids = [vocab.encode(tokens) for tokens in df["claim_tokens"]]
        self.evidence_ids = [vocab.encode(tokens) for tokens in df["evidence_tokens"]]
        self.labels = df["label"].tolist() if "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.claim_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "claim_ids": torch.tensor(self.claim_ids[index], dtype=torch.long),
            "evidence_ids": torch.tensor(self.evidence_ids[index], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return item


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    claim_lengths = torch.tensor([item["claim_ids"].numel() for item in batch], dtype=torch.long)
    evidence_lengths = torch.tensor([item["evidence_ids"].numel() for item in batch], dtype=torch.long)

    max_claim = int(claim_lengths.max().item())
    max_evidence = int(evidence_lengths.max().item())

    claim_ids = torch.zeros((len(batch), max_claim), dtype=torch.long)
    evidence_ids = torch.zeros((len(batch), max_evidence), dtype=torch.long)

    labels = []
    has_labels = "labels" in batch[0]

    for idx, item in enumerate(batch):
        claim = item["claim_ids"]
        evidence = item["evidence_ids"]
        claim_ids[idx, : claim.numel()] = claim
        evidence_ids[idx, : evidence.numel()] = evidence
        if has_labels:
            labels.append(item["labels"])

    payload = {
        "claim_ids": claim_ids,
        "claim_lengths": claim_lengths,
        "evidence_ids": evidence_ids,
        "evidence_lengths": evidence_lengths,
    }
    if has_labels:
        payload["labels"] = torch.stack(labels)
    return payload
