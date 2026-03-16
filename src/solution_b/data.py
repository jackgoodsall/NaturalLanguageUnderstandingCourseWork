"""
data.py — Data loading, preprocessing, and batching for Solution B.

Handles:
  - Loading pretrained word embeddings (FastText) via gensim
  - Tokenising and vectorising claim/evidence text pairs
  - Wrapping data in a PyTorch Dataset and DataLoader with padding
"""

import re

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_embeddings(name: str = "fasttext-wiki-news-subwords-300"):
    """Load pretrained word embeddings from gensim's model repository.

    Args:
        name: gensim model identifier. The last segment after '-' is parsed
              as the embedding dimensionality (e.g. 300 for fasttext-wiki-news-subwords-300).

    Returns:
        A gensim KeyedVectors object mapping words to numpy vectors.
    """
    return api.load(name)


def preprocess(text: str) -> list:
    """Tokenise a text string for embedding lookup.

    Steps:
      1. Lowercase the text
      2. Strip all non-alphanumeric characters (except spaces)
      3. Split on whitespace to produce tokens

    Args:
        text: raw input string (e.g. a claim or evidence sentence).

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def tokens_to_vectors(tokens: list, embeddings, dim: int = 300) -> np.ndarray:
    """Convert a list of tokens into a (num_tokens, dim) array of word vectors.

    Tokens not found in the embedding vocabulary are mapped to zero vectors.
    If the token list is empty, returns a single zero vector so downstream
    code always receives a valid 2-D array.

    Args:
        tokens:     list of string tokens.
        embeddings: gensim KeyedVectors (or similar mapping).
        dim:        embedding dimensionality (must match the loaded model).

    Returns:
        np.ndarray of shape (max(len(tokens), 1), dim).
    """
    vecs = [embeddings[t] if t in embeddings else np.zeros(dim) for t in tokens]
    return np.array(vecs) if vecs else np.zeros((1, dim))


def load_and_preprocess(path: str, embeddings, dim: int = 100) -> pd.DataFrame:
    """Load a CSV file and add tokenised + vectorised columns for claims and evidence.

    Expects the CSV to have at least 'Claim' and 'Evidence' columns
    (and optionally 'label' for supervised data).

    Args:
        path:       path to the CSV file.
        embeddings: pretrained word embeddings (gensim KeyedVectors).
        dim:        dimensionality of the embeddings.

    Returns:
        DataFrame with added columns: claim_tokens, evidence_tokens,
        claim_vecs (list of np arrays), evidence_vecs.
    """
    df = pd.read_csv(path)
    # Tokenise raw text
    df["claim_tokens"] = df["Claim"].apply(preprocess)
    df["evidence_tokens"] = df["Evidence"].apply(preprocess)
    # Map each token list to a (num_tokens, dim) embedding matrix
    df["claim_vecs"] = df["claim_tokens"].apply(lambda t: tokens_to_vectors(t, embeddings, dim))
    df["evidence_vecs"] = df["evidence_tokens"].apply(lambda t: tokens_to_vectors(t, embeddings, dim))
    return df


class ClaimEvidenceDataset(Dataset):
    """PyTorch Dataset wrapping pre-vectorised claim/evidence pairs.

    Each sample is a dict with 'claim' and 'evidence' tensors (variable length)
    plus an optional scalar label.
    """

    def __init__(self, claims, evidences, labels=None):
        # Convert numpy arrays to float tensors up-front so DataLoader is fast
        self.claims = [torch.tensor(c, dtype=torch.float32) for c in claims]
        self.evidences = [torch.tensor(e, dtype=torch.float32) for e in evidences]
        self.labels = (
            torch.tensor(labels.values, dtype=torch.float32)
            if labels is not None
            else None
        )

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        item = {"claim": self.claims[idx], "evidence": self.evidences[idx]}
        label = self.labels[idx] if self.labels is not None else None
        return item, label


def collate_fn(batch):
    """Custom collate function that pads claim and evidence sequences to equal
    length within each batch and records the original sequence lengths.

    This is required because claims and evidence have different numbers of
    tokens, so they cannot be stacked directly into a single tensor.

    Args:
        batch: list of (item_dict, label) tuples from the Dataset.

    Returns:
        A tuple (X, y) where:
          X is a dict with keys:
            - 'claim':         (B, max_claim_len, dim) padded tensor
            - 'evidence':      (B, max_evidence_len, dim) padded tensor
            - 'claim_lens':    (B,) original lengths (needed for pack_padded_sequence)
            - 'evidence_lens': (B,) original lengths
          y is a (B,) label tensor, or None if labels are absent.
    """
    items, labels = zip(*batch)

    claim_tensors = [x["claim"] for x in items]
    evidence_tensors = [x["evidence"] for x in items]

    result = {
        # pad_sequence pads shorter sequences with zeros to match the longest
        "claim": pad_sequence(claim_tensors, batch_first=True),
        "evidence": pad_sequence(evidence_tensors, batch_first=True),
        # Store original lengths so the LSTM can ignore padding via packing
        "claim_lens": torch.tensor([c.size(0) for c in claim_tensors], dtype=torch.long),
        "evidence_lens": torch.tensor(
            [e.size(0) for e in evidence_tensors], dtype=torch.long
        ),
    }

    if labels[0] is not None:
        return result, torch.stack(labels)
    return result, None


def get_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    batch_size: int = 32,
):
    """Build train and dev DataLoaders from preprocessed DataFrames.

    Args:
        train_df: DataFrame with claim_vecs, evidence_vecs, label columns.
        dev_df:   same format as train_df.
        batch_size: mini-batch size.

    Returns:
        (train_dataloader, dev_dataloader) tuple.
    """
    train_ds = ClaimEvidenceDataset(
        train_df["claim_vecs"], train_df["evidence_vecs"], train_df["label"]
    )
    dev_ds = ClaimEvidenceDataset(
        dev_df["claim_vecs"], dev_df["evidence_vecs"], dev_df["label"]
    )
    # Training data is shuffled; dev data is not since not needed
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_dl, dev_dl
