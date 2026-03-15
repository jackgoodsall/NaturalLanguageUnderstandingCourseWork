import re

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_embeddings(name: str = "fasttext-wiki-news-subwords-300"):
    """Function for loading pretrained embeddings from gensim"""
    return api.load(name)


def preprocess(text: str) -> list:
    """Simple function for preprocessing, converts to lower case then basic regex into
    splitting into tokens"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def tokens_to_vectors(tokens: list, embeddings, dim: int = 300) -> np.ndarray:
    """Converts the tokens into vector representations using pretrained embeddings"""
    vecs = [embeddings[t] if t in embeddings else np.zeros(dim) for t in tokens]
    return np.array(vecs) if vecs else np.zeros((1, dim))


def load_and_preprocess(path: str, glove, dim: int = 100) -> pd.DataFrame:
    """Loads and preprocesses claim evidence data from a csv file. Returns data frame with vectorised
    representation of data"""
    df = pd.read_csv(path)
    df["claim_tokens"] = df["Claim"].apply(preprocess)
    df["evidence_tokens"] = df["Evidence"].apply(preprocess)
    df["claim_vecs"] = df["claim_tokens"].apply(lambda t: tokens_to_vectors(t, glove, dim))
    df["evidence_vecs"] = df["evidence_tokens"].apply(lambda t: tokens_to_vectors(t, glove, dim))
    return df


class ClaimEvidenceDataset(Dataset):
    """
    Torch dataset for representing the dataset
    """
    def __init__(self, claims, evidences, labels=None):
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
    """Function for collating the data, required due to differing length of input sequence."""
    items, labels = zip(*batch)

    claim_tensors = [x["claim"] for x in items]
    evidence_tensors = [x["evidence"] for x in items]

    result = {
        "claim": pad_sequence(claim_tensors, batch_first=True),
        "evidence": pad_sequence(evidence_tensors, batch_first=True),
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
    """Helper for getting Dataloaders of train and dev dataset"""
    train_ds = ClaimEvidenceDataset(
        train_df["claim_vecs"], train_df["evidence_vecs"], train_df["label"]
    )
    dev_ds = ClaimEvidenceDataset(
        dev_df["claim_vecs"], dev_df["evidence_vecs"], dev_df["label"]
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_dl, dev_dl
