from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from torch.utils.data import Dataset


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


class PairClassificationDataset(Dataset[dict[str, list[int] | int]]):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int] | None):
        self.encodings = encodings
        self.labels = labels

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
    ) -> "PairClassificationDataset":
        encodings = tokenizer(
            df["claim"].tolist(),
            df["evidence"].tolist(),
            truncation=True,
            max_length=max_length,
        )
        labels = df["label"].tolist() if "label" in df.columns else None
        return cls(encodings=encodings, labels=labels)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        item = {
            key: values[index]
            for key, values in self.encodings.items()
        }
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item
