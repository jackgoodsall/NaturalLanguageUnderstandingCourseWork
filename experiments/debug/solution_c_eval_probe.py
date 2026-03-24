#!/usr/bin/env python3
"""Evaluate a saved Solution C checkpoint directly and inspect prediction collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PairDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: np.ndarray | None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[idx], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        output[key] = torch.stack([item[key] for item in batch])
    return output


def metrics_dict(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "binary_precision": bin_p,
        "binary_recall": bin_r,
        "binary_f1": bin_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "matthews_corrcoef": matthews_corrcoef(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def threshold_predictions(pos_probs: np.ndarray, threshold: float) -> np.ndarray:
    return (pos_probs >= threshold).astype(int)


def build_threshold_grid(
    threshold_min: float, threshold_max: float, threshold_step: float
) -> list[float]:
    thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2), threshold_step)
    return [float(x) for x in np.round(np.clip(thresholds, 0.0, 1.0), 4)]


def sweep_thresholds(
    labels: np.ndarray,
    pos_probs: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    tracked_metrics = ("accuracy", "binary_f1", "macro_f1", "matthews_corrcoef")
    best_by_metric: dict[str, dict[str, Any]] = {}
    sweep_rows: list[dict[str, float]] = []

    for threshold in thresholds:
        preds = threshold_predictions(pos_probs, threshold)
        metric_values = metrics_dict(labels, preds)
        sweep_row = {"threshold": threshold}
        for metric_name in tracked_metrics:
            sweep_row[metric_name] = float(metric_values[metric_name])
            current_best = best_by_metric.get(metric_name)
            candidate = {
                "threshold": threshold,
                metric_name: float(metric_values[metric_name]),
                "prediction_rate_positive": float((preds == 1).mean()),
            }
            if current_best is None:
                best_by_metric[metric_name] = candidate
            else:
                current_value = candidate[metric_name]
                best_value = current_best[metric_name]
                # Prefer the better metric, then the threshold closer to 0.5 on ties.
                if current_value > best_value or (
                    np.isclose(current_value, best_value)
                    and abs(threshold - 0.5) < abs(current_best["threshold"] - 0.5)
                ):
                    best_by_metric[metric_name] = candidate
        sweep_rows.append(sweep_row)

    midpoint_threshold = 0.5
    midpoint_preds = threshold_predictions(pos_probs, midpoint_threshold)
    midpoint_metrics = metrics_dict(labels, midpoint_preds)

    return {
        "threshold_count": len(thresholds),
        "grid": sweep_rows,
        "baseline_threshold_0_5": {
            "threshold": midpoint_threshold,
            **{metric: float(midpoint_metrics[metric]) for metric in tracked_metrics},
            "prediction_rate_positive": float((midpoint_preds == 1).mean()),
        },
        "best_by_metric": best_by_metric,
    }


def evaluate(
    model_dir: Path,
    data_csv: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
    threshold_sweep: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    predictions_out: Path | None,
) -> dict[str, Any]:
    df = pd.read_csv(data_csv)
    has_labels = "label" in df.columns
    labels = df["label"].astype(int).to_numpy() if has_labels else None

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    encoded = tokenizer(
        df["Claim"].tolist(),
        df["Evidence"].tolist(),
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    dataset = PairDataset(encoded, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    all_logits: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_pos_probs: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            labels_tensor = batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().float().cpu().numpy()
            probs = torch.softmax(outputs.logits.detach().float(), dim=-1).cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            all_logits.append(logits)
            all_preds.append(preds)
            all_pos_probs.append(probs[:, 1])
            if labels_tensor is not None:
                labels_tensor = labels_tensor.cpu().numpy()

    logits = np.concatenate(all_logits, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    pos_probs = np.concatenate(all_pos_probs, axis=0)

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "data_csv": str(data_csv),
        "rows": int(len(df)),
        "device": str(device),
        "batch_size": batch_size,
        "max_length": max_length,
        "prediction_counts": {
            str(label): int((preds == label).sum()) for label in np.unique(preds)
        },
        "prediction_rate_positive": float((preds == 1).mean()),
        "positive_probability_mean": float(pos_probs.mean()),
        "positive_probability_min": float(pos_probs.min()),
        "positive_probability_max": float(pos_probs.max()),
        "mean_logits": np.mean(logits, axis=0).tolist(),
        "std_logits": np.std(logits, axis=0).tolist(),
        "sample_rows": pd.DataFrame(
            {
                "Claim": df["Claim"].head(5),
                "Evidence": df["Evidence"].head(5),
                "pred": preds[:5],
                "pos_prob": pos_probs[:5],
            }
        ).to_dict(orient="records"),
    }

    if labels is not None:
        summary["label_counts"] = {
            str(label): int((labels == label).sum()) for label in np.unique(labels)
        }
        summary["metrics"] = metrics_dict(labels, preds)
        if threshold_sweep:
            summary["threshold_probe"] = sweep_thresholds(
                labels=labels,
                pos_probs=pos_probs,
                thresholds=build_threshold_grid(
                    threshold_min=threshold_min,
                    threshold_max=threshold_max,
                    threshold_step=threshold_step,
                ),
            )

    if predictions_out:
        out_df = df.copy()
        out_df["pred"] = preds.astype(int)
        out_df["positive_probability"] = pos_probs
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(predictions_out, index=False)
        summary["predictions_out"] = str(predictions_out)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--json-out")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()

    summary = evaluate(
        model_dir=Path(args.model_dir),
        data_csv=Path(args.data_csv),
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=resolve_device(args.device),
        threshold_sweep=args.threshold_sweep,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        predictions_out=Path(args.predictions_out) if args.predictions_out else None,
    )

    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
