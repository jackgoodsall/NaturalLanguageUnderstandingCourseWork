from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import PairClassificationDataset, load_pair_dataframe
from .metrics import compute_metrics, probabilities_from_logits, threshold_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate solo Solution C checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--predictions-csv")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_probabilities(model, dataloader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []
    for batch in dataloader:
        labels = batch.pop("labels", None)
        moved = {
            key: value.to(device)
            for key, value in batch.items()
        }
        outputs = model(**moved)
        probs = probabilities_from_logits(outputs.logits.detach().cpu().numpy())
        all_probs.append(probs)
        if labels is not None:
            _ = labels
    return np.concatenate(all_probs, axis=0)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    df = load_pair_dataframe(args.data, max_rows=args.max_rows)
    labels = df["label"].to_numpy(dtype=int) if "label" in df.columns else None

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0], use_fast=False)
    dataset = PairClassificationDataset.from_dataframe(df, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    probability_sets = []
    for checkpoint in args.checkpoints:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
        )
        model.to(torch.float32)
        model.to(device)
        probability_sets.append(predict_probabilities(model, dataloader, device))

    mean_probs = np.mean(np.stack(probability_sets, axis=0), axis=0)

    payload: dict[str, object] = {
        "checkpoints": args.checkpoints,
        "threshold": args.threshold,
        "probability_summary": {
            "nan_count": int(np.isnan(mean_probs).sum()),
            "min": float(np.nanmin(mean_probs)) if not np.isnan(mean_probs).all() else None,
            "max": float(np.nanmax(mean_probs)) if not np.isnan(mean_probs).all() else None,
            "mean": float(np.nanmean(mean_probs)) if not np.isnan(mean_probs).all() else None,
            "positive_rate_at_0_5": float((mean_probs >= 0.5).mean()) if not np.isnan(mean_probs).all() else 0.0,
        },
    }

    if labels is not None:
        payload["metrics"] = compute_metrics(labels, mean_probs, threshold=args.threshold)
        if args.sweep:
            payload["threshold_probe"] = threshold_sweep(labels, mean_probs)
    else:
        payload["prediction_rate_positive"] = float((mean_probs >= args.threshold).mean())

    if args.predictions_csv:
        output_df = df.copy()
        output_df["probability_positive"] = mean_probs
        output_df["prediction"] = (mean_probs >= args.threshold).astype(int)
        Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.predictions_csv, index=False)

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
