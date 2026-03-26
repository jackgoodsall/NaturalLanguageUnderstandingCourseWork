from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClaimEvidenceDataset, Vocabulary, add_token_columns, collate_batch, load_pair_dataframe
from .metrics import compute_metrics, threshold_sweep
from .models import ESIMModel
from .train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a solo Solution B checkpoint.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_checkpoint_model(path: str | Path, device: torch.device) -> tuple[ESIMModel, Vocabulary, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = payload["config"]
    vocab_tokens = payload["vocab_tokens"]
    vocab = Vocabulary(tokens=vocab_tokens, token_to_id={token: idx for idx, token in enumerate(vocab_tokens)})
    model = ESIMModel(
        embedding_matrix=payload["embedding_matrix"].to(torch.float32),
        hidden_size=config["hidden_size"],
        projection_size=config["projection_size"],
        dropout=config["dropout"],
        trainable_embeddings=config["trainable_embeddings"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, vocab, payload


def get_probs(model: ESIMModel, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = torch.sigmoid(model(**batch)).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    models = []
    payloads = []
    vocab = None
    for checkpoint in args.checkpoints:
        model, model_vocab, payload = load_checkpoint_model(checkpoint, device)
        models.append(model)
        payloads.append(payload)
        vocab = model_vocab if vocab is None else vocab

    if vocab is None:
        raise RuntimeError("No checkpoint loaded.")

    df = add_token_columns(load_pair_dataframe(args.data, max_rows=args.max_rows))
    dataset = ClaimEvidenceDataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    probs = np.mean([get_probs(model, dataloader, device) for model in models], axis=0)

    labels = df["label"].to_numpy() if "label" in df.columns else None
    output = {
        "checkpoints": args.checkpoints,
        "threshold": args.threshold,
    }
    if labels is not None:
        output["metrics"] = compute_metrics(labels, probs, threshold=args.threshold)
        if args.sweep:
            output["threshold_probe"] = threshold_sweep(labels, probs)
    else:
        predictions = (probs >= args.threshold).astype(int)
        output["prediction_rate_positive"] = float(predictions.mean())

    if args.predictions_out is not None:
        pred_path = Path(args.predictions_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prediction"])
            for pred in (probs >= args.threshold).astype(int):
                writer.writerow([int(pred)])

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
