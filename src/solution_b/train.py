from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import (
    ClaimEvidenceDataset,
    Vocabulary,
    add_token_columns,
    build_embedding_matrix,
    collate_batch,
    load_pair_dataframe,
)
from .metrics import compute_metrics, threshold_sweep
from .models import ESIMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a solo Solution B ESIM baseline.")
    parser.add_argument("--train", default="training_data/train.csv")
    parser.add_argument("--dev", default="training_data/dev.csv")
    parser.add_argument("--output-dir", default="outputs/solution_b_b0")
    parser.add_argument("--embeddings", default="fasttext-wiki-news-subwords-300")
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--trainable-embeddings", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_probs(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_token_columns(load_pair_dataframe(args.train, max_rows=args.max_train_rows))
    dev_df = add_token_columns(load_pair_dataframe(args.dev, max_rows=args.max_dev_rows))

    vocab = Vocabulary.build(
        list(train_df["claim_tokens"]) + list(train_df["evidence_tokens"]),
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    embedding_matrix_np = build_embedding_matrix(vocab, args.embeddings, args.embedding_dim, seed=args.seed)
    embedding_matrix = torch.tensor(embedding_matrix_np)

    train_dataset = ClaimEvidenceDataset(train_df, vocab)
    dev_dataset = ClaimEvidenceDataset(dev_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = ESIMModel(
        embedding_matrix=embedding_matrix,
        hidden_size=args.hidden_size,
        projection_size=args.projection_size,
        dropout=args.dropout,
        trainable_embeddings=args.trainable_embeddings,
    ).to(device)

    labels = train_df["label"].to_numpy()
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    pos_weight_value = negative / max(positive, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_macro_f1 = float("-inf")
    best_payload = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())

        dev_labels, dev_probs = predict_probs(model, dev_loader, device)
        dev_metrics = compute_metrics(dev_labels, dev_probs, threshold=args.threshold)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            **{key: value for key, value in dev_metrics.items() if isinstance(value, float)},
        }
        history.append(epoch_record)

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(dev_metrics["macro_f1"])
            epochs_without_improvement = 0
            best_payload = {
                "config": {
                    "embeddings": args.embeddings,
                    "embedding_dim": args.embedding_dim,
                    "hidden_size": args.hidden_size,
                    "projection_size": args.projection_size,
                    "dropout": args.dropout,
                    "trainable_embeddings": args.trainable_embeddings,
                },
                "vocab_tokens": vocab.tokens,
                "state_dict": model.state_dict(),
                "embedding_matrix": embedding_matrix,
                "dev_metrics": dev_metrics,
                "training_args": vars(args),
            }
            torch.save(best_payload, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_payload is None:
        raise RuntimeError("Training finished without a best model checkpoint.")

    model.load_state_dict(best_payload["state_dict"])
    final_labels, final_probs = predict_probs(model, dev_loader, device)
    meta = {
        "best_dev_metrics": best_payload["dev_metrics"],
        "threshold_sweep": threshold_sweep(final_labels, final_probs),
        "history": history,
        "pos_weight": pos_weight_value,
        "device": str(device),
        "train_rows": len(train_df),
        "dev_rows": len(dev_df),
        "vocab_size": len(vocab.tokens),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(json.dumps({
        "best_dev_metrics": best_payload["dev_metrics"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
