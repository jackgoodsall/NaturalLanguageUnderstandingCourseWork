import argparse
import copy
import json
import os

import torch
import torch.nn as nn

from src.solution_b.data import get_dataloaders, load_and_preprocess, load_embeddings
from src.solution_b.models import build_model

## Core training loops

def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: str,
    X: dict,
    y: torch.Tensor,
) -> float:
    model.train()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(-1), y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for X, y in dataloader:
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)
        total += loss_fn(model(X).squeeze(-1), y).item()
    return total / len(dataloader)


def train_loop(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    train_dl,
    val_dl=None,
    device: str = None,
    n_epochs: int = 10,
    patience: int = 5,
    checkpoint_path: str = None,
    verbose: bool = True,
) -> list:
    """
    Train a model and return a list of per-epoch metric dicts.

    Args:
        checkpoint_path: if provided, saves best weights here as a .pt file
        verbose:         set False to silence per-epoch prints (useful in HPO)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    for epoch in range(n_epochs):
        train_loss = 0.0
        for X, y in train_dl:
            X = {k: v.to(device) for k, v in X.items()}
            y = y.to(device)
            train_loss += train_step(model, loss_fn, optimiser, device, X, y)
        train_loss /= len(train_dl)

        row = {"epoch": epoch + 1, "train_loss": train_loss}
        msg = f"Epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}"

        if val_dl is not None:
            val_loss = eval_epoch(model, loss_fn, val_dl, device)
            row["val_loss"] = val_loss
            msg += f"  val={val_loss:.4f}"
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                if checkpoint_path:
                    torch.save({"model_state": best_weights}, checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    history.append(row)
                    break

        history.append(row)
        if verbose:
            print(msg)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESIM-LSTM")
    parser.add_argument("--train",       default="training_data/train.csv")
    parser.add_argument("--dev",         default="training_data/dev.csv")
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--embeddings",  default="fasttext-wiki-news-subwords-300")
    # model
    parser.add_argument("--hidden-size", type=int,   default=128)
    parser.add_argument("--num-layers",  type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--head",        default="mlp",
                        choices=["mlp", "linear", "deep_mlp"])
    # training
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading embeddings")
    embeddings = load_embeddings(args.embeddings)
    dim = int(args.embeddings.split("-")[-1])

    print("Preprocessing data")
    train_df = load_and_preprocess(args.train, embeddings, dim)
    dev_df = load_and_preprocess(args.dev, embeddings, dim)
    train_dl, dev_dl = get_dataloaders(train_df, dev_df, args.batch_size)

    config = {
        "embedding_size": dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "head": args.head,
    }
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  head={args.head}")

    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    ### Used weight in the loss to deal with the class imbalance.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    history = train_loop(
        model,
        loss_fn,
        optimiser,
        train_dl,
        dev_dl,
        n_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    meta = {"config": config, "history": history, "args": vars(args)}
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
