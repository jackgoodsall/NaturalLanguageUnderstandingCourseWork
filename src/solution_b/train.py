"""
train.py — Training loop and CLI entry point for Solution B.

Handles:
  - Single-batch training step and full-epoch evaluation
  - Training loop with early stopping, LR scheduling, and checkpointing
  - CLI for launching training runs with configurable hyperparameters
"""

import argparse
import copy
import json
import os

import torch
import torch.nn as nn

from src.solution_b.data import get_dataloaders, load_and_preprocess, load_embeddings
from src.solution_b.models import build_model
from src.solution_b.runtime import resolve_device


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: str,
    X: dict,
    y: torch.Tensor,
) -> float:
    """Perform a single gradient-update step on one mini-batch.

    Args:
        model:     the model to train.
        loss_fn:   loss function (e.g. BCEWithLogitsLoss).
        optimiser: optimiser instance (e.g. AdamW).
        device:    'cuda' or 'cpu'.
        X:         batch dict with 'claim', 'evidence', lengths.
        y:         (B,) ground-truth labels.

    Returns:
        Scalar loss value for this batch.
    """
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
    """Evaluate the model on an entire dataloader and return mean loss.

    Runs in no-grad mode for efficiency (no gradient computation needed).

    Args:
        model:      the model to evaluate.
        loss_fn:    loss function.
        dataloader: validation/test DataLoader.
        device:     'cuda' or 'cpu'.

    Returns:
        Average loss across all batches.
    """
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
    """Train a model with early stopping and LR scheduling.

    Features:
      - ReduceLROnPlateau: halves the LR if val loss plateaus for 2 epochs.
      - Early stopping: stops training if val loss doesn't improve for
        `patience` consecutive epochs.
      - Checkpointing: saves the best model weights (by val loss) to disk.
      - Best-weight restoration: loads the best weights back into the model
        at the end of training.

    Args:
        model:           the model to train.
        loss_fn:         loss function.
        optimiser:       optimiser instance.
        train_dl:        training DataLoader.
        val_dl:          validation DataLoader (if None, no validation is done).
        device:          accelerator target ('cuda', 'mps', or 'cpu').
        n_epochs:        maximum number of epochs.
        patience:        early stopping patience (epochs without improvement).
        checkpoint_path: file path to save best model weights (.pt).
        verbose:         if False, suppresses per-epoch print output (useful in HPO).

    Returns:
        List of dicts, one per epoch, with keys 'epoch', 'train_loss',
        and optionally 'val_loss'.
    """
    if device is None:
        device = resolve_device()

    model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history = []

    # Reduce LR by half when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    for epoch in range(n_epochs):
        # --- Training pass ---
        train_loss = 0.0
        for X, y in train_dl:
            # Move batch to device
            X = {k: v.to(device) for k, v in X.items()}
            y = y.to(device)
            train_loss += train_step(model, loss_fn, optimiser, device, X, y)
        train_loss /= len(train_dl)

        row = {"epoch": epoch + 1, "train_loss": train_loss}
        msg = f"Epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}"

        # --- Validation pass ---
        if val_dl is not None:
            val_loss = eval_epoch(model, loss_fn, val_dl, device)
            row["val_loss"] = val_loss
            msg += f"  val={val_loss:.4f}"
            scheduler.step(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                # Save checkpoint if a path was provided
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

    # Restore the best weights found during training
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history


def parse_args():
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Train ESIM-LSTM")
    # Data paths
    parser.add_argument("--train",       default="training_data/train.csv")
    parser.add_argument("--dev",         default="training_data/dev.csv")
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--embeddings",  default="fasttext-wiki-news-subwords-300")
    # Model architecture
    parser.add_argument("--hidden-size", type=int,   default=128)
    parser.add_argument("--num-layers",  type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--head",        default="mlp",
                        choices=["mlp", "linear", "deep_mlp"])
    # Training hyperparameters
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--device",      default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # --- Load pretrained word embeddings ---
    print("Loading embeddings")
    embeddings = load_embeddings(args.embeddings)
    # Infer embedding dimension from the model name (e.g. "...-300" → 300)
    dim = int(args.embeddings.split("-")[-1])

    # --- Preprocess data: tokenise and vectorise ---
    print("Preprocessing data")
    train_df = load_and_preprocess(args.train, embeddings, dim)
    dev_df = load_and_preprocess(args.dev, embeddings, dim)
    train_dl, dev_dl = get_dataloaders(train_df, dev_df, args.batch_size)

    # --- Build model ---
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

    # --- Set up loss with class-imbalance weighting ---
    # pos_weight = num_negatives / num_positives, so the loss upweights
    # the minority class to counteract label imbalance
    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    # --- Train ---
    history = train_loop(
        model,
        loss_fn,
        optimiser,
        train_dl,
        dev_dl,
        device=device,
        n_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    # --- Save run metadata (config + training history) ---
    meta = {"config": config, "history": history, "args": vars(args)}
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
