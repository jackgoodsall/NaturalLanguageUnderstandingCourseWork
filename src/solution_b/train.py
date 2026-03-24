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
import torch.nn.functional as F

from src.solution_b.data import get_dataloaders, load_and_preprocess, load_embeddings
from src.solution_b.models import build_model


class FocalLoss(nn.Module):
    """Focal loss for binary classification (operates on raw logits).

    Down-weights easy examples so the model focuses on hard ones.
    With gamma=0 this is equivalent to BCEWithLogitsLoss.

    Args:
        gamma:      focusing parameter (higher = more focus on hard examples).
        pos_weight: scalar weight for positive class (like BCEWithLogitsLoss).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Standard BCE per-element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = probability of the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        # Apply pos_weight to positive samples
        if self.pos_weight is not None:
            weight = targets * (self.pos_weight - 1) + 1
            loss = loss * weight
        return loss.mean()


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: str,
    X: dict,
    y: torch.Tensor,
    max_grad_norm: float = 0.0,
) -> float:
    """Perform a single gradient-update step on one mini-batch.

    Args:
        model:         the model to train.
        loss_fn:       loss function (e.g. BCEWithLogitsLoss or FocalLoss).
        optimiser:     optimiser instance (e.g. AdamW).
        device:        'cuda' or 'cpu'.
        X:             batch dict with 'claim', 'evidence', lengths.
        y:             (B,) ground-truth labels.
        max_grad_norm: if > 0, clip gradient norms to this value.

    Returns:
        Scalar loss value for this batch.
    """
    model.train()
    outputs = model(X)
    loss = loss_fn(outputs.squeeze(-1), y)
    optimiser.zero_grad()
    loss.backward()
    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    max_grad_norm: float = 0.0,
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
        device:          'cuda' or 'cpu' (auto-detected if None).
        n_epochs:        maximum number of epochs.
        patience:        early stopping patience (epochs without improvement).
        checkpoint_path: file path to save best model weights (.pt).
        verbose:         if False, suppresses per-epoch print output (useful in HPO).

    Returns:
        List of dicts, one per epoch, with keys 'epoch', 'train_loss',
        and optionally 'val_loss'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
            train_loss += train_step(model, loss_fn, optimiser, device, X, y, max_grad_norm)
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
    parser.add_argument("--num-layers",  type=int,   default=1)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--head",        default="mlp",
                        choices=["mlp", "linear", "deep_mlp"])
    parser.add_argument("--embed-proj",  action="store_true",
                        help="Add a trainable projection layer on top of frozen embeddings.")
    # Training hyperparameters
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    # Data augmentation
    parser.add_argument("--token-dropout", type=float, default=0.0,
                        help="Probability of dropping each token embedding during training.")
    parser.add_argument("--noise-sigma",   type=float, default=0.0,
                        help="Std of Gaussian noise added to embeddings during training.")
    parser.add_argument("--oversample",    type=float, default=0.0,
                        help="Oversample minority class to this ratio of majority (1.0=balanced).")
    # Training improvements
    parser.add_argument("--focal-gamma",   type=float, default=0.0,
                        help="Focal loss gamma (0=standard BCE, 2.0=recommended focal).")
    parser.add_argument("--grad-clip",     type=float, default=0.0,
                        help="Max gradient norm for clipping (0=disabled, 1.0=recommended).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load pretrained word embeddings ---
    print("Loading embeddings")
    embeddings = load_embeddings(args.embeddings)
    # Infer embedding dimension from the model name (e.g. "...-300" → 300)
    dim = int(args.embeddings.split("-")[-1])

    # --- Preprocess data: tokenise and vectorise ---
    print("Preprocessing data")
    train_df = load_and_preprocess(args.train, embeddings, dim)
    dev_df = load_and_preprocess(args.dev, embeddings, dim)
    train_dl, dev_dl = get_dataloaders(
        train_df, dev_df, args.batch_size,
        token_dropout_p=args.token_dropout,
        noise_sigma=args.noise_sigma,
        oversample_ratio=args.oversample,
    )

    # --- Build model ---
    config = {
        "embedding_size": dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "head": args.head,
        "embed_projection": args.embed_proj,
    }
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  head={args.head}")

    # --- Set up loss with class-imbalance weighting ---
    # pos_weight = num_negatives / num_positives, so the loss upweights
    # the minority class to counteract label imbalance
    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    if args.focal_gamma > 0:
        loss_fn = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
        print(f"Using focal loss (gamma={args.focal_gamma})")
    else:
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
        n_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        max_grad_norm=args.grad_clip,
    )

    # --- Save run metadata (config + training history) ---
    meta = {"config": config, "history": history, "args": vars(args)}
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
