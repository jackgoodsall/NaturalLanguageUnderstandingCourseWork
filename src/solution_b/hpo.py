"""
hpo.py — Hyperparameter optimisation for Solution B using Optuna.

Searches over model architecture (hidden size, layers, dropout, head type)
and training hyperparameters (learning rate, batch size) to minimise
validation loss.  Supports resumable studies via SQLite storage.
"""

import argparse
import json
import os

import optuna
import torch
import torch.nn as nn

from src.solution_b.data import get_dataloaders, load_and_preprocess, load_embeddings
from src.solution_b.models import HEAD_REGISTRY, build_model
from src.solution_b.train import train_loop


def make_objective(train_df, dev_df, embedding_dim: int, n_epochs: int, patience: int, device: str):
    """Create an Optuna objective function closed over the data and settings.

    The returned function takes an Optuna Trial, samples hyperparameters,
    trains a model, and returns the best validation loss achieved.

    Search space:
      - hidden_size:  {64, 128, 256}
      - num_layers:   1–3
      - dropout:      0.1–0.5
      - head:         all registered heads (mlp, linear, deep_mlp)
      - lr:           1e-4 to 1e-2 (log-uniform)
      - batch_size:   {16, 32, 64}

    Args:
        train_df:      preprocessed training DataFrame.
        dev_df:        preprocessed dev DataFrame.
        embedding_dim: dimensionality of the pretrained embeddings (e.g. 300).
        n_epochs:      max epochs per trial (shorter than full training for speed).
        patience:      early stopping patience.
        device:        'cuda' or 'cpu'.

    Returns:
        A callable suitable for study.optimize().
    """

    def objective(trial: optuna.Trial) -> float:
        # --- Sample model architecture ---
        config = {
            "embedding_size": embedding_dim,
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "head": trial.suggest_categorical("head", list(HEAD_REGISTRY.keys())),
        }

        # --- Sample training hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # --- Build dataloaders and model ---
        train_dl, dev_dl = get_dataloaders(train_df, dev_df, batch_size)
        model = build_model(config)

        # Class-imbalance weighting (same as in train.py)
        n_pos = (train_df["label"] == 1).sum()
        n_neg = (train_df["label"] == 0).sum()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

        # --- Train (silently) and return best validation loss ---
        history = train_loop(
            model,
            loss_fn,
            optimiser,
            train_dl,
            dev_dl,
            device=device,
            n_epochs=n_epochs,
            patience=patience,
            verbose=False,  # suppress per-epoch output during HPO
        )

        # Return the best (lowest) validation loss seen during training
        val_losses = [r["val_loss"] for r in history if "val_loss" in r]
        return min(val_losses) if val_losses else float("inf")

    return objective



def parse_args():
    """Parse command-line arguments for hyperparameter optimisation."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation with Optuna")
    # Data paths
    parser.add_argument("--train",      default="training_data/train.csv")
    parser.add_argument("--dev",        default="training_data/dev.csv")
    parser.add_argument("--output-dir", default="outputs/hpo")
    parser.add_argument("--embeddings", default="fasttext-wiki-news-subwords-300")
    # HPO settings
    parser.add_argument("--n-trials",   type=int, default=30)
    parser.add_argument("--epochs",     type=int, default=20,
                        help="Max epochs per trial (use fewer than full training).")
    parser.add_argument("--patience",   type=int, default=5)
    # Optuna study management
    parser.add_argument("--study-name", default="esim_lstm_hpo")
    parser.add_argument("--storage",    default=None,
                        help="Optuna storage URL for resumable studies, e.g. sqlite:///hpo.db")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Infer embedding dimension from model name
    dim = int(args.embeddings.split("-")[-1])

    # --- Load embeddings and preprocess data (done once, shared across trials) ---
    print("Loading embeddings")
    embeddings = load_embeddings(args.embeddings)

    print("Preprocessing data...")
    train_df = load_and_preprocess(args.train, embeddings, dim)
    dev_df = load_and_preprocess(args.dev, embeddings, dim)

    # --- Create or resume an Optuna study ---
    study = optuna.create_study(
        direction="minimize",          # minimise validation loss
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,           # resume if the study already exists in storage
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,        # don't prune the first 5 trials
            n_warmup_steps=3,          # let each trial run at least 3 epochs
        ),
    )

    # --- Run the optimisation ---
    objective = make_objective(train_df, dev_df, dim, args.epochs, args.patience, device)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # --- Report and save results ---
    best = study.best_trial
    print(f"\nBest trial #{best.number}  val_loss={best.value:.4f}")
    print(f"Params: {best.params}")

    results = {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial": best.number,
        "n_trials_completed": len(study.trials),
    }
    out_path = os.path.join(args.output_dir, "hpo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
