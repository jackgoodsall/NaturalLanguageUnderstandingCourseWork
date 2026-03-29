"""
evaluate.py — Evaluation and inference for Solution B.

Handles:
  - Loading trained checkpoints (single model or ensemble)
  - Computing predicted probabilities from a model
  - Calculating accuracy, F1 (binary & macro), and classification reports
  - Threshold sweep to find the optimal decision boundary
  - CLI for running evaluation from the command line
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.solution_b.data import ClaimEvidenceDataset, collate_fn, load_and_preprocess, load_embeddings
from src.solution_b.models import EnsembleModel, build_model
from src.solution_b.runtime import resolve_device



@torch.no_grad()
def get_probs(model: torch.nn.Module, dataloader, device: str) -> np.ndarray:
    """Run inference and return predicted probabilities for all samples.

    Applies sigmoid to the raw logits to convert them to [0, 1] probabilities.

    Args:
        model:      trained model (single or ensemble).
        dataloader: DataLoader over the evaluation dataset.
        device:     accelerator target ('cuda', 'mps', or 'cpu').

    Returns:
        1-D numpy array of probabilities, one per sample.
    """
    model.eval()
    all_probs = []
    for X, _ in dataloader:
        X = {k: v.to(device) for k, v in X.items()}
        logits = model(X).squeeze(-1)
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(all_probs)


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics at a given probability threshold.

    Uses the same metrics as the NLU shared-task scorer.
    """
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy_score": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_macro_precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "weighted_macro_recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "weighted_macro_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(labels, preds)),
    }


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    start: float = 0.1,
    stop: float = 1.0,
    step: float = 0.05,
) -> list:
    """Try many thresholds and return results sorted by F1 (descending).

    Useful for finding the optimal decision boundary, which may differ from
    the default 0.5 — especially when class distributions are imbalanced.

    Args:
        probs:  predicted probabilities.
        labels: ground-truth labels.
        start:  lowest threshold to try.
        stop:   upper bound (exclusive).
        step:   increment between thresholds.

    Returns:
        List of dicts with keys: threshold, f1_binary, accuracy.
        Sorted by f1_binary descending (best threshold first).
    """
    results = []
    for t in np.arange(start, stop, step):
        preds = (probs >= t).astype(int)
        results.append(
            {
                "threshold": round(float(t), 2),
                "f1_binary": round(
                    float(f1_score(labels, preds, average="binary", zero_division=0)), 4
                ),
                "accuracy": round(float((preds == labels).mean()), 4),
            }
        )
    return sorted(results, key=lambda r: r["f1_binary"], reverse=True)


def load_checkpoint(checkpoint_path: str, meta_path: str = None):
    """Load a single trained model from a checkpoint file.

    Automatically looks for a 'run_meta.json' in the same directory as the
    checkpoint to reconstruct the model architecture from saved config.

    Args:
        checkpoint_path: path to 'best_model.pt'.
        meta_path:       path to 'run_meta.json' (auto-detected if None).

    Returns:
        (model, meta) tuple — the loaded model and its metadata dict.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Auto-detect metadata file alongside the checkpoint
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(checkpoint_path), "run_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    # Rebuild the model architecture from the saved config, then load weights
    model = build_model(meta["config"])
    model.load_state_dict(ckpt["model_state"])
    return model, meta


def load_ensemble(checkpoint_paths: list, meta_paths: list = None, weights: list = None):
    """Load multiple checkpoints and wrap them in an EnsembleModel.

    Args:
        checkpoint_paths: list of paths to 'best_model.pt' files.
        meta_paths:       corresponding 'run_meta.json' paths (auto-detected if None).
        weights:          optional ensemble weights (defaults to uniform).

    Returns:
        An EnsembleModel that averages predictions from all loaded models.
    """
    if meta_paths is None:
        meta_paths = [None] * len(checkpoint_paths)
    models = [load_checkpoint(cp, mp)[0] for cp, mp in zip(checkpoint_paths, meta_paths)]
    return EnsembleModel(models, weights=weights)


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Path(s) to best_model.pt. Multiple paths → ensemble."
    )
    parser.add_argument(
        "--metas", nargs="*",
        help="Corresponding run_meta.json paths (auto-detected if omitted)."
    )
    parser.add_argument(
        "--weights", nargs="*", type=float,
        help="Optional ensemble weights (must match --checkpoints count)."
    )
    parser.add_argument("--data",      default="training_data/dev.csv")
    parser.add_argument("--embeddings", default="fasttext-wiki-news-subwords-300")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep",     action="store_true",
                        help="Print top-10 threshold sweep results.")
    parser.add_argument("--output",    default=None,
                        help="Optional path to save evaluation results as JSON.")
    parser.add_argument("--submission", default=None,
                        help="Path to save scorer-compatible submission file (one pred per line).")
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    # Infer embedding dimension from model name
    dim = int(args.embeddings.split("-")[-1])

    # --- Load embeddings and preprocess the evaluation data ---
    print("Loading embeddings")
    embeddings = load_embeddings(args.embeddings)

    print("Preprocessing data.")
    df = load_and_preprocess(args.data, embeddings, dim)
    labels = df["label"].values if "label" in df.columns else None

    # Build a DataLoader for the evaluation set
    ds = ClaimEvidenceDataset(
        df["claim_vecs"], df["evidence_vecs"],
        df["label"] if labels is not None else None,
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # --- Load model(s) ---
    metas = args.metas or [None] * len(args.checkpoints)

    if len(args.checkpoints) == 1:
        # Single model evaluation
        model, _ = load_checkpoint(args.checkpoints[0], metas[0])
    else:
        # Ensemble: load multiple models and average their predictions
        print(f"Ensembling {len(args.checkpoints)} models...")
        model = load_ensemble(args.checkpoints, metas, weights=args.weights)

    model.to(device)

    # --- Get predictions ---
    probs = get_probs(model, dl, device)

    if labels is not None:
        # --- Labelled data: compute and print metrics ---
        results = evaluate(probs, labels, args.threshold)
        print(f"\nThreshold: {results['threshold']}")
        for key, val in results.items():
            if key == "threshold":
                continue
            print(f"  {key:30s} {val:.6f}")

        # Optional: sweep thresholds to find the best decision boundary
        if args.sweep:
            sweep = threshold_sweep(probs, labels)
            print("Threshold sweep (top 10):")
            for r in sweep[:10]:
                print(f"  t={r['threshold']:.2f}  F1={r['f1_binary']}  acc={r['accuracy']}")

        # Optional: save results to JSON
        if args.output:
            results_out = {k: v for k, v in results.items() if k != "report"}
            if args.sweep:
                results_out["sweep"] = sweep
            with open(args.output, "w") as f:
                json.dump(results_out, f, indent=2)
            print(f"\nResults saved to {args.output}")

        # Optional: save scorer-compatible submission file (one pred per line)
        if args.submission:
            preds = (probs >= args.threshold).astype(int)
            with open(args.submission, "w") as f:
                for p in preds:
                    f.write(f"{p}\n")
            print(f"Submission file saved to {args.submission} ({len(preds)} rows)")
    else:
        # --- Unlabelled data: output raw predictions ---
        print("No labels found — writing raw predictions.")
        preds = (probs >= args.threshold).astype(int)
        df["prob"] = probs
        df["pred"] = preds
        out = args.output or "predictions.csv"
        df[["Claim", "Evidence", "prob", "pred"]].to_csv(out, index=False)
        print(f"Predictions saved to {out}")
        if args.submission:
            with open(args.submission, "w") as f:
                for p in preds:
                    f.write(f"{p}\n")
            print(f"Submission file saved to {args.submission} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
