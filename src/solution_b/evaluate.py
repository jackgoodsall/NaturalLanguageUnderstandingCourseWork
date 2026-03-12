import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.solution_b.data import ClaimEvidenceDataset, collate_fn, load_and_preprocess, load_glove
from src.solution_b.models import EnsembleModel, build_model


@torch.no_grad()
def get_probs(model: torch.nn.Module, dataloader, device: str) -> np.ndarray:
    model.eval()
    all_probs = []
    for X, _ in dataloader:
        X = {k: v.to(device) for k, v in X.items()}
        logits = model(X).squeeze(-1)
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(all_probs)


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float((preds == labels).mean()),
        "f1_binary": float(f1_score(labels, preds, average="binary")),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "report": classification_report(labels, preds),
    }


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    start: float = 0.1,
    stop: float = 1.0,
    step: float = 0.05,
) -> list:
    """Returns results sorted by F1 descending."""
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
    """Load a model from a checkpoint. Auto-detects run_meta.json if not given."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(checkpoint_path), "run_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    model = build_model(meta["config"])
    model.load_state_dict(ckpt["model_state"])
    return model, meta


def load_ensemble(checkpoint_paths: list, meta_paths: list = None, weights: list = None):
    """Load multiple checkpoints and wrap them in an EnsembleModel."""
    if meta_paths is None:
        meta_paths = [None] * len(checkpoint_paths)
    models = [load_checkpoint(cp, mp)[0] for cp, mp in zip(checkpoint_paths, meta_paths)]
    return EnsembleModel(models, weights=weights)


def parse_args():
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
    parser.add_argument("--glove",     default="glove-wiki-gigaword-100")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep",     action="store_true",
                        help="Print top-10 threshold sweep results.")
    parser.add_argument("--output",    default=None,
                        help="Optional path to save evaluation results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = int(args.glove.split("-")[-1])

    print("Loading GloVe")
    glove = load_glove(args.glove)

    print("Preprocessing data.")
    df = load_and_preprocess(args.data, glove, dim)
    labels = df["label"].values if "label" in df.columns else None

    ds = ClaimEvidenceDataset(
        df["claim_vecs"], df["evidence_vecs"],
        df["label"] if labels is not None else None,
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    metas = args.metas or [None] * len(args.checkpoints)

    if len(args.checkpoints) == 1:
        model, _ = load_checkpoint(args.checkpoints[0], metas[0])
    else:
        print(f"Ensembling {len(args.checkpoints)} models...")
        model = load_ensemble(args.checkpoints, metas, weights=args.weights)

    model.to(device)
    probs = get_probs(model, dl, device)

    if labels is not None:
        results = evaluate(probs, labels, args.threshold)
        print(f"\nThreshold : {results['threshold']}")
        print(f"Accuracy  : {results['accuracy']:.4f}")
        print(f"F1 binary : {results['f1_binary']:.4f}")
        print(f"F1 macro  : {results['f1_macro']:.4f}")
        print(results["report"])

        if args.sweep:
            sweep = threshold_sweep(probs, labels)
            print("Threshold sweep (top 10):")
            for r in sweep[:10]:
                print(f"  t={r['threshold']:.2f}  F1={r['f1_binary']}  acc={r['accuracy']}")

        if args.output:
            results_out = {k: v for k, v in results.items() if k != "report"}
            if args.sweep:
                results_out["sweep"] = sweep
            with open(args.output, "w") as f:
                json.dump(results_out, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # No labels — just dump predictions
        print("No labels found — writing raw predictions.")
        df["prob"] = probs
        df["pred"] = (probs >= args.threshold).astype(int)
        out = args.output or "predictions.csv"
        df[["Claim", "Evidence", "prob", "pred"]].to_csv(out, index=False)
        print(f"Predictions saved to {out}")


if __name__ == "__main__":
    main()
