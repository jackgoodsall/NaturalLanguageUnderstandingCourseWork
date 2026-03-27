#!/usr/bin/env python3
"""Evaluate a saved Solution B checkpoint directly with Solution C-style metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solution_b.data import (  # noqa: E402
    ClaimEvidenceDataset,
    collate_fn,
    load_and_preprocess,
    load_embeddings,
)
from src.solution_b.evaluate import (  # noqa: E402
    get_probs,
    load_checkpoint,
    load_ensemble,
)
from src.solution_b.runtime import resolve_device  # noqa: E402


def metrics_dict(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "binary_precision": float(bin_p),
        "binary_recall": float(bin_r),
        "binary_f1": float(bin_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "matthews_corrcoef": float(matthews_corrcoef(labels, preds)),
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
    grid_rows: list[dict[str, float]] = []

    for threshold in thresholds:
        preds = threshold_predictions(pos_probs, threshold)
        metric_values = metrics_dict(labels, preds)
        grid_row = {"threshold": threshold}
        for metric_name in tracked_metrics:
            metric_value = float(metric_values[metric_name])
            grid_row[metric_name] = metric_value
            candidate = {
                "threshold": threshold,
                metric_name: metric_value,
                "prediction_rate_positive": float((preds == 1).mean()),
            }
            current_best = best_by_metric.get(metric_name)
            if current_best is None or metric_value > current_best[metric_name] or (
                np.isclose(metric_value, current_best[metric_name])
                and abs(threshold - 0.5) < abs(current_best["threshold"] - 0.5)
            ):
                best_by_metric[metric_name] = candidate
        grid_rows.append(grid_row)

    midpoint_threshold = 0.5
    midpoint_preds = threshold_predictions(pos_probs, midpoint_threshold)
    midpoint_metrics = metrics_dict(labels, midpoint_preds)

    return {
        "threshold_count": len(thresholds),
        "grid": grid_rows,
        "baseline_threshold_0_5": {
            "threshold": midpoint_threshold,
            **{metric: float(midpoint_metrics[metric]) for metric in tracked_metrics},
            "prediction_rate_positive": float((midpoint_preds == 1).mean()),
        },
        "best_by_metric": best_by_metric,
    }


def load_model(
    checkpoint_paths: list[Path],
    meta_paths: list[Path] | None,
    weights: list[float] | None,
):
    if len(checkpoint_paths) == 1:
        meta_path = meta_paths[0] if meta_paths else None
        model, meta = load_checkpoint(checkpoint_paths[0], meta_path)
        config = meta.get("config", {}) if meta else {}
        return model, meta, config

    model = load_ensemble(checkpoint_paths, meta_paths, weights=weights)
    meta = None
    config = {}
    if meta_paths:
        with meta_paths[0].open() as handle:
            meta = json.load(handle)
        config = meta.get("config", {})
    return model, meta, config


def infer_embeddings_name(
    meta: dict[str, Any] | None,
    fallback_embeddings_name: str,
) -> str:
    if not meta:
        return fallback_embeddings_name
    return meta.get("args", {}).get("embeddings", fallback_embeddings_name)


def infer_embedding_dim(
    config: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    embeddings_name: str,
) -> int:
    if config:
        for key in ("embedding_dim", "embedding_size"):
            if key in config:
                return int(config[key])

    if meta:
        meta_embeddings = meta.get("args", {}).get("embeddings")
        if meta_embeddings:
            embeddings_name = str(meta_embeddings)

    match = re.search(r"(\d+)(?!.*\d)", embeddings_name)
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not infer embedding dimension from config={config} or embeddings='{embeddings_name}'."
    )


def evaluate(
    checkpoint_paths: list[Path],
    meta_paths: list[Path] | None,
    weights: list[float] | None,
    data_csv: Path,
    embeddings_name: str,
    batch_size: int,
    threshold: float,
    threshold_sweep: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    device: str,
    predictions_output: Path | None,
) -> dict[str, Any]:
    model, meta, config = load_model(checkpoint_paths, meta_paths, weights)
    runtime_device = resolve_device(device)
    model.to(runtime_device)

    resolved_embeddings_name = infer_embeddings_name(meta, embeddings_name)
    embedding_dim = infer_embedding_dim(config, meta, resolved_embeddings_name)

    embeddings = load_embeddings(resolved_embeddings_name)
    df = load_and_preprocess(
        str(data_csv),
        embeddings=embeddings,
        dim=embedding_dim,
    )
    labels = df["label"].astype(int).to_numpy()

    dataset = ClaimEvidenceDataset(
        df["claim_vecs"],
        df["evidence_vecs"],
        df["label"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    pos_probs = get_probs(model, dataloader, runtime_device)
    preds = threshold_predictions(pos_probs, threshold)

    summary: dict[str, Any] = {
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "meta_paths": [str(path) for path in meta_paths] if meta_paths else [],
        "data_csv": str(data_csv),
        "rows": int(len(df)),
        "device": runtime_device,
        "batch_size": batch_size,
        "threshold": threshold,
        "embeddings_name": resolved_embeddings_name,
        "embedding_dim": embedding_dim,
        "meta_paths": [str(path) for path in meta_paths] if meta_paths else [],
        "label_counts": {
            str(label): int((labels == label).sum()) for label in np.unique(labels)
        },
        "prediction_counts": {
            str(label): int((preds == label).sum()) for label in np.unique(preds)
        },
        "prediction_rate_positive": float((preds == 1).mean()),
        "positive_probability_mean": float(pos_probs.mean()),
        "positive_probability_min": float(pos_probs.min()),
        "positive_probability_max": float(pos_probs.max()),
        "metrics": metrics_dict(labels, preds),
    }

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

    if predictions_output:
        out_df = df.copy()
        out_df["pred"] = preds.astype(int)
        out_df["positive_probability"] = pos_probs
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(predictions_output, index=False)
        summary["predictions_output"] = str(predictions_output)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--metas", nargs="*")
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--data", required=True)
    parser.add_argument("--embeddings", default="fasttext-wiki-news-subwords-300")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--predictions-output")
    args = parser.parse_args()

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    meta_paths = [Path(path) for path in args.metas] if args.metas else None

    if meta_paths and len(meta_paths) != len(checkpoint_paths):
        raise SystemExit("Number of --metas entries must match --checkpoints.")
    if args.weights and len(args.weights) != len(checkpoint_paths):
        raise SystemExit("Number of --weights entries must match --checkpoints.")

    summary = evaluate(
        checkpoint_paths=checkpoint_paths,
        meta_paths=meta_paths,
        weights=args.weights,
        data_csv=Path(args.data),
        embeddings_name=args.embeddings,
        batch_size=args.batch_size,
        threshold=args.threshold,
        threshold_sweep=args.threshold_sweep,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        device=args.device,
        predictions_output=Path(args.predictions_output) if args.predictions_output else None,
    )

    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
