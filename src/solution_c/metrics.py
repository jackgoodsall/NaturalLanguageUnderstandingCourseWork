from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits)
    if array.ndim == 1:
        return 1.0 / (1.0 + np.exp(-array))
    if array.shape[1] == 1:
        squeezed = array[:, 0]
        return 1.0 / (1.0 + np.exp(-squeezed))

    shifted = array - np.max(array, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs[:, 1]


def compute_metrics(labels: Iterable[int], probs: Iterable[float], threshold: float = 0.5) -> dict[str, float | dict]:
    y_true = np.asarray(list(labels), dtype=int)
    y_prob = np.asarray(list(probs), dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_macro_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_macro_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "binary_precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "binary_recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "prediction_rate_positive": float(y_pred.mean()),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def threshold_sweep(
    labels: Iterable[int],
    probs: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 91, 5)]

    scores = []
    for threshold in thresholds:
        metrics = compute_metrics(labels, probs, threshold=threshold)
        scores.append(metrics)

    best_by_metric = {}
    for metric_name in ["accuracy", "macro_f1", "binary_f1", "matthews_corrcoef"]:
        best = max(scores, key=lambda row: row[metric_name])
        best_by_metric[metric_name] = {
            "threshold": best["threshold"],
            metric_name: best[metric_name],
            "prediction_rate_positive": best["prediction_rate_positive"],
        }

    scores.sort(key=lambda row: (row["macro_f1"], row["matthews_corrcoef"]), reverse=True)
    return {
        "best_by_metric": best_by_metric,
        "top_thresholds_by_macro_f1": [
            {
                "threshold": row["threshold"],
                "macro_f1": row["macro_f1"],
                "matthews_corrcoef": row["matthews_corrcoef"],
                "binary_f1": row["binary_f1"],
                "accuracy": row["accuracy"],
            }
            for row in scores[:5]
        ],
    }
