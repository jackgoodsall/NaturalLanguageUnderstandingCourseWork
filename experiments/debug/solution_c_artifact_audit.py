#!/usr/bin/env python3
"""Audit Solution C training artifacts and summarize checkpoint state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def eval_events_from_state(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        item
        for item in trainer_state.get("log_history", [])
        if any(key.startswith("eval_") for key in item)
    ]
    events.sort(key=lambda item: (item.get("step", -1), item.get("epoch", -1)))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in events:
        signature = (
            item.get("step"),
            item.get("epoch"),
            item.get("eval_accuracy"),
            item.get("eval_macro_f1"),
            item.get("eval_f1"),
            item.get("eval_mcc"),
            item.get("eval_loss"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def primary_metric_name(event: dict[str, Any]) -> str | None:
    if event.get("eval_macro_f1") is not None:
        return "eval_macro_f1"
    if event.get("eval_f1") is not None:
        return "eval_f1"
    return None


def primary_metric_value(event: dict[str, Any]) -> float | None:
    metric_name = primary_metric_name(event)
    if metric_name is None:
        return None
    value = event.get(metric_name)
    return float(value) if value is not None else None


def resolve_checkpoint_reference(output_root: Path, reference: str | None) -> str | None:
    if not reference:
        return None
    reference_path = Path(reference)
    if reference_path.is_absolute():
        return str(reference_path)
    if reference.startswith("./"):
        return str(reference_path.resolve())
    return str((output_root / reference_path).resolve())


def analyze_checkpoint_selection(
    output_root: Path,
    checkpoint_dirs: list[Path],
    checkpoint_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not checkpoint_dirs:
        return {
            "canonical_checkpoint": None,
            "best_epoch": None,
            "best_metric_name": None,
            "best_metric_value": None,
            "latest_epoch": None,
            "latest_metric_value": None,
            "regressed_after_best": None,
            "trainer_state_source": None,
            "eval_events": [],
        }

    latest_checkpoint = checkpoint_dirs[-1]
    latest_state = load_json(latest_checkpoint / "trainer_state.json") or {}
    eval_events = eval_events_from_state(latest_state)

    best_event = None
    best_metric = None
    for event in eval_events:
        metric_value = primary_metric_value(event)
        if metric_value is None:
            continue
        if best_metric is None or metric_value > best_metric:
            best_metric = metric_value
            best_event = event

    latest_event = eval_events[-1] if eval_events else None
    latest_metric = primary_metric_value(latest_event) if latest_event else None

    canonical_checkpoint = resolve_checkpoint_reference(
        output_root,
        latest_state.get("best_model_checkpoint"),
    )

    if canonical_checkpoint is None and best_event is not None:
        best_step = best_event.get("step")
        for summary in checkpoint_summaries:
            if summary["step"] == best_step:
                canonical_checkpoint = summary["path"]
                break

    return {
        "canonical_checkpoint": canonical_checkpoint,
        "best_epoch": best_event.get("epoch") if best_event else None,
        "best_metric_name": primary_metric_name(best_event or {}),
        "best_metric_value": best_metric,
        "latest_epoch": latest_event.get("epoch") if latest_event else None,
        "latest_metric_value": latest_metric,
        "regressed_after_best": (
            latest_metric is not None and best_metric is not None and latest_metric < best_metric
        ),
        "trainer_state_source": str(latest_checkpoint / "trainer_state.json"),
        "eval_events": eval_events,
    }


def summarize_checkpoint(path: Path) -> dict[str, Any]:
    trainer_state = load_json(path / "trainer_state.json") or {}
    config = load_json(path / "config.json") or {}
    eval_events = eval_events_from_state(trainer_state)
    return {
        "path": str(path),
        "step": checkpoint_step(path),
        "exists": path.exists(),
        "dtype": config.get("dtype"),
        "num_labels": config.get("num_labels"),
        "best_metric": trainer_state.get("best_metric"),
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
        "num_eval_events": len(eval_events),
        "eval_events": eval_events,
    }


def build_summary(output_root: Path) -> dict[str, Any]:
    checkpoint_dirs = sorted(
        [path for path in output_root.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    checkpoint_summaries = [summarize_checkpoint(path) for path in checkpoint_dirs]
    checkpoint_analysis = analyze_checkpoint_selection(
        output_root,
        checkpoint_dirs,
        checkpoint_summaries,
    )

    best_model_dir = output_root / "best_model"
    best_model_config = load_json(best_model_dir / "config.json") or {}

    trainer_states = [
        load_json(path / "trainer_state.json") or {} for path in checkpoint_dirs
    ]
    best_metric_values = [state.get("best_metric") for state in trainer_states if state]
    best_checkpoint_values = [
        state.get("best_model_checkpoint") for state in trainer_states if state
    ]

    eval_metric_signatures = []
    for event in checkpoint_analysis["eval_events"]:
        eval_metric_signatures.append(
            {
                "step": event.get("step"),
                "epoch": event.get("epoch"),
                "eval_accuracy": event.get("eval_accuracy"),
                "eval_f1": event.get("eval_f1"),
                "eval_macro_f1": event.get("eval_macro_f1"),
                "eval_mcc": event.get("eval_mcc"),
                "eval_loss": event.get("eval_loss"),
            }
        )

    return {
        "output_root": str(output_root),
        "exists": output_root.exists(),
        "best_model_dir_exists": best_model_dir.exists(),
        "best_model_config": {
            "dtype": best_model_config.get("dtype"),
            "num_labels": best_model_config.get("num_labels"),
            "architectures": best_model_config.get("architectures"),
            "id2label": best_model_config.get("id2label"),
            "label2id": best_model_config.get("label2id"),
        },
        "checkpoint_count": len(checkpoint_summaries),
        "checkpoint_steps": [item["step"] for item in checkpoint_summaries],
        "checkpoint_best_metrics": best_metric_values,
        "checkpoint_best_model_references": best_checkpoint_values,
        "checkpoint_analysis": checkpoint_analysis,
        "eval_metric_signatures": eval_metric_signatures,
        "checkpoints": checkpoint_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/deberta_v3_baseline",
        help="Training output directory that contains checkpoint-* and best_model",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save the audit summary as JSON",
    )
    args = parser.parse_args()

    summary = build_summary(Path(args.output_root))
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
