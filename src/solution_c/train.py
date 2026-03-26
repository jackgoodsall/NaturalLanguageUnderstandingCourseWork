from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import PairClassificationDataset, load_pair_dataframe
from .metrics import compute_metrics, probabilities_from_logits, threshold_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train solo Solution C baseline.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-dev-rows", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    use_cpu = args.device == "cpu"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_pair_dataframe(args.train, max_rows=args.max_train_rows)
    dev_df = load_pair_dataframe(args.dev, max_rows=args.max_dev_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_dataset = PairClassificationDataset.from_dataframe(train_df, tokenizer, max_length=args.max_length)
    dev_dataset = PairClassificationDataset.from_dataframe(dev_df, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "not_relevant", 1: "relevant"},
        label2id={"not_relevant": 0, "relevant": 1},
        torch_dtype=torch.float32,
    )
    model.to(torch.float32)
    model.config.use_cache = False

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() and not use_cpu else None)

    def compute_hf_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        probs = probabilities_from_logits(logits)
        metrics = compute_metrics(labels, probs, threshold=0.5)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "matthews_corrcoef": metrics["matthews_corrcoef"],
            "binary_f1": metrics["binary_f1"],
        }

    signature = inspect.signature(TrainingArguments.__init__)
    training_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": False,
        "bf16": False,
        "use_cpu": use_cpu,
        "optim": "adamw_torch",
        "adam_epsilon": 1e-6,
        "dataloader_pin_memory": torch.cuda.is_available() and not use_cpu,
        "disable_tqdm": True,
    }
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": collator,
        "compute_metrics": compute_hf_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    best_model_dir = output_dir / "best_model"
    prediction_output = trainer.predict(dev_dataset)
    dev_probs = probabilities_from_logits(prediction_output.predictions)
    dev_labels = dev_df["label"].to_numpy(dtype=int)
    best_dev_metrics = compute_metrics(dev_labels, dev_probs, threshold=0.5)
    threshold_probe = threshold_sweep(dev_labels, dev_probs)
    probability_summary = {
        "nan_count": int(np.isnan(dev_probs).sum()),
        "min": float(np.nanmin(dev_probs)) if not np.isnan(dev_probs).all() else None,
        "max": float(np.nanmax(dev_probs)) if not np.isnan(dev_probs).all() else None,
        "mean": float(np.nanmean(dev_probs)) if not np.isnan(dev_probs).all() else None,
        "positive_rate_at_0_5": float((dev_probs >= 0.5).mean()) if not np.isnan(dev_probs).all() else 0.0,
    }

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    run_meta = {
        "model_name": args.model_name,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_metrics": best_dev_metrics,
        "threshold_probe": threshold_probe,
        "probability_summary": probability_summary,
        "train_rows": len(train_df),
        "dev_rows": len(dev_df),
        "training_args": vars(args),
        "log_history": trainer.state.log_history,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(json.dumps({"best_dev_metrics": best_dev_metrics, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
