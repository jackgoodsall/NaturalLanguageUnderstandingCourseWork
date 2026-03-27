#!/usr/bin/env python3
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path('/workspace/NaturalLanguageUnderstandingCourseWork')
RUN_DIR = ROOT / 'experiments/runs/C_REMOTE_A40_017_transfer_smoke'
OUTPUT_ROOT = ROOT / 'outputs/deberta_v3_transfer_smoke_1ep'
TRAIN_CSV = ROOT / 'training_data/train.csv'
DEV_CSV = ROOT / 'training_data/dev.csv'
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
SEED = 42
CONFIG = {
    'model_name': MODEL_NAME,
    'max_length': 256,
    'train_bs': 16,
    'eval_bs': 32,
    'grad_accum': 1,
    'lr': 1e-5,
    'epochs': 1,
    'weight_decay': 0.01,
    'warmup_ratio': 0.06,
    'lr_scheduler_type': 'linear',
    'metric_for_best_model': 'f1',
    'seed': SEED,
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)


def logits_to_pos_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return probs[:, 1]


def metrics_dict(labels: np.ndarray, preds: np.ndarray) -> dict:
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'binary_precision': float(bin_p),
        'binary_recall': float(bin_r),
        'binary_f1': float(bin_f1),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'matthews_corrcoef': float(matthews_corrcoef(labels, preds)),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = logits_to_pos_probs(logits)
    preds = (probs >= 0.5).astype(int)
    metrics = metrics_dict(labels, preds)
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['binary_precision'],
        'recall': metrics['binary_recall'],
        'f1': metrics['binary_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
        'mcc': metrics['matthews_corrcoef'],
    }


def threshold_predictions(pos_probs: np.ndarray, threshold: float) -> np.ndarray:
    return (pos_probs >= threshold).astype(int)


def sweep_thresholds(labels: np.ndarray, pos_probs: np.ndarray) -> dict:
    thresholds = [round(float(x), 4) for x in np.linspace(0.05, 0.95, 181)]
    tracked_metrics = ('accuracy', 'binary_f1', 'macro_f1', 'matthews_corrcoef')
    best_by_metric = {}
    for threshold in thresholds:
        preds = threshold_predictions(pos_probs, threshold)
        metric_values = metrics_dict(labels, preds)
        for metric_name in tracked_metrics:
            current_best = best_by_metric.get(metric_name)
            candidate = {
                'threshold': threshold,
                metric_name: float(metric_values[metric_name]),
                'prediction_rate_positive': float((preds == 1).mean()),
            }
            if current_best is None or candidate[metric_name] > current_best[metric_name] or (
                math.isclose(candidate[metric_name], current_best[metric_name])
                and abs(threshold - 0.5) < abs(current_best['threshold'] - 0.5)
            ):
                best_by_metric[metric_name] = candidate
    return {'best_by_metric': best_by_metric}


def build_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[['Claim', 'Evidence', 'label']], preserve_index=False)

    def tok_fn(batch):
        return TOKENIZER(batch['Claim'], batch['Evidence'], truncation=True, max_length=CONFIG['max_length'])

    return ds.map(tok_fn, batched=True, remove_columns=['Claim', 'Evidence'])


train_df = pd.read_csv(TRAIN_CSV)
dev_df = pd.read_csv(DEV_CSV)
train_df['label'] = train_df['label'].astype(int)
dev_df['label'] = dev_df['label'].astype(int)

dev_labels = dev_df['label'].to_numpy(dtype=np.int64)
train_tok = build_dataset(train_df)
dev_tok = build_dataset(dev_df)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
    id2label={0: 'not_relevant', 1: 'relevant'},
    label2id={'not_relevant': 0, 'relevant': 1},
    torch_dtype=torch.float32,
)
model.to(dtype=torch.float32)

args = TrainingArguments(
    output_dir=str(OUTPUT_ROOT),
    learning_rate=CONFIG['lr'],
    per_device_train_batch_size=CONFIG['train_bs'],
    per_device_eval_batch_size=CONFIG['eval_bs'],
    gradient_accumulation_steps=CONFIG['grad_accum'],
    num_train_epochs=CONFIG['epochs'],
    weight_decay=CONFIG['weight_decay'],
    warmup_ratio=CONFIG['warmup_ratio'],
    lr_scheduler_type=CONFIG['lr_scheduler_type'],
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model=CONFIG['metric_for_best_model'],
    greater_is_better=True,
    save_total_limit=1,
    fp16=False,
    optim='adamw_torch',
    adam_epsilon=1e-6,
    dataloader_pin_memory=torch.cuda.is_available(),
    report_to='none',
    seed=SEED,
    data_seed=SEED,
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
    processing_class=TOKENIZER,
    data_collator=COLLATOR,
    compute_metrics=compute_metrics,
)

train_output = trainer.train()
pred_output = trainer.predict(dev_tok)
pos_probs = logits_to_pos_probs(pred_output.predictions)
preds_05 = threshold_predictions(pos_probs, 0.5)
metrics_05 = metrics_dict(dev_labels, preds_05)
threshold_probe = sweep_thresholds(dev_labels, pos_probs)
trainer.state.save_to_json(str(RUN_DIR / 'trainer_state_final.json'))
trainer.save_model(str(OUTPUT_ROOT / 'best_model'))
TOKENIZER.save_pretrained(str(OUTPUT_ROOT / 'best_model'))

summary = {
    'run_id': 'C_REMOTE_A40_017',
    'variant': 'deberta_v3_transfer_smoke_1ep',
    'device': DEVICE,
    'config': CONFIG,
    'best_model_checkpoint': trainer.state.best_model_checkpoint,
    'best_metric_name': CONFIG['metric_for_best_model'],
    'best_metric_value': float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
    'metrics_at_threshold_0_5': metrics_05,
    'threshold_probe': threshold_probe,
    'train_metrics': train_output.metrics,
}
with open(RUN_DIR / 'result.json', 'w') as f:
    json.dump(summary, f, indent=2)

for child in OUTPUT_ROOT.iterdir():
    if child.is_dir() and child.name.startswith('checkpoint-'):
        shutil.rmtree(child, ignore_errors=True)

print(json.dumps(summary, indent=2))
