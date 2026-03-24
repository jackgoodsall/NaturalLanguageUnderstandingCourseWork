# H001 — Solution C Baseline Stabilization On MPS

## Component

`WS2 — Solution C Single-Model Stabilization`

## Hypothesis

The patched `Solution C` baseline notebook will complete a full-data run on this machine using the new `mps`-safe configuration, without OOM, while producing a saved checkpoint and dev metrics that are directly comparable against the official ED baselines.

## Why This Hypothesis Exists

Before the runtime patch:

1. the notebook failed on missing tokenizer dependencies
2. after dependency repair, the notebook failed with `MPS backend out of memory`

After the patch:

1. tokenizer loading uses `use_fast=False`
2. `mps` fallback is enabled
3. train and eval micro-batches are reduced
4. gradient accumulation is added

## Run Pair

Primary run:
- `C_BASELINE_FULL_MPS_001`

Repeat run for variance check:
- `C_BASELINE_FULL_MPS_002`

## Exact Command

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
mkdir -p experiments/runs/C_BASELINE_FULL_MPS_001
jupyter nbconvert \
  --to notebook \
  --execute solution_c_baseline_development.ipynb \
  --output executed_solution_c_baseline.ipynb \
  --output-dir experiments/runs/C_BASELINE_FULL_MPS_001 \
  --ExecutePreprocessor.timeout=0
```

## Required Artifacts

1. executed notebook:
   - `experiments/runs/C_BASELINE_FULL_MPS_001/executed_solution_c_baseline.ipynb`
2. best checkpoint:
   - `outputs/deberta_v3_baseline/best_model`
3. run result JSON:
   - `experiments/runs/C_BASELINE_FULL_MPS_001/result.json`
4. prediction CSV from saved artifact:
   - `predictions.csv` or a copied run-specific path

## Metrics To Record

Primary:
- `macro_f1`

Secondary:
- `accuracy`
- `macro_precision`
- `macro_recall`
- `matthews_corrcoef`
- `binary_f1`

Operational:
- `train_runtime_seconds`
- `eval_runtime_seconds`
- `infer_runtime_seconds`
- `peak_memory_gb`
- `oom_count`

## Pass Conditions

Mandatory:
1. `oom_count = 0`
2. run completes end to end
3. checkpoint exists
4. prediction path works from saved artifact

Quality gate:
1. metrics must be recorded in `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/experiments/run_result.schema.json` format

Comparison gate:
1. compare against `OFFICIAL_BERT_DEV_001`
2. if `macro_f1` is within `0.02` of the official BERT baseline and the run is stable, proceed to optimization
3. if it beats the official BERT baseline, promote it immediately to candidate final single-model C

## Failure Conditions

1. any OOM
2. missing saved checkpoint
3. missing prediction CSV
4. notebook completes but metrics cannot be extracted or trusted

## Better / Worse Definition

The upgrade is better if:

1. stability improves from failure to completion
2. `oom_count` drops from `>0` to `0`
3. the run becomes reproducible enough to repeat

The upgrade is not yet better if:

1. it still OOMs
2. it completes only in test mode
3. it needs manual patching between runs
