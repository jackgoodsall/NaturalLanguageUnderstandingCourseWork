# H002 — Solution C Baseline With Reduced Context On MPS

## Component

`WS2 — Solution C Single-Model Stabilization`

## Hypothesis

If the `Solution C` baseline keeps the same model but reduces the `mps` path to:

- `max_length = 192`
- `train_bs = 1`
- `eval_bs = 2`
- `infer_eval_bs = 4`
- `grad_accum = 16`
- gradient checkpointing enabled on non-CUDA devices

then the full notebook run should complete without OOM while keeping enough effective batch size to remain a valid baseline candidate.

## Why This Hypothesis Exists

`H001` established that the first `mps`-safe notebook still failed after `37` seconds with an out-of-memory error during `trainer.train()`.

That means the previous reduction:

- `train_bs = 2`
- `eval_bs = 4`
- `max_length = 256`
- `grad_accum = 8`

was still too aggressive for this machine.

## Run ID

- `C_BASELINE_FULL_MPS_002`

## Exact Command

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
ln -sf official_coursework/trial_data/ED_trial.csv test.csv
source .venv/bin/activate
mkdir -p experiments/runs/C_BASELINE_FULL_MPS_002
jupyter nbconvert \
  --to notebook \
  --execute solution_c_baseline_development.ipynb \
  --output executed_solution_c_baseline.ipynb \
  --output-dir experiments/runs/C_BASELINE_FULL_MPS_002 \
  --ExecutePreprocessor.timeout=0
```

## Required Artifacts

1. executed notebook:
   - `experiments/runs/C_BASELINE_FULL_MPS_002/executed_solution_c_baseline.ipynb`
2. checkpoint:
   - `outputs/deberta_v3_baseline/best_model`
3. run result:
   - `experiments/runs/C_BASELINE_FULL_MPS_002/result.json`
4. prediction CSV:
   - `predictions.csv`

## Success Criteria

Mandatory:
1. `oom_count = 0`
2. full notebook completes
3. checkpoint exists
4. prediction CSV exists

Quality:
1. produce full dev metrics
2. record the run in the baseline ledger

## Failure Criteria

1. another OOM
2. checkpoint missing
3. output notebook missing
4. run completes only partially

## Decision Rule

If `H002` still OOMs, the next move should not be another blind rerun.
The next move should be one of:

1. switch the single-model C baseline to a smaller transformer
2. convert the notebook path into a script where memory can be controlled more tightly
3. fall back to CPU only if the runtime remains practical enough for coursework timelines
