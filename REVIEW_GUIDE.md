# Review Guide

This file is the shortest path through the branch for technical review.

## Current winners

### Solution B

Competitive thresholded `B` result:

- `experiments/runs/B_REMOTE_FULL_001_probe/result.json`

At threshold `0.60`:

- accuracy `0.813702328720891`
- binary_f1 `0.6823935558112774`
- macro_f1 `0.7752941991090772`
- matthews_corrcoef `0.5529387441032668`

Primary `B` code:

- `src/solution_b/train.py`
- `src/solution_b/evaluate.py`
- `src/solution_b/hpo.py`
- `src/solution_b/runtime.py`
- `src/solution_b/models.py`

### Solution C

Primary `C` result:

- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

Default-threshold metrics:

- accuracy `0.8889638879514006`
- binary_f1 `0.8132803632236095`
- macro_f1 `0.8671348982304408`
- matthews_corrcoef `0.7383867359239816`

Best tuned metrics from the same run:

- best accuracy `0.8958825514681067`
- best binary_f1 `0.8147492625368732`
- best macro_f1 `0.8702979131582779`
- best matthews_corrcoef `0.7414099636644351`

Primary `C` notebooks:

- `solution_c_baseline_development.ipynb`
- `solution_c_5_seed_ensemble_development.ipynb`
- `solution_c_5_fold_ensemble_development.ipynb`

Latest transfer run scripts:

- `experiments/runs/C_REMOTE_A40_017_transfer_smoke/run_transfer_smoke.py`
- `experiments/runs/C_REMOTE_A40_018_transfer_3ep/run_transfer_3ep.py`
- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/run_transfer_seed3.py`

## Decision trail

Read these in order if you want the short technical story:

1. `experiments/decisions/D009_solution_path_lock.md`
2. `experiments/decisions/D010_solution_c_transfer_ensemble.md`
3. `experiments/baseline_ledger.csv`

## What is intentionally not in git

The branch does not include:

- large generated remote checkpoints
- pod output directories
- transient logs
- generated prediction CSVs that are already covered by recorded metrics

Those are intentionally excluded to keep the repo reviewable and avoid committing large generated artifacts.
