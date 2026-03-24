# H003 — Solution C With DeBERTa V3 XSmall

## Component

`WS2 — Solution C Single-Model Stabilization`

## Research Basis

Research record:

- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/research/briefs/R001_solution_c_smaller_model_selection.md`

Primary source rationale:

1. `microsoft/deberta-v3-xsmall` keeps the same DeBERTa V3 family as the current baseline.
2. Official model-card sizing is materially smaller than the current base model.
3. It is a lower-risk substitution than switching immediately to a different family.

## Hypothesis

Replacing `microsoft/deberta-v3-base` with `microsoft/deberta-v3-xsmall` will allow the full `Solution C` baseline notebook to complete on this machine without OOM, while remaining competitive enough to keep as the primary transformer candidate.

## Proposed Config

- `model_name = microsoft/deberta-v3-xsmall`
- `max_length = 192`
- `train_bs = 2`
- `eval_bs = 4`
- `infer_eval_bs = 8`
- `grad_accum = 8`

## Run ID

- `C_XSMALL_FULL_MPS_003`

## Success Criteria

Mandatory:

1. `oom_count = 0`
2. full notebook completes
3. checkpoint exists
4. prediction CSV exists

Quality floor:

1. `macro_f1 >= 0.708348`

That floor is the verified official ED LSTM baseline.

Promotion condition:

1. if the run completes and materially exceeds the B/LSTM floor, keep it as the lead `Solution C` candidate

## Failure Criteria

1. any OOM
2. missing checkpoint
3. missing prediction file
4. quality below the accepted floor

## Exact Next Action

Update the `Solution C` baseline notebook to use:

- `microsoft/deberta-v3-xsmall`

Then run a full-data notebook execution under a new run directory:

- `experiments/runs/C_XSMALL_FULL_MPS_003`

## Reversal Condition

If this run still fails or underperforms, move to:

- `microsoft/MiniLM-L12-H384-uncased`
