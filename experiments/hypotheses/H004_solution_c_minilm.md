# H004 — Solution C With MiniLM-L12-H384-Uncased

## Component

`WS2 — Solution C Single-Model Stabilization`

## Research Basis

Research record:

- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/research/briefs/R001_solution_c_smaller_model_selection.md`

Local invalidation:

1. `C_XSMALL_FULL_MPS_003` still failed with `mps` OOM after `220` seconds
2. `deberta-v3-xsmall` therefore does not clear the local hardware-fit gate

## Hypothesis

Replacing `microsoft/deberta-v3-xsmall` with `microsoft/MiniLM-L12-H384-uncased` will reduce the memory footprint enough for the full `Solution C` baseline path to complete on this machine, while keeping the model competitive enough to remain a valid transformer solution for the coursework.

## Proposed Config

- `model_name = microsoft/MiniLM-L12-H384-uncased`
- `max_length = 192`
- `train_bs = 2`
- `eval_bs = 4`
- `infer_eval_bs = 8`
- `grad_accum = 8`

## Run ID

- `C_MINILM_FULL_MPS_004`

## Success Criteria

Mandatory:

1. `oom_count = 0`
2. full notebook completes
3. checkpoint exists
4. prediction CSV exists

Quality floor:

1. `macro_f1 >= 0.708348`

That floor is the verified official ED LSTM baseline.

## Failure Criteria

1. any OOM
2. missing checkpoint
3. missing prediction file
4. quality below the accepted floor

## Exact Next Action

Update the `Solution C` baseline notebook to use:

- `microsoft/MiniLM-L12-H384-uncased`

Then run a full-data notebook execution under a new run directory:

- `experiments/runs/C_MINILM_FULL_MPS_004`
