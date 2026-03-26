# Execution Board

## Goal

Deliver a submission-ready `Evidence Detection` coursework repo with two validated solutions (`B + C`), reproducible execution paths, unified metrics, and no undocumented performance claims.

## Definition Of Done

The project is done when all of the following are true:

1. `Solution B` trains, evaluates, and generates prediction files from saved artifacts.
2. `Solution C` single-model baseline trains, evaluates, and generates prediction files from saved artifacts.
3. Any retained ensemble path is numerically justified against the best single model.
4. Every reported metric is backed by a run record that follows `experiments/run_result.schema.json`.
5. Submission artifacts are complete: README, poster, model cards, prediction files, and runnable demo paths.

## Verified Current State

- Track: `Evidence Detection`
- Train rows: `21,508`
- Dev rows: `5,926`
- Train label split: `{0: 15654, 1: 5854}`
- Dev label split: `{0: 4286, 1: 1640}`
- Official verified ED baselines:
  - `SVM`: accuracy `0.797165`, macro F1 `0.721557`, MCC `0.456869`
  - `LSTM`: accuracy `0.805771`, macro F1 `0.708348`, MCC `0.467441`
  - `BERT`: accuracy `0.874452`, macro F1 `0.834774`, MCC `0.674792`

Source of truth for baseline metrics:
- `official_coursework/nlu_bundle-feature-unified-local-scorer`
- `experiments/baseline_ledger.csv`

## State Model

### Current State

- `Solution B` is implemented but operationally heavy.
- `Solution C` is implemented in notebooks and was only partially reproducible before the recent runtime fixes.
- Evaluation records are fragmented.
- Submission-layer artifacts are incomplete.

### Target State

- Every candidate model has:
  - one run ID
  - one command path
  - one metrics record
  - one artifact location
  - one comparison row in the ledger

### Progress Variables

- `B_full_run_complete`: `false`
- `C_full_run_complete`: `false`
- `Unified_metrics_table_complete`: `false`
- `Final_model_selection_locked`: `false`
- `Submission_artifacts_complete`: `false`

## Workstreams

### WS1 — Baseline Control Plane

Objective:
- Standardize how runs, metrics, artifacts, and comparisons are recorded.

Exit Criteria:
- run schema exists
- run template exists
- baseline ledger exists
- canonical commands exist

Acceptance Test:
- a new run can be described entirely with the files under `experiments`

### WS2 — Solution C Single-Model Stabilization

Objective:
- Get one full-data transformer baseline to complete on this machine without OOM and with saved artifacts.

Exit Criteria:
- end-to-end full run completes
- best checkpoint exists
- prediction CSV exists
- metrics are stored in schema-compliant JSON

Acceptance Test:
- rerunning the canonical command does not fail due to environment or device configuration

### WS3 — Solution B Operational Stabilization

Objective:
- Make Solution B fully measurable, reproducible, and cheaper to rerun.

Exit Criteria:
- train path completes
- eval path completes
- prediction path completes
- runtime bottlenecks are measured and documented

Acceptance Test:
- one full B run and one evaluation run are stored in the experiment ledger

### WS4 — Unified Evaluation Layer

Objective:
- Compare B, C single, and any ensemble candidates on identical metrics.

Exit Criteria:
- one comparison table exists
- every row is backed by a run record

Acceptance Test:
- final model choice can be defended numerically without relying on notebook screenshots or memory

### WS5 — Ensemble Retention Decision

Objective:
- Keep an ensemble only if it materially improves on the best single model.

Exit Criteria:
- seed and fold ensemble paths are either validated or explicitly dropped

Acceptance Test:
- each retained ensemble beats the best single model on the agreed threshold, or it is removed from the final plan

### WS6 — Submission Layer

Objective:
- Complete the coursework-facing artifacts after metrics are locked.

Exit Criteria:
- poster complete
- model cards complete
- README complete
- final prediction packaging complete

Acceptance Test:
- all required coursework files exist and contain validated numbers only

## Numeric Decision Rules

Use these rules unless the coursework brief or instructor guidance forces a different standard:

1. Quality upgrade accepted:
   - `macro_f1` improves by at least `+0.005`
2. Runtime upgrade accepted:
   - runtime improves by at least `20%` with no more than `0.005` macro F1 loss
3. Memory stabilization accepted:
   - `oom_count = 0`
4. Reproducibility accepted:
   - repeated run stays within `+-0.01` macro F1
5. Ensemble retention accepted:
   - beats best single model by `+0.005` macro F1 or `+0.01` MCC

## Canonical Metrics

Primary ranking metric:
- `macro_f1`

Secondary metrics:
- `accuracy`
- `macro_precision`
- `macro_recall`
- `matthews_corrcoef`
- `binary_f1`

Operational metrics:
- `train_runtime_seconds`
- `eval_runtime_seconds`
- `infer_runtime_seconds`
- `peak_memory_gb`
- `oom_count`

## File Contract

Run metadata must follow:
- `experiments/run_result.schema.json`

Canonical template:
- `experiments/run_result.template.json`

Baseline table:
- `experiments/baseline_ledger.csv`

Canonical commands:
- `experiments/RUN_COMMANDS.md`

First hypothesis-run card:
- `experiments/hypotheses/H001_solution_c_baseline_mps.md`
