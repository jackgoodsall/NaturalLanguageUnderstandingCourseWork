# D009 - Lock Current Primary Solution Path

## Chosen option
Lock `Solution C` as the current primary coursework path, using `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/experiments/runs/C_REMOTE_A40_003/result.json` as the canonical default-threshold baseline.

Keep:
- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/experiments/runs/C_REMOTE_A40_010/result.json` as a calibrated `Solution C` candidate
- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/experiments/runs/B_REMOTE_FULL_001/result.json` as the validated `Solution B` fallback/comparison baseline

## Comparison table

| Path | Evidence | Accuracy | Macro F1 | Binary F1 | MCC | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Solution C default | `C_REMOTE_A40_003` | 0.8399 | 0.8084 | 0.7308 | 0.6206 | Primary baseline |
| Solution C calibrated | `C_REMOTE_A40_010` tuned | 0.8498 | 0.8144 | 0.7381 | 0.6314 | Candidate only |
| Solution B default | `B_REMOTE_FULL_001` | 0.7904 | 0.7615 | 0.6784 | 0.5415 | Fallback |
| Solution B calibrated | `B_REMOTE_FULL_001` tuned | 0.8210 | 0.7753 | 0.6833 | 0.5529 | Fallback candidate |

## Why this path is locked
- `Solution C` is ahead of `Solution B` on all primary default-threshold metrics.
- `Solution C` has already been validated through controlled remote runs and checkpoint audits.
- `Solution C` class weights create a potentially stronger calibrated branch, but they do not clearly beat the default baseline under the locked default-threshold evaluation policy.
- `Solution B` is no longer a doc-only claim; it is a real fallback, but it is not the frontrunner.

## Alternatives considered
- Promote `Solution C` calibrated thresholding immediately as the main path.
- Reopen `Solution B` as the main path.
- Continue broad first-order tuning before locking a primary solution.

## Trade-offs
- Locking `Solution C` now reduces ambiguity and prevents more low-value branch churn.
- Keeping `C_REMOTE_A40_010` as a candidate preserves upside without forcing threshold-policy ambiguity into the main baseline.
- Keeping `Solution B` as fallback protects the submission if `Solution C` packaging or reproducibility fails later.

## Reversal condition
Reopen the primary-path decision only if one of the following happens:
- a future `Solution C` branch beats `C_REMOTE_A40_003` clearly under the same evaluation policy
- the team decides to allow locked calibrated thresholds as the main reported policy
- `Solution B` or an ensemble beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without lowering MCC

## Locked next step
Stop broad baseline tuning and move to submission-oriented work:
1. Make the chosen `Solution C` path reproducible end-to-end from saved artifacts.
2. Build the final `B vs C` summary table for the report/poster.
3. Prepare final prediction generation and submission-safe packaging.
