# D007 - Keep Class Weights As A Calibrated Candidate, Not The Default Baseline

## Chosen option
Keep `experiments/runs/C_REMOTE_A40_003/result.json` as the canonical default-threshold `Solution C` baseline, and retain `C_REMOTE_A40_010` only as a calibrated-threshold candidate.

## Alternatives considered
- Replace the current baseline with the class-weight branch.
- Reject the class-weight branch entirely.
- Keep the current baseline while preserving the class-weight branch for threshold-tuned evaluation.

## Trade-offs
- At the default threshold (`0.5`), `C_REMOTE_A40_010` regressed slightly on `accuracy` and `macro_f1` relative to `C_REMOTE_A40_003`.
- It improved `binary_f1` and `MCC` slightly at the default threshold, but not enough to justify replacing the canonical baseline.
- Under tuned thresholds, the class-weight branch improved `macro_f1`, `binary_f1`, and `MCC`, so discarding it would throw away a useful calibrated variant.

## Reversal condition
Promote the class-weight branch to the canonical baseline only if a future controlled run or locked thresholding policy makes its primary reported metric clearly better than `C_REMOTE_A40_003` without increasing evaluation ambiguity.
