# SOLO_D005 - Do Not Promote Five-Seed `C3` Ensemble

## Chosen option

Keep `experiments/runs/SOLO_C2_ENSEMBLE_001/result.json` as the solo `Solution C` incumbent.

Record `experiments/runs/SOLO_C3_ENSEMBLE_001/result.json` as a measured but non-promoted branch.

## Why

`C3` improved some thresholded metrics, but it does not clear the current promotion rule over `C2`.

Compared with `C2` at the active incumbent thresholding policy:

- default accuracy: lower
- default macro F1: lower
- default binary F1: lower
- default MCC: lower

At tuned thresholds, `C3` reaches:

- macro F1: `0.8688523930525125`
- MCC: `0.7418570888709031`

But the macro F1 gain over `C2` is only about `+0.00028`, which is below the current `+0.003` promotion threshold in `notes/SOLO_MODEL_RESEARCH_PLAN.md`.

## Alternatives considered

- Promote `C3` anyway because its tuned binary F1 is higher.
- Replace `C2` with `C3` as the calibrated incumbent.

## Trade-offs

- `C3` shows that more ensemble diversity can move binary F1 and MCC slightly.
- But the gain is too small to justify promotion under the current search rule.
- The next move should change a different `C` component or shift effort back to widening the `B` lead.
