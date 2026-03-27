# D001 - Keep Linear Scheduler Baseline

## Chosen option
Keep the epoch-4 linear-scheduler checkpoint from `C_REMOTE_A40_003` as the canonical `Solution C` baseline.

## Alternatives considered
- Switch the five-epoch remote A40 run to a cosine learning-rate scheduler (`C_REMOTE_A40_004`)

## Trade-offs
- Linear keeps the stronger measured dev-set metrics and avoids reopening a branch that already regressed under controlled conditions.
- Cosine remains available for future retesting, but it is currently lower value than training-side changes that could move the baseline beyond `macro_f1 = 0.8084`.

## Reversal condition
Reopen the cosine branch only if a later controlled run changes some other training variable and cosine beats the current linear baseline by at least `+0.005` macro F1 without reducing MCC.
