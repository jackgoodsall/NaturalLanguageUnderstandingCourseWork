# D004 - Reject warmup_ratio=0.10 Branch

## Chosen option
Keep the `C_REMOTE_A40_003` linear-scheduler baseline and reject the `warmup_ratio=0.10` branch from `C_REMOTE_A40_007`.

## Alternatives considered
- Keep the current baseline with its existing warmup behavior.
- Increase the warmup ratio to `0.10` and rerun with all other variables fixed.

## Trade-offs
- `warmup_ratio=0.10` completed cleanly and still selected epoch 4 / `checkpoint-5380` as best.
- It regressed on the more decision-critical metrics: accuracy, macro F1, binary F1, and MCC.
- There was no compensating improvement large enough to justify keeping the branch.

## Reversal condition
Reopen the higher-warmup branch only if it is paired with another controlled change and beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without lowering MCC.
