# D003 - Reject weight_decay=0.0 Branch

## Chosen option
Keep the `C_REMOTE_A40_003` linear-scheduler baseline and reject the `weight_decay=0.0` branch from `C_REMOTE_A40_006`.

## Alternatives considered
- Keep the current baseline with its existing weight decay.
- Remove weight decay entirely and rerun with all other variables fixed.

## Trade-offs
- `weight_decay=0.0` completed cleanly and still selected epoch 4 / `checkpoint-5380` as best.
- The direct probe only improved `binary_f1` by a negligible amount.
- It regressed on the more decision-critical metrics: accuracy, macro F1, and MCC.

## Reversal condition
Reopen the zero-weight-decay branch only if it is paired with another controlled change and beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without lowering MCC.
