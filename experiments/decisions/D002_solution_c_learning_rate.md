# D002 - Reject lr=3e-5 Branch

## Chosen option
Keep the `C_REMOTE_A40_003` linear-scheduler baseline and reject the `lr=3e-5` branch from `C_REMOTE_A40_005`.

## Alternatives considered
- Lower the learning rate from the current baseline setting to `3e-5` while keeping the rest of the training configuration unchanged.

## Trade-offs
- `lr=3e-5` completed cleanly and preserved the same best-checkpoint pattern, so it is a valid branch.
- It still regressed on the only metrics that matter for the current stage: accuracy, macro F1, binary F1, and MCC.
- Rejecting it now keeps the search space smaller and preserves a stronger canonical baseline for the next single-variable test.

## Reversal condition
Reopen the lower-learning-rate branch only if a future controlled run changes one other training variable and the resulting `lr=3e-5` run beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without reducing MCC.
