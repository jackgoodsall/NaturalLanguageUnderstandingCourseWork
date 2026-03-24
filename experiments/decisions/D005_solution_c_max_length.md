# D005 - Reject max_length=192 Branch

## Chosen option
Keep the `C_REMOTE_A40_003` linear-scheduler baseline and reject the `max_length=192` branch from `C_REMOTE_A40_008`.

## Alternatives considered
- Keep the current baseline at `max_length=256`.
- Reduce sequence length to `192` while holding scheduler, learning rate, epoch budget, and best-checkpoint selection constant.

## Trade-offs
- `max_length=192` completed cleanly only after removing obsolete remote output directories that had filled the pod workspace.
- Best checkpoint still occurred at epoch 4 / `checkpoint-5380`.
- The direct evaluation probe regressed on accuracy, macro F1, binary F1, and MCC versus `C_REMOTE_A40_003`.

## Reversal condition
Reopen the shorter-context branch only if it is paired with another controlled change and beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without lowering MCC.
