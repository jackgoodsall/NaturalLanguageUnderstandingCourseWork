# D006 - Reject grad_accum=2 Branch

## Chosen option
Keep the `C_REMOTE_A40_003` linear-scheduler baseline and reject the `grad_accum=2` branch from `C_REMOTE_A40_009`.

## Alternatives considered
- Keep the current baseline with its existing effective batch size.
- Reduce gradient accumulation to `2` and rerun with all other variables fixed.

## Trade-offs
- `grad_accum=2` completed cleanly and still selected epoch 4 as best.
- It regressed on accuracy, macro F1, binary F1, and MCC.
- There was no compensating gain that justifies replacing the current baseline.

## Reversal condition
Reopen the lower-gradient-accumulation branch only if it is paired with another controlled change and beats `C_REMOTE_A40_003` by at least `+0.005` macro F1 without lowering MCC.
