# SOLO_D002 - Promote Patched `C0` As The Solo Transformer Incumbent

## Chosen option

Promote `experiments/runs/SOLO_C0_FULL_002/result.json` as the current solo `Solution C` incumbent.

## Why

The patched fp32 DeBERTa training path is now numerically stable, and the salvaged epoch-3 checkpoint beat the solo `B1` incumbent on the tracked metrics:

- accuracy: `0.8043`
- macro F1: `0.7809`
- binary F1: `0.7094`
- MCC: `0.5898`

Under threshold tuning, `C0` improves further to:

- best accuracy: `0.8361`
- best macro F1: `0.7949`
- best binary F1: `0.7160`
- best MCC: `0.5976`

## Alternatives considered

- Keep `Solution B` as the overall solo incumbent.
- Discard the run because the optimizer-checkpoint save failed at the end of epoch 3.
- Skip `C0` and move straight to transfer-initialized `C1`.

## Trade-offs

- The final trainer save failed due pod disk pressure, but the actual epoch-3 model checkpoint was written and re-evaluated cleanly, so the run is still valid as a measured incumbent.
- `C0` is strong enough to establish the solo transformer baseline, but it is still well below the team-branch `Solution C` target. The next bounded move should be `C1` with transfer initialization, not open-ended tuning on plain `C0`.
