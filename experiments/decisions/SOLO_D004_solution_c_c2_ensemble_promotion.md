# SOLO_D004 - Promote Three-Seed Transfer Ensemble As The Solo `Solution C` Leader

## Chosen option

Promote `experiments/runs/SOLO_C2_ENSEMBLE_001/result.json` as the current solo `Solution C` leader.

## Why

The three-seed ensemble keeps the `C1` transfer path fixed and changes only the aggregation component. It improves on `C1` across all tracked metrics:

- accuracy: `0.8925`
- macro F1: `0.8686`
- binary F1: `0.8125`
- MCC: `0.7381`

Compared with `C1`, this is the first solo branch to close almost all of the remaining gap to the team-branch `Solution C` result.

## Alternatives considered

- Keep `C1` as the solo `Solution C` leader.
- Attempt threshold recalibration only.
- Reopen single-model tuning instead of ensembling.

## Trade-offs

- `C2` is clearly stronger than `C1`, so it should replace it as the incumbent.
- The ensemble still misses the team-branch `Solution C` binary F1 by a very small margin, so the solo `C` search is not finished.
- Since threshold tuning does not improve binary F1 beyond the default ensemble threshold, the next bounded move should likely be more ensemble diversity rather than more calibration.
