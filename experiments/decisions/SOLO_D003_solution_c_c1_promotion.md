# SOLO_D003 - Promote Transfer-Initialized `C1` As The Solo Transformer Leader

## Chosen option

Promote `experiments/runs/SOLO_C1_FULL_002/result.json` as the current solo `Solution C` leader.

## Why

`C1` changes only the initialization checkpoint relative to `C0`, keeps the patched fp32 training path fixed, and materially improves every tracked metric:

- accuracy: `0.8893`
- macro F1: `0.8650`
- binary F1: `0.8076`
- MCC: `0.7311`

Under threshold tuning, `C1` improves further to:

- best accuracy: `0.8935`
- best macro F1: `0.8658`
- best binary F1: `0.8083`
- best MCC: `0.7321`

## Alternatives considered

- Keep plain `C0` as the solo transformer incumbent.
- Continue tuning plain DeBERTa before testing transfer initialization.
- Stop the `C` search after establishing the first valid transformer baseline.

## Trade-offs

- `C1` is clearly better than `C0`, so it should replace it as the incumbent.
- It still does not beat the current team-branch `Solution C` target on binary F1, so the solo `C` search is not finished.
- The next bounded move should change one component only from `C1`, most likely calibration policy or a small ensemble branch.
