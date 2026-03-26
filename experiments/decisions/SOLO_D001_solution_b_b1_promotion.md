# SOLO_D001 - Promote Trainable-Embedding `Solution B`

## Chosen option

Promote `experiments/runs/SOLO_B1_TRAINABLE_001/result.json` as the current solo `Solution B` leader.

## Why

Compared with `SOLO_B0_FULL_001`, enabling trainable FastText embeddings improved all target metrics under threshold tuning:

- accuracy: `0.8157`
- macro F1: `0.7765`
- binary F1: `0.6845`
- MCC: `0.5547`

This run also clears the current team-branch `Solution B` target.

## Reversal condition

Revisit this decision only if a later bounded `Solution B` run beats `SOLO_B1_TRAINABLE_001` on `macro_f1` by at least `0.003` without lowering `matthews_corrcoef` by more than `0.002`.
