# Solo Execution State

## Current state

`SOLO_C1_FULL_002` is the current solo `Solution C` leader, and `SOLO_B1_TRAINABLE_001` remains the current solo `Solution B` leader.

- model: ESIM-style BiLSTM pair classifier
- embeddings: `fasttext-wiki-news-subwords-300`
- setting change: embeddings made trainable
- runtime: RunPod A40
- `Solution B` run record: `experiments/runs/SOLO_B1_TRAINABLE_001/result.json`
- `Solution B` promotion decision: `experiments/decisions/SOLO_D001_solution_b_b1_promotion.md`
- `Solution C` run record: `experiments/runs/SOLO_C1_FULL_002/result.json`
- `Solution C` promotion decision: `experiments/decisions/SOLO_D003_solution_c_c1_promotion.md`

## Result summary

### Solution B

At threshold `0.5`:

- accuracy: `0.7830`
- macro F1: `0.7577`
- binary F1: `0.6793`
- MCC: `0.5437`

Best thresholded metrics:

- best accuracy: `0.8157` at threshold `0.6`
- best macro F1: `0.7765` at threshold `0.6`
- best binary F1: `0.6845` at threshold `0.55`
- best MCC: `0.5547` at threshold `0.6`

### Solution C

At threshold `0.5`:

- accuracy: `0.8893`
- macro F1: `0.8650`
- binary F1: `0.8076`
- MCC: `0.7311`

Best thresholded metrics:

- best accuracy: `0.8935` at threshold `0.7`
- best macro F1: `0.8658` at threshold `0.55`
- best binary F1: `0.8083` at threshold `0.45`
- best MCC: `0.7321` at threshold `0.55`

## Interpretation

`Solution B` beats the official non-transformer floor and also clears the current team-branch `Solution B` target on the tracked metrics.

`Solution C` now has a strong transfer-initialized incumbent on the solo branch, but it is still slightly below the team-branch `Solution C` target on binary F1. The next bounded move should keep the `C1` backbone fixed and change only one downstream component, most likely calibration or a small ensemble branch.

## Next bounded step

Start `C2` from the locked research plan:

1. keep the patched fp32 transfer path fixed
2. change exactly one downstream component from `C1`
3. test the smallest high-value move next, likely calibration or a 3-seed transfer ensemble
