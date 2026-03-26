# Solo Execution State

## Current state

`SOLO_C2_ENSEMBLE_001` is the current solo `Solution C` leader, and `SOLO_B1_TRAINABLE_001` remains the current solo `Solution B` leader.

- model: ESIM-style BiLSTM pair classifier
- embeddings: `fasttext-wiki-news-subwords-300`
- setting change: embeddings made trainable
- runtime: RunPod A40
- `Solution B` run record: `experiments/runs/SOLO_B1_TRAINABLE_001/result.json`
- `Solution B` promotion decision: `experiments/decisions/SOLO_D001_solution_b_b1_promotion.md`
- `Solution C` run record: `experiments/runs/SOLO_C2_ENSEMBLE_001/result.json`
- `Solution C` promotion decision: `experiments/decisions/SOLO_D004_solution_c_c2_ensemble_promotion.md`

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

- accuracy: `0.8925`
- macro F1: `0.8686`
- binary F1: `0.8125`
- MCC: `0.7381`

Best thresholded metrics:

- best accuracy: `0.8949` at threshold `0.6`
- best macro F1: `0.8686` at threshold `0.6`
- best binary F1: `0.8125` at threshold `0.5`
- best MCC: `0.7381` at threshold `0.5`

## Interpretation

`Solution B` beats the official non-transformer floor and also clears the current team-branch `Solution B` target on the tracked metrics.

`Solution C` now has a strong three-seed transfer ensemble on the solo branch. It beats the team-branch `Solution C` on accuracy and macro F1, but is still fractionally below it on binary F1 and MCC. The next bounded move should preserve the transfer backbone and increase ensemble diversity rather than reopening single-model tuning.

## Next bounded step

Start `C3` from the locked research plan:

1. keep the patched fp32 transfer path fixed
2. keep the transfer checkpoint fixed
3. increase only the ensemble diversity next, most likely a 5-seed transfer ensemble
