# Solo Execution State

## Current state

`SOLO_C0_FULL_002` is the current solo `Solution C` incumbent, and `SOLO_B1_TRAINABLE_001` remains the current solo `Solution B` leader.

- model: ESIM-style BiLSTM pair classifier
- embeddings: `fasttext-wiki-news-subwords-300`
- setting change: embeddings made trainable
- runtime: RunPod A40
- `Solution B` run record: `experiments/runs/SOLO_B1_TRAINABLE_001/result.json`
- `Solution B` promotion decision: `experiments/decisions/SOLO_D001_solution_b_b1_promotion.md`
- `Solution C` run record: `experiments/runs/SOLO_C0_FULL_002/result.json`
- `Solution C` promotion decision: `experiments/decisions/SOLO_D002_solution_c_c0_promotion.md`

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

- accuracy: `0.8043`
- macro F1: `0.7809`
- binary F1: `0.7094`
- MCC: `0.5898`

Best thresholded metrics:

- best accuracy: `0.8361` at threshold `0.85`
- best macro F1: `0.7949` at threshold `0.75`
- best binary F1: `0.7160` at threshold `0.6`
- best MCC: `0.5976` at threshold `0.6`

## Interpretation

`Solution B` beats the official non-transformer floor and also clears the current team-branch `Solution B` target on the tracked metrics.

`Solution C` is now a valid measured transformer incumbent on the solo branch, but it is still below the team-branch `Solution C` target. The next bounded move should be the transfer-initialized `C1` branch rather than more open-ended tuning on plain DeBERTa.

## Next bounded step

Start `C1` from the locked research plan:

1. keep the patched fp32 trainer path fixed
2. switch only the initialization to the transfer checkpoint
3. run the first full `C1` transfer baseline on RunPod
