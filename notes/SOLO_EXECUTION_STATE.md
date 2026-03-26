# Solo Execution State

## Current state

`SOLO_B1_TRAINABLE_001` is the current solo `Solution B` leader.

- model: ESIM-style BiLSTM pair classifier
- embeddings: `fasttext-wiki-news-subwords-300`
- setting change: embeddings made trainable
- runtime: RunPod A40
- run record: `experiments/runs/SOLO_B1_TRAINABLE_001/result.json`
- promotion decision: `experiments/decisions/SOLO_D001_solution_b_b1_promotion.md`

## Result summary

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

## Interpretation

This solo run beats the official non-transformer floor and also clears the current team-branch `Solution B` target on the tracked metrics.

The `B` path is now strong enough that the next highest-value move is to stop broad `B` tuning and shift effort to solo `Solution C`.

## Next bounded step

Start solo `Solution C` from the locked research plan:

1. build the stable `C0` DeBERTa cross-encoder path
2. validate the single-model baseline on RunPod
3. then run the first bounded `C` improvement branch only after the baseline is measured cleanly
