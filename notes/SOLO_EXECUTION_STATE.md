# Solo Execution State

## Current state

`SOLO_B0_FULL_001` is the first full-data solo `Solution B` baseline.

- model: ESIM-style BiLSTM pair classifier
- embeddings: `fasttext-wiki-news-subwords-300`
- runtime: RunPod A40
- run record: `experiments/runs/SOLO_B0_FULL_001/result.json`

## Result summary

At threshold `0.5`:

- accuracy: `0.7945`
- macro F1: `0.7556`
- binary F1: `0.6581`
- MCC: `0.5157`

Best thresholded metrics:

- best accuracy: `0.8058` at threshold `0.65`
- best macro F1: `0.7556` at threshold `0.5`
- best binary F1: `0.6631` at threshold `0.45`
- best MCC: `0.5200` at threshold `0.45`

## Interpretation

This solo baseline clears the official non-transformer floor on `macro_f1` and `matthews_corrcoef`, so the fresh rebuild path is viable.

It does not yet beat the current team-branch `Solution B` target, so the next `B` work should be bounded tuning, not architecture churn.

## Next bounded tuning step

Promote exactly one high-value `B1` change:

1. keep the ESIM architecture fixed
2. keep FastText fixed
3. test trainable embeddings against the frozen-embedding baseline
4. keep the same evaluation contract:
   - primary: `macro_f1`
   - guardrail: `matthews_corrcoef`
