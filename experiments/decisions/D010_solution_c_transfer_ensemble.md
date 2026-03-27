# D010 - Promote Transfer-Ensemble Solution C As The Primary Path

## Chosen option
Promote `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json` to the primary `Solution C` path and current technical frontrunner for submission work.

Keep:
- `experiments/runs/C_REMOTE_A40_018_transfer_3ep/result.json` as the strongest single-seed transfer baseline
- `experiments/runs/B_REMOTE_FULL_001/result.json` as the validated `Solution B` fallback

## Alternatives considered
- Keep `experiments/runs/C_REMOTE_A40_003/result.json` as the primary path.
- Promote `experiments/runs/C_REMOTE_A40_018_transfer_3ep/result.json` and avoid ensemble complexity.
- Reopen broad single-model tuning instead of locking the stronger transfer path.

## Trade-offs
- The transfer-initialized checkpoint materially outperformed the plain DeBERTa baseline, so continuing to treat `C_REMOTE_A40_003` as primary would ignore the strongest current evidence.
- The three-seed ensemble adds some packaging complexity, but it is the first confirmed `Solution C` run to beat the visible Google Doc `C` targets on both accuracy and binary_f1.
- `Solution B` remains useful as a fallback and comparison baseline, but it is not the lead path anymore.

## Reversal condition
Replace `C_REMOTE_A40_019` only if a future run clearly beats it on the locked primary metrics or if ensemble packaging/reproducibility turns out to be materially riskier than the score gain is worth.

## Locked next step
Stop broad model tuning and move to submission-oriented work:
1. Pull the winning artifacts into the repo and branch history.
2. Freeze the final `B` vs `C` comparison table.
3. Prepare reproducible inference, final predictions, and submission packaging.
