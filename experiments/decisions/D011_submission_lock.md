# D011 - Lock The Official Submission Systems

## Chosen systems

Use the following two systems as the official coursework submission pair for `Evidence Detection`:

### Solution B

- locked run record: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- model family: ESIM-style BiLSTM with FastText embeddings
- official operating threshold: `0.60`
- locked dev metrics at threshold `0.60`:
  - accuracy `0.813702328720891`
  - binary_f1 `0.6823935558112774`
  - macro_f1 `0.7752941991090772`
  - matthews_corrcoef `0.5529387441032668`
- marker-facing demo path: `solution_b_demo_inference.ipynb`
- training/evaluation code: `src/solution_b/train.py`, `src/solution_b/evaluate.py`

### Solution C

- locked run record: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- model family: transfer-initialized DeBERTa-v3 three-seed ensemble
- official default operating threshold: `0.50`
- locked dev metrics at threshold `0.50`:
  - accuracy `0.8889638879514006`
  - binary_f1 `0.8132803632236095`
  - macro_f1 `0.8671348982304408`
  - matthews_corrcoef `0.7383867359239816`
- marker-facing demo path: `solution_c_demo_inference.ipynb`
- supporting runner and result record:
  - `experiments/runs/C_REMOTE_A40_019_transfer_seed3/run_transfer_seed3.py`
  - `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

## Alternatives considered

- keep `Solution C` on the earlier plain DeBERTa baseline `C_REMOTE_A40_003`
- leave `Solution B` unlocked and continue broad threshold or architecture search
- defer lock-in until test release

## Why this lock exists

- poster content, model cards, README guidance, and test-day execution all need a single canonical answer for which systems are being submitted
- the locked pair above is the strongest reviewed `B + C` combination currently on `main`
- future work can still happen, but any replacement must explicitly beat this locked pair and update this decision record

## Reversal condition

Replace this lock only if a newly validated run clearly supersedes one of these systems and the README, poster, model cards, and demo path references are updated in the same change set.
