## Natural Language Understanding Coursework

Track: `Evidence Detection`  
Solution pairing: `B + C`

## Review Start Here

If you are reviewing the branch and want the fastest path through the repo, start with:

1. `REVIEW_GUIDE.md`
2. `experiments/decisions/D011_submission_lock.md`
3. `experiments/decisions/D010_solution_c_transfer_ensemble.md`
4. `experiments/baseline_ledger.csv`

Current locked submission systems:

- `Solution B` official submission record:
  - `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- `Solution C` official submission record:
  - `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

This repository contains two coursework solutions for the COMP34812 NLU coursework:

1. `Solution B`: ESIM-style BiLSTM with pretrained FastText embeddings in `src/solution_b`
2. `Solution C`: transformer-based DeBERTa notebooks in:
   - `solution_c_baseline_development.ipynb`
   - `solution_c_5_seed_ensemble_development.ipynb`
   - `solution_c_5_fold_ensemble_development.ipynb`

Official coursework files are stored under `official_coursework`.

## Environment Setup

Create the local environment:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name nlu-coursework --display-name "NLU CourseWork"
```

Start Jupyter Lab if needed:

```bash
source .venv/bin/activate
jupyter lab
```

## Data Layout

Training data used by the current code lives in:

- `training_data/train.csv`
- `training_data/dev.csv`

Official trial files and the local scorer live in:

- `official_coursework/trial_data`
- `official_coursework/nlu_bundle-feature-unified-local-scorer`

## Solution B

Primary code lives in:

- `src/solution_b`
- `solution_b_development.ipynb`
- `solution_b_demo_inference.ipynb`

Locked submission record:

- `experiments/decisions/D011_submission_lock.md`
- `experiments/runs/B_REMOTE_FULL_001_probe/result.json`

Official thresholded metrics for the submission path:

- threshold `0.60`
- accuracy `0.813702328720891`
- binary_f1 `0.6823935558112774`
- macro_f1 `0.7752941991090772`
- matthews_corrcoef `0.5529387441032668`

Train:

```bash
source .venv/bin/activate
python -m src.solution_b.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_b_run1 \
  --device auto
```

Evaluate:

```bash
source .venv/bin/activate
python -m src.solution_b.evaluate \
  --checkpoints outputs/solution_b_run1/best_model.pt \
  --data training_data/dev.csv \
  --device auto \
  --sweep
```

Notes:

- `--device auto` now supports `cuda`, `mps`, and `cpu`.
- The default embedding model is `fasttext-wiki-news-subwords-300`, which triggers a large one-time gensim download.
- For marker-facing inference/demo use `solution_b_demo_inference.ipynb`.
- For unlabelled CSVs, `src.solution_b.evaluate` can now write both a detailed predictions CSV and a scorer-style one-label-per-line submission file via `--submission`.

## Solution C

Primary notebooks live at the repo root:

- `solution_c_baseline_development.ipynb`
- `solution_c_5_seed_ensemble_development.ipynb`
- `solution_c_5_fold_ensemble_development.ipynb`

Current strongest validated run:

- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

Current lock and promotion decisions:

- `experiments/decisions/D011_submission_lock.md`
- `experiments/decisions/D010_solution_c_transfer_ensemble.md`

Run the notebooks from the project root with the `NLU CourseWork` kernel.

The transformer notebooks now:

- use `use_fast=False` tokenizers to avoid the dependency failure seen during the audit
- auto-detect `cuda`, `mps`, or `cpu`
- reduce batch sizes and increase gradient accumulation on `mps`/`cpu`
- use smaller inference batch sizes on memory-constrained devices

Recommended execution order:

1. `solution_c_baseline_development.ipynb`
2. `solution_c_5_seed_ensemble_development.ipynb`
3. `solution_c_5_fold_ensemble_development.ipynb`

For the current official submission system, use the transfer-ensemble record in:

- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/run_transfer_seed3.py`

## Local Scorer

Run the official scorer:

```bash
cd official_coursework/nlu_bundle-feature-unified-local-scorer
python3 -m unittest tests.test_local_scorer -v
python3 local_scorer/main.py --task ed
```

## Experiment Record

Run history, decision records, and audit tooling live under:

- `experiments/runs`
- `experiments/decisions`
- `experiments/debug`

## Current Submission Gaps

Still incomplete on this branch:

- final trained model artifacts and cloud links
- final predictions on released test data
- final poster export / layout polish
- final attribution and cloud-link section in this README

## Use of Generative AI Tools

This branch has used AI-assisted engineering support for audit, environment setup, and runtime refactoring.  
Before final submission, this section should be updated with the exact tools used and the final human-reviewed scope of accepted changes, in line with the coursework brief.
