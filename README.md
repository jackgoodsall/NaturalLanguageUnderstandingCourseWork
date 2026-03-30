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
2. `Solution C`: transformer-based DeBERTa notebooks plus an explicit marker-facing inference notebook:
   - `solution_c_demo_inference.ipynb`
   - `solution_c_baseline_development.ipynb`
   - `solution_c_5_seed_ensemble_development.ipynb`
   - `solution_c_5_fold_ensemble_development.ipynb`

Official coursework files are stored under `official_coursework`.

Poster sources:

- content draft: `poster.md`
- poster-ready layout source: `poster.html`
- exported poster PDF: `poster.pdf`

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

- `solution_c_demo_inference.ipynb`
- `solution_c_baseline_development.ipynb`
- `solution_c_5_seed_ensemble_development.ipynb`
- `solution_c_5_fold_ensemble_development.ipynb`
- `src/solution_c/README.md`

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

1. `solution_c_demo_inference.ipynb` for marker-facing inference on the locked submission system
2. `solution_c_baseline_development.ipynb`
3. `solution_c_5_seed_ensemble_development.ipynb`
4. `solution_c_5_fold_ensemble_development.ipynb`

For the current official submission system, use the transfer-ensemble record in:

- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- `experiments/runs/C_REMOTE_A40_019_transfer_seed3/run_transfer_seed3.py`
- `solution_c_demo_inference.ipynb`

Notes:

- `solution_c_demo_inference.ipynb` is the explicit marker-facing demo path for the locked three-seed transfer ensemble.
- `solution_c_5_seed_ensemble_development.ipynb` remains the historical development notebook, not the primary submission demo.

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

Test-day execution checklist:

- `ops/TEST_DAY_RUNBOOK.md`

Final prediction filenames on submission day:

- `Group_n_B.csv`
- `Group_n_C.csv`

## Attribution and Reused Resources

This coursework branch builds on the official coursework materials and standard external model/software resources listed below.

### Official coursework resources

- task specification: `official_coursework/NaturalLanguageUnderstandingCourseWorkSpec.pdf`
- official train/dev data: `training_data/train.csv`, `training_data/dev.csv`
- official trial files: `official_coursework/trial_data`
- official local scorer: `official_coursework/nlu_bundle-feature-unified-local-scorer`
- official model-card template resources:
  - `COMP34812 Coursework - Model Card Resources/COMP34812_modelcard_template.md`
  - `COMP34812 Coursework - Model Card Resources/Model Card Creation.ipynb`

### External software and model resources

- PyTorch for model training and inference
- Hugging Face `transformers` / `datasets` for the transformer pipeline
- `gensim` pretrained `fasttext-wiki-news-subwords-300` vectors for `Solution B`
- transfer-initialised checkpoint `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for the locked `Solution C`
- scikit-learn, pandas, and NumPy for evaluation and experiment analysis

### Closed-track note

No external task datasets were added to the Evidence Detection training/evaluation workflow. The work stayed within the closed-track coursework setting, with the only external model resource being the public pretrained checkpoint used to initialise the transformer submission path.

## Model Artifact Hosting

Large trained model artifacts are intentionally not committed to git. The final submission package should therefore include cloud-hosted links for the locked `B` and `C` resources.

Current cloud links:

- `Solution B` locked checkpoint / artifact link: `https://drive.google.com/drive/folders/1n7I2mY2hIYhh2yJbz4PNHRUur5ztfW0h`
- `Solution C` seed/ensemble artifact link: `https://drive.google.com/drive/folders/1Qcdft2luk5o9nyy0kPpwk0ozXDZeFPTq`

Locked systems that these cloud links must correspond to:

- `Solution B`: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- `Solution C`: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

## Current Submission Gaps

Remaining step on this branch:

- final group review and merge to `main`

## Use of Generative AI Tools

We used generative AI tools, including OpenAI Codex/ChatGPT, as engineering support tools during parts of the coursework workflow.

Their role included:

- debugging runtime and environment issues
- suggesting refactors and code-structure improvements
- helping draft and revise documentation, summaries, and submission-facing notes
- assisting with experiment bookkeeping and result comparison

The final technical decisions, including method selection, experiment design, hyperparameter choices, interpretation of results, and acceptance of code changes, were made by the team. All submitted code, model cards, poster content, and result summaries were reviewed and edited by us before submission.
