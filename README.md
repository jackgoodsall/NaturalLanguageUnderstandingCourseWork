## Natural Language Understanding Coursework

Track: `Evidence Detection`  
Solution pairing: `B + C`

This repository contains two coursework solutions for the COMP34812 NLU coursework:

1. `Solution B`: ESIM-style BiLSTM with pretrained FastText embeddings in `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/src/solution_b`
2. `Solution C`: transformer-based DeBERTa notebooks in:
   - `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_baseline_development.ipynb`
   - `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_5_seed_ensemble_development.ipynb`
   - `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_5_fold_ensemble_development.ipynb`

Official coursework files are stored under `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/official_coursework`.

## Environment Setup

Create the local environment:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
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

- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/training_data/train.csv`
- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/training_data/dev.csv`

Official trial files and the local scorer live in:

- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/official_coursework/trial_data`
- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/official_coursework/nlu_bundle-feature-unified-local-scorer`

## Solution B

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

HPO:

```bash
source .venv/bin/activate
python -m src.solution_b.hpo \
  --output-dir outputs/hpo \
  --device auto
```

Notes:

- `--device auto` now supports `cuda`, `mps`, and `cpu`.
- The default embedding model is `fasttext-wiki-news-subwords-300`, which triggers a large one-time gensim download.

## Solution C

Run the notebooks from the project root with the `NLU CourseWork` kernel.

The transformer notebooks now:

- use `use_fast=False` tokenizers to avoid the dependency failure seen during the audit
- auto-detect `cuda`, `mps`, or `cpu`
- reduce batch sizes and increase gradient accumulation on `mps`/`cpu`
- use smaller inference batch sizes on memory-constrained devices

Recommended execution order:

1. `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_baseline_development.ipynb`
2. `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_5_seed_ensemble_development.ipynb`
3. `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/solution_c_5_fold_ensemble_development.ipynb`

## Local Scorer

Run the official scorer:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/official_coursework/nlu_bundle-feature-unified-local-scorer
python3 -m unittest tests.test_local_scorer -v
python3 local_scorer/main.py --task ed
```

## Current Submission Gaps

Still incomplete on this branch:

- final trained model artifacts and cloud links
- final predictions on released test data
- `model_card_b.md`
- completed `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/model_card_c.md`
- completed `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/poster.md`

## Use of Generative AI Tools

This branch has used AI-assisted engineering support for audit, environment setup, and runtime refactoring.  
Before final submission, this section should be updated with the exact tools used and the final human-reviewed scope of accepted changes, in line with the coursework brief.
