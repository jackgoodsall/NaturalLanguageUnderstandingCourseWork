# Canonical Run Commands

All commands assume the working directory is:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
```

## 1. Official ED Baselines

Use this to refresh the official scorer metrics:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork/official_coursework/nlu_bundle-feature-unified-local-scorer
python3 -m unittest tests.test_local_scorer -v
python3 local_scorer/main.py --task ed
```

## 2. Solution B Smoke Train

Use this only to verify the pipeline shape on a small sample:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
python -m src.solution_b.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_b_smoke \
  --epochs 1 \
  --batch-size 8 \
  --hidden-size 64 \
  --num-layers 1 \
  --device auto
```

## 3. Solution B Full Train

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
python -m src.solution_b.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_b_full_001 \
  --epochs 50 \
  --batch-size 16 \
  --hidden-size 128 \
  --num-layers 3 \
  --dropout 0.2 \
  --head mlp \
  --lr 5e-4 \
  --patience 5 \
  --device auto
```

## 4. Solution B Evaluate

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
python -m src.solution_b.evaluate \
  --checkpoints outputs/solution_b_full_001/best_model.pt \
  --data training_data/dev.csv \
  --device auto \
  --sweep \
  --output outputs/solution_b_full_001/eval.json
```

## 5. Solution C Baseline Full Run

Canonical measurement command for the single-model transformer baseline:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
mkdir -p experiments/runs/C_BASELINE_FULL_MPS_001
jupyter nbconvert \
  --to notebook \
  --execute solution_c_baseline_development.ipynb \
  --output executed_solution_c_baseline.ipynb \
  --output-dir experiments/runs/C_BASELINE_FULL_MPS_001 \
  --ExecutePreprocessor.timeout=0
```

## 6. Solution C Baseline Test-Mode Verification

Use this when you need a cheap preflight before the full notebook run:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import nbformat
src = Path('solution_c_baseline_development.ipynb')
dst = Path('.verify_solution_c_baseline_testmode.ipynb')
nb = nbformat.read(src, as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace('TEST_MODE = False', 'TEST_MODE = True')
        cell.source = cell.source.replace('TEST_PATH = "./test.csv"', 'TEST_PATH = "./official_coursework/trial_data/ED_trial.csv"')
        cell.source = cell.source.replace('OUT_PATH  = "./predictions.csv"', 'OUT_PATH  = "./.tmp_verify/predictions.csv"')
nbformat.write(nb, dst)
PY
jupyter nbconvert \
  --to notebook \
  --execute .verify_solution_c_baseline_testmode.ipynb \
  --output .verify_solution_c_baseline_testmode.executed.ipynb \
  --ExecutePreprocessor.timeout=0
```

## 7. Result Recording Rule

After every full run:

1. copy the relevant metrics into a new JSON file that matches `experiments/run_result.schema.json`
2. add or update the corresponding row in `experiments/baseline_ledger.csv`
3. record whether the run is:
   - `official_verified`
   - `local_verified`
   - `doc_claimed`

## 8. H003 Candidate Run

After updating the notebook model name to `microsoft/deberta-v3-xsmall`, run:

```bash
cd /path/to/NaturalLanguageUnderstandingCourseWork
ln -sf official_coursework/trial_data/ED_trial.csv test.csv
source .venv/bin/activate
mkdir -p experiments/runs/C_XSMALL_FULL_MPS_003
jupyter nbconvert \
  --to notebook \
  --execute solution_c_baseline_development.ipynb \
  --output executed_solution_c_baseline.ipynb \
  --output-dir experiments/runs/C_XSMALL_FULL_MPS_003 \
  --ExecutePreprocessor.timeout=0
```
