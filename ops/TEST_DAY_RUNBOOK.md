# Test-Day Runbook

This document is the operational checklist for generating the final Evidence Detection submission files once the official test CSV is released.

## Locked systems

- `Solution B`: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- `Solution C`: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

Expected submission filenames:

- `Group_n_B.csv`
- `Group_n_C.csv`

Replace `n` with the actual Canvas group number before submission.

## Preconditions

Before running the final test pass, make sure all of the following are true:

1. The official ED test CSV has been downloaded from Canvas.
2. The final large model artifacts have been downloaded from the cloud-hosted links into local paths.
3. The repository environment is installed:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Decide the final group number and replace `Group_n_B.csv` / `Group_n_C.csv` with the correct filenames before submission.

## Recommended local layout on test day

Use these paths consistently:

- input test CSV: `official_coursework/test_data/ED_test.csv`
- temporary detailed `B` output: `outputs/submission/b_predictions_debug.csv`
- final `B` submission file: `outputs/submission/Group_n_B.csv`
- temporary detailed `C` output: `outputs/submission/c_predictions_debug.csv`
- final `C` submission file: `outputs/submission/Group_n_C.csv`

Create the directory if needed:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
mkdir -p outputs/submission official_coursework/test_data
```

## Solution B Generation

### Locked operating settings

- notebook path: `solution_b_demo_inference.ipynb`
- operating threshold: `0.60`

### Inputs you must set in the notebook

In `solution_b_demo_inference.ipynb`, update the path configuration cell to:

- `MODEL_CHECKPOINT` = local downloaded checkpoint path for locked `Solution B`
- `INPUT_CSV` = `official_coursework/test_data/ED_test.csv`
- `OUTPUT_CSV` = `outputs/submission/b_predictions_debug.csv`
- `SUBMISSION_FILE` = `outputs/submission/Group_n_B.csv`
- `THRESHOLD` = `0.60`

### Expected outputs

- detailed CSV with probabilities and predictions:
  - `outputs/submission/b_predictions_debug.csv`
- scorer-style submission file with one label per line:
  - `outputs/submission/Group_n_B.csv`

## Solution C Generation

### Locked operating settings

- demo notebook path: `solution_c_5_seed_ensemble_development.ipynb`
- official submission threshold: `0.50`
- locked model family: transfer-initialised three-seed ensemble

### Inputs you must set in the notebook

Open `solution_c_5_seed_ensemble_development.ipynb` and use it only as the inference shell for the locked transfer ensemble.

Update the relevant variables so that:

1. `seed_model_dirs` points to the three downloaded locked model directories for the transfer ensemble
2. `best_threshold` is set to `0.50`
3. `TEST_PATH` is `./official_coursework/test_data/ED_test.csv`
4. `OUT_PATH` is `./outputs/submission/c_predictions_debug.csv`

The locked seed count is `3`, not `5`, so do not use extra model directories that are not part of `C_REMOTE_A40_019`.

### Convert the detailed C output into the final submission file

The notebook writes a detailed CSV containing `prob_relevant` and `pred`. Convert that file into the required one-label-per-line submission file with:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/submission/c_predictions_debug.csv')
df[['pred']].to_csv('outputs/submission/Group_n_C.csv', index=False, header=False)
PY
```

### Expected outputs

- detailed CSV with probabilities and predictions:
  - `outputs/submission/c_predictions_debug.csv`
- scorer-style submission file with one label per line:
  - `outputs/submission/Group_n_C.csv`

## Sanity Checks

Run these checks before zipping or uploading anything.

### 1. Input row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('official_coursework/test_data/ED_test.csv')
print('test_rows', len(df))
PY
```

### 2. B submission row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
wc -l outputs/submission/Group_n_B.csv
```

### 3. C submission row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
wc -l outputs/submission/Group_n_C.csv
```

Both submission line counts must match the number of rows in the released test CSV.

### 4. Quick spot-check first few lines

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
head outputs/submission/Group_n_B.csv
head outputs/submission/Group_n_C.csv
```

The files should contain only `0` and `1` labels, one per line, with no header.

## Final Packaging

Before Canvas upload:

1. rename the files with the real group number if needed
2. copy the final pair out of `outputs/submission/` if you want a clean upload directory
3. include the required code and model-card deliverables in the final zip

The minimum final prediction deliverables are:

- `Group_n_B.csv`
- `Group_n_C.csv`

## Failure Handling

If one of the notebooks or scripts fails on test day:

1. do not start broad debugging
2. first confirm the model artifact paths are correct
3. confirm the test CSV has the same column names as the trial file
4. confirm the output line counts match the input row count
5. if `Solution C` notebook friction is too high, keep the locked model directories and convert the inference path into a one-off script rather than changing the chosen model
