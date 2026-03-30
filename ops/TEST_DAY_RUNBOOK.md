# Test-Day Runbook

This document is the operational checklist for generating the final Evidence Detection submission files once the official test CSV is released.

## Locked systems

- `Solution B`: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- `Solution C`: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

Expected submission filenames:

- `Group_52_B.csv`
- `Group_52_C.csv`

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

4. Keep the confirmed Canvas group number `52` in the final filenames.

## Recommended local layout on test day

Use these paths consistently:

- input test CSV: `official_coursework/released_test_data/test_data/ED/test.csv`
- temporary detailed `B` output: `outputs/submission/b_predictions_debug.csv`
- final `B` submission file: `outputs/submission/Group_52_B.csv`
- temporary detailed `C` output: `outputs/submission/c_predictions_debug.csv`
- final `C` submission file: `outputs/submission/Group_52_C.csv`

Create the directory if needed:

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
mkdir -p outputs/submission
```

## Solution B Generation

### Locked operating settings

- notebook path: `solution_b_demo_inference.ipynb`
- operating threshold: `0.60`

### Inputs you must set in the notebook

In `solution_b_demo_inference.ipynb`, confirm the path configuration cell is:

- `MODEL_CHECKPOINT` = `official_coursework/local_artifacts/B_REMOTE_FULL_001_best_model.pt`
- `INPUT_CSV` = `official_coursework/released_test_data/test_data/ED/test.csv`
- `OUTPUT_CSV` = `outputs/submission/b_predictions_debug.csv`
- `SUBMISSION_FILE` = `outputs/submission/Group_52_B.csv`
- `THRESHOLD` = `0.60`

### Expected outputs

- detailed CSV with probabilities and predictions:
  - `outputs/submission/b_predictions_debug.csv`
- scorer-style submission file with one label per line:
  - `outputs/submission/Group_52_B.csv`

## Solution C Generation

### Locked operating settings

- demo notebook path: `solution_c_demo_inference.ipynb`
- official submission threshold: `0.50`
- locked model family: transfer-initialised three-seed ensemble

### Inputs you must set in the notebook

Open `solution_c_demo_inference.ipynb` and use it as the inference path for the locked transfer ensemble.

Update the configuration cell so that:

1. `GROUP_NUMBER` matches the confirmed Canvas group number
2. `SEED_MODEL_DIRS` points to the three downloaded locked model directories for the transfer ensemble
3. `INPUT_CSV` points at the released ED test CSV
4. `OUTPUT_CSV` is `outputs/submission/c_predictions_debug.csv`
5. `SUBMISSION_FILE` is `outputs/submission/Group_52_C.csv`
6. `THRESHOLD` is `0.50`

The locked seed count is `3`, not `5`, so do not use extra model directories that are not part of `C_REMOTE_A40_019`.

### Expected outputs

- detailed CSV with probabilities and predictions:
  - `outputs/submission/c_predictions_debug.csv`
- scorer-style submission file with one label per line:
  - `outputs/submission/Group_52_C.csv`

## Sanity Checks

Run these checks before zipping or uploading anything.

### 1. Input row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('official_coursework/released_test_data/test_data/ED/test.csv')
print('test_rows', len(df))
PY
```

### 2. B submission row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
wc -l outputs/submission/Group_52_B.csv
```

### 3. C submission row count

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
wc -l outputs/submission/Group_52_C.csv
```

Both submission line counts must match the number of rows in the released test CSV.

### 4. Quick spot-check first few lines

```bash
cd /Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork
head outputs/submission/Group_52_B.csv
head outputs/submission/Group_52_C.csv
```

The files should contain only `0` and `1` labels, one per line, with no header.

## Final Packaging

Before Canvas upload:

1. keep the final filenames as `Group_52_B.csv` and `Group_52_C.csv`
2. copy the final pair out of `outputs/submission/` if you want a clean upload directory
3. include the required code and model-card deliverables in the final zip

The minimum final prediction deliverables are:

- `Group_52_B.csv`
- `Group_52_C.csv`

## Failure Handling

If one of the notebooks or scripts fails on test day:

1. do not start broad debugging
2. first confirm the model artifact paths are correct
3. confirm the test CSV has the same column names as the trial file
4. confirm the output line counts match the input row count
5. if `Solution C` notebook friction is too high, keep the locked model directories and convert the inference path into a one-off script rather than changing the chosen model
