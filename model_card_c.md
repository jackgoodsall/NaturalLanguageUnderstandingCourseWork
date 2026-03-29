---
language: en
license: cc-by-4.0
tags:
  - text-classification
  - evidence-detection
  - deberta-v3
  - comp34812
---

# Model Card for COMP34812 ED Solution C

This is the locked transformer-based submission system for the COMP34812 Evidence Detection coursework. It predicts whether a candidate evidence snippet is relevant to a given claim.

## Model Details

### Model Description

`Solution C` is a DeBERTa-v3 cross-encoder trained for binary claim-evidence relevance classification. The locked submission system uses transfer initialisation from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, trains three seed variants on the coursework ED data, and averages their predictions into a three-seed ensemble.

- **Developed by:** COMP34812 Evidence Detection coursework team
- **Language(s):** English
- **Model type:** Supervised binary text-pair classification
- **Model architecture:** DeBERTa-v3 sequence-pair classifier with three-seed ensemble averaging
- **Finetuned from model [optional]:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

### Model Resources

- **Repository:** [https://github.com/jackgoodsall/NaturalLanguageUnderstandingCourseWork](https://github.com/jackgoodsall/NaturalLanguageUnderstandingCourseWork)
- **Paper or documentation:** Transfer checkpoint: [https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli); DeBERTa paper: [https://arxiv.org/abs/2006.03654](https://arxiv.org/abs/2006.03654)

## Training Details

### Training Data

The model was trained on the official closed-track Evidence Detection data provided for the coursework:

- training file: `training_data/train.csv`
- dev file: `training_data/dev.csv`
- train rows: `21,508`
- dev rows: `5,926`

The claim-evidence pairs are tokenised as a sequence pair and truncated to a maximum length of `256` tokens. The work stayed within the closed-track coursework setting and did not introduce external task datasets beyond the transfer-initialised starting checkpoint.

### Training Procedure

The locked `Solution C` system is defined in:

- submission lock: `experiments/decisions/D011_submission_lock.md`
- promotion record: `experiments/decisions/D010_solution_c_transfer_ensemble.md`
- locked run record: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- runner script: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/run_transfer_seed3.py`
- marker-facing notebook: `solution_c_5_seed_ensemble_development.ipynb`

#### Training Hyperparameters

Locked settings:

- base checkpoint: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- max length: `256`
- train batch size: `16`
- eval batch size: `32`
- gradient accumulation: `1`
- learning rate: `1e-5`
- epochs: `3`
- weight decay: `0.01`
- warmup ratio: `0.06`
- learning-rate scheduler: `linear`
- metric used for best checkpoint selection: binary `f1`
- seeds: `42`, `52`, `62`
- default submission threshold: `0.50`

#### Speeds, Sizes, Times

Each of the three seed runs completed in roughly `581-586` seconds of training time on CUDA hardware, based on the archived run record. The full locked system is an ensemble over three saved checkpoints. Large model artifacts are not committed to git and are expected to be stored externally when preparing the final submission package.

## Evaluation

### Testing Data & Metrics

#### Testing Data

Evaluation for the locked result was carried out on the official coursework dev split:

- file: `training_data/dev.csv`
- rows: `5,926`

The result record includes both the default-threshold ensemble performance and a threshold probe for tuned analysis.

#### Metrics

The main tracked metrics are:

- accuracy
- binary precision / recall / F1
- macro precision / recall / F1
- Matthews correlation coefficient (MCC)

The system is reported both at the default threshold `0.50` and under threshold sweeps so that calibration effects are visible.

### Results

Locked default threshold `0.50`:

- accuracy: `0.8890`
- binary precision: `0.7606`
- binary recall: `0.8738`
- binary F1: `0.8133`
- macro F1: `0.8671`
- MCC: `0.7384`

Best tuned dev metrics from the same ensemble:

- best accuracy: `0.8959` at threshold `0.91`
- best binary F1: `0.8147` at threshold `0.66`
- best macro F1: `0.8703` at threshold `0.73`
- best MCC: `0.7414` at threshold `0.66`

This was the strongest reviewed `Solution C` path on the submission branch and the reason the transformer approach was locked as the primary system.

## Technical Specifications

### Hardware

The locked run was executed on CUDA hardware. The archived run record shows the ensemble was trained on remote A40 GPU infrastructure.

### Software

Core software components:

- Python
- PyTorch
- transformers
- datasets
- NumPy
- pandas
- scikit-learn

The notebook-based demo path also relies on a Jupyter environment and the package set listed in:

- `requirements.txt`
- `solution_c_5_seed_ensemble_development.ipynb`

## Bias, Risks, and Limitations

- The dataset is imbalanced, so strong accuracy still needs to be checked against macro F1 and MCC.
- The model captures pairwise semantic compatibility, but it does not explain why a piece of evidence is predicted as relevant.
- Transfer initialisation can improve performance, but it also means behaviour is influenced by the source checkpoint’s prior task knowledge.
- The system may confuse genuinely relevant evidence with text that is semantically related but not sufficiently supportive.
- As a closed-track coursework model, it does not include any external retrieval, ranking, or evidence-grounding component beyond the provided pairwise data.

## Additional Information

- marker-facing demo notebook: `solution_c_5_seed_ensemble_development.ipynb`
- supporting notebooks: `solution_c_baseline_development.ipynb`, `solution_c_5_fold_ensemble_development.ipynb`
- locked run record: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- promotion record: `experiments/decisions/D010_solution_c_transfer_ensemble.md`
- submission lock record: `experiments/decisions/D011_submission_lock.md`

The final test-set prediction file was not yet available at the time this card was drafted, so this card reports the locked dev result rather than hidden-test performance.
