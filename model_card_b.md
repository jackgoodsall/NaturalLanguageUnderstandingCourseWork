---
language: en
license: cc-by-4.0
tags:
  - text-classification
  - evidence-detection
  - bilstm
  - comp34812
---

# Model Card for COMP34812 ED Solution B

This is the locked non-transformer submission system for the COMP34812 Evidence Detection coursework. It predicts whether a candidate evidence snippet is relevant to a given claim.

## Model Details

### Model Description

`Solution B` is an ESIM-style BiLSTM claim-evidence classifier. It uses pretrained `fasttext-wiki-news-subwords-300` embeddings, shared sequence encoders for the claim and evidence, a co-attention interaction layer, a second BiLSTM composition layer, pooled pair representations, and an MLP classification head.

- **Developed by:** COMP34812 Evidence Detection coursework team
- **Language(s):** English
- **Model type:** Supervised binary text-pair classification
- **Model architecture:** ESIM-style BiLSTM with FastText embeddings and co-attention
- **Finetuned from model [optional]:** Not applicable. The network is trained from scratch on the coursework ED data while using pretrained FastText token embeddings.

### Model Resources

- **Repository:** [https://github.com/jackgoodsall/NaturalLanguageUnderstandingCourseWork](https://github.com/jackgoodsall/NaturalLanguageUnderstandingCourseWork)
- **Paper or documentation:** ESIM paper: [https://arxiv.org/abs/1609.06038](https://arxiv.org/abs/1609.06038); FastText English vectors: [https://fasttext.cc/docs/en/english-vectors.html](https://fasttext.cc/docs/en/english-vectors.html)

## Training Details

### Training Data

The model was trained on the official closed-track Evidence Detection data provided for the coursework:

- training file: `training_data/train.csv`
- dev file: `training_data/dev.csv`
- train rows: `21,508`
- dev rows: `5,926`

The positive class is the minority label. Preprocessing lowercases text, removes non-alphanumeric characters, tokenises on whitespace, and maps tokens to 300-dimensional FastText vectors. Claim and evidence are both treated as sequences so the model can learn token-level interactions.

### Training Procedure

The locked `Solution B` system is defined in:

- training code: `src/solution_b/train.py`
- evaluation code: `src/solution_b/evaluate.py`
- locked run record: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- submission lock: `experiments/decisions/D011_submission_lock.md`

#### Training Hyperparameters

Locked architecture and training settings:

- embeddings: `fasttext-wiki-news-subwords-300`
- embedding dimension: `300`
- hidden size: `128`
- number of LSTM layers: `3`
- dropout: `0.2`
- classification head: `mlp`
- batch size: `16`
- learning rate: `5e-4`
- requested maximum epochs: `50`
- early stopping patience: `5`
- official operating threshold for submission: `0.60`

The archived training metadata shows that the saved artifact trained through `14` logged epochs before stopping.

#### Speeds, Sizes, Times

The archived metadata for the locked run does not include an exact total wall-clock training time. The trained checkpoint is not committed to git because model artifacts are treated as generated resources and may exceed convenient repository size limits. The marker-facing inference notebook expects the checkpoint to be available locally before execution:

- demo notebook: `solution_b_demo_inference.ipynb`

## Evaluation

### Testing Data & Metrics

#### Testing Data

Evaluation for the locked result was carried out on the official coursework dev split:

- file: `training_data/dev.csv`
- rows: `5,926`

The run record also includes a threshold sweep over the dev probabilities to choose the operating threshold for the submission path.

#### Metrics

The main metrics used are:

- accuracy
- binary precision / recall / F1
- macro precision / recall / F1
- Matthews correlation coefficient (MCC)

Threshold selection was important because the class distribution is imbalanced and the default `0.50` threshold over-predicted the positive class for this model family.

### Results

Default threshold `0.50`:

- accuracy: `0.7904`
- binary F1: `0.6784`
- macro F1: `0.7615`
- MCC: `0.5415`

Locked submission threshold `0.60`:

- accuracy: `0.8137`
- binary F1: `0.6824`
- macro F1: `0.7753`
- MCC: `0.5529`

This thresholded `Solution B` beats the official bundled ED LSTM baseline on the tracked dev metrics and serves as the locked non-transformer submission system.

## Technical Specifications

### Hardware

The archived validation record for the locked artifact resolved `device=auto` to Apple `mps`. The implementation itself supports `cpu`, `cuda`, and `mps` through device auto-detection.

### Software

Core software components:

- Python
- PyTorch
- gensim
- NumPy
- pandas
- scikit-learn

Installation and demo execution instructions are documented in:

- `README.md`
- `solution_b_demo_inference.ipynb`

## Bias, Risks, and Limitations

- The dataset is imbalanced, so raw accuracy alone can hide poor minority-class behaviour.
- `Solution B` is materially threshold-sensitive, which means careless threshold choice can degrade both macro F1 and MCC.
- The model does not provide natural-language explanations for its decisions; it only outputs binary predictions and probabilities.
- Because this is a closed-track coursework system, it does not use any external retrieval or evidence-ranking pipeline beyond the provided pairwise data.
- Errors in either direction are meaningful:
  - false positives can mark irrelevant evidence as useful
  - false negatives can miss genuinely relevant evidence

## Additional Information

- marker-facing demo notebook: `solution_b_demo_inference.ipynb`
- training notebook kept in repo: `solution_b_development.ipynb`
- locked run record: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- submission lock record: `experiments/decisions/D011_submission_lock.md`

The final test-set prediction file was not yet available at the time this card was drafted, so this card reports the locked dev result rather than hidden-test performance.
