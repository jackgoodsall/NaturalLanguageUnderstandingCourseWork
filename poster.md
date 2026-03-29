# Evidence Detection With Non-Transformer and Transformer Models

## Track and Task

Track: `Evidence Detection (ED)`

Given a claim and a candidate piece of evidence, predict whether the evidence is relevant to the claim.

This coursework was developed in the shared-task closed setting, so the task-specific train/dev data came from the official coursework release.

## Dataset Summary

### Data splits

| Split | Rows | Negative (`0`) | Positive (`1`) |
|---|---:|---:|---:|
| Train | 21,508 | 15,654 | 5,854 |
| Dev | 5,926 | 4,286 | 1,640 |
| Combined | 27,434 | 19,940 | 7,494 |

### Pair characteristics

| Statistic | Claims | Evidence |
|---|---:|---:|
| Mean words (train) | 5.60 | 31.13 |
| Mean words (dev) | 5.62 | 31.20 |
| Median words (combined) | 5 | 28 |

### Main dataset challenge

- The label distribution is imbalanced: the positive class is only about `27.3%` of the combined train+dev data.
- Claims are short, while evidence snippets are materially longer.
- This makes threshold choice, calibration, and false-positive control important, especially for `Solution B`.

## Solution B: ESIM-Style BiLSTM

### Objective

Provide a strong non-transformer baseline that still models interactions between the claim and evidence rather than treating them as independent texts.

### Method

1. Tokenise both texts with lightweight preprocessing.
2. Map tokens to pretrained `fasttext-wiki-news-subwords-300` vectors.
3. Encode claim and evidence with a shared BiLSTM.
4. Apply co-attention so each sequence is contextualised by the other.
5. Build enhanced pairwise features using original vectors, attended context, difference, and element-wise product.
6. Run a second BiLSTM composition layer and pool with mean + max pooling.
7. Use a classification head to produce a binary relevance score.

### Why this design

- It gives a meaningful deep-learning `B` solution without using transformers.
- Co-attention is the key step: it lets the model compare claim and evidence token-by-token instead of only using pooled sentence vectors.
- Threshold tuning is part of the locked path because the class imbalance materially affects the decision boundary.

### Locked dev result

Official `Solution B` submission path:
- run record: `experiments/runs/B_REMOTE_FULL_001_probe/result.json`
- operating threshold: `0.60`

Metrics:
- accuracy `0.8137`
- binary F1 `0.6824`
- macro F1 `0.7753`
- MCC `0.5529`

## Solution C: Transfer-Initialised DeBERTa-v3 Ensemble

### Objective

Build the strongest transformer-based system while keeping the training/evaluation path reproducible and measurable.

### Method

1. Use a DeBERTa-v3 cross-encoder for pairwise classification.
2. Start from the transfer-initialised checkpoint `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.
3. Fine-tune on the ED train split with stable float32 execution.
4. Train three seed variants with the same recipe.
5. Average the seed predictions to produce the final ensemble score.

Locked training settings:
- max length `256`
- learning rate `1e-5`
- epochs `3`
- weight decay `0.01`
- warmup ratio `0.06`
- linear scheduler
- seeds `42`, `52`, `62`

### Why this design

- The earlier plain DeBERTa baseline was stable after runtime fixes but plateaued below the strongest visible target.
- Transfer initialisation gave a much larger gain than more open-ended baseline tuning.
- A three-seed ensemble improved robustness without the weak return observed from scaling further to five seeds.

### Locked dev result

Official `Solution C` submission path:
- run record: `experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`
- operating threshold: `0.50`

Metrics:
- accuracy `0.8890`
- binary F1 `0.8133`
- macro F1 `0.8671`
- MCC `0.7384`

Best tuned dev metrics from the same run:
- best accuracy `0.8959`
- best binary F1 `0.8147`
- best macro F1 `0.8703`
- best MCC `0.7414`

## Dev Results Table

| System | Category | Accuracy | Binary F1 | Macro F1 | MCC | Notes |
|---|---|---:|---:|---:|---:|---|
| Official ED LSTM baseline | `B` | 0.8058 | - | 0.7083 | 0.4674 | bundled scorer baseline |
| Official ED BERT baseline | `C` | 0.8745 | - | 0.8348 | 0.6748 | bundled scorer baseline |
| Locked `Solution B` | `B` | 0.8137 | 0.6824 | 0.7753 | 0.5529 | threshold `0.60` |
| Locked `Solution C` | `C` | 0.8890 | 0.8133 | 0.8671 | 0.7384 | three-seed ensemble |

## Error Analysis

### Solution B

- `Solution B` is materially more threshold-sensitive than `Solution C`.
- At the default threshold `0.50`, dev performance was weaker and the positive prediction rate was too high relative to the class prior.
- Moving to threshold `0.60` improved both `macro F1` and `MCC`, which suggests that calibration and false-positive control were important error sources for the BiLSTM path.

### Solution C

- `Solution C` is much less threshold-sensitive and remains strong at the default `0.50` threshold.
- The locked confusion matrix still contains `451` false positives and `207` false negatives on dev, so the remaining errors are not solved by raw accuracy alone.
- This pattern suggests the model still confuses genuinely relevant evidence with evidence that is semantically related but not sufficiently supportive.

## Why Solution C Is The Lead Path

`Solution C` is the primary submission system because it is ahead of `Solution B` on every tracked dev metric:

| Metric | Solution B | Solution C |
|---|---:|---:|
| Accuracy | 0.8137 | 0.8890 |
| Binary F1 | 0.6824 | 0.8133 |
| Macro F1 | 0.7753 | 0.8671 |
| MCC | 0.5529 | 0.7384 |

`Solution B` is still valuable because it satisfies the second required approach category and provides a strong non-transformer comparison point.

## Limitations and Ethical Considerations

- The dataset is imbalanced, so a model can look superficially strong while still mishandling the minority positive class.
- Both systems reduce a nuanced relevance judgement to a binary label, so they do not provide explanatory rationales for why a claim-evidence pair is predicted as relevant.
- Errors matter in both directions:
  - false positives may surface irrelevant evidence as if it were useful
  - false negatives may hide evidence that should have been retrieved
- The coursework is closed-track, so performance is constrained by the quality and scope of the provided data rather than any external retrieval pipeline.

## Demo and Reproducibility Paths

- `Solution B` marker-facing demo notebook: `solution_b_demo_inference.ipynb`
- `Solution C` marker-facing demo notebook: `solution_c_5_seed_ensemble_development.ipynb`
- submission lock record: `experiments/decisions/D011_submission_lock.md`
- final `Solution C` promotion record: `experiments/decisions/D010_solution_c_transfer_ensemble.md`
