# Solo Model Research Plan

## 1. Task definition

Track: `Evidence Detection (ED)`

Problem:

- input: one `Claim` and one `Evidence` sequence
- output: binary label indicating whether the evidence is relevant to the claim

Coursework constraints from `official_coursework/NaturalLanguageUnderstandingCourseWorkSpec.pdf`:

- closed-mode shared task
- build two solutions from different categories
- `Solution B`: deep learning without transformers
- `Solution C`: transformer-based
- code must be reproducible and demo-able in notebook form

Important interpretation:

- closed mode forbids using extra datasets for development
- pretrained embeddings / pretrained transformer checkpoints are assumed acceptable because the coursework itself expects a transformer baseline and prior branch work already relies on standard pretrained models
- if this assumption is challenged later, the fallback is to keep the same architecture but remove external static embeddings for `B`

## 2. Local data profile

Computed from:

- `training_data/train.csv`
- `training_data/dev.csv`

Train:

- rows: `21,508`
- positive labels: `5,854`
- negative labels: `15,654`
- positive rate: `0.2722`
- mean claim length: `5.60` words
- mean evidence length: `31.13` words

Dev:

- rows: `5,926`
- positive labels: `1,640`
- negative labels: `4,286`
- positive rate: `0.2767`
- mean claim length: `5.62` words
- mean evidence length: `31.20` words

Implication:

- the task is materially imbalanced
- claims are short
- evidence is moderate length, so sequence-pair classification is the right framing
- class handling and threshold selection must be treated as first-class experimental axes

## 3. Official evaluation contract

The released scorer in `official_coursework/nlu_bundle-feature-unified-local-scorer/local_scorer/metric.txt` uses:

1. `accuracy_score`
2. `macro_precision`
3. `macro_recall`
4. `macro_f1`
5. `weighted_macro_precision`
6. `weighted_macro_recall`
7. `weighted_mmacro_f1`
8. `matthews_corrcoef`

Locked metric hierarchy for this solo rebuild:

1. primary: `macro_f1`
2. guardrail: `matthews_corrcoef`
3. secondary: `accuracy_score`
4. diagnostic only: positive-class `binary_f1`

Reason:

- `macro_f1` is in the official scorer and is robust to the class imbalance
- `matthews_corrcoef` is also in the official scorer and punishes trivial majority-class behaviour
- `accuracy` is useful but not sufficient on this label distribution
- `binary_f1` is still worth tracking because it tells us whether the positive class is collapsing

## 4. Baseline floors and competitive targets

Official released ED baselines from `official_coursework/nlu_bundle-feature-unified-local-scorer/baseline/25_DEV_ED.csv`:

### Non-transformer floor to beat

`LSTM`

- accuracy: `0.8057711778602767`
- macro_f1: `0.7083478233131942`
- matthews_corrcoef: `0.46744082752748667`

### Transformer floor to beat

`BERT`

- accuracy: `0.874451569355383`
- macro_f1: `0.834773608112048`
- matthews_corrcoef: `0.6747924534061905`

Internal stretch targets from the current team branch:

### Team `B` target

`experiments/runs/B_REMOTE_FULL_001_probe/result.json` at threshold `0.60`

- accuracy: `0.813702328720891`
- macro_f1: `0.7752941991090772`
- matthews_corrcoef: `0.5529387441032668`
- binary_f1: `0.6823935558112774`

### Team `C` target

`experiments/runs/C_REMOTE_A40_019_transfer_seed3/result.json`

- accuracy: `0.8889638879514006`
- macro_f1: `0.8671348982304408`
- matthews_corrcoef: `0.7383867359239816`
- binary_f1: `0.8132803632236095`

## 5. Research-backed baseline choice

### 5.1 `Solution B`

Chosen baseline:

- `ESIM`-style sequence-pair model
- BiLSTM encoders
- token-level interaction / local inference
- pooled classification head

Why this is the right `B` baseline:

- ED is a pairwise sequence classification task
- ESIM was designed for sentence-pair reasoning and remains a strong non-transformer baseline for pairwise tasks
- it is materially better aligned to claim-evidence interaction than a plain single-sequence BiLSTM

Initial baseline recipe:

- tokenizer: simple whitespace / regex tokenization
- embeddings: `fasttext-wiki-news-subwords-300`
- encoder: shared BiLSTM
- interaction: ESIM local inference
- loss: `BCEWithLogitsLoss` with positive-class weighting from train prevalence
- model selection: best `macro_f1` on dev
- threshold policy: default `0.5` plus post-training sweep

Baseline acceptance test:

- must beat official `LSTM` on `macro_f1`
- must avoid any positive-class collapse
- if it fails to beat official `LSTM`, we do not tune around it; we redesign the architecture first

### 5.2 `Solution C`

Chosen baseline:

- DeBERTa-style cross-encoder sequence-pair classifier
- full fine-tuning
- stable full-precision recipe first

Why this is the right `C` baseline:

- DeBERTa is a strong pretrained encoder for sequence classification
- the task is a standard claim-evidence pair classification setup, so a cross-encoder is the correct default transformer form
- prior branch evidence already shows that stable DeBERTa fine-tuning beats weaker notebook paths
- the highest-value follow-up branch after plain DeBERTa is a task-aligned transfer checkpoint, not random hyperparameter drift

Initial baseline recipe:

- base checkpoint: `microsoft/deberta-v3-base`
- sequence form: `[CLS] claim [SEP] evidence [SEP]`
- max length: `256`
- precision: `float32`
- learning rate: `1e-5`
- epochs: `3`
- weight decay: `0.01`
- warmup ratio: `0.06`
- scheduler: `linear`
- batch size: `16`
- model selection: best `macro_f1` on dev

High-value first branch if plain baseline is stable but not competitive:

- transfer-initialized checkpoint aligned to claim/evidence reasoning
- current empirical prior from the team branch:
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

Baseline acceptance test:

- must at least reach the official `BERT` band on `macro_f1` or come within `0.01`
- must show no `NaN`, no positive-class collapse, and no severe seed instability

## 6. Literature-backed experimental priors

We are not treating tuning as open-ended. These papers set the initial rules:

1. ESIM for pairwise sequence classification:
   - Chen et al., [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)

2. DeBERTa as the transformer architecture prior:
   - He et al., [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

3. Stable transformer fine-tuning matters more than random knob changes:
   - Mosbach et al., [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884)

4. Class imbalance should be handled with a deliberate reweighting branch, not vague intuition:
   - Cui et al., [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)

5. Threshold calibration is a post-training operation and should be tested separately:
   - Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

## 7. Closed experimental system

Every experiment must have:

1. one changed variable only
2. one hypothesis
3. one primary metric target (`macro_f1`)
4. one guardrail (`matthews_corrcoef`)
5. one end-to-end result record

Promotion rule for a new run:

- promote only if `macro_f1` improves by at least `+0.003`
- and `matthews_corrcoef` does not drop by more than `0.002`
- if gains are smaller than that, prefer the simpler path

This prevents noise-driven branch churn.

## 8. Research pipeline

### Phase 1: baseline establishment

`B0`

- build ESIM baseline
- run on dev
- threshold sweep

`C0`

- build stable plain DeBERTa baseline
- run on dev
- threshold sweep

### Phase 2: component-isolated tuning

`B` component axes:

1. imbalance handling:
   - naive `pos_weight`
   - class-balanced loss
2. embedding policy:
   - frozen static embeddings
   - lightly fine-tuned embeddings
3. classifier threshold:
   - default `0.5`
   - tuned dev threshold

`C` component axes:

1. base checkpoint:
   - plain DeBERTa
   - task-aligned transfer DeBERTa
2. stability recipe:
   - learning rate
   - epoch budget
   - full precision
3. prediction policy:
   - default threshold
   - tuned threshold
4. variance reduction:
   - single seed
   - seed ensemble

### Phase 3: end-to-end promotion

Only after a component branch clearly wins do we:

1. rerun it cleanly end to end
2. record the official metrics
3. compare it against the locked floor and the locked branch leader

## 9. Immediate implementation order

1. Implement `B0` first in `src/solution_b/`
2. Get `B0` above the official `LSTM` floor
3. Implement `C0` next in `src/solution_c/`
4. Stabilize `C0` before any ensemble attempt
5. Only then open tuning branches

## 10. Stop conditions

Stop tuning a branch when:

1. it is below the official baseline and structurally weak
2. it fails stability acceptance twice
3. it improves less than the promotion threshold
4. a simpler stronger branch already exists

## 11. Current recommendation

For the solo rebuild, the correct next move is not more research. It is:

1. implement `B0` as ESIM
2. validate it against the official `LSTM` floor
3. then implement `C0` as stable DeBERTa

That keeps the project inside a closed, measurable system from the start.
