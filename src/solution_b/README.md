# Solution B — ESIM-style BiLSTM for Claim–Evidence Classification

A sequence-pair classification system that determines whether a piece of evidence supports a given claim. Uses an ESIM (Enhanced Sequential Inference Model) architecture with BiLSTM encoders and pretrained FastText embeddings.

## How It Works

1. **Embedding**: Claims and evidence are tokenised and mapped to 300-dimensional FastText vectors.
2. **Encoding**: A shared BiLSTM encodes both sequences independently.
3. **Co-Attention**: Cross-attention aligns claim and evidence tokens, producing enriched representations that capture how the two sequences interact.
4. **Composition**: A second BiLSTM reads the attention-enhanced representations.
5. **Pooling**: Mean and max pooling over both composed sequences produces a fixed-size vector.
6. **Classification**: A swappable head (MLP, linear, or deep MLP) maps the pooled vector to a binary prediction.

## File Structure

```
src/solution_b/
├── data.py      — Data loading, tokenisation, vectorisation, Dataset & DataLoader
├── models.py    — BiLSTM encoder, ESIM model, classification heads, ensemble
├── train.py     — Training loop with early stopping & LR scheduling, CLI
├── evaluate.py  — Inference, metrics (accuracy/F1), threshold sweep, CLI
└── hpo.py       — Hyperparameter optimisation with Optuna, CLI
```

All scripts are run as modules from the **project root**:

```bash
cd /path/to/NLUCourseWork
python -m src.solution_b.<module> [args]
```

---

## Data Pipeline (`data.py`)

Handles the full path from CSV to batched tensors:

- **`load_embeddings(name)`** — Downloads and loads pretrained embeddings from gensim (default: `fasttext-wiki-news-subwords-300`).
- **`preprocess(text)`** — Lowercases text, strips non-alphanumeric characters, splits into tokens.
- **`tokens_to_vectors(tokens, embeddings, dim)`** — Maps each token to its embedding vector (unknown tokens get zero vectors).
- **`load_and_preprocess(path, embeddings, dim)`** — Reads a CSV and adds tokenised + vectorised columns for claims and evidence.
- **`ClaimEvidenceDataset`** — PyTorch Dataset wrapping vectorised claim/evidence pairs with optional labels.
- **`collate_fn(batch)`** — Pads variable-length sequences within a batch and records original lengths (needed for LSTM packing).
- **`get_dataloaders(train_df, dev_df, batch_size)`** — Convenience function that returns train and dev DataLoaders.

---

## Model Architecture (`models.py`)

### Classification Heads

Three interchangeable heads, selected via the `--head` flag:

| Head | Description |
|---|---|
| `mlp` (default) | 2-layer MLP with LayerNorm, GELU, dropout |
| `linear` | Single linear layer (no hidden layers) — useful as a baseline |
| `deep_mlp` | 3-layer MLP with a residual skip connection |

New heads can be added by defining a class and registering it in `HEAD_REGISTRY`.

### ESIMLSTMModel

The main model. Takes a dict with `claim`, `evidence`, `claim_lens`, `evidence_lens` and outputs a raw logit `(B, 1)`. Apply sigmoid to get probabilities.

### EnsembleModel

Wraps multiple trained models and averages their logits (optionally with custom weights).

### `build_model(config)`

Factory function that constructs an `ESIMLSTMModel` from a config dict.

---

## Training (`train.py`)

### Quick Start

```bash
python -m src.solution_b.train \
    --train training_data/train.csv \
    --dev   training_data/dev.csv \
    --output-dir outputs/run1
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--train` | `training_data/train.csv` | Path to training CSV |
| `--dev` | `training_data/dev.csv` | Path to dev/validation CSV |
| `--output-dir` | `outputs` | Directory for checkpoints and metadata |
| `--embeddings` | `fasttext-wiki-news-subwords-300` | Gensim embedding model name |
| `--hidden-size` | 128 | BiLSTM hidden size per direction |
| `--num-layers` | 3 | Number of stacked LSTM layers |
| `--dropout` | 0.2 | Dropout probability |
| `--head` | `mlp` | Classification head (`mlp`, `linear`, `deep_mlp`) |
| `--lr` | 5e-4 | Learning rate (AdamW optimiser) |
| `--batch-size` | 16 | Training batch size |
| `--epochs` | 50 | Maximum number of epochs |
| `--patience` | 5 | Early stopping patience (epochs without val loss improvement) |

### What It Does

- Uses **BCEWithLogitsLoss** with `pos_weight` to handle class imbalance (upweights the minority class).
- **ReduceLROnPlateau** scheduler halves the LR if validation loss plateaus for 2 epochs.
- **Early stopping** halts training if validation loss doesn't improve for `patience` epochs.
- Saves the **best model checkpoint** (by validation loss) to `<output-dir>/best_model.pt`.
- Saves **run metadata** (config, training history, CLI args) to `<output-dir>/run_meta.json`.

---

## Evaluation (`evaluate.py`)

### Single Model

```bash
python -m src.solution_b.evaluate \
    --checkpoints outputs/run1/best_model.pt \
    --data training_data/dev.csv
```

### With Threshold Sweep

Find the optimal decision boundary (useful when classes are imbalanced):

```bash
python -m src.solution_b.evaluate \
    --checkpoints outputs/run1/best_model.pt \
    --data training_data/dev.csv \
    --threshold 0.45 \
    --sweep
```

### Ensemble Evaluation

Pass multiple checkpoints to average predictions:

```bash
python -m src.solution_b.evaluate \
    --checkpoints outputs/run1/best_model.pt outputs/run2/best_model.pt \
    --data training_data/dev.csv
```

With custom weights:

```bash
python -m src.solution_b.evaluate \
    --checkpoints outputs/run1/best_model.pt outputs/run2/best_model.pt \
    --weights 0.6 0.4
```

### Save Results

```bash
python -m src.solution_b.evaluate \
    --checkpoints outputs/run1/best_model.pt \
    --data training_data/dev.csv \
    --output outputs/run1/eval.json
```

### Unlabelled Data

If the CSV has no `label` column, the script outputs raw predictions to a CSV file instead of computing metrics.

### Key Functions

- **`get_probs(model, dataloader, device)`** — Run inference, return probabilities.
- **`evaluate(probs, labels, threshold)`** — Compute accuracy, F1 (binary & macro), classification report.
- **`threshold_sweep(probs, labels)`** — Try thresholds from 0.1 to 1.0, return sorted by F1.
- **`load_checkpoint(path)`** — Load a single model from a `.pt` file (auto-detects `run_meta.json`).
- **`load_ensemble(paths, weights)`** — Load multiple models into an `EnsembleModel`.

---

## Hyperparameter Optimisation (`hpo.py`)

Uses [Optuna](https://optuna.org/) to search for the best model configuration.

### Quick Start

```bash
python -m src.solution_b.hpo \
    --n-trials 30 \
    --epochs 20 \
    --output-dir outputs/hpo
```

### Search Space

| Parameter | Range |
|---|---|
| `hidden_size` | {64, 128, 256} |
| `num_layers` | 1–3 |
| `dropout` | 0.1–0.5 |
| `head` | mlp, linear, deep_mlp |
| `lr` | 1e-4 to 1e-2 (log-uniform) |
| `batch_size` | {16, 32, 64} |

### Resumable Studies

Use SQLite storage to pause and resume a search:

```bash
# Start
python -m src.solution_b.hpo --storage sqlite:///hpo.db --n-trials 30

# Resume later (adds more trials to the same study)
python -m src.solution_b.hpo --storage sqlite:///hpo.db --n-trials 20
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--train` | `training_data/train.csv` | Training data path |
| `--dev` | `training_data/dev.csv` | Validation data path |
| `--output-dir` | `outputs/hpo` | Directory for results |
| `--embeddings` | `fasttext-wiki-news-subwords-300` | Embedding model |
| `--n-trials` | 30 | Number of Optuna trials to run |
| `--epochs` | 20 | Max epochs per trial (fewer than full training for speed) |
| `--patience` | 5 | Early stopping patience per trial |
| `--study-name` | `esim_lstm_hpo` | Optuna study name |
| `--storage` | None | Optuna storage URL (e.g. `sqlite:///hpo.db`) |

Results are saved to `<output-dir>/hpo_results.json` with the best parameters, trial number, and validation loss.

---

## Using from a Notebook

```python
from src.solution_b.data import load_embeddings, load_and_preprocess, get_dataloaders
from src.solution_b.models import build_model
from src.solution_b.train import train_loop
from src.solution_b.evaluate import get_probs, evaluate, threshold_sweep

# Load embeddings and data
embeddings = load_embeddings()
train_df = load_and_preprocess("training_data/train.csv", embeddings, dim=300)
dev_df   = load_and_preprocess("training_data/dev.csv",   embeddings, dim=300)
train_dl, dev_dl = get_dataloaders(train_df, dev_df, batch_size=16)

# Build and train
config = {"embedding_size": 300, "hidden_size": 128, "num_layers": 3, "head": "mlp"}
model  = build_model(config)
# ... set up loss_fn and optimiser, then call train_loop(...)
```
