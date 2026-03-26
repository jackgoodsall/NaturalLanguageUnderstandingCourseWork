# Solo Solution B

Fresh solo baseline for the coursework `Solution B` path.

## Files

- `data.py`: CSV loading, tokenization, vocabulary building, embedding matrix creation
- `models.py`: ESIM-style pair classifier
- `metrics.py`: official-style classification metrics plus threshold sweep
- `train.py`: training CLI for the solo `B0` baseline
- `evaluate.py`: evaluation CLI for saved checkpoints

## Baseline objective

The first baseline on this branch is `B0`:

- ESIM-style BiLSTM pair model
- FastText embeddings by default
- positive-class weighting in the loss
- best-model selection by `macro_f1`
- threshold sweep after training

## Example commands

Train:

```bash
python3 -m src.solution_b.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_b_b0
```

Evaluate:

```bash
python3 -m src.solution_b.evaluate \
  --checkpoints outputs/solution_b_b0/best_model.pt \
  --data training_data/dev.csv \
  --sweep
```

Fast smoke test without downloading pretrained embeddings:

```bash
python3 -m src.solution_b.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_b_smoke \
  --embeddings random \
  --embedding-dim 64 \
  --hidden-size 32 \
  --projection-size 32 \
  --epochs 1 \
  --batch-size 32 \
  --max-train-rows 256 \
  --max-dev-rows 64 \
  --device cpu
```
