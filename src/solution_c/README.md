# Solo Solution C

Fresh solo transformer baseline for the coursework `Solution C` path.

## Files

- `data.py`: CSV loading and pair-tokenized dataset creation
- `metrics.py`: official-style classification metrics plus threshold sweep
- `train.py`: training CLI for the solo `C0` baseline
- `evaluate.py`: evaluation CLI for saved checkpoints

## Baseline objective

The first baseline on this branch is `C0`:

- DeBERTa cross-encoder pair classifier
- full fine-tuning in float32
- best-model selection by `macro_f1`
- threshold sweep after training

## Example commands

Train:

```bash
python3 -m src.solution_c.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_c_c0
```

Evaluate:

```bash
python3 -m src.solution_c.evaluate \
  --checkpoints outputs/solution_c_c0/best_model \
  --data training_data/dev.csv \
  --sweep
```

Fast smoke test:

```bash
python3 -m src.solution_c.train \
  --train training_data/train.csv \
  --dev training_data/dev.csv \
  --output-dir outputs/solution_c_smoke \
  --model-name prajjwal1/bert-tiny \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --max-train-rows 64 \
  --max-dev-rows 32 \
  --device cpu
```
