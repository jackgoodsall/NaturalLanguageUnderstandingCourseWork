# R001 — Smaller Model Selection For Solution C

## Question

Which smaller transformer should replace `microsoft/deberta-v3-base` for `Solution C` on this MPS-constrained machine?

## Local Evidence

Verified local failures:

1. `C_BASELINE_FULL_MPS_001`
   - failed after `37` seconds
   - `oom_count = 1`
   - failed during `trainer.train()`
2. `C_BASELINE_FULL_MPS_002`
   - failed after `34` seconds
   - `oom_count = 1`
   - still failed after reducing max length to `192`, train batch to `1`, eval batch to `2`, and enabling gradient checkpointing

Conclusion from local evidence:

- `microsoft/deberta-v3-base` is not a viable full-data baseline on this machine through the current notebook execution path.

## Primary Sources Considered

1. PyTorch MPS Environment Variables
   - `https://docs.pytorch.org/docs/stable/mps_environment_variables.html`
2. PyTorch `torch.mps` documentation
   - `https://docs.pytorch.org/docs/stable/mps.html`
3. `microsoft/deberta-v3-base`
   - `https://huggingface.co/microsoft/deberta-v3-base`
4. `microsoft/deberta-v3-small`
   - `https://huggingface.co/microsoft/deberta-v3-small`
5. `microsoft/deberta-v3-xsmall`
   - `https://huggingface.co/microsoft/deberta-v3-xsmall`
6. `microsoft/MiniLM-L12-H384-uncased`
   - `https://huggingface.co/microsoft/MiniLM-L12-H384-uncased`
7. `albert-base-v2`
   - `https://huggingface.co/albert/albert-base-v2`
8. `distilbert-base-uncased`
   - `https://huggingface.co/distilbert/distilbert-base-uncased`

## Candidate Comparison

| Candidate | Official Source Signal | Hardware Fit Signal | Decision |
|---|---|---|---|
| `microsoft/deberta-v3-base` | strongest listed MNLI result among current candidates | already failed twice locally | reject for this machine |
| `microsoft/deberta-v3-small` | smaller than base, still strong | embedding layer still large | fallback only |
| `microsoft/deberta-v3-xsmall` | much smaller while retaining DeBERTa V3 family | strongest balance of size and continuity | choose next |
| `microsoft/MiniLM-L12-H384-uncased` | 33M params, 2.7x faster than BERT-Base | very plausible fallback | second choice |
| `albert-base-v2` | memory-conscious design, 11M params | architecture shift and compute tradeoff | third choice |
| `distilbert-base-uncased` | smaller/faster than BERT | still larger than xsmall and MiniLM | deprioritize |

## Recommendation

Recommended next model:

- `microsoft/deberta-v3-xsmall`

Why:

1. It stays within the same DeBERTa V3 family as the current notebook path.
2. It cuts the backbone size sharply relative to the current baseline.
3. It keeps the model class and overall fine-tuning path familiar, which reduces code churn.
4. It has stronger official NLU signals than the most obvious fallback (`MiniLM`) while remaining much smaller than the current base model.

## Rejected Or Deferred Alternatives

### `microsoft/deberta-v3-small`

Rejected as first choice because:

1. it is smaller than base, but still carries a large embedding layer
2. it is not the sharpest memory reduction available in the same family

### `distilbert-base-uncased`

Rejected as first choice because:

1. it is still significantly larger than the most compact candidates under consideration
2. the architecture shift is less justified when `deberta-v3-xsmall` exists

### `albert-base-v2`

Deferred because:

1. it is attractive on parameter count and memory design
2. but it introduces a more meaningful architecture change than necessary for the next experiment

## Safety Decision

Do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` as the default branch fix.

Reason:

- PyTorch documentation explicitly warns that disabling the high watermark may cause system failure if system-wide OOM occurs.

## H003

Next hypothesis to run:

- `H003 — Solution C with microsoft/deberta-v3-xsmall`

Recommended starting config on this machine:

- `model_name = microsoft/deberta-v3-xsmall`
- `max_length = 192`
- `train_bs = 2`
- `eval_bs = 4`
- `infer_eval_bs = 8`
- `grad_accum = 8`

## Reversal Condition

This recommendation should be reversed if:

1. `deberta-v3-xsmall` still OOMs on the same path, or
2. it completes but performs materially below the accepted quality floor

If either happens, move to:

- `microsoft/MiniLM-L12-H384-uncased`

## Status Update After H003

`H003` was executed and failed after `220` seconds with another verified `mps` out-of-memory error.

That means the original recommendation in this brief is now invalidated by local evidence.

Updated next model candidate:

- `microsoft/MiniLM-L12-H384-uncased`
