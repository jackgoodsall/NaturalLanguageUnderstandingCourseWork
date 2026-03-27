# H006 — Solution C Canonical Checkpoint Selection

## Objective
Turn checkpoint selection for the remote `Solution C` baseline into a repeatable audit step rather than relying on notebook-side inspection.

## Known State
- `C_REMOTE_A40_003` is the strongest current remote baseline
- output root: `./outputs/deberta_v3_baseline_plain_trainer_5ep`
- remote run completed on the RunPod `A40`
- epoch `4` appeared to be the best checkpoint
- epoch `5` appeared to regress

## Probe
Use the artifact audit to answer three questions mechanically:
1. which checkpoint is marked as best by the trainer state
2. which epoch has the highest primary validation metric
3. whether the latest epoch regressed after that best checkpoint

## Command
```bash
ssh -p 22141 -i ~/.ssh/id_ed25519 root@194.68.245.210 \
  'cd /workspace/NaturalLanguageUnderstandingCourseWork && \
   source .venv/bin/activate && \
   python experiments/debug/solution_c_artifact_audit.py \
     --output-root outputs/deberta_v3_baseline_plain_trainer_5ep'
```

## Acceptance Criteria
1. `canonical_checkpoint` resolves to `checkpoint-5380`
2. `best_epoch = 4`
3. `best_metric_name = eval_macro_f1`
4. `regressed_after_best = true`

## Decision Rule
If all acceptance criteria pass, promote the epoch-4 checkpoint as the canonical `Solution C` baseline for further optimization work.
