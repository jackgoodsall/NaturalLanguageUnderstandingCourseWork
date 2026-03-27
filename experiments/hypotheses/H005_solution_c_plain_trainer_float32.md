# H005 — Solution C With Plain Trainer And Float32 Loading

## Hypothesis
The current remote `Solution C` collapse is being amplified by the custom weighted-loss path and mixed checkpoint/output reuse. Replacing the custom `WeightedTrainer` with the standard `Trainer`, forcing float32 model loading, and writing into a fresh output directory should remove the degenerate all-class-0 behaviour and produce a materially healthier validation signal on the remote A40 pod.

## Motivation
Known baseline:
- `C_REMOTE_A40_001`
- completed on remote A40 hardware
- produced checkpoints
- collapsed to class `0` on the full dev set
- `accuracy = 0.7232534593317583`
- `macro_f1 = 0.41970231100665883`
- `binary_f1 = 0.0`
- `mcc = 0.0`

This run isolates one defect cluster only:
1. custom weighted-loss training path
2. checkpoint dtype handling
3. stale output directory reuse

## Controlled Change Set
1. remove `WeightedTrainer`
2. use plain `Trainer`
3. load training model with `torch_dtype=torch.float32`
4. load inference model with `torch_dtype=torch.float32`
5. move outputs to `./outputs/deberta_v3_baseline_plain_trainer`
6. keep the same model family and core optimisation settings otherwise

## Run ID
- `C_REMOTE_A40_002`

## Command
Run the same remote notebook execution path against the patched notebook:

```bash
ssh -p 22141 -i ~/.ssh/id_ed25519 root@194.68.245.210 \
  'cd /workspace/NaturalLanguageUnderstandingCourseWork && \
   ln -sf official_coursework/trial_data/ED_trial.csv test.csv && \
   source .venv/bin/activate && \
   mkdir -p experiments/runs/C_REMOTE_A40_002 && \
   jupyter nbconvert --to notebook --execute solution_c_baseline_development.ipynb \
     --output executed_solution_c_baseline.ipynb \
     --output-dir experiments/runs/C_REMOTE_A40_002 \
     --ExecutePreprocessor.timeout=0 \
     2>&1 | tee experiments/runs/C_REMOTE_A40_002/stdout.log'
```

## Acceptance Criteria
1. run completes end to end
2. `eval_loss` is finite
3. checkpoint exists in `./outputs/deberta_v3_baseline_plain_trainer`
4. dev predictions are not all one class
5. `mcc > 0.0`
6. `macro_f1 > 0.41970231100665883`

## Rejection Condition
Reject this patch if any of the following remain true:
1. `eval_loss` becomes `NaN`
2. predictions are still all class `0`
3. `mcc = 0.0`
4. `macro_f1` does not improve over `C_REMOTE_A40_001`
