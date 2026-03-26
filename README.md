# Solo Coursework Rebuild

This branch is an isolated solo-work branch intended to answer one question:

Can the coursework be rebuilt cleanly from the official materials alone, without relying on the existing team `Solution B` and `Solution C` implementations?

## Branch intent

- Branch name: `codex/solo-coursework-rebuild`
- Starting point: a pre-solution repo state
- Included inputs:
  - `training_data/`
  - `official_coursework/`
  - `COMP34812 Coursework - Model Card Resources/`
- Excluded on purpose:
  - prior team `Solution B` code
  - prior team `Solution C` notebooks
  - prior experiment history
  - prior decision records

## Current structure

- `training_data/`
  - coursework train/dev data
- `official_coursework/`
  - coursework spec, trial data, and local scorer bundle
- `src/solution_b/`
  - empty solo implementation area for a non-transformer solution
- `src/solution_c/`
  - empty solo implementation area for a transformer solution
- `experiments/`
  - solo experiment records for this branch only
- `notes/`
  - planning and decision notes for the solo rebuild

## Immediate next steps

1. Build a fresh `Solution B` baseline in `src/solution_b/`.
2. Build a fresh `Solution C` baseline in `src/solution_c/`.
3. Record all solo runs under `experiments/`.
4. Compare the solo rebuild against the team branch only after both paths are real.

## Validation assets

- scorer:
  - `official_coursework/nlu_bundle-feature-unified-local-scorer`
- spec:
  - `official_coursework/NaturalLanguageUnderstandingCourseWorkSpec.pdf`
- trial files:
  - `official_coursework/trial_data/`

## Notes

This branch is intentionally clean and minimal. It is a rebuild workspace, not a continuation of the current team solution path.
