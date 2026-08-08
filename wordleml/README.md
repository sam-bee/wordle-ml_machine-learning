# Wordle ML application

This module contains the Wordle policy model, offline imitation-data reader, fixed supervised-training proof workflow,
and checkpoint evaluation.
The current components are:

- `policy` implements and tests the configurable GoMLX model architecture;
- `vocabulary` loads and validates the canonical action, solution, and split word lists;
- `modelstate` converts a candidate bitset and turn into the policy's four inputs;
- `imitationdata` validates the frozen WDIT v3 teacher files and expands each
  compact record through the shared model-state encoder;
- `cmd/inspect` prints a mini/train/validation record, its candidates and
  teacher top-16, without opening the final-test dataset;
- `supervised` applies an availability mask outside the raw-logit policy,
  optimizes sparse cross-entropy against the teacher's top action, tracks
  top-1/5/16 agreement, and retains optimizer diagnostics;
- `proofrun` defines the fixed `overfit`, `mini`, and `full` stages, including
  checkpoint/resume and proof gates. A fresh mini run must stop at 500 before
  resuming to 1,000; the runner owns the overfit run-zero 10-game baseline;
- `proofgames` runs the shared game-engine evaluation used by both the
  runner-owned overfit baseline and later checkpoint evaluation;
- `proofeval` and `cmd/evaluate` independently reload only the allowed
  post-training checkpoints: mini latest 10 games, then full initial 100
  games, full best 100 games, and full best ablations; never final-test data;
- `proofreport` and `cmd/report` consume the three completed run IDs, verify
  four proof-report rows, TensorBoard event files for all proof stages, and
  game-summary tags before atomically writing
  `docs/ml/initial-training-proof-report.md`;
- `tensorboard` writes scalar and histogram TensorBoard event files using only
  the Go standard library;
- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies exactly one isolated RTX 5070 Ti or RTX 5050
  (including RTX 5050 Laptop GPU), rejects all other devices, and runs a CUDA
  kernel compiled only for `sm_120`.

Use `make smoke` from the repository root to build and execute both in the development container. Project-level
documentation belongs in `docs/`; see [`docs/ml/model-structure.md`](../docs/ml/model-structure.md) for the policy inputs,
layers, and parameter count, [`docs/ml/board-state-encoding.md`](../docs/ml/board-state-encoding.md) for the shared input
representation, and [`docs/ml/supervised-training.md`](../docs/ml/supervised-training.md) for the data and training
boundary. The proof workflow is implemented, but no successful proof-run result is claimed and final-test records
remain untouched.

The train and validation solution IDs are disjoint. The runner nevertheless
records the known exact overlap in their encoded state distributions—190 of
2,445 unique validation states—as provenance with agreeing teacher labels; it
is not solution-split leakage.

Inside the development container, inspect the first mini record with:

```console
go run ./cmd/inspect
```
