# Wordle ML application

This module contains the Wordle policy model, offline imitation-data reader, and initial supervised-training plumbing.
The current components are:

- `policy` implements and tests the configurable GoMLX model architecture;
- `vocabulary` loads and validates the canonical action, solution, and split word lists;
- `modelstate` converts a candidate bitset and turn into the policy's four inputs;
- `imitationdata` validates the frozen WDIT v3 teacher files and expands each
  compact record through the shared model-state encoder;
- `cmd/inspect` prints a mini/train/validation record, its candidates and
  teacher top-16, without opening the final-test dataset;
- `supervised` applies an availability mask outside the raw-logit policy, optimizes
  sparse cross-entropy against the teacher's top action with Adam (default
  learning rate 0.001), tracks top-1/5/16 agreement, and saves checkpoints;
- `cmd/train` is the bounded training entry point; its default zero-step mode
  only prints the resolved configuration;
- `tensorboard` writes scalar TensorBoard event files using only the Go
  standard library;
- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies the isolated RTX 5070 Ti and runs a CUDA kernel compiled only for `sm_120`.

Use `make smoke` from the repository root to build and execute both in the development container. Project-level
documentation belongs in `docs/`; see [`docs/ml/model-structure.md`](../docs/ml/model-structure.md) for the policy inputs,
layers, and parameter count, [`docs/ml/board-state-encoding.md`](../docs/ml/board-state-encoding.md) for the shared input
representation, and [`docs/ml/supervised-training.md`](../docs/ml/supervised-training.md) for the data and training
boundary. The plumbing is in place, but no training experiment or result has been run and final-test records remain
untouched.

Inside the development container, inspect the first mini record with:

```console
go run ./cmd/inspect
```
