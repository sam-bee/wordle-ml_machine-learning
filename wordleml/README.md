# Wordle ML application

This module contains the Wordle policy model and will later contain training code and CUDA integration. The current
components are:

- `policy` implements and tests the configurable GoMLX model architecture;
- `imitationdata` validates the frozen WDIT v3 teacher files and expands each
  compact record through the shared model-state encoder;
- `cmd/inspect` prints a mini/train/validation record, its candidates and
  teacher top-16, without opening the final-test dataset;
- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies the isolated RTX 5070 Ti and runs a CUDA kernel compiled only for `sm_120`.

Use `make smoke` from the repository root to build and execute both in the development container. Project-level
documentation belongs in `docs/`; see [`docs/ml/model-structure.md`](../docs/ml/model-structure.md) for the policy inputs,
layers, and parameter count.

Inside the development container, inspect the first mini record with:

```console
go run ./cmd/inspect
```
