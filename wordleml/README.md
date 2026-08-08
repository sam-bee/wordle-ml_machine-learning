# Wordle ML application

This module contains the Wordle policy model and will later contain training code and CUDA integration. The current
components are:

- `policy` implements and tests the configurable GoMLX model architecture;
- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies the isolated RTX 5070 Ti and runs a CUDA kernel compiled only for `sm_120`.

Use `make smoke` from the repository root to build and execute both in the development container. Project-level
documentation belongs in `docs/`; see [`docs/ml/model-structure.md`](../docs/ml/model-structure.md) for the policy inputs,
layers, and parameter count.
