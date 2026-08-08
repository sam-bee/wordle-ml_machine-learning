# Wordle ML application

This module will contain the Wordle model, training code, and CUDA integration. It currently contains two executable
smoke tests:

- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies the isolated RTX 5070 Ti and runs a CUDA kernel compiled only for `sm_120`.

Use `make smoke` from the repository root to build and execute both in the development container. Project-level
documentation belongs in `docs/`.
