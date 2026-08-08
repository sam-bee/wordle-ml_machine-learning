# Wordle ML application

This module will contain the Wordle model, training code, and CUDA integration. It currently contains a data preparation
command and two executable smoke tests:

- `cmd/split-solutions` reproducibly partitions the complete solution vocabulary;
- `cmd/smoke` evaluates a small graph with GoMLX;
- `cuda/smoke.cu` verifies the isolated RTX 5070 Ti and runs a CUDA kernel compiled only for `sm_120`.

Use `make data-split` or `make smoke` from the repository root to run the relevant command in the development container.
Project-level documentation belongs in `docs/`.
