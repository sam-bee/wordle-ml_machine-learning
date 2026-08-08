# Wordle ML

Wordle ML is a Go and CUDA project preparing a GoMLX policy to play Wordle. It now contains the frozen imitation
corpus, shared state encoder and batcher, and the initial supervised-training plumbing, alongside the model,
reproducible development stack, GPU smoke test, TensorBoard, and placeholder web visualiser. No training experiment or
result has been run yet; gameplay is still future work.

## Quick start

The development commands run builds and tests in Docker:

```console
make smoke
make monitoring
```

The smoke test passes only the configured RTX 5070 Ti into the container, compiles a CUDA kernel for `sm_120`, runs it,
and executes a small GoMLX graph using the `xla:cuda` backend.

Once monitoring is running, open:

- splash page: <http://127.0.0.1:8082>
- TensorBoard: <http://127.0.0.1:6007>

The training command writes scalar events here when a run is started. See
[`docs/development.md`](docs/development.md) for the container layout, GPU selection, and other commands.

The training, validation, and final-test solution splits and proposed model-action vocabulary live in [`data/`](data/).
Their format, generation, and provenance are documented in
[`docs/data/overview-of-wordlists.md`](docs/data/overview-of-wordlists.md).

The policy consumes precomputed candidate-set features and emits one raw logit per action. Its deliberately small
residual architecture and exact parameter accounting are documented in
[`docs/ml/model-structure.md`](docs/ml/model-structure.md).

The offline WDIT v3 corpus comes from synthetic-data generator release `v0.1.0`: 52,726 training records (including
one opening record), 1,600 mini records, and 2,500 records in each validation and final-test split. The final test is
held back and untouched. The data contract and first supervised-run plumbing are described in
[`docs/ml/supervised-training.md`](docs/ml/supervised-training.md).

The training command is safe to inspect before the first experiment: with its
default `-steps=0`, it only prints the resolved configuration.

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train
```

The first bounded experiment is the next task; see
[`docs/project-plan/next-steps.md`](docs/project-plan/next-steps.md).
