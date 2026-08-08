# Wordle ML

Wordle ML is a Go and CUDA project that will train a GoMLX model to play Wordle. The model itself is still to come; the
repository currently provides the model vocabularies, reproducible development stack, a GPU smoke test, TensorBoard,
and a placeholder web visualiser.

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

TensorBoard is intentionally empty until the model starts emitting summaries. See
[`docs/development.md`](docs/development.md) for the container layout, GPU selection, and other commands.

The training, validation, and final-test solution splits and proposed model-action vocabulary live in [`data/`](data/).
Their format, generation, and provenance are documented in
[`docs/data/overview-of-wordlists.md`](docs/data/overview-of-wordlists.md).
