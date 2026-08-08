# Wordle ML

Wordle ML is a Go and CUDA project for training a GoMLX policy to play Wordle. The repository currently provides the
policy architecture, fixed model vocabularies, shared state encoder, frozen offline teacher corpus, deterministic
batching, reproducible development stack, GPU smoke test, TensorBoard service, and placeholder web visualiser. The
supervised training loop and gameplay are the next stages.

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

The checked-in WDIT v3 corpus contains 52,726 training records (including the opening state), 1,600 mini records, and
2,500 records in each of the validation and sealed final-test splits. `wordleml/imitationdata` validates and batches
these records without rerunning the expensive teacher.

The policy consumes precomputed candidate-set features and emits one raw logit per action. Its deliberately small
residual architecture and exact parameter accounting are documented in
[`docs/ml/model-structure.md`](docs/ml/model-structure.md).

The remaining work before the first experiment is tracked in
[`docs/project-plan/next-steps.md`](docs/project-plan/next-steps.md).
