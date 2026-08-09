# Wordle ML

Wordle ML is a Go and CUDA project preparing a GoMLX policy to play Wordle. It contains the frozen imitation corpus,
shared state encoder and batcher, fixed supervised-training proof stages, independently reloadable checkpoint
evaluation, a reproducible development stack, GPU smoke test, TensorBoard, and a live checkpoint-backed web visualiser.
The proof workflow completed its first fixed validation proof; see the
[initial training proof report](docs/ml/initial-training-proof-report.md).

The best full checkpoint reduced validation loss from 8.3005 to 3.1633 and
raised top-1 agreement from 0.0056 to 0.5008. On the fixed 100-game validation
population it solved 97/100 games versus 4/100 at initialization, reducing mean
guesses from 5.86 to 3.65; the mini stop/resume proof also passed. These are
proof-run results, not a claim of broader generalization: the final-test split
remains sealed and the recorded state-distribution overlap is documented in the
report.

## Quick start

The development commands run builds and tests in Docker:

```console
make smoke
make monitoring
```

The smoke test passes exactly one configured, approved GPU into the container: an RTX 5070 Ti or RTX 5050 (including
the RTX 5050 Laptop GPU name). It rejects every other visible device, including an RTX 3060, requires compute
capability 12.0, compiles a CUDA kernel for `sm_120`, and executes a small GoMLX graph using `xla:cuda`.

Once monitoring is running, open:

- live Wordle policy: <http://127.0.0.1:8082>
- TensorBoard: <http://127.0.0.1:6007>

The visualiser loads the best checkpoint from `WORDLEML_INFERENCE_RUN_ID`
(`proof-full-20260808` by default), offers the fixed validation solutions, and
animates a complete game returned by the internal CUDA inference API. The
browser and web container never receive direct GPU access.

Proof runs write scalar and histogram events under `runs/<run-id>/events`, which TensorBoard discovers beneath
`runs/`. See
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

Run one fixed proof stage with a stable, unused run ID. The run records its
immutable configuration and provenance, checkpoints, metrics, log, and
TensorBoard events under `runs/<run-id>/`.

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=overfit-001 -stage=overfit
```

The fixed stages must run in proof order. The runner owns the overfit run-zero
baseline: it independently reloads the initial checkpoint and records its first
10 validation games, so `cmd/evaluate` deliberately cannot rerun
`overfit initial games10`. A **fresh** mini run must stop at update 500, then
the same run ID resumes without `-stop-at`:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini -stop-at=500
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=mini-001 -checkpoint=latest -mode=games10
```

After mini evaluation has passed, run the full stage and create its
validation-only artifacts in order:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=full-001 -stage=full
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=initial -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=ablations
docker compose run --rm --no-deps wordleml go run ./cmd/report -overfit-run-id=overfit-001 -mini-run-id=mini-001 -full-run-id=full-001 -output=../docs/ml/initial-training-proof-report.md
```

`cmd/report` consumes the three run IDs, verifies its four rendered rows
(run-zero baseline, overfit, mini, and full), re-verifies TensorBoard event
files for every proof stage and their game-summary tags, and atomically writes
the report. A failed gate returns non-zero and writes a visibly incomplete
report with the stopping reason, without replacing an existing successful
report.
The final-test split remains sealed: neither training nor evaluation offers a
test-split mode.

## Fixed production run

The first production run is deliberately separate from the retained proof
stages. After `make format`, `make test`, and `make smoke` have passed and the
implementation is committed and pushed, start it from initialization with a
fresh timestamped ID:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/production -run-id=<timestamped-id>
```

It fixes the existing full training split, policy/encoder/objective and
opening-state sampling for 10,000 updates, then independently validates the
best checkpoint and plays all 100 validation games. It writes the separate
validation-only comparison report to
[`docs/ml/production-training-report.md`](docs/ml/production-training-report.md).

The completed first run, `production-20260809-005026Z`, selected update 2,200:
validation loss improved from the retained proof's 3.1633 to 3.1341, while
both checkpoints solved 97/100 validation games and mean guesses changed from
3.65 to 3.68. This is a validation-only result, not a generalization claim;
the final-test split remains sealed. See
[`docs/ml/supervised-training.md`](docs/ml/supervised-training.md) for fixed
configuration, resume, and handoff details.
