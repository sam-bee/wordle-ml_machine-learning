# Development Environment

All compilation and tests are intended to run in Docker. This keeps the Go, CUDA, PJRT, and eventual front-end
toolchains out of the host environment and gives demonstrations used in the talk a reproducible starting point.

## Services

The Compose project has four services:

- `wordleml` is the development image. It uses CUDA 13.1, Go 1.26.5, GoMLX 0.28.0, go-xla 0.3.0, and the CUDA 13 PJRT
  plugin. Source is bind-mounted at `/workspace` so container commands can edit and build it.
- `inference` loads one passed full-run best checkpoint, keeps its GoMLX session warm on the configured CUDA device,
  and exposes an internal-only complete-game API on port 8090.
- `web` serves the gameplay visualiser and proxies same-origin `/api` requests to `inference`. A multi-stage Docker
  build tests and compiles the Go server and its embedded HTML, CSS, and JavaScript into a small runtime image.
- `tensorboard` runs TensorBoard from the official TensorFlow 2.20 image and reads self-contained run event directories
  below `runs/`. No GPU is passed to `web` or `tensorboard`.

`make monitoring` starts inference, web, and TensorBoard. Only web and TensorBoard bind host ports, both on loopback;
the GPU API remains private to the Compose network.

## GPU isolation

Compose reserves exactly one device by the UUID in `.env`; it does not use `gpus: all` or rely on
`CUDA_VISIBLE_DEVICES`. The selected device must be an RTX 5070 Ti or RTX 5050; `NVIDIA GeForce RTX 5050 Laptop GPU`
is an approved laptop name. This excludes all other devices, including the desktop's RTX 3060.

The UUID is stable for the physical card, but `.env` must name this machine's approved card. Resolve UUIDs without
changing the host system:

```console
nvidia-smi --query-gpu=index,uuid,name,compute_cap --format=csv,noheader
```

The CUDA smoke program adds three further checks before running a tiny kernel:

1. exactly one CUDA device is visible;
2. its name is `NVIDIA GeForce RTX 5070 Ti`, `NVIDIA GeForce RTX 5050`, or
   `NVIDIA GeForce RTX 5050 Laptop GPU`;
3. its compute capability is 12.0.

It is compiled with `-arch=sm_120`; no fallback architecture or PTX target is added. The GoMLX smoke program then runs
a Euclidean-distance graph through `GOMLX_BACKEND=xla:cuda` and checks its result.

## Commands

Run `make help` for the short list. The usual workflow is:

```console
make docker-build  # build the development and web images
make build         # compile both Go modules in the development container
make test          # test both Go modules in the development container
make smoke         # run CUDA and GoMLX GPU smoke tests
make inference     # start the warm CUDA inference API
make monitoring    # start the gameplay visualiser and TensorBoard
make down          # stop this project's services
```

`WORDLEML_INFERENCE_RUN_ID` in `.env` selects a passed full proof run. The
default is `proof-full-20260808`. Once `make monitoring` reports the inference
service healthy, open <http://127.0.0.1:8082> and select a validation solution.
See [inference serving](ml/inference-serving.md) for the REST contract and the
host/device execution split.

`make tidy` and `make format` also execute the Go tools inside the development container. The generated files remain
owned by the host UID and GID configured in `.env`.

The inspection, training, and post-checkpoint evaluation entry points can also
be run directly in that container. Training creates or resumes a self-contained
run under `runs/<run-id>/`; it requires both a run ID and a fixed stage. To
inspect a mini record without running the proof sequence:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/inspect
```

The [completed initial proof](ml/initial-training-proof-report.md) passed its
mini resume gate and, on the fixed validation population, reduced full-run
validation loss from 8.3005 to 3.1633, raised top-1 from 0.0056 to 0.5008, and
improved games from 4/100 at initialization to 97/100 at best (mean guesses
5.86 to 3.65). This is not a final-test or generalization claim.

For a new proof, use fresh matching run IDs and this operator sequence:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=overfit-001 -stage=overfit
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini -stop-at=500
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=mini-001 -checkpoint=latest -mode=games10
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=full-001 -stage=full
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=initial -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=ablations
docker compose run --rm --no-deps wordleml go run ./cmd/report -overfit-run-id=overfit-001 -mini-run-id=mini-001 -full-run-id=full-001 -output=../docs/ml/initial-training-proof-report.md
```

`overfit`, `mini`, and `full` are fixed configurations. A fresh mini run must
use `-stop-at=500`; the second invocation resumes the same ID without it.
Overfit's run-zero initial 10-game baseline is runner-owned, so
`cmd/evaluate` cannot rerun `overfit initial games10`. Once mini has passed,
evaluate its latest checkpoint with `-checkpoint=latest -mode=games10`.

The commands above are the required proof order: overfit; stopped and resumed
mini; mini latest-checkpoint games; full training; full initial games, full best
games, and full ablations; then report generation.

The report command consumes the three run IDs, verifies four rendered report
rows (run-zero baseline, overfit, mini, and full), and atomically writes the
Markdown report. It re-verifies TensorBoard event files for every proof stage,
including the game-summary tags, and re-derives the mini run's continuous event
proof. It fails until every required proof, event, and evaluation artifact has
passed validation. On failure it still exits non-zero, but writes a clearly
marked incomplete report with the discovered stage, update, and blocking gate;
it never replaces an existing successful report with that failure artifact.

Each run contains `config.json`, `metadata.json`, `run-state.json`, `final-metrics.json`, `training.log`, events,
`checkpoints/{initial,latest,best}`, and an `evaluations/` directory. Evaluation writes structured results and game
JSONL there, appends game summary scalars to the run's events, and only evaluates the frozen validation split.

The recorded train/validation state-overlap audit is not solution-split
leakage: solution IDs remain disjoint, while 190 of 2,445 unique validation
encoded states also occur in training with agreeing teacher labels.
