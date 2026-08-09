# Development Environment

All compilation and tests are intended to run in Docker. This keeps the Go, CUDA, PJRT, and eventual front-end
toolchains out of the host environment and gives demonstrations used in the talk a reproducible starting point.

## Services

The Compose project has five services:

- `wordleml` is the development image. It uses CUDA 13.1, Go 1.26.5, GoMLX 0.28.0, go-xla 0.3.0, and the CUDA 13 PJRT
  plugin. Source is bind-mounted at `/workspace` so container commands can edit and build it.
- `inference` loads one passed full-run best checkpoint, keeps its GoMLX session warm on the configured CUDA device,
  and exposes an internal-only complete-game API on port 8090.
- `web` serves the gameplay visualiser and proxies same-origin `/api` requests to `inference`. A multi-stage Docker
  build tests and compiles the Go server and its embedded HTML, CSS, and JavaScript into a small runtime image.
- `tensorboard` runs TensorBoard from the official TensorFlow 2.20 image and reads self-contained run event directories
  below `runs/`. No GPU is passed to `web` or `tensorboard`.
- `cudaweb` is a separate, GPU-enabled one-process demo on port 8083. It loads one validated exported FP32 model and
  serves the browser and game API directly through the hand-written CUDA/cgo backend; it does not proxy to `inference`
  and does not implement model hot swapping.

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

The CUDA/cgo backend applies the same visible-device checks when its native
model is created, and reports the approved device name, compute capability,
CUDA runtime version, and driver version through its model metadata.

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

### CUDA/cgo direct inference

The existing `inference` plus `web` services above are the retained GoMLX/XLA
route. The CUDA/cgo route is a distinct one-process service: it loads one
portable artifact, creates a locked CUDA worker, and serves its browser/API on
<http://127.0.0.1:8083>. It is the only service that should be described as
hand-written CUDA via cgo.

The usual CUDA/cgo sequence is:

```console
make cuda-cgo-export RUN_ID=seed-replication-20260809-132505Z CHECKPOINT=best
make cuda-cgo-build
make cuda-cgo-test
make cuda-cgo-verify MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-bench MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-demo MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

`cuda-cgo-export` is the only step permitted to use GoMLX: it restores and
exports the checkpoint. `cuda-cgo-build`, verifier, benchmark, and `cudaweb`
use `CGO_ENABLED=1` and the `cuda_cgo` build tag; their runtime dependency
graph must exclude GoMLX, PJRT, and XLA. The export command's default run is
the selected seed-replication run recorded in the CUDA/cgo working notes. The
artifact manifest, not the default, is the authoritative run/checkpoint/update
identity.

For reproducible profiler collections, run:

```console
make cuda-cgo-profile-systems MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-profile-compute MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

They write generated reports below `artifacts/cuda-cgo/`. The Systems target
uses the locally installed host Nsight Systems 2025.6.3 directory through a
read-only mount into the unprivileged container, leaving host software and
driver settings unchanged. See
[`artifacts/cuda-cgo/README.md`](../artifacts/cuda-cgo/README.md) for the
mount/reproduction details and report interpretation. The full design,
artifact layout, and measured verification gates are in
[CUDA/cgo inference](ml/cuda-cgo-inference.md).

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

## Production-run handoff

The proof sequence above remains historical and fixed. The first production
run is a separate, single fixed chain; it must not reuse a proof run ID or
checkpoint. Run its preflight first and do not launch if any step fails:

```console
make format
make test
make smoke
```

Commit and push the implementation, verify a clean worktree, and use a fresh
timestamped ID to start the chain:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/production -run-id=<timestamped-id>
```

This command fixes the full training split, existing opening-state sampling,
FP32 Adam, batch size 256, constant learning rate 0.0003, global clip norm 5,
seed 20260808, and 10,000 updates. It emits scalars every 10 updates;
validates and writes `latest` every 100; and writes `best` on validation-loss
improvement. Its immutable run directory records configuration, provenance,
checkpoints, complete optimizer/sampler resume state, TensorBoard events,
training log, validation snapshots, finite-number safety checks, and atomic
final metrics.

To hand an interrupted or detached run to another operator, retain the run ID
and use the same command to resume it. Check the outer chain status at
`runs/<timestamped-id>.status.json` and the checkpoint state at
`runs/<timestamped-id>/run-state.json`, follow
`runs/<timestamped-id>/training.log`, and use
`tensorboard --logdir runs/<timestamped-id>/events` (or the project
TensorBoard service) for live status. After successful training, the command
independently verifies its best checkpoint and runs all 100 validation games,
then atomically creates the validation-only comparison at
`docs/ml/production-training-report.md`. The final-test split is never opened;
any failure stops the post-training chain with retained status and logs.

The recorded train/validation state-overlap audit is not solution-split
leakage: solution IDs remain disjoint, while 190 of 2,445 unique validation
encoded states also occur in training with agreeing teacher labels.
