# Development Environment

All compilation and tests are intended to run in Docker. This keeps the Go, CUDA, PJRT, and eventual front-end
toolchains out of the host environment and gives demonstrations used in the talk a reproducible starting point.

## Services

The Compose project has three services:

- `wordleml` is the development image. It uses CUDA 13.1, Go 1.26.5, GoMLX 0.28.0, go-xla 0.3.0, and the CUDA 13 PJRT
  plugin. Source is bind-mounted at `/workspace` so container commands can edit and build it.
- `web` is the placeholder visualiser. A multi-stage Docker build tests and compiles the Go server, including its
  embedded HTML and CSS, and copies the resulting binary into a small runtime image.
- `tensorboard` runs TensorBoard from the official TensorFlow 2.20 image and reads `data/tensorboard`. No GPU is passed
  to either monitoring service.

`make monitoring` starts the web and TensorBoard services. They bind only to host loopback because neither has an
authentication layer.

## GPU isolation

The desktop contains an RTX 3060 at host index 0 and the target RTX 5070 Ti at host index 1. Compose reserves one device
by the 5070 Ti's UUID from `.env`; it does not use `gpus: all` or rely on `CUDA_VISIBLE_DEVICES`. This means the 3060 is
not passed into the container at all.

The UUID is stable for the physical card, but `.env.example` must be updated if that card is replaced. Resolve UUIDs
without changing the host system:

```console
nvidia-smi --query-gpu=index,uuid,name,compute_cap --format=csv,noheader
```

The CUDA smoke program adds three further checks before running a tiny kernel:

1. exactly one CUDA device is visible;
2. its name is `NVIDIA GeForce RTX 5070 Ti`;
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
make monitoring    # start the splash page and TensorBoard
make down          # stop this project's services
```

`make tidy` and `make format` also execute the Go tools inside the development container. The generated files remain
owned by the host UID and GID configured in `.env`.
