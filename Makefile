.DEFAULT_GOAL := help

.PHONY: help configure docker-build build test tidy format gpu-check smoke shell \
	monitoring web web-logs tensorboard tensorboard-logs logs down

CUDA_FLAGS := -std=c++17 -arch=sm_120 -Xcompiler=-Wall,-Wextra
CUDA_SMOKE_BINARY := /tmp/wordleml-cuda-smoke

help:
	@echo "Wordle ML development commands (all builds and tests run in containers):"
	@echo "  make configure     Create .env from .env.example when it is missing"
	@echo "  make docker-build  Build the GoMLX/CUDA and web images"
	@echo "  make build         Compile both Go modules"
	@echo "  make test          Run both Go test suites"
	@echo "  make smoke         Verify the 5070 Ti, sm_120 CUDA, and GoMLX/XLA"
	@echo "  make monitoring    Start the splash page and TensorBoard"
	@echo "  make shell         Open a shell in the GoMLX/CUDA container"
	@echo "  make down          Stop the project containers"

configure: .env

docker-build: .env
	docker compose build wordleml web

build: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go build -o /tmp/wordleml-smoke ./cmd/smoke && go build -o /tmp/wordleml-inspect ./cmd/inspect && cd /workspace/web && go build -o /tmp/wordleml-web ./cmd/server'

test: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go test ./... && cd /workspace/web && go test ./...'

tidy: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go mod tidy && cd /workspace/web && go mod tidy'

format: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go fmt ./... && cd /workspace/web && go fmt ./...'

gpu-check: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'nvcc $(CUDA_FLAGS) cuda/smoke.cu -o $(CUDA_SMOKE_BINARY) && $(CUDA_SMOKE_BINARY)'

smoke: gpu-check
	docker compose run --rm --no-deps wordleml go run ./cmd/smoke

shell: docker-build
	docker compose run --rm --no-deps wordleml bash

monitoring: .env
	docker compose up -d --build web tensorboard

web: .env
	docker compose up -d --build web

web-logs:
	docker compose logs --follow web

tensorboard: .env
	docker compose up -d tensorboard

tensorboard-logs:
	docker compose logs --follow tensorboard

logs:
	docker compose logs --follow

down:
	docker compose down

.env:
	cp .env.example .env
