.DEFAULT_GOAL := help

.PHONY: help configure docker-build build report test tidy format gpu-check smoke shell \
	monitoring inference inference-logs web web-logs tensorboard tensorboard-logs logs down

CUDA_FLAGS := -std=c++17 -arch=sm_120 -Xcompiler=-Wall,-Wextra
CUDA_SMOKE_BINARY := /tmp/wordleml-cuda-smoke

help:
	@echo "Wordle ML development commands (all builds and tests run in containers):"
	@echo "  make configure     Create .env from .env.example when it is missing"
	@echo "  make docker-build  Build the GoMLX/CUDA and web images"
	@echo "  make build         Compile both Go modules"
	@echo "  make report        Show the command that verifies report rows and TensorBoard evidence, then writes Markdown"
	@echo "  make test          Run both Go test suites"
	@echo "  make smoke         Verify one approved RTX 5070 Ti/5050, sm_120 CUDA, and GoMLX/XLA"
	@echo "  make inference     Start the warm CUDA inference API"
	@echo "  make monitoring    Start the gameplay visualiser and TensorBoard"
	@echo "  make shell         Open a shell in the GoMLX/CUDA container"
	@echo "  make down          Stop the project containers"

configure: .env

docker-build: .env
	docker compose build wordleml web

build: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go build -o /tmp/wordleml-smoke ./cmd/smoke && go build -o /tmp/wordleml-inspect ./cmd/inspect && go build -o /tmp/wordleml-train ./cmd/train && go build -o /tmp/wordleml-production ./cmd/production && go build -o /tmp/wordleml-evaluate ./cmd/evaluate && go build -o /tmp/wordleml-report ./cmd/report && go build -o /tmp/wordleml-serve ./cmd/serve && cd /workspace/web && go build -o /tmp/wordleml-web ./cmd/server'

report: docker-build
	@echo "Use: docker compose run --rm --no-deps wordleml go run ./cmd/report -overfit-run-id=<id> -mini-run-id=<id> -full-run-id=<id> -output=../docs/ml/initial-training-proof-report.md"

test: docker-build
	docker compose run --rm --no-deps wordleml bash -lc \
		'go test -p 1 ./... && cd /workspace/web && go test ./...'

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

inference: .env
	docker compose up -d --build inference

inference-logs:
	docker compose logs --follow inference

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
