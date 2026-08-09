.DEFAULT_GOAL := help

.PHONY: help configure docker-build build report test tidy format gpu-check smoke shell \
	monitoring inference inference-logs web web-logs tensorboard tensorboard-logs logs down \
	cuda-cgo-export cuda-cgo-native cuda-cgo-build cuda-cgo-test cuda-cgo-verify cuda-cgo-final-test cuda-cgo-final-test-preflight \
	cuda-cgo-dependency-audit cuda-cgo-bench cuda-cgo-demo cuda-cgo-demo-logs cuda-cgo-profile-systems \
	cuda-cgo-profile-compute

CUDA_FLAGS := -std=c++17 -arch=sm_120 -Xcompiler=-Wall,-Wextra
CUDA_SMOKE_BINARY := /tmp/wordleml-cuda-smoke

# CUDA/cgo inference is deliberately a fixed, single-model demonstration.
# MODEL_DIR is relative to the repository on the host, and becomes an absolute
# path below because the wordleml container works from /workspace/wordleml.
RUN_ID ?= seed-replication-20260809-132505Z
CHECKPOINT ?= best
MODEL_DIR ?= runs/$(RUN_ID)/exports/cuda-f32-v1/$(CHECKPOINT)
CUDA_MODEL_DIR := $(if $(filter /%,$(MODEL_DIR)),$(MODEL_DIR),/workspace/$(MODEL_DIR))

CUDA_BUILD_DIR := build/cuda
CUDA_NATIVE_OBJECT := $(CUDA_BUILD_DIR)/wordle_cuda.o
CUDA_NATIVE_LIBRARY := $(CUDA_BUILD_DIR)/libwordle_cuda.a
CUDA_NATIVE_FLAGS := -O3 -lineinfo -std=c++17 -arch=sm_120 -Xcompiler=-fPIC,-Wall,-Wextra
CUDA_CGO_TAG := cuda_cgo
CUDA_BIN_DIR := bin
CUDA_ARTIFACT_DIR := artifacts/cuda-cgo
CUDA_SYSTEMS_DIR := $(CUDA_ARTIFACT_DIR)/nsight-systems
CUDA_COMPUTE_DIR := $(CUDA_ARTIFACT_DIR)/nsight-compute
CUDA_SYSTEMS_CONTAINER_DIR := /workspace/$(CUDA_SYSTEMS_DIR)
CUDA_COMPUTE_CONTAINER_DIR := /workspace/$(CUDA_COMPUTE_DIR)
CUDA_BENCH_WARMUP ?= 20
CUDA_BENCH_ITERATIONS ?= 200
CUDA_PROFILE_WARMUP ?= 20
CUDA_PROFILE_ITERATIONS ?= 20
CUDA_NCU_LAUNCH_SKIP ?= 20
CUDA_NCU_PROFILE_ITERATIONS ?= 40

# The host-installed Nsight Systems target is mounted read-only into the
# container.  Its sibling host-linux-x64 directory supplies nsys stats reports,
# so mount the version directory rather than just the CLI executable.
NSYS_HOST_DIR ?= $(shell nsys_binary=$$(readlink -f "$$(command -v nsys)" 2>/dev/null); test -n "$$nsys_binary" && dirname "$$(dirname "$$nsys_binary")")
NSYS_CONTAINER_ROOT := /opt/wordleml-nsys
NSYS_CONTAINER_BINARY := /tmp/wordleml-nsys

CUDA_RUN := docker compose run --rm --no-deps wordleml
# NVIDIA driver policy on the development machines restricts performance
# counters.  Root plus SYS_ADMIN in this short-lived profiling container is
# sufficient; it neither changes the host driver policy nor the normal demo.
CUDA_NCU_RUN := docker compose run --rm --no-deps --user root --cap-add SYS_ADMIN \
	-e WORDLEML_HOST_UID=$(shell id -u) -e WORDLEML_HOST_GID=$(shell id -g) wordleml
CUDA_NSYS_RUN := docker compose run --rm --no-deps \
	-v $(NSYS_HOST_DIR):$(NSYS_CONTAINER_ROOT):ro wordleml

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
	@echo "  make cuda-cgo-export RUN_ID=<run> CHECKPOINT=best"
	@echo "                       Export the selected GoMLX checkpoint for CUDA"
	@echo "  make cuda-cgo-build Build the sm_120 CUDA archive and tagged binaries"
	@echo "  make cuda-cgo-test  Run tagged CUDA/cgo tests on the approved GPU"
	@echo "  make cuda-cgo-dependency-audit"
	@echo "                       Prove runtime binaries contain no GoMLX/PJRT/XLA"
	@echo "  make cuda-cgo-verify MODEL_DIR=<path>"
	@echo "                       Verify golden vectors and validation games"
	@echo "  make cuda-cgo-final-test MODEL_DIR=<path>"
	@echo "                       One-time, confirmation-gated final-test gameplay evaluation"
	@echo "  make cuda-cgo-bench MODEL_DIR=<path>"
	@echo "                       Run the deterministic CUDA benchmark"
	@echo "  make cuda-cgo-demo MODEL_DIR=<path>"
	@echo "                       Start the one-process CUDA/cgo browser demo on :8083"
	@echo "  make cuda-cgo-profile-systems MODEL_DIR=<path>"
	@echo "                       Capture an Nsight Systems report using host nsys read-only"
	@echo "  make cuda-cgo-profile-compute MODEL_DIR=<path>"
	@echo "                       Capture the policy-logits Nsight Compute report"
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

# The neutral exported artifact is produced offline.  It may use GoMLX, unlike
# the CUDA runtime commands below.
cuda-cgo-export: docker-build
	$(CUDA_RUN) go run ./cmd/exportcuda -run-id=$(RUN_ID) -checkpoint=$(CHECKPOINT)

# cgo cannot compile .cu sources itself.  Keep the native compile and archive
# explicit so it is visible in the talk and reproducible in the dev container.
cuda-cgo-native: docker-build
	$(CUDA_RUN) bash -lc \
		'mkdir -p /workspace/$(CUDA_BUILD_DIR) && \
		nvcc $(CUDA_NATIVE_FLAGS) -Icuda/inference -c cuda/inference/wordle_cuda.cu \
			-o /workspace/$(CUDA_NATIVE_OBJECT) && \
		ar rcs /workspace/$(CUDA_NATIVE_LIBRARY) /workspace/$(CUDA_NATIVE_OBJECT)'

cuda-cgo-build: cuda-cgo-native
	$(CUDA_RUN) bash -lc \
		'mkdir -p /workspace/$(CUDA_BIN_DIR) && \
		CGO_ENABLED=1 go build -tags $(CUDA_CGO_TAG) -o /workspace/$(CUDA_BIN_DIR)/cudaverify ./cmd/cudaverify && \
		CGO_ENABLED=1 go build -tags $(CUDA_CGO_TAG) -o /workspace/$(CUDA_BIN_DIR)/cudabench ./cmd/cudabench && \
		CGO_ENABLED=1 go build -tags $(CUDA_CGO_TAG) -ldflags "-X main.evaluatorCommit=$$(git -C /workspace rev-parse HEAD)" -o /workspace/$(CUDA_BIN_DIR)/cudafinal ./cmd/cudafinal && \
		CGO_ENABLED=1 go build -tags $(CUDA_CGO_TAG) -o /workspace/$(CUDA_BIN_DIR)/cudaweb ./cmd/cudaweb'

cuda-cgo-test: cuda-cgo-native
	$(CUDA_RUN) bash -lc 'CGO_ENABLED=1 go test -p 1 -tags $(CUDA_CGO_TAG) ./...'

cuda-cgo-dependency-audit: cuda-cgo-build
	$(CUDA_RUN) bash -lc \
		'dependencies="$$(go list -deps -tags $(CUDA_CGO_TAG) ./cmd/cudaweb ./cmd/cudaverify ./cmd/cudabench ./cmd/cudafinal)"; \
		if printf "%s\n" "$$dependencies" | grep -Eiq "gomlx|go-xla|pjrt|(^|/)xla($$|/)|github.com/sam-bee/wordle-ml_machine-learning/(proofgames|serving|proofrun|supervised)($$|/)"; then \
			printf "%s\n" "$$dependencies" | grep -Ei "gomlx|go-xla|pjrt|(^|/)xla($$|/)|github.com/sam-bee/wordle-ml_machine-learning/(proofgames|serving|proofrun|supervised)($$|/)"; \
			echo "forbidden CUDA runtime dependency found" >&2; \
			exit 1; \
		fi; \
		for binary in cudaweb cudaverify cudabench cudafinal; do \
			go version -m /workspace/$(CUDA_BIN_DIR)/$$binary; \
			ldd /workspace/$(CUDA_BIN_DIR)/$$binary; \
		done; \
		echo "dependency audit passed: cudaweb, cudaverify, cudabench, and cudafinal have no forbidden runtime package dependency"'

cuda-cgo-verify: cuda-cgo-dependency-audit
	$(CUDA_RUN) /workspace/$(CUDA_BIN_DIR)/cudaverify -model-dir=$(CUDA_MODEL_DIR)

# This check is intentionally separate and runs before the recursive build
# below. Every tracked runtime input must match HEAD before a binary can be
# built for the only final-test read. AGENTS.md is the sole exception because
# it is an operator instruction file, not compiled input, and currently has a
# preserved user edit.
cuda-cgo-final-test-preflight:
	@git diff --quiet HEAD -- . ':(exclude)AGENTS.md' && \
		test -z "$$(git ls-files --others --exclude-standard)" || \
		{ echo "commit every runtime source change before final-test evaluation" >&2; exit 1; }

# This is intentionally not a prerequisite of any other target. Its command
# must be invoked explicitly; cudafinal claims one fixed report path before
# opening the final-test word list, so a later invocation cannot retry it.
cuda-cgo-final-test: cuda-cgo-final-test-preflight
	$(MAKE) cuda-cgo-dependency-audit
	$(CUDA_RUN) bash -lc \
		'git diff --quiet HEAD -- . ":(exclude)AGENTS.md" && \
		test -z "$$(git ls-files --others --exclude-standard)" || \
			{ echo "runtime sources changed after final-test preflight" >&2; exit 1; }; \
		CGO_ENABLED=1 go build -tags $(CUDA_CGO_TAG) \
			-ldflags "-X main.evaluatorCommit=$$(git -C /workspace rev-parse HEAD)" \
			-o /workspace/$(CUDA_BIN_DIR)/cudafinal ./cmd/cudafinal && \
		/workspace/$(CUDA_BIN_DIR)/cudafinal -confirm-final-test \
			-model-dir=$(CUDA_MODEL_DIR)'

cuda-cgo-bench: cuda-cgo-dependency-audit
	$(CUDA_RUN) /workspace/$(CUDA_BIN_DIR)/cudabench -model-dir=$(CUDA_MODEL_DIR) \
		-warmup=$(CUDA_BENCH_WARMUP) -iterations=$(CUDA_BENCH_ITERATIONS)

cuda-cgo-demo: cuda-cgo-build
	CUDA_CGO_MODEL_DIR=$(CUDA_MODEL_DIR) docker compose up -d --no-deps --force-recreate cudaweb

cuda-cgo-demo-logs:
	docker compose logs --follow cudaweb

# The CUDA apt sources have no compatible Nsight Systems CLI package.  Instead,
# mount the complete, locally installed host target read-only.  nsys needs to
# be invoked through a symlink which points at target-linux-x64/nsys; that
# symlink lives only in the short-lived container's /tmp directory.
cuda-cgo-profile-systems: cuda-cgo-build
	test -x "$(NSYS_HOST_DIR)/target-linux-x64/nsys" || { echo "host nsys target not found; install Nsight Systems or set NSYS_HOST_DIR" >&2; exit 127; }
	mkdir -p $(CUDA_SYSTEMS_DIR)
	$(CUDA_NSYS_RUN) bash -lc \
		'ln -sf $(NSYS_CONTAINER_ROOT)/target-linux-x64/nsys $(NSYS_CONTAINER_BINARY) && \
		$(NSYS_CONTAINER_BINARY) profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
			--force-overwrite=true -o $(CUDA_SYSTEMS_CONTAINER_DIR)/wordle-inference \
			/workspace/$(CUDA_BIN_DIR)/cudabench -model-dir=$(CUDA_MODEL_DIR) \
			-report=/tmp/wordleml-cudabench-systems.json \
			-warmup=$(CUDA_PROFILE_WARMUP) -iterations=$(CUDA_PROFILE_ITERATIONS) && \
		$(NSYS_CONTAINER_BINARY) stats --force-export=true --force-overwrite=true --format=column \
			--output=$(CUDA_SYSTEMS_CONTAINER_DIR)/wordle-inference-summary \
			--report=cuda_api_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace,nvtx_pushpop_sum \
			$(CUDA_SYSTEMS_CONTAINER_DIR)/wordle-inference.nsys-rep'

# Nsight Compute is run in an ephemeral privileged container only.  This is
# needed solely for the driver's performance-counter policy; normal builds,
# tests and the web demo remain the unprivileged wordleml user.  The final
# find/chown repairs only files in this generated report directory.
cuda-cgo-profile-compute: cuda-cgo-build
	mkdir -p $(CUDA_COMPUTE_DIR)
	$(CUDA_NCU_RUN) bash -lc \
		'ncu --set full -k regex:^policy_logits_with_bonus.* -s $(CUDA_NCU_LAUNCH_SKIP) -c 1 \
			-o $(CUDA_COMPUTE_CONTAINER_DIR)/policy-logits -f \
			/workspace/$(CUDA_BIN_DIR)/cudabench -model-dir=$(CUDA_MODEL_DIR) \
			-report=/tmp/wordleml-cudabench-compute.json \
			-warmup=$(CUDA_PROFILE_WARMUP) -iterations=$(CUDA_NCU_PROFILE_ITERATIONS) && \
		ncu -i $(CUDA_COMPUTE_CONTAINER_DIR)/policy-logits.ncu-rep --page details --csv \
			> $(CUDA_COMPUTE_CONTAINER_DIR)/policy-logits-summary.csv && \
		ncu -i $(CUDA_COMPUTE_CONTAINER_DIR)/policy-logits.ncu-rep --page source --print-source cuda,sass \
			> $(CUDA_COMPUTE_CONTAINER_DIR)/policy-logits-source.txt; \
		status=$$?; \
		find $(CUDA_COMPUTE_CONTAINER_DIR) -maxdepth 1 -type f \
			-exec chown "$$WORDLEML_HOST_UID:$$WORDLEML_HOST_GID" {} +; \
		exit $$status'

.env:
	cp .env.example .env
