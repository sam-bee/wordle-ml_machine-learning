# GoMLX development image with CUDA support for compute capability 12.0 only.
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

ARG DOCKERCOMPOSE_UID=1000
ARG DOCKERCOMPOSE_GID=1000
ARG GO_VERSION=1.26.5
ARG GO_SHA256=5c2c3b16caefa1d968a94c1daca04a7ca301a496d9b086e17ad77bb81393f053
ARG PJRT_INSTALLER_VERSION=v0.4.0
ARG PJRT_CUDA_VERSION=0.11.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/go.tar.gz "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
    && echo "${GO_SHA256}  /tmp/go.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/go.tar.gz -C /usr/local \
    && rm /tmp/go.tar.gz

ENV PATH=/usr/local/go/bin:${PATH}

RUN groupadd --gid "${DOCKERCOMPOSE_GID}" wordleml \
    && useradd --create-home --uid "${DOCKERCOMPOSE_UID}" --gid "${DOCKERCOMPOSE_GID}" \
        --shell /bin/bash wordleml

ENV HOME=/home/wordleml
ENV PATH=/home/wordleml/go/bin:${PATH}

USER wordleml
WORKDIR /workspace/wordleml

# Install the CUDA 13 PJRT plugin at image-build time. Runtime downloads are
# disabled below, which keeps smoke tests and eventual training reproducible.
RUN go run "github.com/gomlx/go-xla/cmd/pjrt_installer@${PJRT_INSTALLER_VERSION}" \
        -plugin cuda13 \
        -version "${PJRT_CUDA_VERSION}" \
        -path /home/wordleml/.local/lib/go-xla \
        -verbosity 1

COPY --chown=wordleml:wordleml wordleml/go.mod wordleml/go.sum* ./
RUN go mod download

ENV GOMLX_BACKEND=xla:cuda
ENV GOMLX_NO_AUTO_INSTALL=1
ENV PJRT_PLUGIN_LIBRARY_PATH=/home/wordleml/.local/lib/go-xla/nvidia

CMD ["go", "run", "./cmd/smoke"]
