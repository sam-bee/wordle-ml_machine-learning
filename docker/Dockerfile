# GoMLX development/runtime image with CUDA support for compute capability 12.0.
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

ARG DOCKERCOMPOSE_UID=1000
ARG DOCKERCOMPOSE_GID=1000
ARG GO_VERSION=1.26.5
ARG PJRT_INSTALLER_VERSION=v0.2.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
    | tar -xz -C /usr/local

ENV PATH=/usr/local/go/bin:${PATH}

RUN groupadd -g "${DOCKERCOMPOSE_GID}" squarl \
    && useradd -m -u "${DOCKERCOMPOSE_UID}" -g "${DOCKERCOMPOSE_GID}" -s /bin/bash squarl

ENV HOME=/home/squarl
ENV PATH=/home/squarl/go/bin:${PATH}

USER squarl
WORKDIR /workspace/squarl

# Production containers should carry their PJRT plugin instead of downloading it at startup.
RUN go run "github.com/gomlx/go-xla/cmd/pjrt_installer@${PJRT_INSTALLER_VERSION}" \
        -plugin cuda13 \
        -version latest \
        -path /home/squarl/.local/lib/go-xla

COPY --chown=squarl:squarl squarl/go.mod squarl/go.sum* ./
RUN go mod download

ENV GOMLX_BACKEND=xla:cuda
ENV GOMLX_NO_AUTO_INSTALL=1
# The supported GeForce GPUs use CUDA minor-version compatibility, not the
# forward-compatibility libcuda bundled in the CUDA image.
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

CMD ["go", "run", "./cmd/squarl"]
