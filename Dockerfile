# syntax=docker/dockerfile:1

# CUDA 12.9 with cuDNN 9. CTranslate2 4.5 and later link against cuDNN 9, so an
# older cuDNN 8 image loads the model and then fails at inference time.
ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# --------------------------------------------------------------------------
# Builder: resolve dependencies and install the application into a virtualenv.
# --------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

WORKDIR /app

# Dependencies first. This layer is rebuilt only when the lockfile changes, so
# editing a source file no longer re-downloads CTranslate2 and onnxruntime.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# The application itself.
COPY src/ src/
COPY README.md LICENSE ./
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --------------------------------------------------------------------------
# Runtime: the CUDA runtime, FFmpeg, and the finished virtualenv.
# --------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime

LABEL org.opencontainers.image.title="aisrt" \
      org.opencontainers.image.description="Hardware-aware pipeline for broadcast-quality subtitles" \
      org.opencontainers.image.source="https://github.com/arvarik/aisrt" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    XDG_CONFIG_HOME=/config

# python3 for the interpreter, ffmpeg for decoding, tini for signal forwarding
# and zombie reaping at PID 1.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 ffmpeg tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# An unprivileged user. Running as root would write root-owned .srt files onto
# a media share that the owning user then cannot delete.
RUN groupadd --gid 1000 aisrt && \
    useradd --uid 1000 --gid 1000 --create-home --home-dir /home/aisrt aisrt && \
    mkdir -p /config/aisrt /media && \
    chown -R 1000:1000 /config /home/aisrt

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv

USER 1000:1000
VOLUME ["/config"]

# The daemon can run for days, so a health check must prove the binary still
# loads. A wedged pipeline needs the container's own logs to diagnose.
HEALTHCHECK --interval=5m --timeout=30s --start-period=10m --retries=3 \
    CMD ["/app/.venv/bin/aisrt", "--version"]

STOPSIGNAL SIGTERM

# tini forwards SIGTERM, which the application handles by draining the queues
# and closing the state database before it exits.
ENTRYPOINT ["/usr/bin/tini", "--", "aisrt"]
CMD ["--help"]
