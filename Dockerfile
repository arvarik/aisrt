# syntax=docker/dockerfile:1

# Base image: NVIDIA CUDA 12.2.2 with cuDNN 8 on Ubuntu 22.04
# Required for optimal CTranslate2 backend performance on GPUs.
# Falls back gracefully to CPU if no GPU is passed to the container.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies:
# - Python 3.11 & venv
# - FFmpeg (required for audio extraction)
# - Tini (required for PID 1 signal handling and zombie process reaping)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ffmpeg \
        tini \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the application working directory
WORKDIR /app

# Create a virtual environment and ensure it's on the PATH
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade core Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project definition and source code
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the application into the virtual environment
RUN pip install --no-cache-dir .

# Create a directory for the SQLite state database
# This should be mapped to a local volume to prevent DB corruption over network shares
RUN mkdir -p /root/.config/aisrt

# Prevent C libraries from thrashing threads, which causes GIL lockups
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Use Tini to handle graceful shutdown (SIGTERM -> SIGINT translation)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command if none is provided in docker-compose
CMD ["aisrt", "--help"]
