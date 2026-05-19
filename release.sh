#!/usr/bin/env bash
# MeDS — build the Docker image and launch an interactive container.
#
# Usage:  bash release.sh
#
# Override any of CONTAINER_NAME / IMAGE_NAME / mount paths via env vars, e.g.:
#   IMAGE_NAME=my/meds:dev DATA_HOST=/mnt/datasets bash release.sh

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-meds}"
IMAGE_NAME="${IMAGE_NAME:-meds:local}"

# Host paths to mount into the container.
# DATA_HOST    — datasets (MVTec-AD, VisA, Real-IAD, noisy variants)
# OUTPUT_HOST  — destination for memory scores / checkpoints / metrics
# CODE_HOST    — this repo (mounted live so source edits don't need a rebuild)
DATA_HOST="${DATA_HOST:-$HOME/datasets}"
OUTPUT_HOST="${OUTPUT_HOST:-$HOME/outputs/meds}"
CODE_HOST="${CODE_HOST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Stop & remove any existing container with the same name
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null || true
    docker rm   "$CONTAINER_NAME" >/dev/null || true
fi

# Remove the old image (optional — comment out to keep build cache across runs)
if [ "$(docker images -q $IMAGE_NAME)" ]; then
    echo "Removing existing image: $IMAGE_NAME"
    docker rmi "$IMAGE_NAME" >/dev/null || true
fi

# Build
echo "Building image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$CODE_HOST"

# Run — GPU access, shared memory for DataLoader, IPC=host for nccl/shm,
# live-mounted source so iterating doesn't need a rebuild.
echo "Launching container: $CONTAINER_NAME"
mkdir -p "$OUTPUT_HOST"
docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    -v "$DATA_HOST":/data \
    -v "$OUTPUT_HOST":/outputs \
    -v "$CODE_HOST":/workspace/MeDS \
    -w /workspace/MeDS \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    /bin/bash
