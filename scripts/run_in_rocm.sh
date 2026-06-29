#!/usr/bin/env bash
# Run any command inside the talens ROCm container on the AMD Strix Halo
# iGPU (gfx1151). Reuses AMD's rocm/pytorch base (shared with AloePri) so
# the ROCm runtime is not duplicated — see Containerfile.
#
# The repo and HF cache are bind-mounted at their SAME host paths so
# absolute paths resolve identically inside and out. talens is made
# importable via PYTHONPATH=src (no in-container editable install, so the
# mounted tree stays clean).
#
# Usage:
#   scripts/run_in_rocm.sh python3 -m talens.cli \
#       --model Qwen/Qwen3-4B --corpus corpora/dev-24.txt --out results/pass1.json
#   scripts/run_in_rocm.sh python3 -c 'import torch; print(torch.cuda.is_available())'

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMAGE="${TALENS_IMAGE:-talens-rocm:latest}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[run_in_rocm] building $IMAGE (thin layer on cached rocm/pytorch base)..." >&2
    docker build -f "$REPO_DIR/Containerfile" -t "$IMAGE" "$REPO_DIR" >&2
fi

mkdir -p "$HF_CACHE"

exec docker run --rm \
    --device /dev/dri --device /dev/kfd \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --user "$(id -u):$(id -g)" \
    --shm-size 16G \
    -v "$REPO_DIR:$REPO_DIR" \
    -v "$HF_CACHE:$HF_CACHE" \
    -v "/tmp:/tmp" \
    -e HF_HOME="$HF_CACHE" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HOME="$HOME" \
    -e PYTHONPATH="$REPO_DIR/src" \
    -e ROCBLAS_USE_HIPBLASLT=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -w "$REPO_DIR" \
    "$IMAGE" "$@"
