#!/usr/bin/env bash
set -e

MODEL_REPO="$(pwd)/triton/model_repository"
IMAGE="nvcr.io/nvidia/tritonserver:24.12-py3"

docker run --rm --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPO}:/models" \
  ${IMAGE} \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=psychological_state_analyzer
