#!/usr/bin/env bash
# Runs training on a Linux GPU machine. Run it from the repository root:
#   bash training/run_on_gpu_host.sh --steps 1000 --remote s3:my-bucket/checkpoints/smoke
# Arguments are passed to training.smoke_train.
set -euo pipefail

for command in docker nvidia-smi; do
  if ! command -v "$command" > /dev/null; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if [ ! -f .env ]; then
  echo "No .env file found. Copy .env.example to .env if you use cloud storage." >&2
fi

mkdir -p checkpoints
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

docker compose --profile training build train
docker compose --profile training run --rm train python -m training.smoke_train "$@"
