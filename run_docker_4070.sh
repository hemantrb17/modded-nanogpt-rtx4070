#!/bin/bash
set -e

echo "=== Modded-NanoGPT Docker Runner (RTX 4070 Laptop) ==="

# 1. Build image if not present
if [[ "$(docker images -q modded-nanogpt 2> /dev/null)" == "" ]]; then
    echo "Building Docker image 'modded-nanogpt' using repository Dockerfile..."
    docker build -t modded-nanogpt .
else
    echo "Docker image 'modded-nanogpt' is ready."
fi

# 2. Download at least 4 shards (~800MB) if not present
if [ ! -f "data/fineweb10B/fineweb_train_000004.bin" ]; then
    echo "Ensuring at least 4 FineWeb training shards are available..."
    docker run --gpus all --ipc=host --rm -v "$(pwd):/modded-nanogpt" -w /modded-nanogpt modded-nanogpt python data/cached_fineweb10B.py 4
fi

# 3. Run training
echo "Launching train_gpt_4070.py inside Docker container..."
docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd):/modded-nanogpt" \
  -w /modded-nanogpt \
  modded-nanogpt \
  torchrun --standalone --nproc_per_node=1 train_gpt_4070.py
