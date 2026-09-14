# PowerShell helper script to build and run modded-nanogpt in Docker
$ErrorActionPreference = "Stop"

Write-Host "=== Modded-NanoGPT Docker Runner (RTX 4070 Laptop) ===" -ForegroundColor Cyan

# 1. Build the official Dockerfile if image does not exist
$imageExists = docker images -q modded-nanogpt
if (-not $imageExists) {
    Write-Host "Building Docker image 'modded-nanogpt' using repository Dockerfile..." -ForegroundColor Yellow
    docker build -t modded-nanogpt .
} else {
    Write-Host "Docker image 'modded-nanogpt' is ready." -ForegroundColor Green
}

# 2. Download at least 4 shards (~800MB) if not present
if (-not (Test-Path "data/fineweb10B/fineweb_train_000004.bin")) {
    Write-Host "Ensuring at least 4 FineWeb training shards are available..." -ForegroundColor Yellow
    docker run --gpus all --ipc=host --rm -v "${PWD}:/modded-nanogpt" -w /modded-nanogpt modded-nanogpt python data/cached_fineweb10B.py 4
}

# 3. Launch training
Write-Host "Launching train_gpt_4070.py inside Docker container..." -ForegroundColor Green
docker run --gpus all --ipc=host --rm -it `
  -v "${PWD}:/modded-nanogpt" `
  -w /modded-nanogpt `
  modded-nanogpt `
  torchrun --standalone --nproc_per_node=1 train_gpt_4070.py
