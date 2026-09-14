# Modded-NanoGPT on RTX 4070 Laptop GPU (8GB VRAM)

This directory documents the adaptation and training results of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) on an **NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM, sm_89 Ada Lovelace)**.

---

## Hardware & Environment

* **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (Mobile, ~120W TGP)
* **VRAM**: 8,188 MiB (~8.0 GB GDDR6)
* **Architecture**: Ada Lovelace (`sm_89`)
* **Host OS**: Windows 11 with WSL2 / Docker Desktop
* **Container**: `modded-nanogpt:latest` built from official `Dockerfile` (Ubuntu 24.04, CUDA 12.6.2, CUDNN 9, Python 3.12.7, PyTorch 2.15.0.dev nightly with cu126)

---

## Key Adaptations for 8GB VRAM & Ada Lovelace

1. **Memory Tuning**:
   - Official records allocate **>30 GB of VRAM per GPU**.
   - Configured micro-batch size **`mbs = 4`** (4 sequences of 1024 tokens = 4,096 tokens per microbatch).
   - Set **`grad_accum_steps = 16`** to maintain an effective batch size of **65,536 tokens per optimizer step**.
   - Peak VRAM allocated: **3,840 MB (~3.84 GB)**, operating safely within 8 GB with over 4 GB of free headroom.
2. **Attention Compatibility (`sm_89`)**:
   - Replaced Hopper-exclusive FlashAttention-3 (`kernels-community/flash-attn3`) with PyTorch native `F.scaled_dot_product_attention`, executing FlashAttention-2 / cuDNN kernels optimized for Ada Lovelace.
3. **Optimizers**:
   - Muon with pure PyTorch Newton-Schulz orthogonalization + fused AdamW.
4. **Single-GPU Execution**:
   - Launched with `torchrun --standalone --nproc_per_node=1`.

---

## Training Results

See [`training_log.txt`](./training_log.txt) for the full raw training output.

* **Initial Validation Loss**: `10.8258`
* **Step 125 Validation Loss**: `5.4178` (in ~9.5 minutes)
* **Average Step Time**: ~3.3 – 4.5 seconds per step
* **Token Throughput**: ~15,000 – 19,500 tokens / second
* **Peak VRAM**: 3,840 MB

---

## How to Reproduce

In PowerShell:
```powershell
.\run_docker_4070.ps1
```
Or in Bash / Linux:
```bash
./run_docker_4070.sh
```
