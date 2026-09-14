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
5. **Multi-Epoch Support**:
   - Uses `itertools.cycle` to seamlessly cycle through training data shards for multi-epoch training.

---

## Training Results

See [`training_log.txt`](./training_log.txt) for the full raw training output across 1,520 steps (~100 Million tokens).

| Step | Validation Loss | Perplexity ($e^{\text{loss}}$) | Train Time | Peak VRAM |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **10.8258** | ~50,300 | 0.0s | 1,043 MB |
| **125** | **5.4162** | ~225.0 | 870.9s | 3,840 MB |
| **250** | **4.9136** | ~136.1 | 1,554.0s | 3,840 MB |
| **500** | **4.5153** | ~91.4 | 2,068.5s | 3,840 MB |
| **750** | **4.3691** | ~78.9 | 2,609.8s | 3,840 MB |
| **1000** | **4.2701** | ~71.5 | 3,140.7s | 3,840 MB |
| **1250** | **4.1935** | ~66.2 | 3,671.8s | 3,840 MB |
| **1500** | **4.1217** | ~61.6 | 4,205.1s | 3,840 MB |

* **Step Speed**: ~2.0 – 2.1 seconds per step (after single-container isolation).
* **Token Throughput**: ~31,000 tokens / second.
* **Peak VRAM**: 3,840 MB (consistent throughout the entire run).

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
