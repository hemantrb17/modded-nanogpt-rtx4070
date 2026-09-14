"""
train_gpt_4070.py

Single-GPU adapted training script for NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM).
Descends from Keller Jordan's modded-nanogpt (Track 3 Optimization / Speedrun).

Key adaptations for RTX 4070 Laptop (8GB VRAM, sm_89 Ada Lovelace):
1. Single-GPU execution (world_size=1, no multi-node NCCL overhead).
2. PyTorch native F.scaled_dot_product_attention (SDPA) leveraging FlashAttention-2
   and cuDNN optimized for sm_89 (avoids Hopper-exclusive FlashAttention-3).
3. Pure PyTorch Newton-Schulz orthogonalization for the Muon optimizer (avoids Hopper TMA).
4. Micro-batch size mbs=16 with gradient accumulation so peak VRAM is strictly < 4.5 GB.
5. Softcapped cross-entropy and ReLU^2 MLP matching the speedrun architecture.
"""

import os
import sys
import uuid
import time
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Dataloader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, f"magic number mismatch in {file}"
    assert header[1] == 1, "unsupported dataset version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

import itertools

def ensure_dataset(data_dir: Path, min_train_shards: int = 4, rank: int = 0, print_fn=print):
    """
    Pre-flight check: Verifies validation and training shards exist.
    If missing, automatically downloads them via huggingface_hub before training begins.
    """
    if rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        # 1. Validation shard
        val_path = data_dir / "fineweb_val_000000.bin"
        if not val_path.exists():
            print_fn(f"[Dataset] Validation shard missing. Downloading fineweb_val_000000.bin...")
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename="fineweb_val_000000.bin",
                            repo_type="dataset", local_dir=str(data_dir))

        # 2. Minimum required training shards
        existing_train_shards = sorted(data_dir.glob("fineweb_train_*.bin"))
        if len(existing_train_shards) < min_train_shards:
            print_fn(f"[Dataset] Found {len(existing_train_shards)} train shard(s). Ensuring at least {min_train_shards} shards are present...")
            for i in range(1, min_train_shards + 1):
                fname = f"fineweb_train_{i:06d}.bin"
                fpath = data_dir / fname
                if not fpath.exists():
                    print_fn(f"[Dataset] Downloading missing shard: {fname}...")
                    hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                                    repo_type="dataset", local_dir=str(data_dir))
            print_fn(f"[Dataset] All {min_train_shards} required shards are ready.")

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len: int = 1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {filename_pattern}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = itertools.cycle(files) # Seamless multi-epoch cycling across available shards
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size : pos + (rank + 1) * local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)

# -----------------------------------------------------------------------------
# Architecture

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        # Scaled Dot Product Attention (FlashAttention-2 / cuDNN path on sm_89)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=0.12, is_causal=True
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.relu().square() # ReLU^2 activation
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int = 50304, num_layers: int = 12, model_dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        # Asymmetric softcap
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

# -----------------------------------------------------------------------------
# Muon Optimizer (Newton-Schulz Polar Express approximation)

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        assert isinstance(params, list) and len(params) >= 1
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

# -----------------------------------------------------------------------------
# Main Setup & Training Loop

def main():
    # PyTorch allocator optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize distributed process group (works for single GPU or multi-GPU)
    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, device_id=device if backend == "nccl" else None)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/run_{run_id}.txt"

    def print0(msg: str, console: bool = True):
        if rank == 0:
            if console:
                print(msg, flush=True)
            with open(logfile, "a") as f:
                f.write(f"{msg}\n")

    print0("=" * 80)
    print0(f"Modded-NanoGPT (RTX 4070 Laptop 8GB Optimized)")
    print0(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print0(f"Device: {torch.cuda.get_device_name(device)} | VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
    print0("=" * 80)

    # Memory & Batch configuration for 8GB VRAM
    # Micro-batch size (mbs): 4 sequences of 1024 tokens = 4,096 tokens per microbatch.
    # Total batch size: 64 * 1024 tokens = 65,536 tokens per optimizer step (accumulated across 16 micro-batches).
    seq_len = 1024
    mbs = 4
    tokens_per_microbatch = mbs * seq_len
    batch_size = 64 * 1024 # 65,536 tokens
    grad_accum_steps = batch_size // (mbs * seq_len * world_size)
    assert grad_accum_steps >= 1, "batch_size must be >= mbs * seq_len * world_size"

    print0(f"Config: mbs={mbs} ({tokens_per_microbatch} tokens) | batch_size={batch_size} tokens | grad_accum_steps={grad_accum_steps}")

    # Ensure minimum required dataset shards are downloaded
    ensure_dataset(Path("data/fineweb10B"), min_train_shards=4, rank=rank, print_fn=print0)

    # Prepare model
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()

    # Checkpoint configuration
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    latest_ckpt_path = checkpoint_dir / "checkpoint_latest.pt"
    best_ckpt_path = checkpoint_dir / "checkpoint_best.pt"

    start_step = 0
    best_val_loss = float("inf")
    training_time = 0.0

    # Initialize weights default
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)

    # Resume weights if checkpoint exists
    resumed = False
    if latest_ckpt_path.exists():
        try:
            print0(f"Found checkpoint at {latest_ckpt_path}. Resuming training...")
            ckpt = torch.load(latest_ckpt_path, map_location="cuda")
            model.load_state_dict(ckpt["model"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            training_time = ckpt.get("training_time", 0.0)
            resumed = True
            print0(f"Resumed from step {start_step} (previous val loss: {ckpt.get('val_loss', 'N/A')}, elapsed time: {training_time:.1f}s)")
        except Exception as e:
            print0(f"Warning: Failed to load checkpoint: {e}. Starting fresh.")

    model = torch.compile(model, dynamic=False)

    # Create optimizers
    optimizer1 = AdamW([
        dict(params=[model.embed.weight], lr=0.7),
        dict(params=[model.proj.weight], lr=0.004),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)

    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.025, weight_decay=0.05)
    optimizers = [optimizer1, optimizer2]

    # Restore optimizer states if resuming
    if resumed and "optimizer1" in ckpt and "optimizer2" in ckpt:
        try:
            optimizer1.load_state_dict(ckpt["optimizer1"])
            optimizer2.load_state_dict(ckpt["optimizer2"])
            print0("Restored optimizer states successfully.")
        except Exception as e:
            print0(f"Warning: Could not restore optimizer states: {e}")

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    train_steps = 3250
    def set_hparams(step: int, cooldown_frac: float = 0.7):
        progress = step / train_steps
        eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    # Data loaders
    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, seq_len=seq_len)

    # Validation: use 2M tokens (instead of 10M) for fast validation iterations on laptop
    val_tokens = 2 * 1024 * 1024
    print0(f"Validation tokens: {val_tokens}")
    val_loader = distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens, seq_len=seq_len)
    val_inputs, val_targets = next(val_loader)

    print0("Starting training...")
    t0 = time.perf_counter()
    last_val_step = start_step

    for step in range(start_step, train_steps + 1):
        # Validation
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or (step % val_step_freq == 0 and step > start_step) or (step == 0 and not resumed):
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / max(step - last_val_step, 1) if step > start_step else float("nan")
            last_val_step = step
            training_time += time_since_last_val

            model.eval()
            val_loss = 0.0
            num_val_batches = len(val_inputs) // mbs
            with torch.no_grad():
                for i in range(num_val_batches):
                    chunk_in = val_inputs[i * mbs : (i + 1) * mbs]
                    chunk_tgt = val_targets[i * mbs : (i + 1) * mbs]
                    val_loss += model(chunk_in, chunk_tgt).item()

            val_loss /= (num_val_batches * tokens_per_microbatch)
            peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            print0(f"step:{step:4d}/{train_steps} | val_loss:{val_loss:.4f} | train_time:{training_time:.1f}s | step_avg:{1000*step_avg:.1f}ms | peak_vram:{peak_vram_mb}MB")

            # Checkpoint management
            if step > 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ckpt_data = {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer1": optimizer1.state_dict(),
                    "optimizer2": optimizer2.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": min(val_loss, best_val_loss),
                    "training_time": training_time,
                }
                # 1. Always save latest checkpoint for resume
                torch.save(ckpt_data, latest_ckpt_path)

                # 2. Save best checkpoint if new all-time low
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ckpt_data, best_ckpt_path)
                    print0(f"🏆 New best validation loss: {best_val_loss:.4f}! Saved to {best_ckpt_path}")

                # 3. Save milestone checkpoints every 1000 steps
                if step % 1000 == 0:
                    milestone_path = checkpoint_dir / f"checkpoint_step_{step:04d}.pt"
                    torch.save(ckpt_data, milestone_path)
                    print0(f"Saved milestone checkpoint to {milestone_path}")

            model.train()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # Training forward & backward across micro-batches
        inputs, targets = next(train_loader)
        num_microbatches = len(inputs) // mbs
        for i in range(num_microbatches):
            loss = model(inputs[i * mbs : (i + 1) * mbs], targets[i * mbs : (i + 1) * mbs])
            # Scale loss for gradient accumulation
            (loss / num_microbatches).backward()

        # Step optimizers
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0 or step < 5:
            approx_time = training_time + (time.perf_counter() - t0)
            peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            print0(f"step:{step+1:4d}/{train_steps} | train_time:{approx_time:.1f}s | step_avg:{1000*approx_time/(step+1):.1f}ms | peak_vram:{peak_vram_mb}MB")

    print0("=" * 80)
    print0("Training complete!")
    if step > 0:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        final_path = checkpoint_dir / "final_model.pt"
        torch.save({"step": step, "model": raw_model.state_dict(), "val_loss": val_loss}, final_path)
        print0(f"Final model weights saved to {final_path}")
    print0(f"Final peak memory allocated: {torch.cuda.max_memory_allocated() // (1024 * 1024)} MiB")
    print0("=" * 80)

if __name__ == "__main__":
    main()
