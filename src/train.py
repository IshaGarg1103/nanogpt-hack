"""Train nanoGPT-style models on fixed-token mixed-data shards."""

from __future__ import annotations

import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from optimizers import configure_optimizer


ROOT = Path(__file__).resolve().parents[1]

# I/O
out_dir = "out/exp1_30m_baseline"
dataset = "smollm_mix_100m"
data_dir = ""
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True
wandb_log = True
wandb_project = "nanogpt-hack"
wandb_run_name = "exp1_30m_baseline"
init_from = "scratch"

# data
gradient_accumulation_steps = 8
batch_size = 16
block_size = 512
max_tokens = 50_000_000

# model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
use_rope = False

# optimizer
optimizer_name = "adamw"
learning_rate = 6e-4
max_iters = -1
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
muon_momentum = 0.95
muon_ns_steps = 5
muon_nesterov = True

# lr schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = -1
min_lr = 6e-5

# system
device = "cuda"
dtype = "bfloat16"
compile = False

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(Path(__file__).with_name("configurator.py")).read())
config = {k: globals()[k] for k in config_keys}

if not data_dir:
    data_dir = str(ROOT / "data" / dataset)
else:
    data_dir = str(Path(data_dir))

out_path = ROOT / out_dir
out_path.mkdir(parents=True, exist_ok=True)

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
if max_iters < 0:
    max_iters = math.ceil(max_tokens / tokens_per_iter)
if lr_decay_iters < 0:
    lr_decay_iters = max_iters

print(f"data_dir: {data_dir}")
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"max_iters: {max_iters:,}")
print(f"max train tokens: {max_iters * tokens_per_iter:,}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50257,
    dropout=dropout,
    use_rope=use_rope,
)
iter_num = 0
running_tokens = 0
best_val_loss = float("inf")

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = GPT(GPTConfig(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_path}")
    ckpt_path = out_path / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    model_args["use_rope"] = checkpoint_model_args.get("use_rope", False)
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    running_tokens = iter_num * tokens_per_iter
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)
raw_model = model

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = configure_optimizer(
    model,
    optimizer_name,
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
    muon_momentum,
    muon_ns_steps,
    muon_nesterov,
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

if compile:
    print("compiling model")
    model = torch.compile(model)
    raw_model = model

if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int) -> float:
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


X, Y = get_batch("train")
t0 = time.time()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, tokens {running_tokens:,}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "tokens": running_tokens,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "best_val_loss": min(best_val_loss, losses["val"]),
                }
            )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if always_save_checkpoint:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_path}")
                torch.save(checkpoint, out_path / "ckpt.pt")

    if iter_num >= max_iters:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    running_tokens = iter_num * tokens_per_iter

    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = tokens_per_iter * log_interval / dt if iter_num > 0 else 0.0
        print(
            f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, "
            f"lr {lr:.2e}, tokens/sec {tokens_per_sec:,.0f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "tokens": running_tokens,
                    "train/iter_loss": loss.item() * gradient_accumulation_steps,
                    "lr": lr,
                    "perf/tokens_per_sec": tokens_per_sec,
                }
            )

print(f"done. best val loss: {best_val_loss:.4f}")
if wandb_log:
    wandb.finish()
