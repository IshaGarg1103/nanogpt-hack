# Small Training Experiment Log

Running log for the `small-training` project. Add new entries at the top.

## Template

```text
## YYYY-MM-DD HH:MM — <run name>

### Goal
- ...

### Command
```bash
...
```

### Config
- Dataset:
- Model:
- Token budget:
- Device:
- dtype:

### Result
- Status:
- Initial train/val loss:
- Best val loss:
- Final train/val loss:
- Tokens/sec:
- Wall-clock:
- Artifacts:

### Learnings
- ...

### Next Step
- ...
```

---

## 2026-06-07 00:23 — Exp 3B RoPE Muon 100M

### Goal
- Test Muon as an optimizer ablation on top of the current winning RoPE architecture.
- Keep dataset, model size, token budget, RoPE, batch setup, eval cadence, and `compile=False` identical to Exp 2.

### Command
```bash
cd ~/small-training
python3 src/train.py configs/exp3b_30m_rope_muon_100m.py
```

### Config
- Dataset: `smollm_mix_100m`
- Model: `8` layers, `8` heads, `512` embedding dim, RoPE positional encoding
- Optimizer: Muon for hidden 2D matrix weights, AdamW for embeddings/norms/biases/other params
- Token budget: `100M` tokens
- Actual tokens logged: `99,614,720`
- Device: `cuda` on Hot Aisle MI300X
- dtype: `bfloat16`
- Eval interval: `50`
- Compile: `False`
- Checkpointing: enabled, best checkpoint saved to `out/exp3b_30m_rope_muon_100m/ckpt.pt`
- Tracking: WandB project `small-training`, run `exp3b_30m_rope_muon_100m`

### Result
- Status: pass
- Final iteration: `1520`
- Best validation loss: `4.47715`
- Final train/val loss: `4.37017` / `4.54312`
- Final iter loss: `4.25272`
- Final learning rate: `6e-05`
- Final observed tokens/sec: `384,609.54997`
- WandB run: `https://wandb.ai/ishagarg-research/small-training/runs/py26nqkj`

### Comparison To Exp 2
- Exp 2 RoPE + AdamW best val loss: `4.04767`
- Exp 3B RoPE + Muon/AdamW best val loss: `4.47715`
- Muon was worse by `0.42948` validation loss at the same token budget.
- Exp 2 final throughput was about `407k` tokens/sec; Exp 3B final throughput was about `385k` tokens/sec.

### Learnings
- Muon did not help this `30M` model at `100M` tokens with the initial hybrid implementation/hyperparameters.
- AdamW remains the best optimizer choice among our completed runs.
- This does not prove Muon is bad generally; it means this setup would need tuning before being competitive.
- RoPE is still the strongest architecture result so far, but the optimizer should remain AdamW for the next serious run.
- The current ranking is Exp 2 RoPE + AdamW, then Exp 1B learned positions + AdamW, then Exp 3B RoPE + Muon/AdamW.

### Next Step
- Pull `out/exp3b_30m_rope_muon_100m/ckpt.pt` and logs back to the Mac.
- Use Exp 2 RoPE + AdamW as the current best checkpoint for future experiments.

---

## 2026-06-07 00:07 — Exp 3A RoPE Compile 100M

### Goal
- Test whether `torch.compile=True` improves throughput for the current winning RoPE model.
- Keep the dataset, token budget, model size, RoPE setting, optimizer, and batch setup identical to Exp 2.

### Command
```bash
cd ~/small-training
python3 src/train.py configs/exp3a_30m_rope_compile_100m.py
```

### Config
- Dataset: `smollm_mix_100m`
- Model: `8` layers, `8` heads, `512` embedding dim, RoPE positional encoding
- Token budget: `100M` tokens
- Device: `cuda` on Hot Aisle MI300X
- dtype: `bfloat16`
- Optimizer: AdamW
- Compile: `True`
- Tracking: WandB project `small-training`, run `exp3a_30m_rope_compile_100m`

### Result
- Status: failed/blocked
- Failure point: first training backward pass after step 0 eval/checkpoint
- Error: `torch.ops.aten._scaled_dot_product_flash_attention_backward.default` stride assertion failure
- WandB run: `https://wandb.ai/ishagarg-research/small-training/runs/qa07o3ls`

### Learnings
- The uncompiled RoPE model is healthy, as shown by Exp 2.
- `torch.compile=True` hit a ROCm/TorchInductor compatibility issue in flash-attention backward.
- This is not a dataset or command issue; it is the compiled GPU execution path.
- Debugging compile further would likely require changing attention behavior, so it is not a clean Exp 3A anymore.

### Next Step
- Move to Exp 3B as a cleaner optimizer ablation: RoPE + Muon/AdamW hybrid with `compile=False`.

---

## 2026-06-06 23:52 — Exp 2 30M RoPE 100M

### Goal
- Test RoPE as the only architectural change against the 100M-token Exp 1 baseline.
- Disable/bypass learned absolute positional embeddings and apply RoPE to `q` and `k` inside attention.

### Command
```bash
cd ~/small-training
python3 src/train.py configs/exp2_30m_rope_100m.py
```

### Config
- Dataset: `smollm_mix_100m`
- Model: `8` layers, `8` heads, `512` embedding dim, RoPE positional encoding
- Token budget: `100M` tokens
- Actual tokens logged: `99,614,720`
- Device: `cuda` on Hot Aisle MI300X
- dtype: `bfloat16`
- Eval interval: `50`
- Checkpointing: enabled, best checkpoint saved to `out/exp2_30m_rope_100m/ckpt.pt`
- Tracking: WandB project `small-training`, run `exp2_30m_rope_100m`

### Result
- Status: pass
- Final iteration: `1520`
- Best validation loss: `4.04767`
- Final train/val loss: `3.94298` / `4.09622`
- Final iter loss: `3.84098`
- Final learning rate: `6e-05`
- Final observed tokens/sec: `407,401.01715`
- WandB run: `https://wandb.ai/ishagarg-research/small-training/runs/2svwntsv`

### Comparison To Exp 1B
- Exp 1B learned absolute positional embeddings best val loss: `4.29455`
- Exp 2 RoPE best val loss: `4.04767`
- RoPE improved best validation loss by `0.24688` at the same dataset, model size, optimizer, batch setup, and token budget.
- Exp 1B final throughput was about `480k` tokens/sec; Exp 2 final throughput was about `407k` tokens/sec.

### Learnings
- RoPE was a clear quality win in this controlled 100M-token ablation.
- The improvement is meaningful because every major training setting was held constant and only the positional encoding changed.
- RoPE added compute overhead from rotating `q` and `k`, so it traded some throughput for better validation loss.
- The best validation loss happened before the final eval, so the checkpoint matters more than the final printed validation loss.
- For the current experiment ladder, RoPE is the best architecture so far and should be the starting point for Exp 3 unless a stronger ablation beats it.

### Next Step
- Pull `out/exp2_30m_rope_100m/ckpt.pt` and logs back to the Mac.
- Plan Exp 3 around throughput/optimization using the RoPE model as the current winner.

---

## 2026-06-06 23:28 — Exp 1B 30M Baseline Continue To 100M

### Goal
- Continue the Exp 1 30M baseline because the 50M-token run was still improving.
- Add denser validation logging with `eval_interval=50` so WandB shows a useful loss curve instead of a nearly straight line.

### Command
```bash
cd ~/small-training
python3 src/train.py configs/exp1_30m_baseline_continue_100m.py
```

### Config
- Dataset: `smollm_mix_100m`
- Model: `8` layers, `8` heads, `512` embedding dim, learned absolute positional embeddings
- Resume: `init_from="resume"` from `out/exp1_30m_baseline/ckpt.pt`
- Token budget: `100M` total tokens
- Actual tokens logged: `99,614,720`
- Device: `cuda` on Hot Aisle MI300X
- dtype: `bfloat16`
- Eval interval: `50`
- Checkpointing: enabled, best checkpoint saved to `out/exp1_30m_baseline/ckpt.pt`
- Tracking: WandB project `small-training`, run `exp1_30m_baseline_continue_100m`

### Result
- Status: pass
- Final iteration: `1520`
- Starting continuation val loss: `5.1889` at step `500`
- Best validation loss: `4.29455`
- Final train/val loss: `4.19933` / `4.29455`
- Final iter loss: `3.748`
- Final learning rate: `6e-05`
- Final observed tokens/sec: `480,592.58199`
- WandB run: `https://wandb.ai/ishagarg-research/small-training/runs/8yhqne9a`

### Learnings
- Extending from 50M to ~100M tokens improved validation loss from `5.21998` to `4.29455`, so the original 50M run was undertrained.
- The validation curve still trended downward through the extended run, though improvements slowed near the end.
- `eval_interval=50` gave a much more informative WandB curve than the original sparse evaluation.
- The 100M-token dataset is small enough that this is roughly one pass over the training shard; larger datasets or longer token budgets are needed for more serious pretraining.
- The baseline is now a stronger control for RoPE: Exp 2 should compare at the same `100M` token budget if budget allows.

### Next Step
- Finish copying `out/exp1_30m_baseline/ckpt.pt` back to the Mac.
- Decide whether Exp 2 RoPE should run at `50M` first for a cheap signal or directly at `100M` for a fair comparison against this stronger baseline.

---

## 2026-06-06 22:57 — Exp 1 30M Baseline

### Goal
- Establish the first serious baseline on the fixed `smollm_mix_100m` dataset before testing RoPE or throughput optimizations.

### Command
```bash
cd ~/small-training
python3 src/train.py
```

### Config
- Dataset: `smollm_mix_100m`
- Model: `8` layers, `8` heads, `512` embedding dim, learned absolute positional embeddings
- Token budget: about `50M` training tokens
- Actual tokens logged: `49,987,286`
- Device: `cuda` on Hot Aisle MI300X
- dtype: `bfloat16`
- Optimizer: AdamW with fused optimizer enabled
- Checkpointing: enabled, best checkpoint saved to `out/exp1_30m_baseline/ckpt.pt`
- Tracking: WandB project `small-training`, run `exp1_30m_baseline`

### Result
- Status: pass
- Final iteration: `760`
- Initial validation loss: about `10.8`
- Best validation loss: `5.21998`
- Final train/val loss: `5.07147` / `5.21998`
- Final iter loss: `4.63632`
- Final learning rate: `6e-05`
- Final observed tokens/sec: `478,610.71412`
- WandB run: `https://wandb.ai/ishagarg-research/small-training/runs/k50v7ayd`

### Learnings
- The `30M` baseline trains successfully on the mixed modern dataset and reaches a much lower validation loss than the smoke runs, confirming the full `smollm_mix_100m` shard is usable for real experiments.
- MI300X throughput is strong for this model size, around `480k` tokens/sec near the end of the run.
- The loss curve kept improving through the planned token budget, so this run did not obviously saturate by `50M` tokens.
- The baseline is now good enough to serve as the control for Exp 2 RoPE; RoPE should use the same dataset, token budget, optimizer, eval schedule, and model size.
- WandB logging worked and is useful for comparing future ablations by validation loss vs tokens.

### Next Step
- Copy `out/exp1_30m_baseline/ckpt.pt` and logs back to the Mac with `rsync`.
- Start Exp 2 by adding RoPE as the only architectural change.

---

## 2026-06-06 22:42 — GPU Smoke 10k

### Goal
- Verify the Hot Aisle MI300X VM can run `small-training` end to end with ROCm PyTorch, CUDA device naming, bf16 autocast, fused AdamW, mixed dataset loading, forward/backward, optimizer, and eval.

### Command
```bash
cd ~/small-training
python3 src/train.py configs/exp1_gpu_smoke_10k.py
```

### Config
- Dataset: `smollm_mix_10k`
- Model: `2` layers, `2` heads, `128` embedding dim, about `6.83M` parameters
- Token budget: `10,240` actual train tokens
- Device: `cuda`
- dtype: `bfloat16`
- Batch/context: `batch_size=8`, `block_size=64`, `gradient_accumulation_steps=1`

### Result
- Status: pass
- GPU check: `torch.cuda.is_available() == True`
- GPU name: `AMD Instinct MI300X VF`
- PyTorch: `2.9.1+rocm6.4`
- Optimizer: fused AdamW enabled
- Initial train/val loss: `10.8201` / `10.8413`
- Final train/val loss: `9.8868` / `9.7639`
- Best val loss: `9.7639`
- Peak observed tokens/sec: about `139k`
- Artifacts: no checkpoint saved (`always_save_checkpoint=False`)

### Learnings
- ROCm exposes the MI300X through PyTorch's normal `cuda` API, so `device="cuda"` is correct.
- The synced `small-training` project and `smollm_mix_10k` data load correctly on the VM.
- bf16 smoke training is stable for the tiny config.

### Next Step
- Run a GPU smoke test against `smollm_mix_100m` with a small token budget before launching the serious Exp 1 baseline.

---

## 2026-06-06 — Dataset Build Complete

### Goal
- Build a modern fixed-token mixed dataset for Exp 1 baseline, Exp 2 RoPE, and Exp 3 throughput optimization.

### Dataset
- Name: `smollm_mix_100m`
- Location: `small-training/data/smollm_mix_100m`
- Tokenizer: GPT-2 BPE via `tiktoken`
- Total tokens: `100,000,000`
- Train tokens: `99,000,000`
- Val tokens: `1,000,000`

### Mix
- `70%` FineWeb-Edu-Dedup: `69,300,000` train, `700,000` val
- `20%` Cosmopedia v2: `19,800,000` train, `200,000` val
- `10%` Python code: `9,900,000` train, `100,000` val

### Result
- Status: complete
- Final files: `train.bin`, `val.bin`, `meta.pkl`, `dataset_manifest.json`
- Source part files were staged under `parts/` and merged into final `train.bin` / `val.bin`.

### Learnings
- Long streaming dataset builds should be staged by source so completed work is reusable after network timeouts.
- `python-code` streaming was the most fragile source; staging prevented losing FineWeb/Cosmopedia progress.

### Next Step
- Run GPU smoke test with `configs/exp1_gpu_smoke_10k.py`.
