# Experiment 3B: RoPE + Muon/AdamW hybrid optimizer.

out_dir = "out/exp3b_30m_rope_muon_100m"
dataset = "smollm_mix_100m"
init_from = "scratch"

eval_interval = 50
eval_iters = 100
log_interval = 10
always_save_checkpoint = True
wandb_log = True
wandb_project = "nanogpt-hack"
wandb_run_name = "exp3b_30m_rope_muon_100m"

gradient_accumulation_steps = 8
batch_size = 16
block_size = 512
max_tokens = 100_000_000

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
use_rope = True

optimizer_name = "muon"
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
muon_momentum = 0.95
muon_ns_steps = 5
muon_nesterov = True

decay_lr = True
warmup_iters = 100
min_lr = 6e-5

device = "cuda"
dtype = "bfloat16"
compile = False
