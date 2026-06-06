# GPU smoke test for the smallest mixed dataset shard.

out_dir = "out/exp1_gpu_smoke_10k"
dataset = "smollm_mix_10k"

eval_interval = 10
eval_iters = 5
log_interval = 1
always_save_checkpoint = False
wandb_log = False

gradient_accumulation_steps = 1
batch_size = 8
block_size = 64
max_tokens = 10_000

n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.0
bias = False

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 5
min_lr = 6e-5

device = "cuda"
dtype = "bfloat16"
compile = False
