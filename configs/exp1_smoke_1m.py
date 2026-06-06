# Tiny local smoke test for the mixed 1M-token dataset.

out_dir = "out/exp1_smoke_1m"
dataset = "smollm_mix_1m"

eval_interval = 10
eval_iters = 5
log_interval = 1
always_save_checkpoint = False
wandb_log = False

gradient_accumulation_steps = 1
batch_size = 4
block_size = 64
max_tokens = 20_000

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

device = "cpu"
dtype = "float32"
compile = False
