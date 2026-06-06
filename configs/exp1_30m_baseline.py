# Experiment 1: learned-position baseline on the mixed 100M-token dataset.

out_dir = "out/exp1_30m_baseline"
dataset = "smollm_mix_100m"

eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True
wandb_log = True
wandb_project = "nanogpt-hack"
wandb_run_name = "exp1_30m_baseline"

gradient_accumulation_steps = 8
batch_size = 16
block_size = 512
max_tokens = 50_000_000

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
min_lr = 6e-5

device = "cuda"
dtype = "bfloat16"
compile = False
