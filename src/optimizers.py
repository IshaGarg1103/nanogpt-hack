"""Optimizer choices for the nanoGPT hack experiments."""

from __future__ import annotations

import inspect
import math

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(
    g: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the orthogonalized Muon update for a 2D gradient matrix."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(0) > g.size(1):
        x = x.T

    x = x / (x.norm() + eps)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if g.size(0) > g.size(1):
        x = x.T
    return x.to(dtype=g.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimizer: Muon for hidden 2D matrices, AdamW for the rest."""

    def __init__(
        self,
        param_groups: list[dict],
        lr: float,
        betas: tuple[float, float],
        muon_momentum: float,
        muon_ns_steps: int,
        muon_nesterov: bool,
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
            muon_nesterov=muon_nesterov,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["optimizer"] == "muon":
                self._muon_step(group)
            elif group["optimizer"] == "adamw":
                self._adamw_step(group)
            else:
                raise ValueError(f"Unknown optimizer group: {group['optimizer']}")

        return loss

    @torch.no_grad()
    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["muon_momentum"]
        ns_steps = group["muon_ns_steps"]
        nesterov = group["muon_nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim != 2:
                raise ValueError("Muon group expects only 2D parameters")

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if nesterov else buf
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            update *= max(1.0, math.sqrt(p.size(0) / p.size(1)))
            p.add_(update, alpha=-lr)

    @torch.no_grad()
    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["adamw_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            p.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


def configure_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
    muon_momentum: float,
    muon_ns_steps: int,
    muon_nesterov: bool,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    if optimizer_name != "muon":
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []

    for name, param in param_dict.items():
        is_embedding = name.endswith("wte.weight") or name.endswith("wpe.weight") or name.endswith("lm_head.weight")
        if param.dim() == 2 and not is_embedding:
            muon_params.append(param)
        elif param.dim() >= 2:
            adamw_decay_params.append(param)
        else:
            adamw_nodecay_params.append(param)

    print(
        f"num Muon parameter tensors: {len(muon_params)}, "
        f"with {sum(p.numel() for p in muon_params):,} parameters"
    )
    print(
        f"num AdamW decayed parameter tensors: {len(adamw_decay_params)}, "
        f"with {sum(p.numel() for p in adamw_decay_params):,} parameters"
    )
    print(
        f"num AdamW non-decayed parameter tensors: {len(adamw_nodecay_params)}, "
        f"with {sum(p.numel() for p in adamw_nodecay_params):,} parameters"
    )
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    print(f"using Muon/AdamW hybrid; fused AdamW path unavailable inside hybrid: {fused_available and device_type == 'cuda'}")

    return MuonAdamW(
        [
            {
                "params": muon_params,
                "optimizer": "muon",
                "weight_decay": 0.0,
            },
            {
                "params": adamw_decay_params,
                "optimizer": "adamw",
                "weight_decay": weight_decay,
            },
            {
                "params": adamw_nodecay_params,
                "optimizer": "adamw",
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=betas,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        muon_nesterov=muon_nesterov,
    )
