"""Pure AWAC actor-loss adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AWACDiagnostics:
    selected_advantage_mean: float
    selected_advantage_std: float
    weight_mean: float
    weight_std: float
    max_weight: float
    clipping_fraction: float
    effective_sample_size: float


def awac_actor_loss(
    log_probs: torch.Tensor,
    selected_action: torch.Tensor,
    advantage: torch.Tensor,
    temperature: float,
    weight_clip: float,
    *,
    stop_gradient_weight: bool = True,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, AWACDiagnostics]:
    if log_probs.shape != advantage.shape or log_probs.ndim != 2:
        raise ValueError("log_probs and advantage must be equal-shape rank-2 tensors")
    if selected_action.shape != (log_probs.shape[0],):
        raise ValueError("selected_action must contain one index per row")
    if temperature <= 0 or weight_clip <= 0:
        raise ValueError("temperature and weight_clip must be positive")
    chosen = selected_action.to(device=log_probs.device, dtype=torch.long)
    if action_mask is not None:
        if action_mask.shape != log_probs.shape:
            raise ValueError("action_mask shape differs")
        if not torch.all(action_mask.to(log_probs.device).gather(1, chosen[:, None])):
            raise ValueError("selected action is illegal")
    selected_logp = log_probs.gather(1, chosen[:, None]).squeeze(1)
    selected_adv = advantage.gather(1, chosen[:, None]).squeeze(1)
    weights_raw = torch.exp(selected_adv / float(temperature))
    weights = torch.clamp(weights_raw, 0.0, float(weight_clip))
    loss_weights = weights.detach() if stop_gradient_weight else weights
    loss = -(loss_weights * selected_logp).mean()
    with torch.no_grad():
        denom = float(torch.square(weights).sum())
        ess = float(weights.sum().square() / denom) if denom > 0 else 0.0
        diagnostics = AWACDiagnostics(
            selected_advantage_mean=float(selected_adv.mean()),
            selected_advantage_std=float(selected_adv.std(unbiased=False)),
            weight_mean=float(weights.mean()),
            weight_std=float(weights.std(unbiased=False)),
            max_weight=float(weights.max()),
            clipping_fraction=float((weights_raw >= weight_clip).to(torch.float64).mean()),
            effective_sample_size=ess,
        )
    return loss, diagnostics

