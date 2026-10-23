"""Clean legal-action policy centering and PDR advantage projection."""

from __future__ import annotations

import torch


def _validated(q: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape != probs.shape:
        raise ValueError("values and policy_probs shapes differ")
    if mask is None:
        mask = torch.ones_like(probs, dtype=torch.bool)
    else:
        if mask.shape != probs.shape:
            raise ValueError("action_mask shape differs")
        mask = mask.to(dtype=torch.bool, device=probs.device)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("every row requires a legal action")
    if torch.any(probs < 0) or not torch.all(torch.isfinite(probs)):
        raise ValueError("policy probabilities must be finite and nonnegative")
    illegal_mass = torch.where(mask, torch.zeros_like(probs), probs).sum(dim=-1)
    if torch.any(illegal_mass != 0):
        raise ValueError("illegal actions must have exactly zero probability")
    legal_sum = torch.where(mask, probs, torch.zeros_like(probs)).sum(dim=-1)
    tol = 1e-12 if probs.dtype == torch.float64 else 1e-6
    if not torch.allclose(legal_sum, torch.ones_like(legal_sum), atol=tol, rtol=0):
        raise ValueError("legal policy probability must sum to one")
    return mask, probs


def policy_expectation(q: torch.Tensor, policy_probs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
    mask, probs = _validated(q, policy_probs, action_mask)
    safe_q = torch.where(mask, q, torch.zeros_like(q))
    return (probs * safe_q).sum(dim=-1, keepdim=True)


def center_under_policy(values: torch.Tensor, policy_probs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
    mask, _ = _validated(values, policy_probs, action_mask)
    centered = values - policy_expectation(values, policy_probs, mask)
    return torch.where(mask, centered, torch.zeros_like(centered))


def pdr_advantage_correction(delta_q: torch.Tensor, policy_probs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
    return center_under_policy(delta_q, policy_probs, action_mask)


def pdr_advantage(q_ref: torch.Tensor, delta_q: torch.Tensor, policy_probs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
    if q_ref.shape != delta_q.shape:
        raise ValueError("q_ref and delta_q shapes differ")
    return center_under_policy(q_ref + delta_q, policy_probs, action_mask)

