"""Mathematically exact legal-action masking for padded batches."""

from __future__ import annotations

import torch


def validate_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask shapes differ")
    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    if mask.ndim < 1 or not torch.all(mask.any(dim=-1)):
        raise ValueError("every batch row must contain at least one legal action")
    return mask


def masked_log_softmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = validate_action_mask(logits, action_mask)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    values = torch.log_softmax(masked_logits, dim=-1)
    return values.masked_fill(~mask, -torch.inf)


def masked_softmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = validate_action_mask(logits, action_mask)
    probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    return probabilities.masked_fill(~mask, 0.0)

