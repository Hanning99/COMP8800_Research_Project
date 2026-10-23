"""
Actor (policy) network for SecureCourier discrete AWAC.

Architecture:
  obs_encoder  : TextEncoder(GRU)   → obs_hidden
  act_encoder  : ActionEncoder(MLP) → act_hidden  [per candidate]
  score        : Linear([obs_hidden, act_hidden] → 1)
  policy       : softmax over candidate scores

forward(obs_tokens, act_tokens) → log_probs over candidates
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder, ActionEncoder, Vocab


class Actor(nn.Module):
    """
    Categorical policy over a dynamic set of admissible actions.

    Parameters
    ----------
    vocab_size : int
    embed_dim  : int
    hidden_dim : int
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_enc = TextEncoder(vocab_size, embed_dim, hidden_dim)
        self.act_enc = ActionEncoder(vocab_size, embed_dim, hidden_dim)
        self.score   = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        obs_tokens: torch.Tensor,    # [B, obs_len]
        act_tokens: torch.Tensor,    # [B, N_acts, act_len]
    ) -> torch.Tensor:
        """Returns log-probabilities: [B, N_acts]."""
        obs_h = self.obs_enc(obs_tokens)                      # [B, H]
        act_h = self.act_enc(act_tokens)                      # [B, N, H]
        obs_e = obs_h.unsqueeze(1).expand(-1, act_h.size(1), -1)
        logits = self.score(torch.cat([obs_e, act_h], dim=-1)).squeeze(-1)  # [B, N]
        return F.log_softmax(logits, dim=-1)

    def get_action_dist(
        self,
        obs_tokens: torch.Tensor,
        act_tokens: torch.Tensor,
    ) -> torch.distributions.Categorical:
        log_probs = self.forward(obs_tokens, act_tokens)
        return torch.distributions.Categorical(logits=log_probs)

    @torch.no_grad()
    def action_probs_np(
        self,
        obs_tokens: torch.Tensor,
        act_tokens: torch.Tensor,
    ) -> "np.ndarray":
        """Return action probabilities as a numpy array (no grad)."""
        import numpy as np
        lp = self.forward(obs_tokens, act_tokens)
        return F.softmax(lp, dim=-1).squeeze(0).cpu().numpy()
