"""
Critic (Q-value) network for SecureCourier discrete AWAC.

Same architecture as the actor but outputs scalar Q values per action.
A target critic (Polyak-averaged copy) is maintained separately.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder, ActionEncoder


class Critic(nn.Module):
    """
    Q-network: Q(s, a) for each candidate action.

    forward(obs_tokens, act_tokens) → Q values [B, N_acts]
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
        self.q_head  = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        obs_tokens: torch.Tensor,   # [B, obs_len]
        act_tokens: torch.Tensor,   # [B, N_acts, act_len]
    ) -> torch.Tensor:
        """Returns Q values: [B, N_acts]."""
        obs_h = self.obs_enc(obs_tokens)
        act_h = self.act_enc(act_tokens)
        obs_e = obs_h.unsqueeze(1).expand(-1, act_h.size(1), -1)
        q = self.q_head(torch.cat([obs_e, act_h], dim=-1)).squeeze(-1)  # [B, N]
        return q

    def make_target(self) -> "Critic":
        """Return a deep copy of this critic (to be used as target network)."""
        tgt = copy.deepcopy(self)
        for p in tgt.parameters():
            p.requires_grad_(False)
        return tgt

    @torch.no_grad()
    def soft_update(self, online: "Critic", tau: float = 0.005) -> None:
        """Polyak update: target ← tau*online + (1-tau)*target."""
        for t_p, o_p in zip(self.parameters(), online.parameters()):
            t_p.data.mul_(1.0 - tau).add_(o_p.data * tau)
