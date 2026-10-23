"""
Discrete AWAC training step for SecureCourier.

Implements one critic update and one actor update per call.

Critic target (1-step TD, per EXPERIMENT_PLAN Section 6.3):
    y = r + gamma * E_{a'~pi(.|s')} Q_target(s', a')

Critic loss: MSE(Q(s,a) - y)

Actor:
    A_hat(s,a) = Q(s,a) - V(s)   where V(s) = sum_a pi(a|s)*Q(s,a)
    w(s,a) = clip(exp(A_hat / lambda), 0, w_max)
    L_pi = -E[w(s,a) * log pi(a|s)]
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def awac_step(
    actor,
    critic,
    target_critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.97,
    awac_lambda: float = 0.3,
    weight_clip: float = 20.0,
    target_tau: float = 0.005,
) -> Dict[str, float]:
    """
    One AWAC gradient step.

    Batch keys:
        obs_tokens       [B, obs_len]
        act_tokens       [B, N, act_len]   (current admissible actions)
        action_idx       [B]               (which action was taken, index into N)
        reward           [B]
        next_obs_tokens  [B, obs_len]
        next_act_tokens  [B, M, act_len]   (next state admissible actions)
        done             [B]

    Returns dict of scalar metrics.
    """
    obs_t  = batch["obs_tokens"]
    act_t  = batch["act_tokens"]
    a_idx  = batch["action_idx"]           # [B]
    rew    = batch["reward"]               # [B]
    nobs_t = batch["next_obs_tokens"]
    nact_t = batch["next_act_tokens"]
    done   = batch["done"]                 # [B]

    B = obs_t.size(0)

    # ---- Critic update ----
    with torch.no_grad():
        # Compute target Q values at next states
        q_next = target_critic(nobs_t, nact_t)          # [B, M]
        log_pi_next = actor(nobs_t, nact_t)             # [B, M]
        pi_next = log_pi_next.exp()                     # [B, M]
        v_next  = (pi_next * q_next).sum(dim=-1)        # [B]
        y = rew + gamma * v_next * (1.0 - done)         # [B]

    q_all = critic(obs_t, act_t)                        # [B, N]
    q_sa  = q_all.gather(1, a_idx.unsqueeze(1)).squeeze(1)  # [B]

    critic_loss = F.mse_loss(q_sa, y)

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
    critic_opt.step()

    # ---- Actor update ----
    q_all_actor = critic(obs_t, act_t).detach()         # [B, N]
    log_pi = actor(obs_t, act_t)                        # [B, N]
    pi     = log_pi.exp()

    v_hat  = (pi * q_all_actor).sum(dim=-1, keepdim=True)  # [B, 1]
    a_hat  = q_all_actor - v_hat                             # [B, N]

    w = torch.clamp(torch.exp(a_hat / awac_lambda), 0.0, weight_clip)  # [B, N]

    # Weighted NLL: -E[w * log pi(a|s)]
    log_pi_sa = log_pi.gather(1, a_idx.unsqueeze(1)).squeeze(1)  # [B]
    w_sa      = w.gather(1, a_idx.unsqueeze(1)).squeeze(1)        # [B]

    actor_loss = -(w_sa * log_pi_sa).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
    actor_opt.step()

    # ---- Target critic Polyak update ----
    target_critic.soft_update(critic, tau=target_tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss":  actor_loss.item(),
        "q_mean":      q_sa.mean().item(),
        "q_target_mean": y.mean().item(),
        "w_mean":      w_sa.mean().item(),
        "v_mean":      v_hat.mean().item(),
    }
