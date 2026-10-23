"""
RQ1 staleness measurement helpers.

Provides:
  - Critic readout at a navigated structural node (Q_hat for all admissible actions)
  - Per-instance DP policy builder (exact Q_true under a specific actor on a specific instance)
  - Staleness metric computation (E_stale, E_Q, E_A, harmful_stale, AWAC weight ratio)

Used by scripts/run_rq1.py.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from secure_courier.env import SecureCourierEnv
from secure_courier.generator import InstanceConfig
from secure_courier.oracle import dp_values
from secure_courier.state import EnvState
from secure_courier.task_graph import (
    DELIVERED,
    ORACLE_SHORTCUT_ACTIONS,
    admissible_actions as struct_admissible,
)
from rq1.exact_value_table import make_dp_policy_from_actor


# ---------------------------------------------------------------------------
# Target historical states (from EXPERIMENT_PLAN Section 10.2)
# Each entry: (label, distance_to_gate, hist_node, hist_action, post_node, oracle_prefix_len)
# oracle_prefix_len = number of ORACLE_SHORTCUT_ACTIONS to execute to reach hist_node
# ---------------------------------------------------------------------------

# From _build_dist_to_decision() in exact_value_table.py:
#   nodes[i] is after i oracle actions; hist_node = nodes[step_idx - 1]
# So oracle_prefix_len = step_idx - 1
TARGET_STATES = [
    # (label,  d,  hist_node,           hist_action,          post_node,        prefix_len)
    ("d12",    12, "locker_room",        "open_locker",        "locker_opened",  1),
    ("d8",      8, "code_known",         "take_access_badge",  "badge_taken",    5),
    ("jxn",     5, "junction",           "take_shortcut",      "sc1",            8),
    ("d4",      4, "sc1",                "go_sc2",             "sc2",            9),
    ("d2",      2, "sc3",                "go_sc4",             "sc4",            11),
    ("d1",      1, "sc4",                "go_sc5",             "sc5",            12),
]

# AWAC hyperparameters (from config)
AWAC_LAMBDA  = 0.3
WEIGHT_CLIP  = 20.0
DELTA_Q_MIN  = 0.02   # pre-declared threshold (matches G0 DP-vs-MC tolerance)


# ---------------------------------------------------------------------------
# Critic readout at a structural node
# ---------------------------------------------------------------------------

def get_qhat_at_node(
    critic,
    actor,
    vocab,
    env: SecureCourierEnv,
    oracle_prefix_actions: List[str],
    target_structural_action: str,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> Dict:
    """
    Navigate env to hist_node via oracle_prefix_actions, then query critic.

    Returns dict with:
      node            -- current structural node after navigation
      adm_structural  -- admissible structural actions at this node
      q_hat           -- {structural_action: Q_hat value}
      v_hat           -- expected Q under actor pi at this node
      a_hat_target    -- advantage of target_structural_action under critic + actor
      pi              -- {structural_action: actor probability}
    """
    env.reset()
    for a in oracle_prefix_actions:
        adm = env.admissible_structural()
        if a not in adm:
            raise ValueError(
                f"Oracle action '{a}' not admissible at node '{env.current_node()}'; "
                f"available: {adm}"
            )
        env.execute_structural(a)

    obs = env._obs()
    adm_disp = obs["admissible_actions"]
    adm_str  = obs["admissible_structural"]
    obs_text = obs["text"]

    obs_ids = torch.tensor(
        [vocab.encode(obs_text, obs_max_len)], dtype=torch.long, device=device
    )
    act_ids = torch.tensor(
        [[vocab.encode(a, act_max_len) for a in adm_disp]],
        dtype=torch.long, device=device,
    )

    critic.eval()
    actor.eval()
    with torch.no_grad():
        q_vals   = critic(obs_ids, act_ids)[0]          # [N]
        log_pis  = actor(obs_ids, act_ids)[0]            # [N] log-probs
        pis      = F.softmax(log_pis, dim=-1)            # [N] probs

    q_dict  = {s: float(q_vals[i])  for i, s in enumerate(adm_str)}
    pi_dict = {s: float(pis[i])     for i, s in enumerate(adm_str)}

    # V_hat = sum_a pi(a|s) * Q_hat(s,a)
    v_hat = sum(pi_dict[s] * q_dict[s] for s in adm_str)
    # A_hat(target_action) = Q_hat(s, target) - V_hat
    q_target = q_dict.get(target_structural_action, 0.0)
    a_hat    = q_target - v_hat

    return {
        "node":            env.current_node(),
        "adm_structural":  adm_str,
        "q_hat":           q_dict,
        "v_hat":           v_hat,
        "a_hat_target":    a_hat,
        "pi":              pi_dict,
        "q_hat_target":    q_target,
    }


# ---------------------------------------------------------------------------
# Q_true extraction from a DP run
# ---------------------------------------------------------------------------

def get_qtrue_at_node(
    V_table: Dict,
    Q_table: Dict,
    hist_node: str,
    hist_action: str,
    t_rem: int,
) -> Dict:
    """Extract Q_true, V_true, A_true at (hist_node, hist_action, t_rem) from DP tables."""
    q_true = Q_table.get((hist_node, hist_action, t_rem), 0.0)
    v_true = V_table.get((hist_node, t_rem), 0.0)
    a_true = q_true - v_true
    return {
        "q_true":  q_true,
        "v_true":  v_true,
        "a_true":  a_true,
    }


# ---------------------------------------------------------------------------
# Per-instance DP run
# ---------------------------------------------------------------------------

def run_dp_for_instance(
    actor,
    vocab,
    instance: InstanceConfig,
    device: torch.device,
    gamma: float = 0.97,
    max_steps: int = 40,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> Tuple[Dict, Dict]:
    """Run exact DP for (actor, instance). Returns (V_table, Q_table)."""
    policy_fn = make_dp_policy_from_actor(
        actor, vocab, instance, device,
        obs_max_len=obs_max_len, act_max_len=act_max_len,
    )
    return dp_values(policy_fn, gamma=gamma, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Junction-specific helpers (two actions: shortcut + safe)
# ---------------------------------------------------------------------------

def get_junction_qhat_both(
    critic,
    actor,
    vocab,
    env: SecureCourierEnv,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> Dict:
    """
    Navigate to junction and return Q_hat + A_hat for BOTH take_shortcut and
    take_safe_route, plus V_hat, pi, and rank.
    """
    jxn_prefix = list(ORACLE_SHORTCUT_ACTIONS[:8])  # 8 actions reach junction
    res = get_qhat_at_node(
        critic, actor, vocab, env,
        oracle_prefix_actions=jxn_prefix,
        target_structural_action="take_shortcut",
        device=device,
        obs_max_len=obs_max_len,
        act_max_len=act_max_len,
    )
    q_sc  = res["q_hat"].get("take_shortcut", 0.0)
    q_sf  = res["q_hat"].get("take_safe_route", 0.0)
    v_hat = res["v_hat"]
    a_sc  = q_sc - v_hat
    a_sf  = q_sf - v_hat
    rank_hat = "shortcut" if q_sc > q_sf else "safe_route"
    return {
        "q_hat_shortcut":    q_sc,
        "q_hat_safe_route":  q_sf,
        "v_hat":             v_hat,
        "a_hat_shortcut":    a_sc,
        "a_hat_safe_route":  a_sf,
        "rank_hat":          rank_hat,
        "pi":                res["pi"],
    }


def get_junction_qtrue_both(
    Q_table: Dict,
    V_table: Dict,
    t_rem: int = 32,   # junction reached after 8 steps → t_rem = 40-8 = 32
) -> Dict:
    """Extract Q_true, A_true for take_shortcut and take_safe_route at junction."""
    q_sc   = Q_table.get(("junction", "take_shortcut",   t_rem), 0.0)
    q_sf   = Q_table.get(("junction", "take_safe_route", t_rem), 0.0)
    v_true = V_table.get(("junction", t_rem), 0.0)
    a_sc   = q_sc - v_true
    a_sf   = q_sf - v_true
    rank_true = "shortcut" if q_sc > q_sf else "safe_route"
    return {
        "q_true_shortcut":    q_sc,
        "q_true_safe_route":  q_sf,
        "v_true":             v_true,
        "a_true_shortcut":    a_sc,
        "a_true_safe_route":  a_sf,
        "rank_true":          rank_true,
    }


# ---------------------------------------------------------------------------
# AWAC weight helpers
# ---------------------------------------------------------------------------

def awac_weight(advantage: float, lam: float = AWAC_LAMBDA,
                clip: float = WEIGHT_CLIP) -> float:
    return min(math.exp(advantage / lam), clip)


def weight_ratio(w1: float, w2: float) -> float:
    """max(w1/w2, w2/w1), handling zeros gracefully."""
    if w2 == 0 and w1 == 0:
        return 1.0
    if w2 == 0:
        return WEIGHT_CLIP
    if w1 == 0:
        return WEIGHT_CLIP
    return max(w1 / w2, w2 / w1)


# ---------------------------------------------------------------------------
# Harmful stale classification (EXPERIMENT_PLAN Section 10.6)
# ---------------------------------------------------------------------------

def is_harmful_stale(
    q_true_old: float,
    q_true_new: float,
    a_true_new: float,
    a_hat_new: float,
    rank_true: Optional[str],   # "shortcut" | "safe_route" | None
    rank_hat: Optional[str],
    delta_q_min: float = DELTA_Q_MIN,
) -> Tuple[bool, List[str]]:
    """
    Returns (harmful_stale: bool, reasons: List[str]).

    Harmful stale requires:
      1. |Q_true_new - Q_true_old| >= delta_q_min (value materially changed)
      2. At least one of:
         a. sign(A_true_new) != sign(A_hat_new)
         b. rank_true != rank_hat (only meaningful at junction)
         c. AWAC weight ratio >= 2
    """
    true_changed = abs(q_true_new - q_true_old) >= delta_q_min
    if not true_changed:
        return False, []

    reasons = []

    # (a) advantage sign error
    def _sign(x): return 1 if x > 0 else (-1 if x < 0 else 0)
    if _sign(a_true_new) != _sign(a_hat_new):
        reasons.append("advantage_sign_error")

    # (b) rank error (only at junction)
    if rank_true is not None and rank_hat is not None:
        if rank_true != rank_hat:
            reasons.append("rank_reversal")

    # (c) AWAC weight ratio
    w_true = awac_weight(a_true_new)
    w_hat  = awac_weight(a_hat_new)
    r = weight_ratio(w_true, w_hat)
    if r >= 2.0:
        reasons.append(f"weight_ratio_error({r:.2f}x)")

    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Full row computation for one instance × one target state
# ---------------------------------------------------------------------------

def compute_staleness_row(
    label: str,
    distance_to_gate: int,
    hist_node: str,
    hist_action: str,
    post_node: str,
    oracle_prefix_len: int,
    t_rem_at_hist: int,           # t_rem when AT hist_node (= 40 - prefix_len)
    # Old policy
    V_old: Dict,
    Q_old: Dict,
    old_critic,
    old_actor,
    old_vocab,
    # New policy
    V_new: Dict,
    Q_new: Dict,
    new_critic,
    new_actor,
    new_vocab,
    # Env + instance
    env: SecureCourierEnv,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> Dict:
    """Compute all RQ1 metrics for one (instance, target_state) pair."""
    oracle_prefix = list(ORACLE_SHORTCUT_ACTIONS[:oracle_prefix_len])

    # ---- Q_true ----
    qt_old = get_qtrue_at_node(V_old, Q_old, hist_node, hist_action, t_rem_at_hist)
    qt_new = get_qtrue_at_node(V_new, Q_new, hist_node, hist_action, t_rem_at_hist)

    q_true_old = qt_old["q_true"]
    q_true_new = qt_new["q_true"]
    a_true_old = qt_old["a_true"]
    a_true_new = qt_new["a_true"]
    v_true_old = qt_old["v_true"]
    v_true_new = qt_new["v_true"]

    # ---- Q_hat ----
    qh_old = get_qhat_at_node(
        old_critic, old_actor, old_vocab, env,
        oracle_prefix, hist_action, device, obs_max_len, act_max_len,
    )
    qh_new = get_qhat_at_node(
        new_critic, new_actor, new_vocab, env,
        oracle_prefix, hist_action, device, obs_max_len, act_max_len,
    )

    q_hat_old = qh_old["q_hat_target"]
    q_hat_new = qh_new["q_hat_target"]
    a_hat_old = qh_old["a_hat_target"]
    a_hat_new = qh_new["a_hat_target"]
    v_hat_old = qh_old["v_hat"]
    v_hat_new = qh_new["v_hat"]

    # ---- Staleness metrics ----
    delta_q_true = q_true_new - q_true_old
    delta_q_hat  = q_hat_new  - q_hat_old
    change_tracking_error = delta_q_hat - delta_q_true   # positive = overestimates, negative = lags

    abs_error_old = abs(q_hat_old - q_true_old)
    abs_error_new = abs(q_hat_new - q_true_new)

    # Advantage sign
    def _sign(x): return 1 if x > 0 else (-1 if x < 0 else 0)
    true_sign_change = _sign(a_true_old) != _sign(a_true_new)
    est_sign_change  = _sign(a_hat_old)  != _sign(a_hat_new)
    sign_agreement   = true_sign_change == est_sign_change

    # Rank (only meaningful at junction; None for corridor nodes)
    true_rank = None
    est_rank  = None
    rank_agreement = None

    # AWAC weights
    w_true_new = awac_weight(a_true_new)
    w_hat_new  = awac_weight(a_hat_new)
    w_ratio    = weight_ratio(w_true_new, w_hat_new)

    # Harmful stale classification
    harmful, reasons = is_harmful_stale(
        q_true_old, q_true_new, a_true_new, a_hat_new,
        true_rank, est_rank,
    )

    return {
        "label":                  label,
        "distance_to_gate":       distance_to_gate,
        "hist_node":              hist_node,
        "hist_action":            hist_action,
        "post_node":              post_node,
        "t_rem_at_hist":          t_rem_at_hist,
        # Q_true
        "Q_true_old":             q_true_old,
        "Q_true_new":             q_true_new,
        "A_true_old":             a_true_old,
        "A_true_new":             a_true_new,
        "V_true_old":             v_true_old,
        "V_true_new":             v_true_new,
        # Q_hat
        "Q_hat_old":              q_hat_old,
        "Q_hat_new":              q_hat_new,
        "A_hat_old":              a_hat_old,
        "A_hat_new":              a_hat_new,
        "V_hat_old":              v_hat_old,
        "V_hat_new":              v_hat_new,
        # Differences
        "Delta_Q_true":           delta_q_true,
        "Delta_Q_hat":            delta_q_hat,
        "change_tracking_error":  change_tracking_error,
        "abs_error_old":          abs_error_old,
        "abs_error_new":          abs_error_new,
        # Sign / rank
        "true_sign_change":       true_sign_change,
        "est_sign_change":        est_sign_change,
        "sign_agreement":         sign_agreement,
        "true_rank":              true_rank,
        "est_rank":               est_rank,
        "rank_agreement":         rank_agreement,
        # AWAC weights
        "w_true_new":             w_true_new,
        "w_hat_new":              w_hat_new,
        "weight_ratio":           w_ratio,
        # Harmful stale
        "harmful_stale":          harmful,
        "harmful_reasons":        reasons,
    }
