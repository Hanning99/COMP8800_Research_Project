"""
Exact true-value-change table for G2.

Computes Q^pi_old and Q^pi_new for historical decisions at distances
d = 1, 2, 4, 8, 12 from the gate capability boundary, plus the junction decision.

Uses the G0 DP oracle (secure_courier/oracle.py) with the learned policy's
action distribution at each structural node, evaluated on a reference instance.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from secure_courier.env import SecureCourierEnv
from secure_courier.generator import InstanceConfig
from secure_courier.oracle import dp_values
from secure_courier.task_graph import (
    GRAPH, JUNCTION, START_NODE, GATE_LOBBY,
    DELIVERED, ORACLE_SHORTCUT_ACTIONS,
    admissible_actions as struct_admissible,
)


# ---------------------------------------------------------------------------
# Distance table for shortcut path
# ---------------------------------------------------------------------------
# Gate capability boundary = first gate action (scan_badge) at gate_lobby.
# d = number of structural actions from the (post-action) node to reach gate_lobby.

# Nodes on shortcut path from start and their distances to gate_lobby:
# depot(14), locker_room(13), locker_opened(12), parcel_taken(11),
# admin_hall(10), code_known(9), badge_taken(8), parcel_verified(7),
# junction(6), sc1(5), sc2(4), sc3(3), sc4(2), sc5(1), gate_lobby(0)
#
# "Historical decision" = (state_before_action, action, post-action-state)
# d = dist(post-action-state, gate_lobby)

_DIST_MAP: Dict[str, int] = {
    "depot":          14,
    "locker_room":    13,
    "locker_opened":  12,
    "parcel_taken":   11,
    "admin_hall":     10,
    "code_known":      9,
    "badge_taken":     8,
    "parcel_verified": 7,
    "junction":        6,
    "sc1":             5,
    "sc2":             4,
    "sc3":             3,
    "sc4":             2,
    "sc5":             1,
    "gate_lobby":      0,
}

# Evaluation distances requested in Section 10.2 + junction
EVAL_DISTANCES = [1, 2, 4, 8, 12]
JUNCTION_DIST  = 6  # junction is d=6 from gate_lobby

# Build: dist → (historical_state_node, historical_action, post_action_node)
# "historical_action" = action taken from state that leads to post-action-node
# post-action-node is d steps from gate_lobby

def _build_dist_to_decision() -> Dict[int, Tuple[str, str, str]]:
    """
    For each distance d, find:
      (node_where_action_is_taken, action, post_action_node_at_dist_d)
    by tracing the oracle shortcut path.
    """
    result = {}
    nodes = [START_NODE]
    for a in ORACLE_SHORTCUT_ACTIONS:
        nxt, _, _ = _step_noexcept(nodes[-1], a)
        nodes.append(nxt)

    # nodes[i] is the node after i actions; action i is ORACLE_SHORTCUT_ACTIONS[i]
    # post_action_node = nodes[i+1], pre_action_node = nodes[i]
    for i, a in enumerate(ORACLE_SHORTCUT_ACTIONS):
        pre  = nodes[i]
        post = nodes[i + 1]
        if post in _DIST_MAP:
            d = _DIST_MAP[post]
            result[d] = (pre, a, post)

    return result


def _step_noexcept(node, action):
    from secure_courier.task_graph import step_transition
    try:
        return step_transition(node, action)
    except ValueError:
        return node, 0.0, False


_DIST_TO_DECISION = _build_dist_to_decision()


# ---------------------------------------------------------------------------
# Extract policy distribution from actor checkpoint
# ---------------------------------------------------------------------------

def make_dp_policy_from_actor(
    actor,
    vocab,
    reference_instance: InstanceConfig,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
):
    """
    Returns a DP-compatible policy function derived from a neural actor.

    policy_fn(structural_node, admissible_structural_actions) -> Dict[str, float]
    """
    from secure_courier.state import EnvState

    actor.eval()
    ref_env = SecureCourierEnv(reference_instance)

    def _get_obs_for_node(node: str) -> Dict:
        """Get observation text for a structural node using the reference instance."""
        dummy_state = EnvState(
            node=node,
            step_count=0,
            done=(node == DELIVERED),
            terminal_success=(node == DELIVERED),
            instance_id=reference_instance.instance_id,
        )
        ref_env.restore_state(dummy_state)
        return ref_env._obs()

    def policy_fn(node: str, admissible: List[str]) -> Dict[str, float]:
        if not admissible:
            return {}
        obs = _get_obs_for_node(node)
        adm_disp = [reference_instance.action_display.get(a, a) for a in admissible]

        obs_ids = torch.tensor([vocab.encode(obs["text"], obs_max_len)],
                                dtype=torch.long, device=device)
        act_ids = torch.tensor(
            [[vocab.encode(a, act_max_len) for a in adm_disp]],
            dtype=torch.long, device=device,
        )
        with torch.no_grad():
            log_probs = actor(obs_ids, act_ids)
        probs = F.softmax(log_probs[0], dim=-1).cpu().tolist()
        return {a: p for a, p in zip(admissible, probs)}

    return policy_fn


# ---------------------------------------------------------------------------
# True value change table
# ---------------------------------------------------------------------------

def compute_true_value_table(
    old_actor,
    new_actor,
    vocab_old,
    vocab_new,
    reference_instance: InstanceConfig,
    device: torch.device,
    gamma: float = 0.97,
    max_steps: int = 40,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> List[Dict]:
    """
    Compute the G2 true-value-change table.

    For each distance d ∈ {1, 2, 4, 8, 12} and for the junction decision:
      historical_state, historical_action, post_action_node
      Q_true_old, Q_true_new, Delta_Q_true
      V_true_old, V_true_new (at post_action_node)
      A_true_old, A_true_new (at historical_state for historical_action)
      advantage_sign_change, rank_change

    Returns list of row dicts (one per evaluated distance).
    """
    old_policy = make_dp_policy_from_actor(
        old_actor, vocab_old, reference_instance, device,
        obs_max_len=obs_max_len, act_max_len=act_max_len,
    )
    new_policy = make_dp_policy_from_actor(
        new_actor, vocab_new, reference_instance, device,
        obs_max_len=obs_max_len, act_max_len=act_max_len,
    )

    V_old, Q_old = dp_values(old_policy, gamma=gamma, max_steps=max_steps)
    V_new, Q_new = dp_values(new_policy, gamma=gamma, max_steps=max_steps)

    rows = []

    # Distances to evaluate: EVAL_DISTANCES + junction (d=6)
    eval_d = sorted(set(EVAL_DISTANCES + [JUNCTION_DIST]))

    for d in eval_d:
        if d not in _DIST_TO_DECISION:
            continue

        hist_node, hist_action, post_node = _DIST_TO_DECISION[d]

        # Horizon at hist_node: it is step (14 - d - 1) in the shortcut path
        # (gate_lobby is step 14, and each dist reduces by 1)
        # But we need the horizon AFTER taking 'hist_action', which depends
        # on how many steps it takes to reach hist_node from depot.
        # The step index is: shortcut_idx_of_post_node
        # post_node = nodes[i+1] where ORACLE_SHORTCUT_ACTIONS[i] = hist_action
        # So step_count when reaching post_node = i+1
        # t_rem at post_node = max_steps - (i+1)

        # Find the step index in the shortcut path
        step_idx = None
        nodes_visited = [START_NODE]
        for k, a in enumerate(ORACLE_SHORTCUT_ACTIONS):
            nxt, _, _ = _step_noexcept(nodes_visited[-1], a)
            nodes_visited.append(nxt)
            if nxt == post_node and a == hist_action:
                step_idx = k + 1  # step count after taking this action
                break

        if step_idx is None:
            continue

        t_rem_post = max_steps - step_idx
        t_rem_hist = t_rem_post + 1  # one step earlier (before hist_action)

        q_old = Q_old.get((hist_node, hist_action, t_rem_hist), 0.0)
        q_new = Q_new.get((hist_node, hist_action, t_rem_hist), 0.0)
        v_old_hist = V_old.get((hist_node, t_rem_hist), 0.0)
        v_new_hist = V_new.get((hist_node, t_rem_hist), 0.0)
        a_old = q_old - v_old_hist
        a_new = q_new - v_new_hist
        delta_q = q_new - q_old

        # For junction: also compute Q for safe route
        rank_change = None
        if hist_node == JUNCTION and hist_action == "take_shortcut":
            q_old_safe = Q_old.get((JUNCTION, "take_safe_route", t_rem_hist), 0.0)
            q_new_safe = Q_new.get((JUNCTION, "take_safe_route", t_rem_hist), 0.0)
            old_shortcut_better = (q_old > q_old_safe)
            new_shortcut_better = (q_new > q_new_safe)
            rank_change = (old_shortcut_better != new_shortcut_better)

        row = {
            "distance_to_gate": d,
            "historical_state_id": hist_node,
            "historical_action": hist_action,
            "post_action_node": post_node,
            "step_idx_in_shortcut": step_idx,
            "t_rem_at_hist": t_rem_hist,
            "Q_true_old": q_old,
            "Q_true_new": q_new,
            "Delta_Q_true": delta_q,
            "V_true_old": v_old_hist,
            "V_true_new": v_new_hist,
            "A_true_old": a_old,
            "A_true_new": a_new,
            "advantage_sign_change": (
                (a_old < 0 and a_new >= 0) or (a_old >= 0 and a_new < 0)
            ),
            "rank_change": rank_change,
            "is_junction_decision": (hist_node == JUNCTION),
        }
        rows.append(row)

    return rows
