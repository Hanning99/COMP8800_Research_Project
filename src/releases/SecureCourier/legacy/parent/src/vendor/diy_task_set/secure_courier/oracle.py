"""
Oracle functions for SecureCourier G0.

Provides:
  1. oracle_shortcut_plan / oracle_safe_plan  — optimal action sequences.
  2. dp_values   — exact finite-horizon DP Q/V for an arbitrary policy.
  3. mc_estimate — Monte Carlo rollout estimate for cross-checking DP.
  4. Audit helpers used by the G0 runner.

Policy representation:
  A policy is a callable:
    policy(node: str, admissible: List[str]) -> Dict[str, float]
  returning a probability distribution over admissible structural action IDs.
  Probabilities must be non-negative and sum to 1.0.

DP state representation:
  V[(node, t_rem)]    — expected discounted return from node with t_rem steps left.
  Q[(node, a, t_rem)] — Q-value for action a at node with t_rem steps left.

The DP correctly handles distractor self-loops via finite-horizon recursion.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from .task_graph import (
    GRAPH,
    ALL_NODES,
    DELIVERED,
    ORACLE_SHORTCUT_ACTIONS,
    ORACLE_SAFE_ACTIONS,
    GATE_SEQUENCE,
    admissible_actions,
    step_transition,
    is_terminal,
    JUNCTION,
    START_NODE,
)

Policy = Callable[[str, List[str]], Dict[str, float]]

# ---------------------------------------------------------------------------
# Oracle route plans
# ---------------------------------------------------------------------------

def oracle_shortcut_plan() -> List[str]:
    """Return the optimal shortcut action sequence (structural IDs, length 22)."""
    return list(ORACLE_SHORTCUT_ACTIONS)


def oracle_safe_plan() -> List[str]:
    """Return the optimal safe-route action sequence (structural IDs, length 27)."""
    return list(ORACLE_SAFE_ACTIONS)


def verify_plan_execution(plan: List[str], start_node: str = START_NODE) -> Dict:
    """
    Trace a plan through the structural graph and return a verification report.

    Returns dict with: nodes_visited, total_steps, final_node,
    terminal_success, reward_step, gate_sequence_observed.
    """
    node = start_node
    nodes_visited = [node]
    reward_step = None
    gate_actions_seen = []

    for step_idx, action in enumerate(plan):
        try:
            nxt, reward, done = step_transition(node, action)
        except ValueError:
            # Inadmissible action — plan fails here
            break
        if action in GATE_SEQUENCE:
            gate_actions_seen.append((step_idx, action))
        if reward > 0:
            reward_step = step_idx
        nodes_visited.append(nxt)
        node = nxt
        if done:
            break

    # Check gate sequence order
    gate_seq_ok = False
    if len(gate_actions_seen) == 3:
        obs_gate = [a for _, a in gate_actions_seen]
        gate_seq_ok = obs_gate == list(GATE_SEQUENCE)

    return {
        "total_steps": len(plan),
        "final_node": node,
        "terminal_success": (node == DELIVERED),
        "reward_step": reward_step,
        "nodes_visited": nodes_visited,
        "gate_actions_seen": gate_actions_seen,
        "gate_sequence_ok": gate_seq_ok,
    }


# ---------------------------------------------------------------------------
# Exact finite-horizon DP
# ---------------------------------------------------------------------------

def dp_values(
    policy: Policy,
    gamma: float = 0.97,
    max_steps: int = 40,
    start_node: str = START_NODE,
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, str, int], float]]:
    """
    Compute exact finite-horizon DP values for *policy*.

    The DP uses (node, t_rem) as the state, where t_rem = remaining steps
    before truncation.

    Base case:
        V(node, 0) = 0 for all nodes (no steps left; can't gain reward).
        V(DELIVERED, t) = 0 for all t (terminal; no future steps matter).

    Recursion (for t_rem >= 1 and node != DELIVERED):
        Q(node, a, t_rem)  = r(node, a) + gamma * V(next(node,a), t_rem-1)
        V(node, t_rem)     = sum_a pi(a|node) * Q(node, a, t_rem)

    Distractor self-loops (next(node, dist) == node) are handled naturally:
        Q(node, dist, t_rem) = 0 + gamma * V(node, t_rem-1)

    Returns
    -------
    V : Dict[(node, t_rem), float]
    Q : Dict[(node, action_id, t_rem), float]
    """
    # Collect all nodes reachable from start_node
    all_nodes = set(GRAPH.keys())

    V: Dict[Tuple[str, int], float] = {}
    Q: Dict[Tuple[str, str, int], float] = {}

    # Base case: t_rem = 0 → V = 0 for all nodes
    for node in all_nodes:
        V[(node, 0)] = 0.0

    # Fill t_rem = 1, 2, ..., max_steps
    for t_rem in range(1, max_steps + 1):
        for node in all_nodes:
            # Terminal node: no future reward
            if node == DELIVERED:
                V[(node, t_rem)] = 0.0
                continue

            actions = admissible_actions(node)
            if not actions:
                # Dead-end non-terminal (only DELIVERED should be here)
                V[(node, t_rem)] = 0.0
                continue

            # Get policy probabilities
            probs = policy(node, actions)
            _validate_probs(probs, actions, node)

            v_sum = 0.0
            for a, p in probs.items():
                if p == 0.0:
                    Q[(node, a, t_rem)] = 0.0
                    continue
                nxt, reward, done = step_transition(node, a)
                # After this action, t_rem decreases by 1
                v_nxt = V.get((nxt, t_rem - 1), 0.0)
                q_val = reward + gamma * v_nxt
                Q[(node, a, t_rem)] = q_val
                v_sum += p * q_val

            # Fill Q=0 for actions with p=0 that were skipped
            for a in actions:
                if (node, a, t_rem) not in Q:
                    nxt, reward, done = step_transition(node, a)
                    v_nxt = V.get((nxt, t_rem - 1), 0.0)
                    Q[(node, a, t_rem)] = reward + gamma * v_nxt

            V[(node, t_rem)] = v_sum

    return V, Q


def _validate_probs(probs: Dict[str, float], actions: List[str], node: str) -> None:
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Policy probs at '{node}' sum to {total:.6f}, not 1.0. "
            f"actions={actions}, probs={probs}"
        )
    for a in actions:
        if a not in probs:
            raise ValueError(
                f"Policy missing action '{a}' at node '{node}'."
            )


# ---------------------------------------------------------------------------
# Monte Carlo rollout estimate
# ---------------------------------------------------------------------------

def mc_estimate(
    policy: Policy,
    gamma: float = 0.97,
    max_steps: int = 40,
    start_node: str = START_NODE,
    n_rollouts: int = 2000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of V(start_node) under policy.

    Runs n_rollouts episodes from start_node, applying policy at each step,
    and returns the mean discounted return.

    Uses a local RNG seeded from *seed* for full reproducibility.
    """
    rng = random.Random(seed)
    returns = []

    for _ in range(n_rollouts):
        node = start_node
        total_r = 0.0
        discount = 1.0

        for _step in range(max_steps):
            if node == DELIVERED:
                break
            actions = admissible_actions(node)
            if not actions:
                break
            probs = policy(node, actions)
            # Sample action
            chosen = rng.choices(actions, weights=[probs[a] for a in actions], k=1)[0]
            nxt, reward, done = step_transition(node, chosen)
            total_r += discount * reward
            discount *= gamma
            node = nxt
            if done:
                break

        returns.append(total_r)

    return sum(returns) / len(returns)


def mc_estimate_per_node(
    policy: Policy,
    nodes_to_check: List[str],
    gamma: float = 0.97,
    max_steps: int = 40,
    n_rollouts: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo V estimate for multiple start nodes.

    For each start node, estimates V(node) by running n_rollouts from that
    node using remaining horizon = max_steps.  Returns dict node -> V_mc.
    """
    rng = random.Random(seed)
    results = {}

    for start in nodes_to_check:
        node_returns = []
        for _ in range(n_rollouts):
            node = start
            total_r = 0.0
            discount = 1.0
            for _step in range(max_steps):
                if node == DELIVERED:
                    break
                actions = admissible_actions(node)
                if not actions:
                    break
                probs = policy(node, actions)
                chosen = rng.choices(
                    actions, weights=[probs[a] for a in actions], k=1
                )[0]
                nxt, reward, done = step_transition(node, chosen)
                total_r += discount * reward
                discount *= gamma
                node = nxt
                if done:
                    break
            node_returns.append(total_r)
        results[start] = sum(node_returns) / len(node_returns)

    return results


# ---------------------------------------------------------------------------
# Built-in test policies
# ---------------------------------------------------------------------------

def uniform_policy(node: str, admissible: List[str]) -> Dict[str, float]:
    """Uniform probability over all admissible actions."""
    n = len(admissible)
    return {a: 1.0 / n for a in admissible}


def gate_expert_policy(node: str, admissible: List[str]) -> Dict[str, float]:
    """
    Policy that knows the gate sequence and always prefers forward progress.

    At gate nodes: P(correct gate action) = 1, P(distractor) = 0.
    At all other nodes: uniform over all admissible actions.
    """
    gate_actions = set(GATE_SEQUENCE)
    gate_at_node = [a for a in admissible if a in gate_actions]
    if gate_at_node:
        # Only pick the gate action
        probs = {a: 0.0 for a in admissible}
        probs[gate_at_node[0]] = 1.0
        return probs
    return uniform_policy(node, admissible)


def shortcut_gate_expert_policy(node: str, admissible: List[str]) -> Dict[str, float]:
    """
    Policy that strongly prefers the shortcut and knows the gate.

    At junction: P(take_shortcut) = 0.95, P(take_safe_route) = 0.05, P(dist) = 0.
    At gate nodes: P(correct gate action) = 1, P(distractor) = 0.
    At all other nodes: uniform over forward actions (excluding distractors if possible).
    """
    # Junction
    if "take_shortcut" in admissible:
        probs = {a: 0.0 for a in admissible}
        if "take_safe_route" in admissible:
            probs["take_shortcut"] = 0.95
            probs["take_safe_route"] = 0.05
        else:
            probs["take_shortcut"] = 1.0
        return probs

    # Gate nodes
    gate_actions = set(GATE_SEQUENCE)
    gate_at_node = [a for a in admissible if a in gate_actions]
    if gate_at_node:
        probs = {a: 0.0 for a in admissible}
        probs[gate_at_node[0]] = 1.0
        return probs

    # Everywhere else: uniform
    return uniform_policy(node, admissible)


def gate_novice_policy(node: str, admissible: List[str],
                       gate_action_prob: float = 0.05) -> Dict[str, float]:
    """
    Policy that has very low probability of executing gate actions correctly.

    Simulates an untrained policy that cannot reliably perform the secure-gate
    sequence.  At gate nodes: P(correct gate action) = gate_action_prob,
    remaining probability shared equally among distractors.
    At junction: uniform over all admissible actions (doesn't know to prefer shortcut).
    Elsewhere: uniform.
    """
    gate_acts = set(GATE_SEQUENCE)
    gate_here = [a for a in admissible if a in gate_acts]
    if gate_here:
        n = len(admissible)
        probs = {}
        for a in admissible:
            if a in gate_acts:
                probs[a] = gate_action_prob
            else:
                probs[a] = (1.0 - gate_action_prob) / (n - 1) if n > 1 else 0.0
        return probs
    return uniform_policy(node, admissible)


def gate_expert_no_junction_bias_policy(
    node: str, admissible: List[str]
) -> Dict[str, float]:
    """
    Policy that knows the gate (P=1 for gate actions) but is uniform at junction.
    Useful for isolating the gate-skill effect while keeping junction uniform.
    """
    gate_acts = set(GATE_SEQUENCE)
    gate_here = [a for a in admissible if a in gate_acts]
    if gate_here:
        probs = {a: 0.0 for a in admissible}
        probs[gate_here[0]] = 1.0
        return probs
    return uniform_policy(node, admissible)


def safe_route_policy(node: str, admissible: List[str]) -> Dict[str, float]:
    """
    Policy that always takes the safe route.

    At junction: P(take_safe_route) = 1.0.
    Gate nodes: uniform (irrelevant since safe route avoids gate).
    Elsewhere: uniform.
    """
    if "take_safe_route" in admissible:
        probs = {a: 0.0 for a in admissible}
        probs["take_safe_route"] = 1.0
        return probs
    return uniform_policy(node, admissible)


# ---------------------------------------------------------------------------
# Exploit-path audit
# ---------------------------------------------------------------------------

def find_all_successful_paths(max_depth: int = 50) -> List[List[str]]:
    """
    DFS over the structural graph to find all paths that reach DELIVERED.

    Only valid structural paths are considered (no repeated nodes).
    Returns list of action sequences.
    """
    paths = []
    # DFS: (current_node, actions_so_far, nodes_so_far)
    stack = [(START_NODE, [], {START_NODE})]

    while stack:
        node, actions, visited = stack.pop()
        if node == DELIVERED:
            paths.append(actions)
            continue
        if len(actions) >= max_depth:
            continue
        for a, nxt in GRAPH.get(node, []):
            # Avoid infinite loops: don't revisit nodes (except self-loops
            # are distractors — skip them in path search)
            if nxt == node:
                continue  # distractor self-loop — skip in path enumeration
            if nxt not in visited:
                stack.append((nxt, actions + [a], visited | {nxt}))

    return paths


def audit_exploit_paths() -> Dict:
    """
    Check that no successful path is shorter than the oracle shortcut (22 steps).

    Returns a dict with:
      - all_path_lengths: sorted list of successful path lengths
      - min_length: minimum path length found
      - oracle_shortcut_length: 22
      - no_exploit: True if min_length >= oracle_shortcut_length
    """
    paths = find_all_successful_paths()
    lengths = sorted(len(p) for p in paths)
    oracle_len = len(ORACLE_SHORTCUT_ACTIONS)

    return {
        "total_successful_paths": len(paths),
        "all_path_lengths": lengths,
        "min_length": min(lengths) if lengths else None,
        "oracle_shortcut_length": oracle_len,
        "no_exploit": (min(lengths) >= oracle_len) if lengths else False,
        "paths": [{"length": len(p), "actions": p} for p in paths],
    }


def verify_gate_sequence_required() -> Dict:
    """
    Verify that DELIVERED cannot be reached via the shortcut without the full
    gate sequence scan_badge -> enter_code -> open_gate.

    Checks:
    1. The oracle shortcut plan contains all three gate actions in order.
    2. Removing any gate action from the plan causes failure to reach DELIVERED.
    3. Reordering gate actions causes failure.
    """
    shortcut = list(ORACLE_SHORTCUT_ACTIONS)
    gate_indices = {a: shortcut.index(a) for a in GATE_SEQUENCE if a in shortcut}

    checks = {}

    # Check 1: gate sequence is in shortcut
    checks["gate_in_shortcut"] = all(a in shortcut for a in GATE_SEQUENCE)
    checks["gate_order_correct"] = (
        gate_indices.get("scan_badge", -1) <
        gate_indices.get("enter_code", 9999) <
        gate_indices.get("open_gate", 9999)
    )

    # Check 2: removing each gate action causes failure
    for removed in GATE_SEQUENCE:
        plan_without = [a for a in shortcut if a != removed]
        result = verify_plan_execution(plan_without)
        checks[f"fails_without_{removed}"] = not result["terminal_success"]

    # Check 3: reordering gate sequence causes failure
    # swap scan_badge and open_gate
    plan_swapped = list(shortcut)
    i1, i2 = gate_indices["scan_badge"], gate_indices["open_gate"]
    plan_swapped[i1], plan_swapped[i2] = plan_swapped[i2], plan_swapped[i1]
    result_swapped = verify_plan_execution(plan_swapped)
    checks["fails_with_swapped_gate_order"] = not result_swapped["terminal_success"]

    checks["all_passed"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Advantage flip verification (G0 question 7)
# ---------------------------------------------------------------------------

def compute_junction_advantage_flip(
    gamma: float = 0.97,
    max_steps: int = 40,
) -> Dict:
    """
    Verify that the junction action can support an advantage sign/rank flip.

    Uses two contrasting policies that represent the 'old' and 'new' policy
    from the value-staleness mechanism:

    Old policy  (gate_novice): very low P(correct gate action) = 0.05 at gate
                               nodes; uniform at junction.
                               Models an agent that hasn't learned the gate.

    New policy  (gate_expert_no_junction_bias): P(gate action) = 1.0 at gate
                               nodes; uniform at junction.
                               Models an agent that has learned the gate.

    Both policies are evaluated with uniform over {shortcut, safe, distractors}
    at the junction so the advantage is purely due to the gate capability, not
    a junction-preference.

    Expected result:
      Under old policy: Q(shortcut) < Q(safe)  →  A(shortcut) < 0
      Under new policy: Q(shortcut) > Q(safe)  →  A(shortcut) > 0

    Returns a dict with DP values and flip status.
    """
    def old_policy(node: str, admissible: List[str]) -> Dict[str, float]:
        return gate_novice_policy(node, admissible, gate_action_prob=0.05)

    # Compute DP under old (gate-novice) policy
    V_old, Q_old = dp_values(old_policy, gamma=gamma, max_steps=max_steps)

    # Compute DP under new (gate-expert, no junction bias) policy
    V_new, Q_new = dp_values(
        gate_expert_no_junction_bias_policy, gamma=gamma, max_steps=max_steps
    )

    # Horizon at junction: after 8 prefix steps, t_rem = max_steps - 8 = 32
    t_junc = max_steps - 8

    q_old_shortcut = Q_old.get((JUNCTION, "take_shortcut", t_junc), 0.0)
    q_old_safe     = Q_old.get((JUNCTION, "take_safe_route", t_junc), 0.0)
    q_new_shortcut = Q_new.get((JUNCTION, "take_shortcut", t_junc), 0.0)
    q_new_safe     = Q_new.get((JUNCTION, "take_safe_route", t_junc), 0.0)

    v_old_junc = V_old.get((JUNCTION, t_junc), 0.0)
    v_new_junc = V_new.get((JUNCTION, t_junc), 0.0)

    a_old_shortcut = q_old_shortcut - v_old_junc
    a_new_shortcut = q_new_shortcut - v_new_junc

    flip_exists = (a_old_shortcut < 0) and (a_new_shortcut > 0)
    rank_flip   = (q_old_shortcut < q_old_safe) and (q_new_shortcut > q_new_safe)

    return {
        "t_junc": t_junc,
        "description": (
            "old_policy = gate_novice (P(gate_action)=0.05); "
            "new_policy = gate_expert_no_junction_bias (P(gate_action)=1.0). "
            "Both use uniform at junction."
        ),
        "old_policy": {
            "name": "gate_novice (P_gate=0.05, uniform at junction)",
            "Q_take_shortcut": q_old_shortcut,
            "Q_take_safe_route": q_old_safe,
            "V_junction": v_old_junc,
            "A_take_shortcut": a_old_shortcut,
        },
        "new_policy": {
            "name": "gate_expert_no_junction_bias (P_gate=1.0, uniform at junction)",
            "Q_take_shortcut": q_new_shortcut,
            "Q_take_safe_route": q_new_safe,
            "V_junction": v_new_junc,
            "A_take_shortcut": a_new_shortcut,
        },
        "advantage_sign_flip": flip_exists,
        "rank_flip": rank_flip,
        "junction_supports_flip": flip_exists or rank_flip,
        "note": (
            "advantage_sign_flip requires A_old < 0 AND A_new > 0. "
            "rank_flip requires Q_old(shortcut) < Q_old(safe) AND "
            "Q_new(shortcut) > Q_new(safe). "
            "Either condition suffices to confirm the mechanism."
        ),
    }
