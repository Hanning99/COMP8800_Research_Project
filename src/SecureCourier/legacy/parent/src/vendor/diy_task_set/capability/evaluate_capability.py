"""
Capability evaluation harness for G1 and G2 gates.

Evaluates a policy checkpoint on:
  - full-task success (unrestricted)
  - safe-route success (force first action = take_safe_route)
  - shortcut corridor arrival-at-gate (force first action = take_shortcut)
  - secure-gate completion (from gate_lobby with prerequisites)
  - forced-shortcut full-task success (force first action = take_shortcut,
    then let policy decide — requires completing gate)
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

import torch

from secure_courier.env import SecureCourierEnv
from secure_courier.generator import InstanceConfig
from secure_courier.state import EnvState
from secure_courier.task_graph import (
    JUNCTION, GATE_LOBBY,
    START_NODE, DELIVERED,
)
from secure_courier.oracle import oracle_shortcut_plan as _sc_plan, oracle_safe_plan as _sf_plan

# How many prefix steps to reach junction
_PREFIX_LEN = 8
# Shortcut prefix reaches gate_lobby at step 14
_SHORTCUT_TO_GATE_LEN = 14


# ---------------------------------------------------------------------------
# Policy wrapper: neural actor → callable for env
# ---------------------------------------------------------------------------

def make_env_policy(
    actor,
    vocab,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
    deterministic: bool = False,
) -> Callable[[SecureCourierEnv], int]:
    """
    Return a function that takes an env and returns an action index.

    Uses greedy (argmax) if deterministic=True, else samples.
    """
    actor.eval()

    def policy_fn(env: SecureCourierEnv) -> int:
        obs = env._obs()
        adm_disp = obs["admissible_actions"]
        if not adm_disp:
            return 0

        obs_ids  = torch.tensor([vocab.encode(obs["text"], obs_max_len)],
                                 dtype=torch.long, device=device)
        act_ids  = torch.tensor(
            [[vocab.encode(a, act_max_len) for a in adm_disp]],
            dtype=torch.long, device=device,
        )
        with torch.no_grad():
            log_probs = actor(obs_ids, act_ids)  # [1, N]
        if deterministic:
            return int(log_probs.argmax(dim=-1).item())
        else:
            return int(torch.distributions.Categorical(logits=log_probs[0]).sample().item())

    return policy_fn


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: SecureCourierEnv,
    policy_fn: Callable[[SecureCourierEnv], int],
    max_steps: Optional[int] = None,
    force_first_action: Optional[str] = None,
    start_state: Optional[EnvState] = None,
) -> Dict[str, Any]:
    """
    Run one episode and return result dict.

    force_first_action: structural action ID to force at the first step.
    start_state: if given, restore env to this state before running.
    """
    if start_state is not None:
        env.restore_state(start_state)
    else:
        env.reset()

    max_steps = max_steps or env.max_steps
    total_reward = 0.0
    gamma = env.gamma
    discount = 1.0
    steps = 0
    terminal_success = False
    visited_nodes = [env.current_node()]

    first_step = True
    while not env.is_done() and steps < max_steps:
        if first_step and force_first_action is not None:
            adm = env.admissible_structural()
            if force_first_action in adm:
                idx = adm.index(force_first_action)
                obs, r, terminated, truncated, info = env.step(idx)
            else:
                break  # can't force action
        else:
            idx = policy_fn(env)
            obs, r, terminated, truncated, info = env.step(idx)

        first_step = False
        total_reward += discount * r
        discount *= gamma
        steps += 1
        visited_nodes.append(env.current_node())
        if env.is_done():
            terminal_success = env._state.terminal_success
            break

    return {
        "total_reward": total_reward,
        "terminal_success": terminal_success,
        "steps": steps,
        "visited_nodes": visited_nodes,
    }


# ---------------------------------------------------------------------------
# G1/G2 evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    actor,
    vocab,
    instances: List[InstanceConfig],
    device: torch.device,
    n_eval_episodes: int = 100,
    seed: int = 99,
    gamma: float = 0.97,
    max_steps: int = 40,
    obs_max_len: int = 128,
    act_max_len: int = 32,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Full G1/G2 evaluation of an actor checkpoint.

    Uses deterministic (greedy argmax) policy by default so that gate
    metrics are seed-independent and match the experiment intent:
      - gate_sequence_success measures whether argmax at each gate node
        is the correct gate action (not retry-inflated stochastic success)
      - forced_shortcut measures whether the greedy policy completes
        the gate after forced shortcut (not retry-inflated)

    Returns dict with keys:
      full_task_success
      safe_route_success
      shortcut_arrival_at_gate
      gate_sequence_success
      forced_shortcut_full_task_success
    """
    rng = random.Random(seed)
    policy_fn = make_env_policy(actor, vocab, device,
                                obs_max_len=obs_max_len, act_max_len=act_max_len,
                                deterministic=deterministic)

    sc_prefix_actions  = list(_sc_plan()[:_PREFIX_LEN])    # up to junction
    sf_prefix_actions  = list(_sf_plan()[:_PREFIX_LEN])    # up to junction

    metrics = {
        "full_task_success": [],
        "safe_route_success": [],
        "shortcut_arrival_at_gate": [],
        "gate_sequence_success": [],
        "forced_shortcut_full_task_success": [],
    }

    for _ in range(n_eval_episodes):
        inst = rng.choice(instances)
        env  = SecureCourierEnv(inst, gamma=gamma, max_steps=max_steps)

        # 1. Full-task success (unrestricted)
        env.reset()
        res = run_episode(env, policy_fn)
        metrics["full_task_success"].append(float(res["terminal_success"]))

        # 2. Safe-route success: force take_safe_route at junction,
        #    then let policy continue
        env.reset()
        # Navigate to junction using oracle prefix
        for a in sf_prefix_actions:
            if env.is_done():
                break
            adm = env.admissible_structural()
            if a in adm:
                env.execute_structural(a)
        # Force safe route
        if not env.is_done() and env.current_node() == JUNCTION:
            adm = env.admissible_structural()
            if "take_safe_route" in adm:
                env.execute_structural("take_safe_route")
            state_after_safe = env.clone_state()
            safe_res = run_episode(env, policy_fn, start_state=state_after_safe)
            metrics["safe_route_success"].append(float(safe_res["terminal_success"]))
        else:
            metrics["safe_route_success"].append(0.0)

        # 3. Shortcut corridor arrival-at-gate:
        #    Force take_shortcut, check if policy reaches gate_lobby
        env.reset()
        for a in sc_prefix_actions:
            if env.is_done():
                break
            adm = env.admissible_structural()
            if a in adm:
                env.execute_structural(a)
        if not env.is_done() and env.current_node() == JUNCTION:
            adm = env.admissible_structural()
            if "take_shortcut" in adm:
                env.execute_structural("take_shortcut")
            # Let policy navigate from sc1 toward gate
            gate_reached = False
            steps_left = max_steps - env.step_count()
            for _ in range(steps_left):
                if env.current_node() == GATE_LOBBY:
                    gate_reached = True
                    break
                if env.is_done():
                    break
                idx = policy_fn(env)
                env.step(idx)
            metrics["shortcut_arrival_at_gate"].append(float(gate_reached))
        else:
            metrics["shortcut_arrival_at_gate"].append(0.0)

        # 4. Secure-gate completion: start from gate_lobby with prerequisites
        #    Already set (reached via shortcut prefix), reset to gate_lobby-like state
        env.reset()
        # Navigate to gate_lobby via oracle shortcut
        sc_to_gate = list(_sc_plan()[:_SHORTCUT_TO_GATE_LEN])
        success_before_gate = True
        for a in sc_to_gate:
            if env.is_done():
                success_before_gate = False
                break
            adm = env.admissible_structural()
            if a in adm:
                env.execute_structural(a)
            else:
                success_before_gate = False
                break
        if success_before_gate and env.current_node() == GATE_LOBBY:
            # Let policy try to complete the gate
            gate_state_before = env.clone_state()
            gate_completed = False
            for _ in range(max_steps - env.step_count()):
                if env.current_node() not in ("gate_lobby", "gate_badged", "gate_coded"):
                    # Either completed gate (gate_passed) or gave up
                    if env.current_node() == "gate_passed":
                        gate_completed = True
                    break
                if env.is_done():
                    break
                idx = policy_fn(env)
                env.step(idx)
            metrics["gate_sequence_success"].append(float(gate_completed))
        else:
            metrics["gate_sequence_success"].append(0.0)

        # 5. Forced-shortcut full-task success:
        #    Navigate prefix, force take_shortcut at junction, then let policy go
        env.reset()
        for a in sc_prefix_actions:
            if env.is_done():
                break
            adm = env.admissible_structural()
            if a in adm:
                env.execute_structural(a)
        if not env.is_done() and env.current_node() == JUNCTION:
            adm = env.admissible_structural()
            if "take_shortcut" in adm:
                env.execute_structural("take_shortcut")
            state_after_shortcut = env.clone_state()
            forced_res = run_episode(env, policy_fn, start_state=state_after_shortcut)
            metrics["forced_shortcut_full_task_success"].append(
                float(forced_res["terminal_success"])
            )
        else:
            metrics["forced_shortcut_full_task_success"].append(0.0)

    return {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}


def evaluate_junction_action_probs(
    actor,
    vocab,
    reference_instance: InstanceConfig,
    device: torch.device,
    obs_max_len: int = 128,
    act_max_len: int = 32,
) -> Dict[str, float]:
    """
    Return the actor's action probabilities at the junction state.

    Uses the reference instance's junction observation.
    """
    import torch.nn.functional as F
    from secure_courier.state import EnvState

    env = SecureCourierEnv(reference_instance)
    env.reset()
    # Navigate to junction
    sc_prefix = list(_sc_plan()[:_PREFIX_LEN])
    for a in sc_prefix:
        if env.current_node() == JUNCTION:
            break
        adm = env.admissible_structural()
        if a in adm:
            env.execute_structural(a)

    if env.current_node() != JUNCTION:
        return {}

    obs  = env._obs()
    adm_disp = obs["admissible_actions"]
    adm_str  = obs["admissible_structural"]

    obs_ids = torch.tensor([vocab.encode(obs["text"], obs_max_len)],
                            dtype=torch.long, device=device)
    act_ids = torch.tensor(
        [[vocab.encode(a, act_max_len) for a in adm_disp]],
        dtype=torch.long, device=device,
    )
    actor.eval()
    with torch.no_grad():
        log_probs = actor(obs_ids, act_ids)
    probs = F.softmax(log_probs[0], dim=-1).cpu().numpy()

    return {a_str: float(probs[i]) for i, a_str in enumerate(adm_str)}
