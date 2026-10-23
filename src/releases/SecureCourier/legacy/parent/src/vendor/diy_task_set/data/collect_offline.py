"""
Offline dataset collection for SecureCourier Phase 1.

Dataset composition (Section 7 targets):
  40%  safe-route scripted successes
  20%  noisy safe-route (distractors injected)
  25%  shortcut-to-gate then gate-fail (distractors at gate)
  15%  uniform exploration

Distractor injection = with probability p, take a distractor (self-loop)
instead of the scripted forward action.

This produces ~20k transitions for the pilot.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from secure_courier.env import SecureCourierEnv
from secure_courier.generator import InstanceConfig, generate_pilot_split
from secure_courier.oracle import oracle_safe_plan, oracle_shortcut_plan
from secure_courier.task_graph import GRAPH, GATE_SEQUENCE, JUNCTION, GATE_LOBBY, GATE_BADGED, GATE_CODED

from .schemas import Transition


# ---------------------------------------------------------------------------
# Scripted behavior policies
# ---------------------------------------------------------------------------

_SHORTCUT_PLAN = oracle_shortcut_plan()
_SAFE_PLAN     = oracle_safe_plan()


def _safe_route_episode(
    env: SecureCourierEnv,
    episode_id: str,
    rng: random.Random,
    noise_prob: float = 0.0,
) -> List[Transition]:
    """
    Follow oracle safe plan.  With noise_prob, replace a forward action with
    a random admissible action (may include distractors).
    """
    transitions = []
    plan = list(_SAFE_PLAN)
    t = 0

    while not env.is_done() and t < env.max_steps and plan:
        state = env.clone_state()
        obs   = env._obs()
        adm_disp = obs["admissible_actions"]
        adm_str  = obs["admissible_structural"]

        intended = plan.pop(0)
        if noise_prob > 0 and rng.random() < noise_prob and len(adm_str) > 1:
            # randomly replace
            action_str = rng.choice([a for a in adm_str if a != intended] or [intended])
        else:
            action_str = intended if intended in adm_str else (adm_str[0] if adm_str else None)

        if action_str is None or action_str not in adm_str:
            break

        action_idx = adm_str.index(action_str)
        log_prob = math.log(1.0 if noise_prob == 0 else max(1.0 / len(adm_str), 1e-8))

        obs_t, reward, terminated, truncated, info = env.step(action_idx)
        next_obs = env._obs()

        transitions.append(Transition(
            transition_id=f"{episode_id}_t{t}",
            instance_id=env.instance.instance_id,
            instance_seed=env.instance.instance_seed,
            episode_id=episode_id,
            t=t,
            full_state=state.to_dict(),
            observation=obs["text"],
            admissible_actions=adm_disp,
            admissible_structural=adm_str,
            action=action_idx,
            action_display=adm_disp[action_idx],
            action_structural=action_str,
            reward=float(reward),
            next_state=env.clone_state().to_dict(),
            next_observation=next_obs["text"],
            next_admissible_actions=next_obs["admissible_actions"],
            next_admissible_structural=next_obs["admissible_structural"],
            done=env.is_done(),
            behavior_policy_id="safe_noisy_scripted" if noise_prob > 0 else "safe_route_scripted",
            behavior_log_prob=log_prob,
            route_label="safe_noisy" if noise_prob > 0 else "safe_success",
        ))
        t += 1

    return transitions


def _shortcut_gate_fail_episode(
    env: SecureCourierEnv,
    episode_id: str,
    rng: random.Random,
) -> List[Transition]:
    """
    Follow shortcut until gate_lobby, then always take distractor at gate nodes.
    Gate procedure never completes → episode times out with reward=0.
    """
    transitions = []
    # Plan up to gate_lobby (first 14 actions of shortcut)
    shortcut_prefix = list(_SHORTCUT_PLAN[:14])  # arrives at gate_lobby
    t = 0

    while not env.is_done() and t < env.max_steps:
        state = env.clone_state()
        obs   = env._obs()
        adm_disp = obs["admissible_actions"]
        adm_str  = obs["admissible_structural"]
        node = state.node

        if node in ("gate_lobby", "gate_badged", "gate_coded"):
            # Take gate distractor (NOT the gate action)
            distractor_actions = [a for a in adm_str if a not in set(GATE_SEQUENCE)]
            if distractor_actions:
                action_str = rng.choice(distractor_actions)
            else:
                break  # shouldn't happen
        elif shortcut_prefix:
            intended = shortcut_prefix.pop(0)
            action_str = intended if intended in adm_str else (adm_str[0] if adm_str else None)
        else:
            # Post-prefix: shouldn't reach here since gate always fails
            break

        if action_str is None or action_str not in adm_str:
            break

        action_idx = adm_str.index(action_str)
        obs_t, reward, terminated, truncated, info = env.step(action_idx)
        next_obs = env._obs()

        transitions.append(Transition(
            transition_id=f"{episode_id}_t{t}",
            instance_id=env.instance.instance_id,
            instance_seed=env.instance.instance_seed,
            episode_id=episode_id,
            t=t,
            full_state=state.to_dict(),
            observation=obs["text"],
            admissible_actions=adm_disp,
            admissible_structural=adm_str,
            action=action_idx,
            action_display=adm_disp[action_idx],
            action_structural=action_str,
            reward=float(reward),
            next_state=env.clone_state().to_dict(),
            next_observation=next_obs["text"],
            next_admissible_actions=next_obs["admissible_actions"],
            next_admissible_structural=next_obs["admissible_structural"],
            done=env.is_done(),
            behavior_policy_id="shortcut_gate_fail_scripted",
            behavior_log_prob=math.log(1.0 / max(len(adm_str), 1)),
            route_label="shortcut_fail",
        ))
        t += 1

        if t >= 40:  # hit step limit
            break

    return transitions


_GATE_NODES = frozenset([GATE_LOBBY, GATE_BADGED, GATE_CODED])
_GATE_ACTIONS = frozenset(GATE_SEQUENCE)


def _exploration_episode(
    env: SecureCourierEnv,
    episode_id: str,
    rng: random.Random,
) -> List[Transition]:
    """Uniform random exploration (gate actions masked at gate nodes).

    At gate nodes, the exploration policy only takes distractor self-loops —
    never gate sequence actions.  This prevents accidental gate completions
    from injecting high-advantage gate trajectories into the offline dataset
    (which would cause AWAC to massively over-weight them and teach the
    gate sequence to the old policy).
    """
    transitions = []
    t = 0

    while not env.is_done() and t < env.max_steps:
        state = env.clone_state()
        obs   = env._obs()
        adm_disp = obs["admissible_actions"]
        adm_str  = obs["admissible_structural"]

        if not adm_str:
            break

        node = state.node
        if node in _GATE_NODES:
            # Mask out gate sequence actions; only take distractors
            non_gate = [a for a in adm_str if a not in _GATE_ACTIONS]
            action_str = rng.choice(non_gate) if non_gate else adm_str[0]
        else:
            action_str = rng.choice(adm_str)
        action_idx = adm_str.index(action_str)
        log_prob   = math.log(1.0 / len(adm_str))

        obs_t, reward, terminated, truncated, info = env.step(action_idx)
        next_obs = env._obs()

        transitions.append(Transition(
            transition_id=f"{episode_id}_t{t}",
            instance_id=env.instance.instance_id,
            instance_seed=env.instance.instance_seed,
            episode_id=episode_id,
            t=t,
            full_state=state.to_dict(),
            observation=obs["text"],
            admissible_actions=adm_disp,
            admissible_structural=adm_str,
            action=action_idx,
            action_display=adm_disp[action_idx],
            action_structural=action_str,
            reward=float(reward),
            next_state=env.clone_state().to_dict(),
            next_observation=next_obs["text"],
            next_admissible_actions=next_obs["admissible_actions"],
            next_admissible_structural=next_obs["admissible_structural"],
            done=env.is_done(),
            behavior_policy_id="exploration_random",
            behavior_log_prob=log_prob,
            route_label="exploration",
        ))
        t += 1

    return transitions


# ---------------------------------------------------------------------------
# Main dataset collector
# ---------------------------------------------------------------------------

def collect_offline_dataset(
    instances: List[InstanceConfig],
    target_transitions: int = 20_000,
    seed: int = 42,
    gamma: float = 0.97,
    max_steps: int = 40,
) -> List[Transition]:
    """
    Collect offline dataset from scripted policies on the given instances.

    Composition:
        40% safe success
        20% noisy safe
        25% shortcut gate fail
        15% exploration
    """
    rng = random.Random(seed)
    all_transitions: List[Transition] = []

    # Target episode counts
    # Average episode lengths:
    #   safe success: 27 steps
    #   noisy safe: ~33 steps (some distractor detours)
    #   shortcut fail: 40 steps (times out)
    #   exploration: ~35 steps (random)
    n_safe    = int(target_transitions * 0.40 / 27)
    n_noisy   = int(target_transitions * 0.20 / 33)
    n_sc_fail = int(target_transitions * 0.25 / 40)
    n_expl    = int(target_transitions * 0.15 / 35)

    episode_counter = 0

    def _run(policy_type, count, noise_prob=0.0):
        nonlocal episode_counter
        for _ in range(count):
            inst = rng.choice(instances)
            env  = SecureCourierEnv(inst, gamma=gamma, max_steps=max_steps)
            env.reset()
            eid = f"ep_{episode_counter:06d}"
            episode_counter += 1

            if policy_type == "safe":
                trs = _safe_route_episode(env, eid, rng, noise_prob=0.0)
            elif policy_type == "noisy":
                trs = _safe_route_episode(env, eid, rng, noise_prob=noise_prob)
            elif policy_type == "sc_fail":
                trs = _shortcut_gate_fail_episode(env, eid, rng)
            else:
                trs = _exploration_episode(env, eid, rng)

            all_transitions.extend(trs)

    _run("safe",    n_safe)
    _run("noisy",   n_noisy,   noise_prob=0.3)
    _run("sc_fail", n_sc_fail)
    _run("expl",    n_expl)

    # Top-up if short (due to rounding)
    while len(all_transitions) < target_transitions:
        inst = rng.choice(instances)
        env  = SecureCourierEnv(inst, gamma=gamma, max_steps=max_steps)
        env.reset()
        eid = f"ep_{episode_counter:06d}"
        episode_counter += 1
        all_transitions.extend(_safe_route_episode(env, eid, rng))

    return all_transitions[:target_transitions]


def save_dataset(transitions: List[Transition], path: Path) -> None:
    path = Path(path)
    with open(path, "w") as f:
        for tr in transitions:
            f.write(tr.to_json() + "\n")


def load_dataset(path: Path) -> List[Transition]:
    transitions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                transitions.append(Transition.from_dict(json.loads(line)))
    return transitions
