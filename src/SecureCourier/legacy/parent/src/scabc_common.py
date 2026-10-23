"""Common clean-AWAC, evaluation, hashing, and checkpoint utilities for SC-A/B/C."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DIY_VENDOR = ROOT / "src/vendor/diy_task_set"
VENDOR = ROOT / "src/vendor"
sys.path[:0] = [str(DIY_VENDOR), str(VENDOR)]

from agents.actor import Actor
from agents.critic import Critic
from agents.text_encoder import Vocab
from capability.evaluate_capability import evaluate_checkpoint
from pdr.adapters.awac import awac_actor_loss
from pdr.masks.action_mask import masked_log_softmax
from pdr.projection.advantage import center_under_policy
from secure_courier.env import SecureCourierEnv
from secure_courier.generator import InstanceConfig, generate_pilot_split
from secure_courier.oracle import dp_values, oracle_shortcut_plan
from secure_courier.task_graph import ALL_NODES, GRAPH, GATE_SEQUENCE, step_transition


GAMMA = 0.97
MAX_STEPS = 40
OBS_MAX = 128
ACT_MAX = 32
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 3e-4
TEMPERATURE = 0.3
WEIGHT_CLIP = 20.0
GRADIENT_CLIP = 5.0
TARGET_TAU = 0.005

GATE_ACTION = {
    "gate_lobby": GATE_SEQUENCE[0],
    "gate_badged": GATE_SEQUENCE[1],
    "gate_coded": GATE_SEQUENCE[2],
}
CURRICULUM_NODES = ("sc5", "sc4", "sc3", "sc2")


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> Any:
    if torch.is_tensor(value):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(clean(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tensor_tree_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return tensor_tree_hash(module.state_dict())


def seed_everything(seed: int, device: torch.device) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        if len(self.rows) >= self.capacity:
            self.rows.pop(0)
        self.rows.append(row)

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    def sample(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        if not self.rows:
            raise RuntimeError("cannot sample an empty replay")
        return rng.choices(self.rows, k=count)

    def __len__(self) -> int:
        return len(self.rows)


def build_vocab(transitions: Sequence[Any], instances: Sequence[InstanceConfig]) -> Vocab:
    vocab = Vocab(max_size=2000)
    for item in transitions:
        row = item.__dict__ if hasattr(item, "__dict__") else item
        vocab.add_text(row["observation"])
        for action in row["admissible_actions"]:
            vocab.add_text(action)
    for instance in instances:
        for action in instance.action_display.values():
            vocab.add_text(action)
    return vocab


def encode_transition(
    row: dict[str, Any], vocab: Vocab, device: torch.device
) -> dict[str, torch.Tensor]:
    current = [vocab.encode(action, ACT_MAX) for action in row["admissible_actions"]]
    next_actions = [
        vocab.encode(action, ACT_MAX) for action in row["next_admissible_actions"]
    ]
    if not next_actions:
        next_actions = [[0] * ACT_MAX]
    return {
        "obs_tokens": torch.tensor(vocab.encode(row["observation"], OBS_MAX), dtype=torch.long, device=device),
        "act_tokens": torch.tensor(current, dtype=torch.long, device=device),
        "action_idx": torch.tensor(row["action"], dtype=torch.long, device=device),
        "reward": torch.tensor(row["reward"], dtype=torch.float32, device=device),
        "next_obs_tokens": torch.tensor(vocab.encode(row["next_observation"], OBS_MAX), dtype=torch.long, device=device),
        "next_act_tokens": torch.tensor(next_actions, dtype=torch.long, device=device),
        "done": torch.tensor(float(row["done"]), dtype=torch.float32, device=device),
    }


def clean_collate(
    rows: Sequence[dict[str, Any]], vocab: Vocab, device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = [encode_transition(row, vocab, device) for row in rows]
    max_current = max(item["act_tokens"].shape[0] for item in encoded)
    max_next = max(item["next_act_tokens"].shape[0] for item in encoded)

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        if value.shape[0] == count:
            return value
        padding = torch.zeros(
            (count - value.shape[0], value.shape[1]), dtype=value.dtype, device=device
        )
        return torch.cat([value, padding], dim=0)

    batch = {
        "obs_tokens": torch.stack([item["obs_tokens"] for item in encoded]),
        "act_tokens": torch.stack([pad(item["act_tokens"], max_current) for item in encoded]),
        "action_idx": torch.stack([item["action_idx"] for item in encoded]),
        "reward": torch.stack([item["reward"] for item in encoded]),
        "next_obs_tokens": torch.stack([item["next_obs_tokens"] for item in encoded]),
        "next_act_tokens": torch.stack([pad(item["next_act_tokens"], max_next) for item in encoded]),
        "done": torch.stack([item["done"] for item in encoded]),
    }
    current_mask = torch.zeros((len(rows), max_current), dtype=torch.bool, device=device)
    next_mask = torch.zeros((len(rows), max_next), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        current_mask[index, : len(row["admissible_actions"])] = True
        next_mask[index, : max(1, len(row["next_admissible_actions"]))] = True
    batch["action_mask"] = current_mask
    batch["next_action_mask"] = next_mask
    return batch


def critic_step(
    actor: Actor,
    critic: Critic,
    target: Critic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    # Diagnostic evaluators deliberately leave modules in eval mode.  Restore
    # the train/eval contract at the update boundary so GRU backward semantics
    # do not depend on which audit ran immediately beforehand.
    actor.eval()
    critic.train()
    target.eval()
    with torch.no_grad():
        next_q = target(batch["next_obs_tokens"], batch["next_act_tokens"])
        next_logp = masked_log_softmax(
            actor(batch["next_obs_tokens"], batch["next_act_tokens"]),
            batch["next_action_mask"],
        )
        next_value = (
            next_logp.exp()
            * torch.where(batch["next_action_mask"], next_q, torch.zeros_like(next_q))
        ).sum(-1)
        target_q = batch["reward"] + GAMMA * (1 - batch["done"]) * next_value
    all_q = critic(batch["obs_tokens"], batch["act_tokens"])
    selected_q = all_q.gather(1, batch["action_idx"][:, None]).squeeze(1)
    loss = torch.nn.functional.mse_loss(selected_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), GRADIENT_CLIP))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "critic_loss": float(loss.detach()),
        "critic_gradient_norm": gradient_norm,
        "selected_q_mean": float(selected_q.detach().mean()),
        "target_q_mean": float(target_q.detach().mean()),
    }


def actor_values(
    actor: Actor, critic: Critic, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = masked_log_softmax(
        actor(batch["obs_tokens"], batch["act_tokens"]), batch["action_mask"]
    )
    with torch.no_grad():
        q = critic(batch["obs_tokens"], batch["act_tokens"])
    advantage = center_under_policy(q, logp.exp(), batch["action_mask"])
    return logp, q, advantage


def actor_step(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    actor.train()
    logp, _, advantage = actor_values(actor, critic, batch)
    loss, diagnostics = awac_actor_loss(
        logp,
        batch["action_idx"],
        advantage,
        TEMPERATURE,
        WEIGHT_CLIP,
        stop_gradient_weight=True,
        action_mask=batch["action_mask"],
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "actor_loss": float(loss.detach()),
        "actor_gradient_norm": gradient_norm,
        **asdict(diagnostics),
    }


def update_target(target: Critic, critic: Critic) -> None:
    target.soft_update(critic, tau=TARGET_TAU)


def clean_awac_step(
    actor: Actor,
    critic: Critic,
    target: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    critic_metrics = critic_step(actor, critic, target, critic_optimizer, batch)
    actor_metrics = actor_step(actor, critic, actor_optimizer, batch)
    update_target(target, critic)
    return {**critic_metrics, **actor_metrics}


def split_instances() -> dict[str, list[InstanceConfig]]:
    return generate_pilot_split(
        seed_train=1000,
        seed_val=2000,
        seed_test=3000,
        n_train=16,
        n_val=8,
        n_test=8,
    )


def capability_macro(
    actor: Actor,
    vocab: Vocab,
    instances: Sequence[InstanceConfig],
    device: torch.device,
) -> dict[str, float]:
    rows = []
    for index, instance in enumerate(instances):
        row = evaluate_checkpoint(
            actor,
            vocab,
            [instance],
            device,
            n_eval_episodes=1,
            seed=10_000 + index,
            gamma=GAMMA,
            max_steps=MAX_STEPS,
            obs_max_len=OBS_MAX,
            act_max_len=ACT_MAX,
            deterministic=True,
        )
        rows.append({"instance_id": instance.instance_id, **row})
    keys = [key for key in rows[0] if key != "instance_id"]
    return {
        "macro": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "instances": rows,
        "evaluation": "one deterministic greedy evaluation per registered instance",
    }


def policy_by_state_time(
    actor: Actor, vocab: Vocab, instance: InstanceConfig, device: torch.device
) -> dict[tuple[str, int], dict[str, float]]:
    actor.eval()
    table: dict[tuple[str, int], dict[str, float]] = {}
    groups: dict[int, list[tuple[str, int, list[int], list[list[int]], list[str]]]] = {}
    env = SecureCourierEnv(instance, gamma=GAMMA, max_steps=MAX_STEPS)
    from secure_courier.state import EnvState

    for step in range(MAX_STEPS):
        for node in sorted(ALL_NODES):
            actions = [action for action, _ in GRAPH.get(node, [])]
            if not actions:
                continue
            env.restore_state(
                EnvState(
                    node=node,
                    step_count=step,
                    done=False,
                    terminal_success=False,
                    instance_id=instance.instance_id,
                )
            )
            observation = env._obs()
            structural = observation["admissible_structural"]
            if structural != actions:
                raise RuntimeError(f"action order mismatch at {node}, step {step}")
            groups.setdefault(len(structural), []).append(
                (
                    node,
                    step,
                    vocab.encode(observation["text"], OBS_MAX),
                    [vocab.encode(action, ACT_MAX) for action in observation["admissible_actions"]],
                    structural,
                )
            )
    with torch.no_grad():
        for records in groups.values():
            obs_tokens = torch.tensor([record[2] for record in records], dtype=torch.long, device=device)
            act_tokens = torch.tensor([record[3] for record in records], dtype=torch.long, device=device)
            probabilities = actor(obs_tokens, act_tokens).exp().cpu().numpy()
            for row_index, (node, step, _, __, structural) in enumerate(records):
                table[(node, step)] = {
                    action: float(probabilities[row_index, action_index])
                    for action_index, action in enumerate(structural)
                }
    return table


def exact_instance_metrics(
    actor: Actor, vocab: Vocab, instance: InstanceConfig, device: torch.device
) -> dict[str, Any]:
    table = policy_by_state_time(actor, vocab, instance, device)
    values = {(node, MAX_STEPS): 0.0 for node in ALL_NODES}
    q_values: dict[tuple[str, int, str], float] = {}
    for step in range(MAX_STEPS - 1, -1, -1):
        for node in sorted(ALL_NODES):
            actions = [action for action, _ in GRAPH.get(node, [])]
            if not actions:
                values[(node, step)] = 0.0
                continue
            value = 0.0
            for action in actions:
                next_node, reward, terminal = step_transition(node, action)
                continuation = 0.0 if terminal else values[(next_node, step + 1)]
                q_value = float(reward + GAMMA * continuation)
                q_values[(node, step, action)] = q_value
                value += table[(node, step)][action] * q_value
            values[(node, step)] = value
    junction_step = 8
    shortcut = q_values[("junction", junction_step, "take_shortcut")]
    safe = q_values[("junction", junction_step, "take_safe_route")]
    distribution = table[("junction", junction_step)]
    junction_value = sum(
        distribution[action] * q_values[("junction", junction_step, action)]
        for action in distribution
    )
    # Forward occupancy makes the route quantities unconditional from the full
    # start, unlike the junction probabilities above, which are conditional on
    # being at the junction at the canonical decision time.
    occupancy: dict[tuple[str, int], float] = {("depot", 0): 1.0}
    full_start_shortcut = 0.0
    full_start_safe = 0.0
    for step in range(MAX_STEPS):
        for node in sorted(ALL_NODES):
            mass = occupancy.get((node, step), 0.0)
            if mass == 0.0 or not GRAPH.get(node):
                continue
            for action, probability in table[(node, step)].items():
                flow = mass * probability
                if node == "junction" and action == "take_shortcut":
                    full_start_shortcut += flow
                elif node == "junction" and action == "take_safe_route":
                    full_start_safe += flow
                next_node, _, terminal = step_transition(node, action)
                if not terminal:
                    key = (next_node, step + 1)
                    occupancy[key] = occupancy.get(key, 0.0) + flow
    return {
        "instance_id": instance.instance_id,
        "J_exact": float(values[("depot", 0)]),
        "Q_shortcut": float(shortcut),
        "Q_safe": float(safe),
        "A_shortcut": float(shortcut - junction_value),
        "p_shortcut": float(distribution["take_shortcut"]),
        "p_safe": float(distribution["take_safe_route"]),
        "full_start_shortcut_route_probability": float(full_start_shortcut),
        "full_start_safe_route_probability": float(full_start_safe),
        "shortcut_rank_one": bool(shortcut > max(q_values[("junction", junction_step, a)] for a in distribution if a != "take_shortcut")),
    }


def state_action_values(
    actor: Actor,
    vocab: Vocab,
    instance: InstanceConfig,
    device: torch.device,
    node: str,
    step: int,
) -> dict[str, Any]:
    """Exact current-policy Q/A table at one structural state and elapsed step."""
    if step < 0 or step >= MAX_STEPS:
        raise ValueError(f"elapsed step outside finite horizon: {step}")
    solution = exact_policy_solution(actor, vocab, instance, device)
    policy = solution["policy"]
    values = solution["V"]
    q_values = solution["Q"]
    actions = [action for action, _ in GRAPH.get(node, [])]
    return {
        "node": node,
        "step": step,
        "V": float(values[(node, step)]),
        "policy": policy[(node, step)],
        "Q": {action: float(q_values[(node, step, action)]) for action in actions},
        "A": {
            action: float(q_values[(node, step, action)] - values[(node, step)])
            for action in actions
        },
    }


def exact_policy_solution(
    actor: Actor, vocab: Vocab, instance: InstanceConfig, device: torch.device
) -> dict[str, Any]:
    """Return the complete finite-horizon current-policy solution for one surface."""
    policy = policy_by_state_time(actor, vocab, instance, device)
    values = {(state, MAX_STEPS): 0.0 for state in ALL_NODES}
    q_values: dict[tuple[str, int, str], float] = {}
    for elapsed in range(MAX_STEPS - 1, -1, -1):
        for state in sorted(ALL_NODES):
            actions = [action for action, _ in GRAPH.get(state, [])]
            if not actions:
                values[(state, elapsed)] = 0.0
                continue
            value = 0.0
            for action in actions:
                next_node, reward, terminal = step_transition(state, action)
                continuation = 0.0 if terminal else values[(next_node, elapsed + 1)]
                q_value = float(reward + GAMMA * continuation)
                q_values[(state, elapsed, action)] = q_value
                value += policy[(state, elapsed)][action] * q_value
            values[(state, elapsed)] = value
    return {"policy": policy, "V": values, "Q": q_values}


def learned_state_action_values(
    actor: Actor,
    critic: Critic,
    vocab: Vocab,
    instance: InstanceConfig,
    device: torch.device,
    node: str = "junction",
    step: int = 8,
) -> dict[str, Any]:
    """Native learned critic quantities using the M2 actor-facing centering."""
    from secure_courier.state import EnvState

    env = SecureCourierEnv(instance, gamma=GAMMA, max_steps=MAX_STEPS)
    env.restore_state(
        EnvState(
            node=node,
            step_count=step,
            done=False,
            terminal_success=False,
            instance_id=instance.instance_id,
        )
    )
    observation = env._obs()
    structural = observation["admissible_structural"]
    obs_tokens = torch.tensor(
        [vocab.encode(observation["text"], OBS_MAX)], dtype=torch.long, device=device
    )
    act_tokens = torch.tensor(
        [[vocab.encode(action, ACT_MAX) for action in observation["admissible_actions"]]],
        dtype=torch.long,
        device=device,
    )
    actor.eval()
    critic.eval()
    with torch.no_grad():
        logp = actor(obs_tokens, act_tokens)[0]
        probabilities = logp.exp()
        q = critic(obs_tokens, act_tokens)[0]
        value = (probabilities * q).sum()
        advantage = q - value
    return {
        "node": node,
        "step": step,
        "policy": {action: float(probabilities[index]) for index, action in enumerate(structural)},
        "Q": {action: float(q[index]) for index, action in enumerate(structural)},
        "A": {action: float(advantage[index]) for index, action in enumerate(structural)},
        "rank": sorted(structural, key=lambda action: (-float(q[structural.index(action)]), action)),
    }


def structural_eligibility(
    old_actor: Actor,
    current_actor: Actor,
    vocab: Vocab,
    instances: Sequence[InstanceConfig],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for instance in instances:
        old = exact_instance_metrics(old_actor, vocab, instance, device)
        current = exact_instance_metrics(current_actor, vocab, instance, device)
        rows.append(
            {
                "instance_id": instance.instance_id,
                "old": old,
                "current": current,
                "delta_Q_shortcut": current["Q_shortcut"] - old["Q_shortcut"],
                "true_sign_flip": old["A_shortcut"] < -0.02 and current["A_shortcut"] > 0.02,
                "true_rank_flip": (not old["shortcut_rank_one"]) and current["shortcut_rank_one"],
            }
        )
    summary = {
        "mean_delta_Q_shortcut": float(np.mean([row["delta_Q_shortcut"] for row in rows])),
        "mean_old_A_shortcut": float(np.mean([row["old"]["A_shortcut"] for row in rows])),
        "mean_current_A_shortcut": float(np.mean([row["current"]["A_shortcut"] for row in rows])),
        "sign_flip_fraction": float(np.mean([row["true_sign_flip"] for row in rows])),
        "rank_flip_fraction": float(np.mean([row["true_rank_flip"] for row in rows])),
    }
    gates = {
        "mean_delta_Q_shortcut_ge_0.20": summary["mean_delta_Q_shortcut"] >= 0.20,
        "mean_old_A_shortcut_lt_-0.02": summary["mean_old_A_shortcut"] < -0.02,
        "mean_current_A_shortcut_gt_0.02": summary["mean_current_A_shortcut"] > 0.02,
        "sign_and_rank_flip_fraction_ge_0.90": min(
            summary["sign_flip_fraction"], summary["rank_flip_fraction"]
        )
        >= 0.90,
    }
    return {
        "status": "ELIGIBLE" if all(gates.values()) else "INELIGIBLE",
        "selection_uses_learned_critic": False,
        "summary": summary,
        "gates": gates,
        "instances": rows,
    }


def collect_assisted_episode(
    actor: Actor,
    vocab: Vocab,
    instances: Sequence[InstanceConfig],
    device: torch.device,
    rng: random.Random,
    cycle: int,
    curriculum_probability: float = 0.8,
    gate_epsilon: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance = rng.choice(list(instances))
    env = SecureCourierEnv(instance, gamma=GAMMA, max_steps=MAX_STEPS)
    env.reset()
    target_node = None
    navigation_steps = 0
    if rng.random() < curriculum_probability:
        target_node = rng.choice(CURRICULUM_NODES)
        for action in oracle_shortcut_plan():
            if env.current_node() == target_node:
                break
            if action in env.admissible_structural():
                env.execute_structural(action)
                navigation_steps += 1
            if env.is_done():
                break

    rows: list[dict[str, Any]] = []
    while not env.is_done() and env.step_count() < MAX_STEPS:
        state = env.clone_state()
        observation = env._obs()
        structural = observation["admissible_structural"]
        if not structural:
            break
        obs_tokens = torch.tensor(
            [vocab.encode(observation["text"], OBS_MAX)], dtype=torch.long, device=device
        )
        act_tokens = torch.tensor(
            [[vocab.encode(action, ACT_MAX) for action in observation["admissible_actions"]]],
            dtype=torch.long,
            device=device,
        )
        actor.eval()
        with torch.no_grad():
            logp = actor(obs_tokens, act_tokens)[0]
        index = int(torch.multinomial(logp.exp(), 1).item())
        forced = False
        if state.node in GATE_ACTION and rng.random() < gate_epsilon:
            index = structural.index(GATE_ACTION[state.node])
            forced = True
        actor.train()
        action = structural[index]
        _, reward, _, _, _ = env.step(index)
        next_observation = env._obs()
        next_state = env.clone_state()
        rows.append(
            {
                "transition_id": f"source_c{cycle}_t{len(rows)}",
                "instance_id": instance.instance_id,
                "instance_seed": instance.instance_seed,
                "episode_id": f"source_c{cycle}",
                "t": len(rows),
                "observation": observation["text"],
                "admissible_actions": observation["admissible_actions"],
                "admissible_structural": structural,
                "action": index,
                "action_display": observation["admissible_actions"][index],
                "action_structural": action,
                "reward": float(reward),
                "next_observation": next_observation["text"],
                "next_admissible_actions": next_observation["admissible_actions"],
                "next_admissible_structural": next_observation["admissible_structural"],
                "done": env.is_done(),
                "behavior_policy_id": "behavior_assisted_source",
                "behavior_log_prob": float(logp[index]),
                "gate_action_forced": forced,
                "route_label": "source_online",
                "full_state": clean(state.__dict__),
                "next_state": clean(next_state.__dict__),
            }
        )
    return rows, {
        "cycle": cycle,
        "instance_id": instance.instance_id,
        "curriculum_target": target_node,
        "curriculum_navigation_steps": navigation_steps,
        "collected_transitions": len(rows),
        "terminal_success": bool(rows and rows[-1]["reward"] > 0),
        "forced_gate_actions": sum(row["gate_action_forced"] for row in rows),
    }


def save_model_bundle(
    path: Path,
    actor: Actor,
    critic: Critic,
    target: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    vocab: Vocab,
    metadata: dict[str, Any],
    rng: random.Random,
    device: torch.device,
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    with (path / "vocab.pkl").open("wb") as handle:
        pickle.dump(vocab, handle)
    payload = {
        "actor": copy.deepcopy(actor.state_dict()),
        "critic": copy.deepcopy(critic.state_dict()),
        "target": copy.deepcopy(target.state_dict()),
        "actor_optimizer": copy.deepcopy(actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(critic_optimizer.state_dict()),
        "python_rng": rng.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "metadata": metadata,
    }
    torch.save(payload, path / "state.pt")
    write_json(
        path / "manifest.json",
        {
            **metadata,
            "actor_hash": module_hash(actor),
            "critic_hash": module_hash(critic),
            "target_hash": module_hash(target),
            "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
            "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
            "state_sha256": sha256(path / "state.pt"),
            "vocab_sha256": sha256(path / "vocab.pkl"),
        },
    )


def load_model_bundle(
    path: Path, device: torch.device
) -> tuple[Actor, Critic, Critic, torch.optim.Optimizer, torch.optim.Optimizer, Vocab, dict[str, Any], random.Random]:
    with (path / "vocab.pkl").open("rb") as handle:
        vocab = pickle.load(handle)
    actor = Actor(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    critic = Critic(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    target = Critic(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
    # Always deserialize the bundle on CPU.  Loading the whole object directly
    # onto CUDA also moves the CPU RNG byte tensor, which makes
    # torch.set_rng_state reject an otherwise valid resumable checkpoint.
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    target.load_state_dict(payload["target"])
    actor_optimizer.load_state_dict(payload["actor_optimizer"])
    critic_optimizer.load_state_dict(payload["critic_optimizer"])
    rng = random.Random()
    rng.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if device.type == "cuda" and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    return (
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        payload["metadata"],
        rng,
    )
