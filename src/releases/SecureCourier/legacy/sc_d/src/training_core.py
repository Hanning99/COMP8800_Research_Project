"""Scorer-free SecureCourier training core copied from the sealed SC-C semantics."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from scd_common import (
    ACT_MAX,
    GAMMA,
    GRADIENT_CLIP,
    MAX_STEPS,
    OBS_MAX,
    PARENT,
    TARGET_TAU,
    clean,
    module_hash,
    sha256,
    tensor_tree_hash,
)


VENDOR = PARENT / "src/vendor/diy_task_set"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(PARENT / "src/vendor"))
from agents.actor import Actor  # noqa: E402
from agents.critic import Critic  # noqa: E402
from agents.text_encoder import Vocab  # noqa: E402
from pdr.masks.action_mask import masked_log_softmax  # noqa: E402
from secure_courier.env import SecureCourierEnv  # noqa: E402
from secure_courier.generator import generate_pilot_split  # noqa: E402
from secure_courier.task_graph import GATE_SEQUENCE, ORACLE_SHORTCUT_ACTIONS  # noqa: E402


EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 3e-4
OFFLINE_BATCH = 6
ONLINE_BATCH = 58
ONLINE_CAPACITY = 20_000
CURRICULUM_NODES = ("sc5", "sc4", "sc3", "sc2")
GATE_ACTION = {"gate_lobby": GATE_SEQUENCE[0], "gate_badged": GATE_SEQUENCE[1], "gate_coded": GATE_SEQUENCE[2]}


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
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


def split_instances():
    return generate_pilot_split(seed_train=1000, seed_val=2000, seed_test=3000, n_train=16, n_val=8, n_test=8)


def encode_transition(row: dict[str, Any], vocab: Vocab, device: torch.device) -> dict[str, torch.Tensor]:
    current = [vocab.encode(action, ACT_MAX) for action in row["admissible_actions"]]
    next_actions = [vocab.encode(action, ACT_MAX) for action in row["next_admissible_actions"]]
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


def clean_collate(rows: Sequence[dict[str, Any]], vocab: Vocab, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = [encode_transition(row, vocab, device) for row in rows]
    max_current = max(item["act_tokens"].shape[0] for item in encoded)
    max_next = max(item["next_act_tokens"].shape[0] for item in encoded)

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        if value.shape[0] == count:
            return value
        padding = torch.zeros((count - value.shape[0], value.shape[1]), dtype=value.dtype, device=device)
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


def critic_step(actor, critic, target, optimizer, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    actor.eval()
    critic.train()
    target.eval()
    with torch.no_grad():
        next_q = target(batch["next_obs_tokens"], batch["next_act_tokens"])
        next_logp = masked_log_softmax(actor(batch["next_obs_tokens"], batch["next_act_tokens"]), batch["next_action_mask"])
        next_value = (
            next_logp.exp() * torch.where(batch["next_action_mask"], next_q, torch.zeros_like(next_q))
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


def update_target(target, critic) -> None:
    target.soft_update(critic, tau=TARGET_TAU)


def collect_assisted_episode(actor, vocab, instances, device, rng: random.Random, cycle: int):
    instance = rng.choice(list(instances))
    env = SecureCourierEnv(instance, gamma=GAMMA, max_steps=MAX_STEPS)
    env.reset()
    target_node = None
    navigation_steps = 0
    if rng.random() < 0.8:
        target_node = rng.choice(CURRICULUM_NODES)
        for action in ORACLE_SHORTCUT_ACTIONS:
            if env.current_node() == target_node:
                break
            if action in env.admissible_structural():
                env.execute_structural(action)
                navigation_steps += 1
            if env.is_done():
                break
    rows = []
    while not env.is_done() and env.step_count() < MAX_STEPS:
        state = env.clone_state()
        observation = env._obs()
        structural = observation["admissible_structural"]
        if not structural:
            break
        obs_tokens = torch.tensor([vocab.encode(observation["text"], OBS_MAX)], dtype=torch.long, device=device)
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
        if state.node in GATE_ACTION and rng.random() < 1.0:
            index = structural.index(GATE_ACTION[state.node])
            forced = True
        actor.train()
        action = structural[index]
        _, reward, _, _, _ = env.step(index)
        next_observation = env._obs()
        next_state = env.clone_state()
        rows.append(
            {
                "transition_id": f"continuation_c{cycle}_t{len(rows)}",
                "instance_id": instance.instance_id,
                "instance_seed": instance.instance_seed,
                "episode_id": f"continuation_c{cycle}",
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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def base_rows(source: Path):
    selected = json.loads((source / "capability_checkpoint/manifest.json").read_text())
    offline = read_jsonl(source / "offline_dataset.jsonl")
    online = read_jsonl(source / "acquisition_trace/online_stream.jsonl", int(selected["online_stream_rows"]))
    rows = []
    for origin, values in (("offline", offline), ("online", online)):
        for index, row in enumerate(values):
            item = dict(row)
            item["_scb_uid"] = f"{origin}:{index:06d}:{row.get('transition_id', 'missing')}"
            item["_scb_origin"] = origin
            rows.append(item)
    return rows


def load_model_bundle(path: Path, device: torch.device):
    with (path / "vocab.pkl").open("rb") as handle:
        vocab = pickle.load(handle)
    actor = Actor(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    critic = Critic(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    target = Critic(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
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
    return actor, critic, target, actor_optimizer, critic_optimizer, vocab, payload["metadata"], rng


def save_model_bundle(path: Path, actor, critic, target, actor_optimizer, critic_optimizer, vocab, metadata, rng, device):
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
    manifest = {
        **metadata,
        "actor_hash": module_hash(actor),
        "critic_hash": module_hash(critic),
        "target_hash": module_hash(target),
        "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
        "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
        "state_sha256": sha256(path / "state.pt"),
        "vocab_sha256": sha256(path / "vocab.pkl"),
    }
    (path / "manifest.json").write_text(json.dumps(clean(manifest), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), sort_keys=True, allow_nan=False) + "\n"
    with path.open("a") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return handle.tell()
