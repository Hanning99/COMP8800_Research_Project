#!/usr/bin/env python3
"""Exact-Qold KeyCorridor RQ1 mechanism experiment.

The six immutable full-task graphs and source policies are read-only inputs.
Fresh evidence consists only of preregistered paired learner schedules.  No
online transition is collected and no learned Qold is trained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import resource
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj


ROOT = Path(__file__).resolve().parents[1] / 'inputs/key_exact'
WORKSPACE = ROOT.parents[2]
LEGACY = Path(__file__).resolve().parents[1] / 'inputs/key'
FAMILY = LEGACY
sys.path.insert(0, str(LEGACY))
import run_replication as base  # noqa: E402


TASKS = {
    "S3R1": ("BabyAI-KeyCorridorS3R1-v0", 270, FAMILY / "S3R1"),
    "S3R2": ("BabyAI-KeyCorridorS3R2-v0", 270, FAMILY / "S3R2"),
    "S3R3": ("BabyAI-KeyCorridorS3R3-v0", 270, FAMILY / "S3R3"),
    "S4R3": ("BabyAI-KeyCorridorS4R3-v0", 480, LEGACY / 'S4R3'),
    "S5R3": ("BabyAI-KeyCorridorS5R3-v0", 750, FAMILY / "S5R3"),
    "S6R3": ("BabyAI-KeyCorridorS6R3-v0", 1080, FAMILY / "S6R3"),
}
ARMS = ["A0_STANDARD_TD", "A1_STATIC_EXACT_QOLD", "A2_FRESH_PDR"]
MASTER_SEED = 20260902
TOTAL_UPDATES = 512
EARLY_UPDATES = 128
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
DIAGNOSTIC_CHECKPOINTS = [32, 128, 256]
GAMMA = 0.97
BETA = 0.3
WEIGHT_CLIP = 20.0
GRAD_CLIP = 5.0
LEARNING_RATE = 3e-4
TARGET_TAU = 0.005
BATCH_SIZE = 64
DEMO_ROWS = 58
UNIFORM_ROWS = 6
QOLD_TOLERANCE = 5e-6
PDR_TOLERANCE = 5e-6
POLICY_TOLERANCE = 5e-7
BOOTSTRAP_REPLICATES = 10_000
DEVICE = torch.device(os.environ.get("KEYCORRIDOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_paths(task: str):
    env_id, horizon, source = TASKS[task]
    return {
        "env_id": env_id,
        "horizon": horizon,
        "source": source,
        "graph": source / "domain" / "graph.npz",
        "states": source / "domain" / "states.pkl",
        "qold": source / "domain" / "Q_OLD_V3.npy",
        "common": source / "COMMON_START_V3.pt",
    }


def load_graph(task: str):
    return dict(np.load(source_paths(task)["graph"]))


def load_common(task: str, map_location="cpu"):
    return torch.load(source_paths(task)["common"], map_location=map_location, weights_only=False)


def family_restore(unwrapped, state, step_count=0):
    unwrapped.grid = Grid.decode(
        np.frombuffer(state[0], dtype=np.uint8).reshape(unwrapped.width, unwrapped.height, 3).copy()
    )[0]
    unwrapped.agent_pos = state[1]
    unwrapped.agent_dir = state[2]
    unwrapped.carrying = None
    unwrapped.step_count = int(step_count)
    if state[3] is not None:
        unwrapped.carrying = WorldObj.decode(*state[3])
        unwrapped.carrying.cur_pos = np.array([-1, -1])
    unwrapped.instrs.reset_verifier(unwrapped)


def actor_policy(actor, features, mask, dtype=torch.float32):
    actor.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), 8192):
            output.append(actor(features[start : start + 8192], mask[start : start + 8192]).exp())
    policy = torch.cat(output).to(dtype)
    outside = torch.where(mask, torch.zeros_like(policy), policy).abs().max().item()
    if outside != 0.0:
        raise RuntimeError(f"ACTOR_OUTSIDE_LEGAL_MASK:{outside}")
    return policy


def exact_metrics(policy, graph, horizon):
    successor = torch.as_tensor(np.maximum(graph["next"], 0), dtype=torch.long, device=DEVICE)
    mask = torch.as_tensor(graph["mask"], dtype=torch.bool, device=DEVICE)
    terminal = torch.as_tensor(graph["terminal"], dtype=torch.bool, device=DEVICE)
    key = torch.as_tensor(graph["key"], dtype=torch.float32, device=DEVICE)
    unlock = torch.as_tensor(graph["unlock"], dtype=torch.float32, device=DEVICE)
    next_return = torch.zeros(len(policy), dtype=torch.float32, device=DEVICE)
    next_success = torch.zeros_like(next_return)
    next_key = key.clone()
    next_unlock = unlock.clone()
    for time_index in range(horizon - 1, -1, -1):
        q_return = torch.zeros((len(policy), 7), dtype=torch.float32, device=DEVICE)
        q_success = torch.zeros_like(q_return)
        q_key = torch.zeros_like(q_return)
        q_unlock = torch.zeros_like(q_return)
        q_return[terminal] = 1.0 - 0.9 * (time_index + 1) / horizon
        q_success[terminal] = 1.0
        if time_index + 1 < horizon:
            nonterminal = mask & ~terminal
            q_return[nonterminal] = GAMMA * next_return[successor[nonterminal]]
            q_success[nonterminal] = next_success[successor[nonterminal]]
            q_key[nonterminal] = next_key[successor[nonterminal]]
            q_unlock[nonterminal] = next_unlock[successor[nonterminal]]
        next_return = (policy * q_return).sum(1)
        next_success = (policy * q_success).sum(1)
        next_key = torch.maximum(key, (policy * q_key).sum(1))
        next_unlock = torch.maximum(unlock, (policy * q_unlock).sum(1))
    initial = int(graph["initial"])
    return {
        "return": float(next_return[initial]),
        "success": float(next_success[initial]),
        "key_capability": float(next_key[initial]),
        "unlock_capability": float(next_unlock[initial]),
    }


def demonstration(graph):
    return base.demonstration(graph)


def build_schedule(task: str, seed: int, updates=TOTAL_UPDATES):
    graph = load_graph(task)
    horizon = TASKS[task][1]
    rng = np.random.default_rng(seed + 700_000)
    demos = demonstration(graph)
    states = np.empty((updates, BATCH_SIZE), np.int32)
    times = np.empty_like(states)
    actions = np.empty((updates, BATCH_SIZE), np.int8)
    for update in range(updates):
        for index in range(DEMO_ROWS):
            row = demos[int(rng.integers(len(demos)))]
            states[update, index], times[update, index], actions[update, index] = map(int, row[:3])
        for index in range(DEMO_ROWS, BATCH_SIZE):
            state = int(rng.integers(len(graph["features"])))
            time_index = int(rng.integers(min(int(graph["min_depth"][state]), horizon - 1), horizon))
            action = int(rng.choice(np.flatnonzero(graph["mask"][state])))
            states[update, index] = state
            times[update, index] = time_index
            actions[update, index] = action
    return states, times, actions


def schedule_path(task: str, seed: int):
    return ROOT / "preregistration" / "schedules" / task / f"seed_{seed}.npz"


def batch_tensors(graph, horizon, qold, states, times, actions):
    successor_np = np.maximum(graph["next"][states, actions], 0)
    terminal_np = graph["terminal"][states, actions]
    done_np = terminal_np | (times + 1 >= horizon)
    reward_np = terminal_np * (1.0 - 0.9 * (times + 1) / horizon)
    next_times = np.minimum(times + 1, horizon - 1)
    return {
        "s_np": states,
        "t_np": times,
        "a_np": actions,
        "ns_np": successor_np,
        "done_np": done_np,
        "s": torch.as_tensor(states, dtype=torch.long, device=DEVICE),
        "t": torch.as_tensor(times, dtype=torch.long, device=DEVICE),
        "a": torch.as_tensor(actions, dtype=torch.long, device=DEVICE),
        "ns": torch.as_tensor(successor_np, dtype=torch.long, device=DEVICE),
        "x": torch.as_tensor(graph["features"][states], dtype=torch.float32, device=DEVICE),
        "mask": torch.as_tensor(graph["mask"][states], dtype=torch.bool, device=DEVICE),
        "tf": torch.as_tensor(times / horizon, dtype=torch.float32, device=DEVICE),
        "nx": torch.as_tensor(graph["features"][successor_np], dtype=torch.float32, device=DEVICE),
        "nmask": torch.as_tensor(graph["mask"][successor_np], dtype=torch.bool, device=DEVICE),
        "ntf": torch.as_tensor(next_times / horizon, dtype=torch.float32, device=DEVICE),
        "done": torch.as_tensor(done_np, dtype=torch.float32, device=DEVICE),
        "reward": torch.as_tensor(reward_np, dtype=torch.float32, device=DEVICE),
        "qold": torch.as_tensor(np.array(qold[times, states], copy=True), dtype=torch.float32, device=DEVICE),
        "qold_next": torch.as_tensor(
            np.array(qold[next_times, successor_np], copy=True), dtype=torch.float32, device=DEVICE
        ),
    }


def initialize_models(task: str, graph, seed: int):
    common = load_common(task)
    actor = base.Actor(graph["features"].shape[1]).to(DEVICE)
    actor.load_state_dict(common["actor"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    actor_optimizer.load_state_dict(copy.deepcopy(common["actor_opt"]))
    torch.manual_seed(seed + 50_000)
    critic = base.Critic(graph["features"].shape[1]).to(DEVICE)
    with torch.no_grad():
        critic.net[-1].weight.zero_()
        critic.net[-1].bias.zero_()
    target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    return common, actor, actor_optimizer, critic, target, critic_optimizer


def critic_step(actor, critic, target, optimizer, batch):
    critic.train()
    target.eval()
    with torch.no_grad():
        next_policy = actor(batch["nx"], batch["nmask"]).exp()
        next_q = batch["qold_next"] + target(batch["nx"], batch["ntf"])
        target_q = batch["reward"] + GAMMA * (1.0 - batch["done"]) * (next_policy * next_q).sum(1)
    live_q = batch["qold"] + critic(batch["x"], batch["tf"])
    prediction = live_q.gather(1, batch["a"][:, None]).squeeze(1)
    loss = F.mse_loss(prediction, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP))
    optimizer.step()
    return float(loss), gradient_norm


def actor_loss(actor, batch, q):
    log_policy = actor(batch["x"], batch["mask"])
    policy = log_policy.exp()
    advantage = q.detach() - (policy * q.detach()).sum(1, keepdim=True)
    weights = torch.clamp(torch.exp(advantage / BETA), 0.0, WEIGHT_CLIP)
    selected = batch["a"][:, None]
    loss = -(weights.gather(1, selected).squeeze(1) * log_policy.gather(1, selected).squeeze(1)).mean()
    return loss, advantage, weights


def actor_step(actor, optimizer, batch, q):
    loss, _, weights = actor_loss(actor, batch, q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP))
    optimizer.step()
    return float(loss), gradient_norm, float(weights.mean()), float((weights >= WEIGHT_CLIP).float().mean())


def gradient_vector(actor, batch, q):
    actor.zero_grad(set_to_none=True)
    loss, advantage, weights = actor_loss(actor, batch, q)
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()))
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).detach(), advantage.detach(), weights.detach()


def dependency_counts(graph, horizon):
    successor = np.maximum(graph["next"], 0)
    legal_counts = graph["mask"].sum(1).astype(np.int64)
    nonterminal = graph["mask"] & ~graph["terminal"]
    by_action = np.zeros_like(graph["next"], dtype=np.int64)
    by_action[nonterminal] = legal_counts[successor[nonterminal]]
    one_pass = (horizon - 1) * int(by_action.sum())
    return by_action, one_pass


def pdr_batch_q(actor, graph, horizon, qold_device, pi_old, batch):
    features = torch.as_tensor(graph["features"], dtype=torch.float32, device=DEVICE)
    mask = torch.as_tensor(graph["mask"], dtype=torch.bool, device=DEVICE)
    successor = torch.as_tensor(np.maximum(graph["next"], 0), dtype=torch.long, device=DEVICE)
    terminal = torch.as_tensor(graph["terminal"], dtype=torch.bool, device=DEVICE)
    nonterminal = mask & ~terminal
    pi_current = actor_policy(actor, features, mask)
    next_d_value = torch.zeros(len(features), dtype=torch.float32, device=DEVICE)
    output = torch.zeros((len(batch["s"]), 7), dtype=torch.float32, device=DEVICE)
    times = batch["t_np"]
    groups = {int(t): np.flatnonzero(times == t) for t in np.unique(times)}
    for time_index in range(horizon - 1, -1, -1):
        d = torch.zeros((len(features), 7), dtype=torch.float32, device=DEVICE)
        if time_index + 1 < horizon:
            source = ((pi_current - pi_old) * qold_device[time_index + 1]).sum(1)
            right = GAMMA * (source + next_d_value)[successor]
            d[nonterminal] = right[nonterminal]
        if time_index in groups:
            indices = groups[time_index]
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
            states = batch["s"][index_tensor]
            output[index_tensor] = qold_device[time_index, states] + d[states]
        next_d_value = (pi_current * d).sum(1)
    return output, pi_current


def exact_current_selected(policy, graph, horizon, sample_states, sample_times):
    successor = torch.as_tensor(np.maximum(graph["next"], 0), dtype=torch.long, device=DEVICE)
    mask = torch.as_tensor(graph["mask"], dtype=torch.bool, device=DEVICE)
    terminal = torch.as_tensor(graph["terminal"], dtype=torch.bool, device=DEVICE)
    nonterminal = mask & ~terminal
    next_value = torch.zeros(len(policy), dtype=torch.float32, device=DEVICE)
    selected = torch.zeros((len(sample_states), 7), dtype=torch.float32, device=DEVICE)
    groups = {int(t): np.flatnonzero(sample_times == t) for t in np.unique(sample_times)}
    for time_index in range(horizon - 1, -1, -1):
        q = torch.zeros((len(policy), 7), dtype=torch.float32, device=DEVICE)
        q[terminal] = 1.0 - 0.9 * (time_index + 1) / horizon
        if time_index + 1 < horizon:
            q[nonterminal] = GAMMA * next_value[successor[nonterminal]]
        if time_index in groups:
            indices = groups[time_index]
            selected[indices] = q[torch.as_tensor(sample_states[indices], dtype=torch.long, device=DEVICE)]
        next_value = (policy * q).sum(1)
    return selected


def diagnostic_signal(arm, actor, critic, graph, horizon, qold, qold_device, pi_old, batch):
    features = torch.as_tensor(graph["features"], dtype=torch.float32, device=DEVICE)
    mask = torch.as_tensor(graph["mask"], dtype=torch.bool, device=DEVICE)
    current_policy = actor_policy(actor, features, mask)
    oracle_q = exact_current_selected(
        current_policy, graph, horizon, batch["s_np"], batch["t_np"]
    )
    if arm == "A0_STANDARD_TD":
        with torch.no_grad():
            actor_q = batch["qold"] + critic(batch["x"], batch["tf"])
    elif arm == "A1_STATIC_EXACT_QOLD":
        actor_q = batch["qold"]
    else:
        actor_q, _ = pdr_batch_q(actor, graph, horizon, qold_device, pi_old, batch)
    local_policy = current_policy[batch["s"]]
    actor_advantage = actor_q - (local_policy * actor_q).sum(1, keepdim=True)
    oracle_advantage = oracle_q - (local_policy * oracle_q).sum(1, keepdim=True)
    actor_gradient, _, actor_weights = gradient_vector(actor, batch, actor_q)
    oracle_gradient, _, oracle_weights = gradient_vector(actor, batch, oracle_q)
    legal = batch["mask"]
    q_difference = (actor_q - oracle_q)[legal]
    advantage_difference = (actor_advantage - oracle_advantage)[legal]
    gradient_norm = torch.linalg.vector_norm(oracle_gradient)
    return {
        "q_mae_to_current_policy": float(q_difference.abs().mean()),
        "q_max_abs_error_to_current_policy": float(q_difference.abs().max()),
        "centered_advantage_mae": float(advantage_difference.abs().mean()),
        "centered_advantage_max_abs_error": float(advantage_difference.abs().max()),
        "awac_weight_mae": float((actor_weights - oracle_weights)[legal].abs().mean()),
        "actor_gradient_cosine": float(F.cosine_similarity(actor_gradient[None], oracle_gradient[None]).item()),
        "actor_gradient_relative_l2": float(torch.linalg.vector_norm(actor_gradient - oracle_gradient) / (gradient_norm + 1e-12)),
    }


def prelock() -> None:
    prereg = ROOT / "preregistration"
    prereg.mkdir(parents=True, exist_ok=True)
    seed_path = prereg / "SEED_PRELOCK.json"
    if seed_path.exists():
        raise RuntimeError("SEED_PRELOCK_ALREADY_EXISTS")
    legacy_seeds = {
        1000,
        314159,
        271828,
        161803,
        141421,
        173205,
        314160,
        271829,
        161804,
        141422,
        173206,
        2054730081,
        1668083863,
        1381560984,
        71539991,
        729512858,
        133347924,
        1473050824,
        100270762,
        1099550647,
        283212493,
    }
    rng = np.random.default_rng(MASTER_SEED)
    selected = set()
    per_task = {}
    for task in TASKS:
        values = []
        while len(values) < 30:
            candidate = int(rng.integers(1, 2**31 - 1))
            if candidate in legacy_seeds or candidate in selected:
                continue
            selected.add(candidate)
            values.append(candidate)
        per_task[task] = {"confirmatory": values[:20], "reserve": values[20:]}
    all_fresh = [seed for task in per_task.values() for group in task.values() for seed in group]
    write_json(
        seed_path,
        {
            "status": "RQ1_FRESH_SEEDS_PRELOCKED",
            "master_seed": MASTER_SEED,
            "task_order": list(TASKS),
            "per_task": per_task,
            "fresh_seed_count": len(all_fresh),
            "fresh_unique_count": len(set(all_fresh)),
            "fresh_legacy_overlap": sorted(set(all_fresh) & legacy_seeds),
            "legacy_development_exclusions": sorted(legacy_seeds),
            "statistical_unit": "paired learner seed conditional on one frozen full-task world per named task",
            "replacement_rule": "ordered reserve only for pre-method input/gate failure; never for method outcome or runtime failure",
        },
    )
    if len(all_fresh) != len(set(all_fresh)) or set(all_fresh) & legacy_seeds:
        raise RuntimeError("SEED_PRELOCK_OVERLAP")


def prepare_schedules() -> None:
    prelock_data = json.loads((ROOT / "preregistration" / "SEED_PRELOCK.json").read_text())
    rows = []
    for task in TASKS:
        seeds = prelock_data["per_task"][task]["confirmatory"]
        for seed in seeds:
            path = schedule_path(task, seed)
            if path.exists():
                raise RuntimeError(f"SCHEDULE_EXISTS:{path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            states, times, actions = build_schedule(task, seed)
            np.savez_compressed(path, states=states, times=times, actions=actions)
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "rows": int(states.size),
                    "demo_rows_per_batch": DEMO_ROWS,
                    "uniform_rows_per_batch": UNIFORM_ROWS,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "preregistration" / "SCHEDULE_MANIFEST.csv", index=False)


def restore_audit(task, graph, cases=1024):
    paths = source_paths(task)
    with paths["states"].open("rb") as stream:
        states = pickle.load(stream)
    env = gym.make(paths["env_id"])
    env.reset(seed=1000)
    unwrapped = env.unwrapped
    rng = random.Random(MASTER_SEED + list(TASKS).index(task))
    passed = 0
    for _ in range(cases):
        state_id = rng.randrange(len(states))
        action = rng.choice(np.flatnonzero(graph["mask"][state_id]).tolist())
        step_count = rng.randrange(paths["horizon"])
        state = states[state_id]
        family_restore(unwrapped, state, step_count)
        observation_a = unwrapped.gen_obs()["image"].copy()
        transition_a = unwrapped.step(action)
        post_a = base.snapshot(unwrapped)
        family_restore(unwrapped, state, step_count)
        observation_b = unwrapped.gen_obs()["image"].copy()
        transition_b = unwrapped.step(action)
        post_b = base.snapshot(unwrapped)
        okay = (
            np.array_equal(observation_a, observation_b)
            and float(transition_a[1]) == float(transition_b[1])
            and transition_a[2:4] == transition_b[2:4]
            and post_a == post_b
        )
        passed += int(okay)
    env.close()
    return {"cases": cases, "passed": passed, "status": "PASS" if passed == cases else "FAIL"}


def qold_bellman_audit(task, graph, common, qold):
    horizon = TASKS[task][1]
    policy = np.asarray(common["pi_old"], dtype=np.float64)
    mask = graph["mask"]
    terminal = graph["terminal"]
    successor = np.maximum(graph["next"], 0)
    nonterminal = mask & ~terminal
    maximum = 0.0
    terminal_maximum = 0.0
    next_value = np.zeros(len(policy), np.float64)
    for time_index in range(horizon - 1, -1, -1):
        observed = np.asarray(qold[time_index], dtype=np.float64)
        expected = np.zeros_like(observed)
        expected[terminal] = 1.0 - 0.9 * (time_index + 1) / horizon
        if time_index + 1 < horizon:
            expected[nonterminal] = GAMMA * next_value[successor[nonterminal]]
        difference = observed - expected
        maximum = max(maximum, float(np.abs(difference[mask]).max()))
        terminal_maximum = max(terminal_maximum, float(np.abs(difference[terminal]).max()))
        next_value = (policy * observed).sum(1)
    return maximum, terminal_maximum


def gate(task: str) -> None:
    output = ROOT / "preflight" / task
    if output.exists():
        raise RuntimeError(f"PREFLIGHT_EXISTS:{output}")
    output.mkdir(parents=True)
    paths = source_paths(task)
    graph = load_graph(task)
    common = load_common(task)
    qold = np.load(paths["qold"], mmap_mode="r")
    horizon = paths["horizon"]
    mask = graph["mask"]
    terminal = graph["terminal"]
    next_state = graph["next"]
    closure_valid = bool(
        np.all(next_state[mask & ~terminal] >= 0)
        and np.all(next_state[mask & ~terminal] < len(graph["features"]))
        and np.all(next_state[terminal] == -1)
    )
    policy = np.asarray(common["pi_old"], dtype=np.float64)
    outside_mass = float(np.abs(np.where(mask, 0.0, policy)).max())
    row_sum_error = float(np.abs(policy.sum(1) - 1.0).max())
    residual, terminal_error = qold_bellman_audit(task, graph, common, qold)
    restore = restore_audit(task, graph)

    features = torch.as_tensor(graph["features"], dtype=torch.float32, device=DEVICE)
    masks = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE)
    actor = base.Actor(features.shape[1]).to(DEVICE)
    actor.load_state_dict(common["actor"])
    reconstructed = actor_policy(actor, features, masks).cpu().numpy()
    reconstruction_error = float(np.abs(reconstructed - np.asarray(common["pi_old"])).max())

    # Zero-D regression and residual-critic representation.
    batch_states, batch_times, batch_actions = build_schedule(task, MASTER_SEED + 99, 1)
    batch = batch_tensors(graph, horizon, qold, batch_states[0], batch_times[0], batch_actions[0])
    common_a, actor_a, opt_a, critic, target, critic_opt = initialize_models(task, graph, MASTER_SEED + 7)
    actor_b = copy.deepcopy(actor_a)
    opt_b = torch.optim.Adam(actor_b.parameters(), lr=LEARNING_RATE)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    initial_residual = float(critic(batch["x"], batch["tf"]).abs().max())
    q_static = batch["qold"]
    actor_step(actor_a, opt_a, batch, q_static)
    actor_step(actor_b, opt_b, batch, q_static + torch.zeros_like(q_static))
    zero_d_error = max(float((left - right).abs().max()) for left, right in zip(actor_a.parameters(), actor_b.parameters()))
    _, residual_gradient = critic_step(actor, critic, target, critic_opt, batch)
    residual_after = float(critic(batch["x"], batch["tf"]).abs().max())

    # Independent deterministic current-policy perturbation for the PDR gate.
    current_actor = copy.deepcopy(actor)
    generator = torch.Generator(device=DEVICE).manual_seed(MASTER_SEED + 1000 + list(TASKS).index(task))
    with torch.no_grad():
        for parameter in current_actor.parameters():
            noise = torch.randn(parameter.shape, generator=generator, device=DEVICE, dtype=parameter.dtype)
            parameter.add_(0.002 * noise)
    qold_device = torch.as_tensor(np.asarray(qold), dtype=torch.float32, device=DEVICE)
    pi_old = torch.as_tensor(np.asarray(common["pi_old"]), dtype=torch.float32, device=DEVICE)
    test_states, test_times, test_actions = build_schedule(task, MASTER_SEED + 123, 4)
    test_batch = batch_tensors(
        graph, horizon, qold, test_states.reshape(-1), test_times.reshape(-1), test_actions.reshape(-1)
    )
    pdr_q, current_policy = pdr_batch_q(current_actor, graph, horizon, qold_device, pi_old, test_batch)
    oracle_q = exact_current_selected(current_policy, graph, horizon, test_batch["s_np"], test_batch["t_np"])
    legal = test_batch["mask"]
    q_error = float((pdr_q - oracle_q)[legal].abs().max())
    local_policy = current_policy[test_batch["s"]]
    pdr_adv = pdr_q - (local_policy * pdr_q).sum(1, keepdim=True)
    oracle_adv = oracle_q - (local_policy * oracle_q).sum(1, keepdim=True)
    advantage_error = float((pdr_adv - oracle_adv)[legal].abs().max())
    pdr_gradient, _, _ = gradient_vector(current_actor, test_batch, pdr_q)
    oracle_gradient, _, _ = gradient_vector(current_actor, test_batch, oracle_q)
    gradient_cosine = float(F.cosine_similarity(pdr_gradient[None], oracle_gradient[None]).item())
    del qold_device
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    passed = bool(
        closure_valid
        and outside_mass == 0.0
        and row_sum_error <= 1e-6
        and residual <= QOLD_TOLERANCE
        and terminal_error <= QOLD_TOLERANCE
        and restore["passed"] == restore["cases"]
        and reconstruction_error <= POLICY_TOLERANCE
        and initial_residual == 0.0
        and residual_gradient > 0.0
        and residual_after > 0.0
        and zero_d_error == 0.0
        and q_error <= PDR_TOLERANCE
        and advantage_error <= PDR_TOLERANCE
        and gradient_cosine >= 0.99999
    )
    result = {
        "task": task,
        "environment": paths["env_id"],
        "status": "PASS" if passed else "FAIL",
        "closure_fraction": 1.0 if closure_valid else 0.0,
        "dropped_policy_mass": outside_mass,
        "policy_row_sum_max_error": row_sum_error,
        "qold_bellman_residual_max_abs": residual,
        "qold_terminal_error_max_abs": terminal_error,
        "policy_reconstruction_max_abs_error": reconstruction_error,
        "restore": restore,
        "residual_critic_initial_max_abs": initial_residual,
        "residual_critic_first_step_gradient_norm": residual_gradient,
        "residual_critic_after_step_max_abs": residual_after,
        "zero_d_actor_parameter_max_abs_error": zero_d_error,
        "pdr_q_max_abs_error": q_error,
        "pdr_centered_advantage_max_abs_error": advantage_error,
        "pdr_actor_gradient_cosine": gradient_cosine,
        "input_hashes": {
            "graph": sha256(paths["graph"]),
            "states": sha256(paths["states"]),
            "qold": sha256(paths["qold"]),
            "common": sha256(paths["common"]),
        },
        "fresh_method_outcomes": 0,
    }
    write_json(output / "GATE_RESULT.json", result)
    if not passed:
        raise RuntimeError(f"{task}_PREFLIGHT_GATE_FAILED:{result}")


def lock_protocol() -> None:
    prereg = ROOT / "preregistration"
    target = prereg / "PREREGISTRATION_LOCK.json"
    if target.exists():
        raise RuntimeError("PREREGISTRATION_LOCK_ALREADY_EXISTS")
    seed_lock = json.loads((prereg / "SEED_PRELOCK.json").read_text())
    schedule_manifest = pd.read_csv(prereg / "SCHEDULE_MANIFEST.csv")
    gate_results = {}
    for task in TASKS:
        path = ROOT / "preflight" / task / "GATE_RESULT.json"
        if not path.exists():
            raise RuntimeError(f"MISSING_PREFLIGHT:{task}")
        gate_results[task] = json.loads(path.read_text())
        if gate_results[task]["status"] != "PASS":
            raise RuntimeError(f"FAILED_PREFLIGHT:{task}")
    lock = {
        "status": "RQ1_EXACT_ANCHOR_CONFIRMATORY_PROTOCOL_LOCKED",
        "experiment": "KEYCORRIDOR_RQ1_EXACT_ANCHOR_MECHANISM",
        "scientific_role": "mechanism/generalization experiment, not an algorithm-efficiency or O2O benchmark",
        "design_sha256": sha256(ROOT / "RQ1_design.md"),
        "runner_sha256": sha256(Path(__file__)),
        "analysis_sha256": sha256(ROOT / "src" / "analyze_rq1_confirmatory.py"),
        "seed_prelock_sha256": sha256(prereg / "SEED_PRELOCK.json"),
        "schedule_manifest_sha256": sha256(prereg / "SCHEDULE_MANIFEST.csv"),
        "seeds": seed_lock["per_task"],
        "tasks": {
            task: {
                "environment": TASKS[task][0],
                "horizon": TASKS[task][1],
                "world_seed": 1000,
                "graph_sha256": gate_results[task]["input_hashes"]["graph"],
                "qold_sha256": gate_results[task]["input_hashes"]["qold"],
                "common_start_sha256": gate_results[task]["input_hashes"]["common"],
                "schedule_hashes": schedule_manifest[schedule_manifest.task == task][["seed", "sha256"]].to_dict("records"),
            }
            for task in TASKS
        },
        "arms": ARMS,
        "primary_task": "S4R3",
        "primary_contrasts": ["A2_MINUS_A0", "A2_MINUS_A1"],
        "secondary_contrast": "A1_MINUS_A0",
        "primary_metric": "exact start-state success AUC over learner cycles 0 through 128",
        "total_updates": TOTAL_UPDATES,
        "early_window": [0, EARLY_UPDATES],
        "checkpoints": CHECKPOINTS,
        "diagnostic_checkpoints": DIAGNOSTIC_CHECKPOINTS,
        "hyperparameters": {
            "gamma": GAMMA,
            "actor_learning_rate": LEARNING_RATE,
            "critic_learning_rate": LEARNING_RATE,
            "awac_beta": BETA,
            "weight_clip": WEIGHT_CLIP,
            "gradient_clip": GRAD_CLIP,
            "target_tau": TARGET_TAU,
            "batch_size": BATCH_SIZE,
            "demo_rows_per_batch": DEMO_ROWS,
            "uniform_rows_per_batch": UNIFORM_ROWS,
            "actor_critic_update_ratio": "1:1",
        },
        "cycle_order": [
            "ordinary residual TD critic update under pi_t",
            "A2 exact-anchor fresh PDR solve under the same pi_t",
            "arm-specific actor-facing Q selection",
            "one native AWAC actor update to pi_t+1",
            "scheduled exact evaluation",
        ],
        "numerical_gates": {
            "qold_bellman_max_abs": QOLD_TOLERANCE,
            "pdr_q_and_centered_advantage_max_abs": PDR_TOLERANCE,
            "actor_gradient_cosine_min": 0.99999,
            "policy_reconstruction_max_abs": POLICY_TOLERANCE,
            "closure": 1.0,
            "dropped_policy_mass": 0.0,
        },
        "statistics": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "paired learner seed",
            "confidence_level": 0.95,
            "S4R3_confirmation": "both A2-A0 and A2-A1 means > 0, CI lower bounds > 0, and >=16/20 positive; final A2-A0 CI lower > -0.03",
            "family_summary": "equal-task hierarchical bootstrap; no requirement that all tasks be individually significant",
            "equivalence_testing": "none; a CI crossing zero is not interpreted as equivalence",
        },
        "cost_role": "separate descriptive boundary only; never a primary RQ1 success gate",
        "cost_counters": [
            "transition_dependency_edge_evaluations",
            "actor_updates",
            "critic_updates",
            "wall_clock_seconds",
            "peak_cpu_memory_mb",
            "peak_gpu_memory_mb",
        ],
        "development_exclusion": {
            "S4R3_development_path": "development/S4R3_world_1000/closed_loop_exact_anchor",
            "included_in_confirmatory_estimates": False,
        },
        "prohibited": [
            "online data collection",
            "learned Qold",
            "task-specific hyperparameters",
            "seed replacement after method outcome",
            "direct current-policy Q injection",
            "PDR-DI",
        ],
        "fresh_method_outcomes_at_lock": 0,
    }
    write_json(target, lock)
    (prereg / "LOCK_SHA256.txt").write_text(f"{sha256(target)}  PREREGISTRATION_LOCK.json\n")


def load_lock():
    path = ROOT / "preregistration" / "PREREGISTRATION_LOCK.json"
    lock = json.loads(path.read_text())
    if lock["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("RUNNER_HASH_MISMATCH_AFTER_LOCK")
    if lock["design_sha256"] != sha256(ROOT / "RQ1_design.md"):
        raise RuntimeError("DESIGN_HASH_MISMATCH_AFTER_LOCK")
    return lock


def run_one(task: str, seed_index: int, arm: str) -> None:
    lock = load_lock()
    seed = int(lock["seeds"][task]["confirmatory"][seed_index])
    output = ROOT / "runs" / task / f"seed_{seed_index:02d}_{seed}" / arm
    if output.exists():
        raise RuntimeError(f"RUN_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    graph = load_graph(task)
    horizon = TASKS[task][1]
    qold_np = np.load(source_paths(task)["qold"], mmap_mode="r")
    schedule_file = schedule_path(task, seed)
    schedule = np.load(schedule_file)
    common, actor, actor_optimizer, critic, target, critic_optimizer = initialize_models(task, graph, seed)
    features = torch.as_tensor(graph["features"], dtype=torch.float32, device=DEVICE)
    masks = torch.as_tensor(graph["mask"], dtype=torch.bool, device=DEVICE)
    pi_old = torch.as_tensor(np.asarray(common["pi_old"]), dtype=torch.float32, device=DEVICE)
    qold_device = None
    if arm == "A2_FRESH_PDR":
        qold_device = torch.as_tensor(np.asarray(qold_np), dtype=torch.float32, device=DEVICE)
    dependencies_by_action, one_dependency_pass = dependency_counts(graph, horizon)
    diagnostic_batch = batch_tensors(
        graph,
        horizon,
        qold_np,
        schedule["states"][:4].reshape(-1),
        schedule["times"][:4].reshape(-1),
        schedule["actions"][:4].reshape(-1),
    )
    curves = []
    updates = []
    signals = []
    td_edges = 0
    pdr_source_edges = 0
    pdr_solve_edges = 0
    scoring_seconds = 0.0
    started = time.perf_counter()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def checkpoint(update: int):
        nonlocal scoring_seconds
        score_started = time.perf_counter()
        policy = actor_policy(actor, features, masks)
        metrics = exact_metrics(policy, graph, horizon)
        scoring_seconds += time.perf_counter() - score_started
        curves.append(
            {
                "task": task,
                "environment": TASKS[task][0],
                "seed": seed,
                "seed_index": seed_index,
                "arm": arm,
                "update": update,
                "success": metrics["success"],
                "return": metrics["return"],
                "key_capability": metrics["key_capability"],
                "unlock_capability": metrics["unlock_capability"],
                "td_edge_evaluations": td_edges,
                "pdr_source_edge_evaluations": pdr_source_edges,
                "pdr_solve_edge_evaluations": pdr_solve_edges,
                "total_transition_dependency_edge_evaluations": td_edges + pdr_source_edges + pdr_solve_edges,
                "training_wall_seconds": time.perf_counter() - started - scoring_seconds,
                "scoring_wall_seconds": scoring_seconds,
                "evidence_status": "FRESH_CONFIRMATORY",
            }
        )
        if update in DIAGNOSTIC_CHECKPOINTS:
            signal_started = time.perf_counter()
            signal = diagnostic_signal(
                arm, actor, critic, graph, horizon, qold_np, qold_device, pi_old, diagnostic_batch
            )
            scoring_seconds += time.perf_counter() - signal_started
            signal.update(
                {
                    "task": task,
                    "seed": seed,
                    "seed_index": seed_index,
                    "arm": arm,
                    "update": update,
                    "evidence_status": "FRESH_CONFIRMATORY",
                }
            )
            signals.append(signal)
        pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
        if signals:
            pd.DataFrame(signals).to_csv(output / "actor_signal.csv", index=False)

    checkpoint(0)
    for update in range(1, TOTAL_UPDATES + 1):
        batch = batch_tensors(
            graph,
            horizon,
            qold_np,
            schedule["states"][update - 1],
            schedule["times"][update - 1],
            schedule["actions"][update - 1],
        )
        critic_loss, critic_gradient = critic_step(actor, critic, target, critic_optimizer, batch)
        row_dependencies = dependencies_by_action[batch["s_np"], batch["a_np"]]
        td_edges += int(np.where(batch["done_np"], 1, row_dependencies).sum())
        pdr_seconds = 0.0
        if arm == "A0_STANDARD_TD":
            actor_q = batch["qold"] + critic(batch["x"], batch["tf"])
        elif arm == "A1_STATIC_EXACT_QOLD":
            actor_q = batch["qold"]
        else:
            pdr_started = time.perf_counter()
            actor_q, _ = pdr_batch_q(actor, graph, horizon, qold_device, pi_old, batch)
            pdr_seconds = time.perf_counter() - pdr_started
            pdr_source_edges += one_dependency_pass
            pdr_solve_edges += one_dependency_pass
        actor_loss_value, actor_gradient, weight_mean, clipping_fraction = actor_step(
            actor, actor_optimizer, batch, actor_q
        )
        with torch.no_grad():
            for target_parameter, parameter in zip(target.parameters(), critic.parameters()):
                target_parameter.mul_(1.0 - TARGET_TAU).add_(parameter, alpha=TARGET_TAU)
        updates.append(
            {
                "task": task,
                "seed": seed,
                "seed_index": seed_index,
                "arm": arm,
                "update": update,
                "critic_loss": critic_loss,
                "critic_gradient_norm": critic_gradient,
                "actor_loss": actor_loss_value,
                "actor_gradient_norm": actor_gradient,
                "awac_weight_mean": weight_mean,
                "awac_clipping_fraction": clipping_fraction,
                "pdr_wall_seconds": pdr_seconds,
                "td_edge_evaluations": td_edges,
                "pdr_source_edge_evaluations": pdr_source_edges,
                "pdr_solve_edge_evaluations": pdr_solve_edges,
                "total_transition_dependency_edge_evaluations": td_edges + pdr_source_edges + pdr_solve_edges,
            }
        )
        if update in CHECKPOINTS:
            checkpoint(update)

    pd.DataFrame(updates).to_csv(output / "updates.csv", index=False)
    pd.DataFrame(signals).to_csv(output / "actor_signal.csv", index=False)
    peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 2**20) if DEVICE.type == "cuda" else 0.0
    peak_cpu_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    status = {
        "status": "COMPLETE",
        "evidence_status": "FRESH_CONFIRMATORY",
        "task": task,
        "seed": seed,
        "seed_index": seed_index,
        "arm": arm,
        "updates": TOTAL_UPDATES,
        "schedule_sha256": sha256(schedule_file),
        "graph_sha256": lock["tasks"][task]["graph_sha256"],
        "qold_sha256": lock["tasks"][task]["qold_sha256"],
        "common_start_sha256": lock["tasks"][task]["common_start_sha256"],
        "protocol_sha256": sha256(ROOT / "preregistration" / "PREREGISTRATION_LOCK.json"),
        "runner_sha256": sha256(Path(__file__)),
        "device": str(DEVICE),
        "wall_seconds_total": time.perf_counter() - started,
        "wall_seconds_scoring": scoring_seconds,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "peak_cpu_memory_mb": peak_cpu_mb,
        "total_transition_dependency_edge_evaluations": td_edges + pdr_source_edges + pdr_solve_edges,
    }
    write_json(output / "STATUS.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prelock", "schedules", "gate", "lock", "run"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.phase == "prelock":
        prelock()
    elif args.phase == "schedules":
        prepare_schedules()
    elif args.phase == "gate":
        gate(args.task)
    elif args.phase == "lock":
        lock_protocol()
    else:
        run_one(args.task, args.seed_index, args.arm)


if __name__ == "__main__":
    main()
