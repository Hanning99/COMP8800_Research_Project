#!/usr/bin/env python3
"""Preregistered M5 clean SecureCourier causal reproduction.

Only this new experiment tree is writable. Historical SecureCourier code and
artifacts are imported read-only as environment/backbone provenance. Canonical
PDR, clean masking, projection, and AWAC loss come from the validated PDR package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch


HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parent
HIST = ROOT / "predict-Qnew"
sys.path.insert(0, str(PROJECT))

from pdr.adapters.awac import awac_actor_loss
from pdr.core.differential import compute_pdr
from pdr.core.types import ReferenceSnapshot
from pdr.masks.action_mask import masked_log_softmax, masked_softmax
from pdr.operators.tabular_replay import TabularReplayOperator
from pdr.projection.advantage import center_under_policy, pdr_advantage, pdr_advantage_correction


os.environ["PDR_CLOSED_LOOP_DEVICE"] = os.environ.get("M5_DEVICE", "cpu")
os.environ["STEPWISE_PDR_DEVICE"] = os.environ.get("M5_DEVICE", "cpu")
cs = importlib.util.spec_from_file_location("m5_closed_loop_source", HIST / "close_loop_experiment/run_closed_loop.py")
C = importlib.util.module_from_spec(cs); assert cs.loader is not None; cs.loader.exec_module(C)
_set_threads, _set_interop = torch.set_num_threads, torch.set_num_interop_threads
torch.set_num_threads = lambda n: None; torch.set_num_interop_threads = lambda n: None
try:
    ss = importlib.util.spec_from_file_location("m5_stepwise_source", HIST / "STEPWISE-PDR/run_stepwise_pdr.py")
    S = importlib.util.module_from_spec(ss); assert ss.loader is not None; ss.loader.exec_module(S)
finally:
    torch.set_num_threads, torch.set_num_interop_threads = _set_threads, _set_interop
B = S.B
DEVICE = C.DEVICE


LABEL = "M5_CLEAN_SECURECOURIER_CAUSAL_REPRODUCTION"
ARMS = ["STANDARD_AWAC_CLEAN", "PDR_AWAC_CLEAN"]
SEEDS = [2086877193, 391970944, 1612870557, 200169987, 277732637, 882909101, 1682238597, 300771054, 1928572942, 682378278]
U_TOTAL = 2048
H_EARLY = 1024
H_EXTENDED = 2048
CHECKPOINTS = list(range(0, U_TOTAL + 1, 16))
MECHANISM_CHECKPOINTS = {0, 16, 64, 256, 1024, 2048}
GAMMA = 0.97
BOOT_N = 20000
BOOT_SEED = 20260902
SUCCESS_TOL = -0.01
GATE_TOL = -0.01
OFFLINE_N, ONLINE_N, ONLINE_CAP = 6, 58, 20000
INIT_N, INIT_SEED = 64, 20260901
CURRICULUM_NODES = ["sc5", "sc4", "sc3", "sc2"]
CRITICAL_JUNCTION = ("junction", 8, "take_shortcut", "take_safe_route")
ADV_TOL = 1e-8

SUPPORT_P = HIST / "pdr_direct_q_full_support/audit/full_support_replay_rows.jsonl"
GRAPH_P = B.GRAPH_P
STRICT_P = B.STRICT_P
CLOSURE_P = B.CLOSURE_P
ACTOR_P, ACTOR_OPT_P, VOCAB_P = C.ACTOR_P, C.ACTOR_OPT_P, C.VOCAB_P
CRITIC_P, TARGET_P, CRITIC_OPT_P = C.CRITIC_P, C.TARGET_P, C.CRITIC_OPT_P
OLD_ACTOR_P, OLD_VOCAB_P = B.ACTOR_OLD_P, B.VOCAB_OLD_P
QTABLE_P = B.INITIAL_PDR_P
PREREG_P = HERE / "PREREGISTRATION.md"
LOCK_P = HERE / "PREREGISTRATION_LOCK.json"


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(float(value)) else None
    if torch.is_tensor(value): return clean(value.detach().cpu().tolist())
    return value


def write_json(path, value):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path): return json.loads(pathlib.Path(path).read_text())


def read_jsonl(path):
    with pathlib.Path(path).open() as handle: return [json.loads(line) for line in handle]


def load_vocab(path):
    with pathlib.Path(path).open("rb") as handle: return pickle.load(handle)


def load_actor(vocab, path=ACTOR_P):
    model = C.Actor(vocab.size, 32, 64).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); return model


def load_critic(path, vocab): return C.critic_load(path, vocab)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def flatten_gradient(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in model.parameters()]).cpu().double()


def cosine(a, b):
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) <= 1e-12 or float(nb) <= 1e-12: return None
    return float(torch.dot(a, b) / (na * nb))


def git_state(path):
    try:
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()),
        }
    except Exception: return {"commit": None, "dirty": None}


def build_operator(pairs, probabilities, rewards):
    transitions = {}
    for pair in pairs:
        transitions[pair] = [
            ((outcome[0], outcome[1], int(outcome[2])), bool(outcome[3]), float(probability))
            for outcome, probability in probabilities[pair].items()
        ]
    return TabularReplayOperator(pairs, transitions, rewards, probability_tolerance=1e-12)


def qvector(pairs, values): return np.asarray([values[pair] for pair in pairs], dtype=np.float64)


def clean_collate(samples, vocab):
    batch = C.collate(samples, vocab)
    current_mask = torch.zeros_like(batch["action_idx"][:, None].expand(-1, batch["act_tokens"].shape[1]), dtype=torch.bool)
    next_mask = torch.zeros((len(samples), batch["next_act_tokens"].shape[1]), dtype=torch.bool, device=DEVICE)
    for i, row in enumerate(samples):
        current_mask[i, :len(row["admissible_actions"])] = True
        next_count = len(row["next_admissible_actions"])
        next_mask[i, :max(1, next_count)] = True  # terminal sentinel is multiplied by zero continuation
    batch["action_mask"] = current_mask
    batch["next_action_mask"] = next_mask
    return batch


def critic_step_clean(actor, critic, target, optimizer, batch):
    with torch.no_grad():
        next_q = target(batch["next_obs_tokens"], batch["next_act_tokens"])
        next_logp = masked_log_softmax(actor(batch["next_obs_tokens"], batch["next_act_tokens"]), batch["next_action_mask"])
        target_value = (next_logp.exp() * torch.where(batch["next_action_mask"], next_q, torch.zeros_like(next_q))).sum(-1)
        y = batch["reward"] + GAMMA * (1 - batch["done"]) * target_value
    q = critic(batch["obs_tokens"], batch["act_tokens"])
    selected_q = q.gather(1, batch["action_idx"][:, None]).squeeze(1)
    loss = torch.nn.functional.mse_loss(selected_q, y)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), C.GCLIP)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"critic_loss": float(loss), "critic_gradient_norm": grad}


def actor_step_clean(actor, optimizer, batch, q_values):
    actor.train(); optimizer.zero_grad(set_to_none=True)
    logp = masked_log_softmax(actor(batch["obs_tokens"], batch["act_tokens"]), batch["action_mask"])
    probs = logp.exp()
    advantage = center_under_policy(q_values.detach(), probs, batch["action_mask"])
    loss, diagnostics = awac_actor_loss(
        logp, batch["action_idx"], advantage, C.LAM, C.WCLIP,
        stop_gradient_weight=True, action_mask=batch["action_mask"],
    )
    loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), C.GCLIP)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"actor_loss": float(loss), "actor_gradient_norm": grad, **asdict(diagnostics)}


def deliver_table_q(samples, batch, pair_index, q_source):
    q = torch.zeros_like(batch["action_mask"], dtype=torch.float32, device=DEVICE)
    for i, row in enumerate(samples):
        state = row["full_state"]
        for j, action in enumerate(row["admissible_structural"]):
            key = (str(row["instance_id"]), str(state["node"]), int(state["step_count"]), str(action))
            if key not in pair_index: raise RuntimeError(f"PDR_SUPPORT_FAILURE:{key}")
            q[i, j] = float(q_source[pair_index[key]])
    return q


def row_key(row, action):
    state = row["full_state"]
    return str(row["instance_id"]), str(state["node"]), int(state["step_count"]), str(action)


def collect_episode(actor, vocab, instance, target_node, generator, update):
    env = C.SecureCourierEnv(instance, gamma=GAMMA, max_steps=C.MAX_STEPS); env.reset(); transitions = 0
    if target_node:
        for action in C.oracle_shortcut_plan():
            if env.current_node() == target_node: break
            env.execute_structural(action); transitions += 1
    rows, t = [], 0
    while not env.is_done():
        state = env.clone_state(); observation = env._obs(); index = C.obs_action(actor, vocab, env, generator, False)
        if state.node in C.GATE_ACTION: index = observation["admissible_structural"].index(C.GATE_ACTION[state.node])
        action = observation["admissible_structural"][index]
        with torch.no_grad():
            obs_tokens = torch.tensor([vocab.encode(observation["text"], C.OBS_MAX)], device=DEVICE)
            act_tokens = torch.tensor([[vocab.encode(x, C.ACT_MAX) for x in observation["admissible_actions"]]], device=DEVICE)
            behavior_log_prob = float(actor(obs_tokens, act_tokens)[0, index])
        _, reward, _, _, _ = env.step(index); transitions += 1
        next_observation, next_state = env._obs(), env.clone_state()
        rows.append({
            "transition_id": f"m5_u{update}_t{t}", "instance_id": instance.instance_id, "t": t,
            "observation": observation["text"], "admissible_actions": observation["admissible_actions"],
            "admissible_structural": observation["admissible_structural"], "action": index,
            "action_display": observation["admissible_actions"][index], "action_structural": action,
            "reward": float(reward), "next_observation": next_observation["text"],
            "next_admissible_actions": next_observation["admissible_actions"],
            "next_admissible_structural": next_observation["admissible_structural"],
            "done": env.is_done(), "behavior_policy_id": "m5_clean_online_actor",
            "behavior_log_prob": behavior_log_prob, "route_label": "m5_online",
            "full_state": C.clean(state.__dict__), "next_state": C.clean(next_state.__dict__) if next_state else None,
        }); t += 1
    return rows, transitions


def exact_gate_capability(policy, instance_id):
    values = {(node, C.MAX_STEPS): float(node == "gate_passed") for node in C.ALL_NODES}
    for step in range(C.MAX_STEPS - 1, -1, -1):
        for node in sorted(C.ALL_NODES):
            if node == "gate_passed": values[(node, step)] = 1.0; continue
            if node == C.DELIVERED: values[(node, step)] = 0.0; continue
            value = 0.0
            for action, probability in policy[(instance_id, node, step)].items():
                nxt, _, terminal = C.step_transition(node, action)
                value += probability * (1.0 if nxt == "gate_passed" else (0.0 if terminal else values[(nxt, step + 1)]))
            values[(node, step)] = value
    return float(values[("gate_lobby", 14)])


def center_vector(q, policy, pairs, pair_index):
    state_value = {}
    supported_states = {pair[:-1] for pair in pairs}
    for state in supported_states:
        distribution = policy[state]
        state_value[state] = sum(float(probability) * float(q[pair_index[state + (action,)]]) for action, probability in distribution.items())
    return np.asarray([float(q[i]) - state_value[pair[:-1]] for i, pair in enumerate(pairs)], dtype=np.float64)


def ranking_accuracy(values, policy, pairs, pair_index, exact):
    scores = []
    supported_states = {pair[:-1] for pair in pairs}
    for state in supported_states:
        distribution = policy[state]
        actions = list(distribution)
        if len(actions) < 2: continue
        predicted = actions[int(np.argmax([values[pair_index[state + (a,)]] for a in actions]))]
        target = actions[int(np.argmax([exact[pair_index[state + (a,)]] for a in actions]))]
        scores.append(predicted == target)
    return float(np.mean(scores)) if scores else 1.0


def gradient_for_q(actor, batch, q):
    actor.train()
    actor.zero_grad(set_to_none=True)
    logp = masked_log_softmax(actor(batch["obs_tokens"], batch["act_tokens"]), batch["action_mask"])
    advantage = center_under_policy(q.detach(), logp.exp(), batch["action_mask"])
    loss, diagnostics = awac_actor_loss(logp, batch["action_idx"], advantage, C.LAM, C.WCLIP, stop_gradient_weight=True, action_mask=batch["action_mask"])
    loss.backward(); gradient = flatten_gradient(actor); actor.zero_grad(set_to_none=True)
    selected = advantage.gather(1, batch["action_idx"][:, None]).squeeze(1).detach()
    return gradient, selected, diagnostics


def save_checkpoint(seed_index, seed, arm, update, actor, critic, target, critic_optimizer):
    actor_dir = HERE / f"actor_checkpoints/seed_{seed_index:02d}_{seed}/{arm}"; actor_dir.mkdir(parents=True, exist_ok=True)
    actor_path = actor_dir / f"actor_u{update:04d}.pt"
    if actor_path.exists(): raise RuntimeError("CHECKPOINT_EXISTS_REFUSING_OVERWRITE")
    torch.save({"seed": seed, "arm": arm, "u": update, "state_dict": cpu_tree(actor.state_dict())}, actor_path)
    records = [{"seed": seed, "arm": arm, "u": update, "kind": "actor", "path": str(actor_path.relative_to(HERE)), "sha256": sha(actor_path), "parameter_hash": C.param_hash(actor)}]
    if update in {0, H_EARLY, H_EXTENDED}:
        critic_dir = HERE / f"critic_checkpoints/seed_{seed_index:02d}_{seed}/{arm}"; critic_dir.mkdir(parents=True, exist_ok=True)
        critic_path = critic_dir / f"critic_u{update:04d}.pt"
        torch.save({"seed": seed, "arm": arm, "u": update, "critic": cpu_tree(critic.state_dict()), "target": cpu_tree(target.state_dict()), "optimizer": cpu_tree(critic_optimizer.state_dict())}, critic_path)
        records.append({"seed": seed, "arm": arm, "u": update, "kind": "critic_target_optimizer", "path": str(critic_path.relative_to(HERE)), "sha256": sha(critic_path), "critic_hash": C.param_hash(critic), "target_hash": C.param_hash(target), "optimizer_hash": C.state_hash(critic_optimizer.state_dict())})
    return records


def source_paths():
    paths = [
        PROJECT / "README.md", PROJECT / "PDR_ALGORITHM_ENGINEERING_BLUEPRINT.md",
        PROJECT / "M0_M4_IMPLEMENTATION_REPORT.md", PROJECT / "M0_M4_VALIDATION.json",
        PROJECT / "pdr/core/differential.py", PROJECT / "pdr/core/solver.py", PROJECT / "pdr/core/types.py",
        PROJECT / "pdr/operators/tabular_replay.py", PROJECT / "pdr/projection/advantage.py",
        PROJECT / "pdr/masks/action_mask.py", PROJECT / "pdr/adapters/awac.py",
        HIST / "close_loop_experiment/run_closed_loop.py", HIST / "STEPWISE-PDR/run_stepwise_pdr.py",
        B.CANONICAL_P, SUPPORT_P, GRAPH_P, STRICT_P, CLOSURE_P,
        ACTOR_P, ACTOR_OPT_P, VOCAB_P, CRITIC_P, TARGET_P, CRITIC_OPT_P,
        OLD_ACTOR_P, OLD_VOCAB_P, QTABLE_P,
    ]
    paths.extend(sorted((HIST / "STEPWISE-PDR/securecourier_env").rglob("*.py")))
    return paths


def prepare():
    if LOCK_P.exists(): raise RuntimeError("PREREGISTRATION_LOCK_EXISTS")
    if not PREREG_P.is_file(): raise RuntimeError("PREREGISTRATION_MISSING")
    validation = read_json(PROJECT / "M0_M4_VALIDATION.json")
    if validation.get("verdict") != "M0_M4_CLEAN_PDR_REFERENCE_IMPLEMENTATION_VALIDATED": raise RuntimeError("M0_M4_VALIDATION_REQUIRED")
    paths = source_paths()
    inventory = {"created_at": now(), "historical_sources_read_only": True, "files": [{"path": str(p), "size": p.stat().st_size, "sha256": sha(p)} for p in paths]}
    write_json(HERE / "SOURCE_INVENTORY.json", inventory)
    replay = read_jsonl(SUPPORT_P)
    if len(replay) != 32771: raise RuntimeError("PDR_SUPPORT_FAILURE: replay occurrence count")
    closure = read_json(CLOSURE_P)
    if closure["closure_fraction"] != 1.0 or closure["dropped_policy_mass"] != 0.0: raise RuntimeError("PDR_SUPPORT_FAILURE")
    init_indices = random.Random(INIT_SEED).choices(range(len(replay)), k=INIT_N)
    split = C.generate_pilot_split(); train = split["train"]
    schedules, diagnostics = {}, {}
    for seed in SEEDS:
        rng = random.Random(seed + 500000); rows = []
        for update in range(1, U_TOTAL + 1):
            instance = rng.choice(train); target = rng.choice(CURRICULUM_NODES) if rng.random() < 0.8 else None
            rows.append({"u": update, "instance_id": instance.instance_id, "curriculum_target": target})
        diagnostic_rng = random.Random(seed + 600000); diagnostic_indices = diagnostic_rng.choices(range(len(replay)), k=64)
        schedules[str(seed)] = {"schedule": rows, "sha256": canonical_hash(rows)}
        diagnostics[str(seed)] = {"indices": diagnostic_indices, "sha256": canonical_hash(diagnostic_indices)}
    write_json(HERE / "FROZEN_CONTINUATION_SCHEDULES.json", {"seeds": schedules})
    write_json(HERE / "FROZEN_DIAGNOSTIC_MINIBATCHES.json", {"seeds": diagnostics})
    config = {
        "label": LABEL, "arms": ARMS, "seeds": SEEDS, "gamma": GAMMA, "updates": U_TOTAL,
        "early_window": H_EARLY, "extended_window": H_EXTENDED, "evaluation_frequency": 16,
        "mechanism_checkpoints": sorted(MECHANISM_CHECKPOINTS), "refresh_interval": 1,
        "batch": {"size": 64, "offline": OFFLINE_N, "online": ONLINE_N}, "online_capacity": ONLINE_CAP,
        "actor_critic_lr": C.LR, "temperature": C.LAM, "weight_clip": C.WCLIP,
        "gradient_clip": C.GCLIP, "target_tau": C.TAU, "stop_gradient_weight": True,
        "legal_action_masking": "exact bool masks; zero illegal probability; legal-only centering",
        "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED, "unit": "paired continuation seed", "CI": 0.95},
        "reliability_tolerances": {"success": SUCCESS_TOL, "gate_capability": GATE_TOL},
        "environment_AUC": "H_env_common=min E_seed_arm(u=1024); linear interpolation; normalized trapezoid",
    }
    write_json(HERE / "CONFIG.json", config)
    vocab = load_vocab(VOCAB_P); actor = load_actor(vocab); critic = load_critic(CRITIC_P, vocab); target = load_critic(TARGET_P, vocab)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=C.LR); actor_optimizer.load_state_dict(torch.load(ACTOR_OPT_P, map_location=DEVICE, weights_only=True))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=C.LR); critic_optimizer.load_state_dict(torch.load(CRITIC_OPT_P, map_location=DEVICE, weights_only=True))
    initial_rows = [replay[i] for i in init_indices]
    state = {
        "name": "M5_STANDARDIZED_CONTINUATION_STATE", "historical_resume_claimed": False,
        "actor_hash": C.param_hash(actor), "critic_hash": C.param_hash(critic), "target_hash": C.param_hash(target),
        "actor_optimizer_hash": C.state_hash(actor_optimizer.state_dict()), "critic_optimizer_hash": C.state_hash(critic_optimizer.state_dict()),
        "offline_replay_sha256": sha(SUPPORT_P), "online_initial_indices": init_indices,
        "online_initial_indices_sha256": canonical_hash(init_indices), "online_initial_buffer_digest": C.Buffer(initial_rows, ONLINE_CAP).digest(),
        "counters": {"u": 0, "episodes": 0, "training_transitions": 0, "evaluation_interactions": 0},
        "curriculum": {"probability": 0.8, "nodes": CURRICULUM_NODES, "gate_action_forced_legal": True},
        "instances": [x.instance_id for x in train], "all_arms_identical_before_training": True,
    }
    write_json(HERE / "STANDARDIZED_CONTINUATION_STATE.json", state)
    write_json(HERE / "PRELOCK_AUDIT.json", {"status": "PASSED", "M0_M4": validation["verdict"], "PDR_solves_before_lock": 0, "actor_updates_before_lock": 0, "critic_updates_before_lock": 0, "environment_interactions_before_lock": 0, "source_inventory_sha256": sha(HERE / "SOURCE_INVENTORY.json"), "state": state})
    print(json.dumps({"status": "PRELOCK_PREPARED", "device": str(DEVICE), "seeds": SEEDS}, indent=2))


def create_lock():
    if LOCK_P.exists(): raise RuntimeError("LOCK_EXISTS_REFUSING_OVERWRITE")
    required = ["PREREGISTRATION.md", "SOURCE_INVENTORY.json", "CONFIG.json", "FROZEN_CONTINUATION_SCHEDULES.json", "FROZEN_DIAGNOSTIC_MINIBATCHES.json", "STANDARDIZED_CONTINUATION_STATE.json", "PRELOCK_AUDIT.json"]
    if any(not (HERE / name).is_file() for name in required): raise RuntimeError("PRELOCK_ARTIFACT_MISSING")
    inventory = read_json(HERE / "SOURCE_INVENTORY.json")
    lock = {
        "immutable": True, "schema_version": 1, "created_at": now(), "label": LABEL,
        "locked_files": {name: sha(HERE / name) for name in required}, "runner_sha256": sha(__file__),
        "source_hashes": {row["path"]: row["sha256"] for row in inventory["files"]},
        "canonical_PDR_core_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in [PROJECT / "pdr/core/differential.py", PROJECT / "pdr/core/solver.py", PROJECT / "pdr/operators/tabular_replay.py"]},
        "clean_actor_path_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in [PROJECT / "pdr/masks/action_mask.py", PROJECT / "pdr/projection/advantage.py", PROJECT / "pdr/adapters/awac.py"]},
        "reference": {"pi_r": sha(OLD_ACTOR_P), "Q_r_checkpoint": sha(CRITIC_P), "Q_r_table": sha(QTABLE_P), "fixed": True},
        "support": {"replay": sha(SUPPORT_P), "graph": sha(GRAPH_P), "strict": sha(STRICT_P), "closure": sha(CLOSURE_P)},
        "arms": ARMS, "seeds": SEEDS, "gamma": GAMMA, "refresh_interval": 1,
        "windows": {"early": H_EARLY, "extended": H_EXTENDED}, "evaluation_checkpoints": CHECKPOINTS,
        "actor_semantics": {"clean_mask": True, "stop_gradient_weight": True, "same_adapter_both_arms": True},
        "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED, "unit": "seed"},
        "verdict_logic": "PREREGISTRATION.md", "git": git_state(ROOT), "device_not_part_of_scientific_protocol": True,
    }
    write_json(LOCK_P, lock); print(json.dumps({"status": "LOCKED", "sha256": sha(LOCK_P)}, indent=2))


def verify_lock():
    lock = read_json(LOCK_P)
    for name, digest in lock["locked_files"].items():
        if sha(HERE / name) != digest: raise RuntimeError(f"M5_SOURCE_PROVENANCE_CHANGED:{name}")
    current_runner_hash = sha(__file__)
    if current_runner_hash != lock["runner_sha256"]:
        amendment_path = HERE / "AMENDMENT_002.json"
        if not amendment_path.is_file(): raise RuntimeError("M5_SOURCE_PROVENANCE_CHANGED:runner")
        amendment = read_json(amendment_path)
        if amendment.get("base_locked_runner_sha256") != lock["runner_sha256"] or amendment.get("prior_amended_runner_sha256") != read_json(HERE / "AMENDMENT_001.json").get("amended_runner_sha256") or amendment.get("amended_runner_sha256") != current_runner_hash or amendment.get("status") != "FROZEN_IMPLEMENTATION_ONLY_FIX":
            raise RuntimeError("M5_SOURCE_PROVENANCE_CHANGED:runner_amendment_invalid")
    for path, digest in lock["source_hashes"].items():
        if not pathlib.Path(path).is_file() or sha(path) != digest: raise RuntimeError(f"M5_SOURCE_PROVENANCE_CHANGED:{path}")
    return lock


def run_seed(seed_index):
    verify_lock(); seed = SEEDS[seed_index]; out = HERE / f"seed_runs/seed_{seed_index:02d}_{seed}"
    if out.exists(): raise RuntimeError("SEED_OUTPUT_EXISTS_REFUSING_OVERWRITE")
    out.mkdir(parents=True); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    vocab, old_vocab = load_vocab(VOCAB_P), load_vocab(OLD_VOCAB_P)
    split = C.generate_pilot_split(); instances = {x.instance_id: x for x in split["train"]}
    evaluator, old_evaluator = B.PolicyEvaluator(split["train"], vocab), B.PolicyEvaluator(split["train"], old_vocab)
    pairs, probabilities = B.load_graph(); pair_index = {pair: i for i, pair in enumerate(pairs)}
    raw_graph = read_json(GRAPH_P); rewards = {tuple(json.loads(k)): float(v) for k, v in raw_graph["reward_mean"].items()}
    operator = build_operator(pairs, probabilities, rewards)
    old_actor = load_actor(old_vocab, OLD_ACTOR_P); policy_ref, policy_ref_audit = old_evaluator.policy(old_actor)
    _, q_env_ref_map = old_evaluator.exact(policy_ref); q_env_ref = qvector(pairs, q_env_ref_map)
    qtable = pd.read_csv(QTABLE_P).set_index(["instance_id", "node", "step_count", "action"])
    q_ref = np.asarray([float(qtable.loc[pair, "Q_theta_old"]) for pair in pairs], dtype=np.float64)
    reference = ReferenceSnapshot(policy_ref, q_ref, GAMMA, label="fixed_securecourier_reference")
    support_rows = read_jsonl(SUPPORT_P); standard_state = read_json(HERE / "STANDARDIZED_CONTINUATION_STATE.json")
    initial = [support_rows[i] for i in standard_state["online_initial_indices"]]
    schedule = read_json(HERE / "FROZEN_CONTINUATION_SCHEDULES.json")["seeds"][str(seed)]["schedule"]
    diagnostic_indices = read_json(HERE / "FROZEN_DIAGNOSTIC_MINIBATCHES.json")["seeds"][str(seed)]["indices"]
    diagnostic_samples = [support_rows[i] for i in diagnostic_indices]; diagnostic_batch = clean_collate(diagnostic_samples, vocab)
    common = {
        "actor_file": sha(ACTOR_P), "critic_file": sha(CRITIC_P), "target_file": sha(TARGET_P),
        "actor_optimizer_file": sha(ACTOR_OPT_P), "critic_optimizer_file": sha(CRITIC_OPT_P),
        "offline_replay": sha(SUPPORT_P), "online_replay": C.Buffer(initial, ONLINE_CAP).digest(),
        "schedule": canonical_hash(schedule), "counters": canonical_hash({"u": 0, "episodes": 0, "E": 0}),
        "replay_rng_seed": seed + 200000, "action_rng_seed": seed + 300000,
        "clean_masking": True, "stop_gradient_weight": True,
    }
    branch = {"status": "PASSED", "seed": seed, "only_advantage_source_differs": True, "rows": [{"arm": arm, **common} for arm in ARMS]}
    if len({canonical_hash({k: v for k, v in row.items() if k != "arm"}) for row in branch["rows"]}) != 1: raise RuntimeError("M5_BRANCH_PARITY_FAILURE")
    write_json(out / "BRANCH_EQUIVALENCE_AUDIT.json", branch)
    exact_rows, junction_rows, solve_rows, advantage_rows, weight_rows, gradient_rows = [], [], [], [], [], []
    update_rows, compute_rows, checkpoint_records = [], [], []

    for arm in ARMS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        actor = load_actor(vocab); critic = load_critic(CRITIC_P, vocab); target = load_critic(TARGET_P, vocab); target.eval()
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=C.LR); actor_optimizer.load_state_dict(torch.load(ACTOR_OPT_P, map_location=DEVICE, weights_only=True))
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=C.LR); critic_optimizer.load_state_dict(torch.load(CRITIC_OPT_P, map_location=DEVICE, weights_only=True))
        offline, online = C.Buffer(support_rows, len(support_rows)), C.Buffer(initial, ONLINE_CAP)
        replay_rng, action_generator = random.Random(seed + 200000), torch.Generator(device=DEVICE).manual_seed(seed + 300000)
        environment_steps = 0; arm_started = time.perf_counter()
        cumulative = {"environment": 0.0, "critic": 0.0, "actor": 0.0, "target_update": 0.0, "policy_enumeration": 0.0, "PDR_solve": 0.0, "exact_DP_evaluation": 0.0}

        def solve_for(policy, label, purpose, update):
            started = time.perf_counter(); result = compute_pdr(operator, reference, policy, row_sum_tolerance=1e-12, residual_tolerance=1e-10); elapsed = time.perf_counter() - started
            cumulative["PDR_solve"] += elapsed
            solve_rows.append({"seed": seed, "arm": arm, "u": update, "purpose": purpose, **asdict(result.solver_diagnostics), **{f"support_{k}": v for k, v in asdict(result.support_diagnostics).items() if k != "unsupported_pairs"}, "unsupported_pair_count": len(result.support_diagnostics.unsupported_pairs), "measured_total_solve_seconds": elapsed})
            return result

        def checkpoint(update):
            started = time.perf_counter(); policy, policy_audit = evaluator.policy(actor); metrics, q_current_map = evaluator.exact(policy); cumulative["exact_DP_evaluation"] += time.perf_counter() - started
            q_current = qvector(pairs, q_current_map); pdr_result = solve_for(policy, f"diagnostic_{arm}_u{update}", "diagnostic", update)
            a_stale = center_vector(q_ref, policy, pairs, pair_index); a_pdr = center_vector(pdr_result.q_pdr, policy, pairs, pair_index); a_exact = center_vector(q_current, policy, pairs, pair_index)
            delta_a = center_vector(pdr_result.differential, policy, pairs, pair_index)
            valid = np.abs(a_exact) > ADV_TOL
            advantage_rows.append({
                "seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps,
                "A_stale_MAE": float(np.mean(np.abs(a_stale - a_exact))), "A_PDR_MAE": float(np.mean(np.abs(a_pdr - a_exact))),
                "DeltaA_PDR_MAE": float(np.mean(np.abs(delta_a - center_vector(q_current - q_env_ref, policy, pairs, pair_index)))),
                "A_PDR_sign_accuracy": float(np.mean(np.sign(a_pdr[valid]) == np.sign(a_exact[valid]))) if valid.any() else 1.0,
                "A_stale_sign_accuracy": float(np.mean(np.sign(a_stale[valid]) == np.sign(a_exact[valid]))) if valid.any() else 1.0,
                "A_PDR_ranking_accuracy": ranking_accuracy(pdr_result.q_pdr, policy, pairs, pair_index, q_current),
                "A_stale_ranking_accuracy": ranking_accuracy(q_ref, policy, pairs, pair_index, q_current),
                "Direct_A_identity_max_abs_error": float(np.max(np.abs(a_pdr - (a_stale + delta_a)))),
                "policy_probability_sum_error": float(policy_audit["maximum_probability_sum_error"]),
                "dropped_policy_mass": float(policy_audit["dropped_policy_mass"]),
            })
            for metric in metrics:
                exact_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, **metric, "P_timeout_or_failure_exact": 1.0 - float(metric["P_success_exact"]), "P_gate_capability_exact": exact_gate_capability(policy, metric["instance_id"])})
            node, step, shortcut, safe = CRITICAL_JUNCTION
            for instance_id in sorted(instances):
                distribution = policy[(instance_id, node, step)]
                junction_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, "instance_id": instance_id, "p_shortcut": distribution[shortcut], "p_safe": distribution[safe], "log_odds": math.log(distribution[shortcut]) - math.log(distribution[safe])})
            if update in MECHANISM_CHECKPOINTS:
                qpdr = deliver_table_q(diagnostic_samples, diagnostic_batch, pair_index, pdr_result.q_pdr)
                qexact = deliver_table_q(diagnostic_samples, diagnostic_batch, pair_index, q_current)
                gpdr, apdr_selected, dpdr = gradient_for_q(actor, diagnostic_batch, qpdr)
                gexact, aexact_selected, dexact = gradient_for_q(actor, diagnostic_batch, qexact)
                cs = cosine(gpdr, gexact)
                if cs is None: raise RuntimeError("PDR_ADVANTAGE_IDENTITY_FAIL: undefined diagnostic gradient")
                wpdr = torch.clamp(torch.exp(apdr_selected / C.LAM), 0, C.WCLIP); wexact = torch.clamp(torch.exp(aexact_selected / C.LAM), 0, C.WCLIP)
                weight_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, "selected_log_weight_MAE": float(torch.mean(torch.abs(torch.log(wpdr) - torch.log(wexact)))), "selected_weight_MAE": float(torch.mean(torch.abs(wpdr - wexact))), "PDR_weight_mean": float(wpdr.mean()), "exact_weight_mean": float(wexact.mean()), "PDR_weight_ESS": dpdr.effective_sample_size, "exact_weight_ESS": dexact.effective_sample_size})
                gradient_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, "gradient_cosine_PDR_to_exact": cs, "PDR_gradient_norm": float(torch.linalg.vector_norm(gpdr)), "exact_gradient_norm": float(torch.linalg.vector_norm(gexact)), "gradient_norm_ratio": float(torch.linalg.vector_norm(gpdr) / torch.linalg.vector_norm(gexact))})
            checkpoint_records.extend(save_checkpoint(seed_index, seed, arm, update, actor, critic, target, critic_optimizer))
            compute_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, "total_training_wall": time.perf_counter() - arm_started, **cumulative})

        checkpoint(0)
        for update in range(1, U_TOTAL + 1):
            actor.eval(); started = time.perf_counter(); schedule_row = schedule[update - 1]
            episode, count = collect_episode(actor, vocab, instances[schedule_row["instance_id"]], schedule_row["curriculum_target"], action_generator, update)
            cumulative["environment"] += time.perf_counter() - started; environment_steps += count
            for row in episode: online.add(row)
            samples = offline.sample(OFFLINE_N, replay_rng) + online.sample(ONLINE_N, replay_rng); batch = clean_collate(samples, vocab)
            actor.train(); critic.train(); started = time.perf_counter(); critic_audit = critic_step_clean(actor, critic, target, critic_optimizer, batch); cumulative["critic"] += time.perf_counter() - started
            started = time.perf_counter()
            if arm == "STANDARD_AWAC_CLEAN":
                q_actor = critic(batch["obs_tokens"], batch["act_tokens"]).detach()
            else:
                policy_started = time.perf_counter(); current_policy, _ = evaluator.policy(actor); cumulative["policy_enumeration"] += time.perf_counter() - policy_started
                pdr_result = solve_for(current_policy, f"deployable_{arm}_u{update}", "deployable", update)
                q_actor = deliver_table_q(samples, batch, pair_index, pdr_result.q_pdr)
            actor_audit = actor_step_clean(actor, actor_optimizer, batch, q_actor); cumulative["actor"] += time.perf_counter() - started
            started = time.perf_counter(); target.soft_update(critic, tau=C.TAU); cumulative["target_update"] += time.perf_counter() - started
            update_rows.append({"seed": seed, "arm": arm, "u": update, "training_env_transitions": environment_steps, **critic_audit, **actor_audit})
            if update in CHECKPOINTS:
                checkpoint(update); print(json.dumps({"seed": seed, "arm": arm, "u": update, "E": environment_steps, "device": str(DEVICE)}), flush=True)
    pd.DataFrame(exact_rows).to_csv(out / "EXACT_POLICY_METRICS.csv", index=False)
    pd.DataFrame(junction_rows).to_csv(out / "JUNCTION_POLICY_TRAJECTORY.csv", index=False)
    pd.DataFrame(solve_rows).to_csv(out / "PDR_SOLVE_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(advantage_rows).to_csv(out / "ADVANTAGE_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "WEIGHT_GEOMETRY.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(out / "GRADIENT_GEOMETRY.csv", index=False)
    pd.DataFrame(update_rows).to_csv(out / "UPDATE_AUDIT.csv", index=False)
    pd.DataFrame(compute_rows).to_csv(out / "COMPUTE_ACCOUNTING.csv", index=False)
    write_json(out / "CHECKPOINT_MANIFEST.json", {"seed": seed, "records": checkpoint_records})
    max_residual = max(row["linear_residual_inf"] for row in solve_rows)
    write_json(out / "STATUS.json", {"status": "PASSED", "seed": seed, "device": str(DEVICE), "arms": ARMS, "updates_per_arm": U_TOTAL, "max_PDR_residual": max_residual, "support_failures": 0, "undefined_gradients": 0})
    print(json.dumps({"status": "PASSED", "seed": seed, "max_residual": max_residual}, indent=2))


def auc(x, y, horizon):
    x, y = np.asarray(x, float), np.asarray(y, float); order = np.argsort(x); x, y = x[order], y[order]
    keep = x < horizon; xx = np.r_[x[keep], horizon]; yy = np.r_[y[keep], np.interp(horizon, x, y)]
    return float(np.trapezoid(yy, xx) / horizon)


def bootstrap(values, salt):
    values = np.asarray(values, float); rng = np.random.default_rng(BOOT_SEED + salt)
    samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(BOOT_N)])
    return {"mean": float(values.mean()), "CI": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "positive_seed_fraction": float(np.mean(values > 0)), "values": values.tolist()}


def crossing_time(group, x_column):
    group = group.sort_values("u"); condition = (group.p_shortcut > group.p_safe).to_numpy()
    for i in range(len(condition) - 2):
        if condition[i:i + 3].all(): return float(group.iloc[i][x_column])
    return None


def finalize():
    verify_lock(); statuses = sorted((HERE / "seed_runs").glob("seed_*/STATUS.json"))
    if len(statuses) != len(SEEDS) or any(read_json(path)["status"] != "PASSED" for path in statuses): raise RuntimeError("M5_SEEDS_INCOMPLETE")
    def all_csv(name): return pd.concat([pd.read_csv(path.parent / name) for path in statuses], ignore_index=True)
    exact = all_csv("EXACT_POLICY_METRICS.csv"); junction = all_csv("JUNCTION_POLICY_TRAJECTORY.csv")
    solves = all_csv("PDR_SOLVE_DIAGNOSTICS.csv"); advantages = all_csv("ADVANTAGE_DIAGNOSTICS.csv")
    weights = all_csv("WEIGHT_GEOMETRY.csv"); gradients = all_csv("GRADIENT_GEOMETRY.csv"); compute = all_csv("COMPUTE_ACCOUNTING.csv")
    macro = exact.groupby(["seed", "arm", "u", "training_env_transitions"], as_index=False).mean(numeric_only=True)
    macro.to_csv(HERE / "LEARNING_CURVES.csv", index=False); junction.to_csv(HERE / "JUNCTION_POLICY_TRAJECTORY.csv", index=False)
    solves.to_csv(HERE / "PDR_SOLVE_DIAGNOSTICS.csv", index=False); advantages.to_csv(HERE / "ADVANTAGE_DIAGNOSTICS.csv", index=False)
    weights.to_csv(HERE / "WEIGHT_GEOMETRY.csv", index=False); gradients.to_csv(HERE / "GRADIENT_GEOMETRY.csv", index=False); compute.to_csv(HERE / "COMPUTE_ACCOUNTING.csv", index=False)
    h_env = int(macro[macro.u == H_EARLY].training_env_transitions.min()); seed_rows = []
    junction_macro = junction.groupby(["seed", "arm", "u", "training_env_transitions"], as_index=False).mean(numeric_only=True)
    for (seed, arm), group in macro.groupby(["seed", "arm"]):
        group = group.sort_values("u"); jgroup = junction_macro[(junction_macro.seed == seed) & (junction_macro.arm == arm)]
        final = group[group.u == H_EXTENDED].iloc[0]
        seed_rows.append({
            "seed": int(seed), "arm": arm,
            "AUC_update_early": auc(group.u, group.J_exact, H_EARLY), "AUC_update_extended": auc(group.u, group.J_exact, H_EXTENDED),
            "AUC_env": auc(group.training_env_transitions, group.J_exact, h_env),
            "J_final": float(final.J_exact), "P_success_final": float(final.P_success_exact),
            "P_gate_final": float(final.P_gate_capability_exact), "P_shortcut_final": float(final.P_shortcut_exact),
            "E_T_final": float(final.E_T_exact), "E_T_given_success_final": float(final.E_T_given_success_exact),
            "E_1024": int(group[group.u == H_EARLY].training_env_transitions.iloc[0]), "E_2048": int(final.training_env_transitions),
            "tau_flip_update": crossing_time(jgroup, "u"), "tau_flip_env": crossing_time(jgroup, "training_env_transitions"),
        })
    summary_long = pd.DataFrame(seed_rows); wide = summary_long.pivot(index="seed", columns="arm"); rows = []
    metrics = [x for x in summary_long.columns if x not in {"seed", "arm"}]
    for seed in SEEDS:
        row = {"seed": seed}
        for arm in ARMS:
            for metric in metrics: row[f"{arm}_{metric}"] = wide.loc[seed, (metric, arm)]
        for metric in metrics:
            a, b = row[f"PDR_AWAC_CLEAN_{metric}"], row[f"STANDARD_AWAC_CLEAN_{metric}"]
            row[f"PDR_minus_STANDARD_{metric}"] = a - b if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(row)
    seed_summary = pd.DataFrame(rows); seed_summary.to_csv(HERE / "SEED_LEVEL_SUMMARY.csv", index=False)
    statistics, salt = {}, 0
    for column in [x for x in seed_summary.columns if x.startswith("PDR_minus_STANDARD_") and seed_summary[x].notna().all()]:
        salt += 1; statistics[column] = bootstrap(seed_summary[column], salt)
    primary = statistics["PDR_minus_STANDARD_AUC_update_early"]
    benefit = primary["mean"] > 0 and primary["CI"][0] > 0
    success_diff = float(seed_summary.PDR_minus_STANDARD_P_success_final.mean()); gate_diff = float(seed_summary.PDR_minus_STANDARD_P_gate_final.mean())
    reliability = success_diff >= SUCCESS_TOL and gate_diff >= GATE_TOL
    early_adv = advantages[advantages.u <= H_EARLY].groupby(["seed", "arm"], as_index=False)[["A_stale_MAE", "A_PDR_MAE"]].mean()
    mechanism_seed = early_adv.groupby("seed").mean(numeric_only=True); mechanism_effect = mechanism_seed.A_PDR_MAE - mechanism_seed.A_stale_MAE
    mechanism_stats = bootstrap(mechanism_effect, 999); mechanism = mechanism_stats["mean"] < 0 and mechanism_stats["CI"][1] < 0
    if benefit and reliability: verdict = "CLEAN_PDR_CLOSED_LOOP_EFFECT_REPRODUCED"
    elif benefit and not reliability: verdict = "CLEAN_PDR_ACCELERATION_WITH_RELIABILITY_HARM"
    elif mechanism: verdict = "CLEAN_PDR_SIGNAL_PRESENT_BUT_CLOSED_LOOP_EFFECT_NOT_REPRODUCED"
    else: verdict = "CLEAN_PDR_CLOSED_LOOP_EFFECT_NOT_REPRODUCED"
    env_stats = statistics["PDR_minus_STANDARD_AUC_env"]; env_supported = env_stats["mean"] > 0 and env_stats["CI"][0] > 0
    branch_records = {path.parent.name: read_json(path.parent / "BRANCH_EQUIVALENCE_AUDIT.json") for path in statuses}
    write_json(HERE / "BRANCH_EQUIVALENCE_AUDIT.json", {"status": "PASSED", "all_seeds": all(x["status"] == "PASSED" for x in branch_records.values()), "records": branch_records})
    checkpoint_records = [row for path in statuses for row in read_json(path.parent / "CHECKPOINT_MANIFEST.json")["records"]]
    write_json(HERE / "CHECKPOINT_MANIFEST.json", {"records": checkpoint_records})
    result = {
        "status": "COMPLETE", "verdict": verdict, "primary_supported": benefit and reliability,
        "primary_AUC_update_early": primary, "H_env_common": h_env, "environment_AUC": env_stats,
        "environment_sample_efficiency_supported": env_supported, "secondary_statistics": statistics,
        "reliability": {"success_difference_mean": success_diff, "success_gate": success_diff >= SUCCESS_TOL, "gate_capability_difference_mean": gate_diff, "gate_capability_gate": gate_diff >= GATE_TOL, "all_pass": reliability},
        "mechanism": {"PDR_minus_stale_advantage_MAE": mechanism_stats, "supported": mechanism},
        "maximum_PDR_solver_residual": float(solves.linear_residual_inf.max()), "PDR_support_failures": 0,
        "fresh_confirmation_seeds": SEEDS, "inference_unit": "paired continuation seed",
        "scope": {"M5_only": True, "AntMaze": False, "scalable_backend": False, "legacy_diagnostic_arm": False},
    }
    write_json(HERE / "FINAL_VERDICT.json", result)
    report = f"""# M5 Clean SecureCourier Final Report

## Verdict

```text
{verdict}
```

This is the preregistered two-arm clean causal reproduction on the support-sufficient 16-instance SecureCourier domain. Both arms used identical legal-action masking, detached-weight AWAC, critic/replay/update schedules, fresh paired seeds, and exact stochastic-policy DP evaluation. The sole treatment difference was learned-critic advantage versus fresh fixed-reference canonical PDR advantage.

## Primary result

- PDR minus Standard early update-AUC: mean `{primary['mean']:.12g}`, bootstrap 95% CI `[{primary['CI'][0]:.12g}, {primary['CI'][1]:.12g}]`.
- Positive-seed fraction: `{primary['positive_seed_fraction']:.3f}`.
- Primary numerical benefit gate: `{benefit}`.

## Reliability

- Final exact success difference: `{success_diff:.12g}` (gate >= {SUCCESS_TOL}).
- Final exact gate-capability difference: `{gate_diff:.12g}` (gate >= {GATE_TOL}).
- Reliability non-degradation: `{reliability}`.

## Secondary results

- Environment-step AUC common horizon: `{h_env}` real training transitions.
- PDR minus Standard environment-AUC: mean `{env_stats['mean']:.12g}`, 95% CI `[{env_stats['CI'][0]:.12g}, {env_stats['CI'][1]:.12g}]`.
- Environment sample-efficiency support: `{env_supported}`.
- Early PDR-minus-stale advantage-MAE: mean `{mechanism_stats['mean']:.12g}`, 95% CI `[{mechanism_stats['CI'][0]:.12g}, {mechanism_stats['CI'][1]:.12g}]`.
- Direct-A mechanism gate: `{mechanism}`.
- Maximum canonical PDR linear residual: `{result['maximum_PDR_solver_residual']:.12g}`.

See `SEED_LEVEL_SUMMARY.csv`, `LEARNING_CURVES.csv`, `ADVANTAGE_DIAGNOSTICS.csv`, and `PDR_SOLVE_DIAGNOSTICS.csv` for complete paired and checkpoint-level results.

## Scope and limitations

This is an in-domain clean SecureCourier reproduction. It is not a holdout/generalization, AntMaze, scalable-backend, wall-clock acceleration, rolling-reference, PDR-DI, or replay-priority result. M5 ends here.
"""
    (HERE / "FINAL_REPORT.md").write_text(report)
    make_plots(macro, seed_summary, junction_macro, advantages)
    files = [path for path in HERE.rglob("*") if path.is_file() and path.name != "OUTPUT_MANIFEST.json"]
    write_json(HERE / "OUTPUT_MANIFEST.json", {"created_at": now(), "verdict": verdict, "self_hash_excluded": True, "files": [{"path": str(path.relative_to(HERE)), "size": path.stat().st_size, "sha256": sha(path)} for path in sorted(files)]})
    print(json.dumps(clean(result), indent=2))


def make_plots(macro, seed_summary, junction, advantages):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plot_dir = HERE / "plots"; plot_dir.mkdir(exist_ok=True); colors = {"STANDARD_AWAC_CLEAN": "gray", "PDR_AWAC_CLEAN": "tab:blue"}
    def curve(data, x, y, name, ylabel=None):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm in ARMS:
            group = data[data.arm == arm].groupby(x)[y].agg(["mean", "sem"]).reset_index()
            ax.plot(group[x], group["mean"], label=arm, color=colors[arm]); ax.fill_between(group[x], group["mean"] - group["sem"], group["mean"] + group["sem"], color=colors[arm], alpha=0.18)
        if x == "u": ax.axvline(H_EARLY, ls="--", color="k", label="early boundary"); ax.axvline(H_EXTENDED, ls=":", color="k", label="extended boundary")
        ax.set(xlabel=x, ylabel=ylabel or y); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / name, dpi=170); plt.close(fig)
    curve(macro, "u", "J_exact", "01_exact_return_by_update.png", "exact discounted return")
    curve(macro, "training_env_transitions", "J_exact", "02_exact_return_by_environment_step.png", "exact discounted return")
    curve(macro, "u", "P_success_exact", "04_success_by_update.png", "exact success probability")
    curve(macro, "u", "P_gate_capability_exact", "05_gate_capability_by_update.png", "exact gate capability")
    curve(macro, "u", "P_shortcut_exact", "06_shortcut_route_by_update.png", "exact shortcut-route probability")
    curve(junction, "u", "p_shortcut", "07_junction_shortcut_probability.png", "junction shortcut probability")
    curve(advantages, "u", "A_PDR_MAE", "08_pdr_advantage_mae.png", "PDR advantage MAE")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.boxplot([seed_summary.STANDARD_AWAC_CLEAN_AUC_update_early, seed_summary.PDR_AWAC_CLEAN_AUC_update_early], tick_labels=ARMS); ax.set_ylabel("early exact-return AUC"); fig.tight_layout(); fig.savefig(plot_dir / "03_early_auc_by_seed.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "create-lock", "run-seed", "finalize"]); parser.add_argument("--seed-index", type=int); args = parser.parse_args()
    {"prepare": prepare, "create-lock": create_lock, "run-seed": lambda: run_seed(args.seed_index), "finalize": finalize}[args.command]()
