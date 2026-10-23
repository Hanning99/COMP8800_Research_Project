#!/usr/bin/env python3
"""
SecureCourier Phase 1: G1 (old-policy gate) + G2 (capability acquisition).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase1.py \
        --config configs/secure_courier_v1.yaml \
        --outdir results/ \
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- project path ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

from secure_courier.generator import generate_pilot_split, InstanceConfig
from secure_courier.env import SecureCourierEnv
from secure_courier.state import EnvState
from secure_courier.task_graph import (
    ORACLE_SHORTCUT_ACTIONS, ORACLE_SAFE_ACTIONS,
    GATE_SEQUENCE, JUNCTION, GATE_LOBBY, START_NODE, DELIVERED, GRAPH,
    admissible_actions as struct_admissible,
    step_transition,
)

# Maps each gate node to the correct gate-sequence action at that node.
# Used for epsilon-greedy gate forcing during online curriculum.
_GATE_NODE_TO_GATE_ACTION = {
    "gate_lobby":  GATE_SEQUENCE[0],  # scan_badge
    "gate_badged": GATE_SEQUENCE[1],  # enter_code
    "gate_coded":  GATE_SEQUENCE[2],  # open_gate
}
from secure_courier.oracle import oracle_shortcut_plan, oracle_safe_plan

from agents.text_encoder import Vocab
from agents.actor import Actor
from agents.critic import Critic
from agents.checkpoint import save_checkpoint, load_checkpoint, load_actor_only
from agents.awac import awac_step

from data.collect_offline import collect_offline_dataset, save_dataset
from data.schemas import Transition

from capability.evaluate_capability import (
    evaluate_checkpoint,
    evaluate_junction_action_probs,
    make_env_policy,
)
from rq1.exact_value_table import (
    compute_true_value_table,
    make_dp_policy_from_actor,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _env_meta(device):
    import importlib.metadata
    def _ver(p):
        try:
            return importlib.metadata.version(p)
        except Exception:
            return "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "gymnasium": _ver("gymnasium"),
        "numpy": _ver("numpy"),
        "device": str(device),
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def _param_hash(model):
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: List[Dict] = []
        self.capacity = capacity

    def add_transition(self, tr: Dict):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append(tr)

    def sample(self, batch_size: int, rng: random.Random) -> List[Dict]:
        return rng.choices(self.buf, k=batch_size)

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def build_vocab(transitions: List[Transition], all_instances: List[InstanceConfig]) -> Vocab:
    vocab = Vocab(max_size=2000)
    for tr in transitions:
        vocab.add_text(tr.observation)
        for a in tr.admissible_actions:
            vocab.add_text(a)
    # Also add all action display strings from all instances
    for inst in all_instances:
        for v in inst.action_display.values():
            vocab.add_text(v)
    return vocab


def encode_transition(tr: Dict, vocab: Vocab,
                      obs_max: int, act_max: int, device: torch.device):
    """
    Encode a single transition dict for the neural network.
    Returns dict of tensors (no batch dimension).
    """
    obs_ids  = vocab.encode(tr["observation"], obs_max)
    adm_disp = tr["admissible_actions"]
    nxt_disp = tr["next_admissible_actions"]

    act_ids  = [vocab.encode(a, act_max) for a in adm_disp]
    nact_ids = [vocab.encode(a, act_max) for a in nxt_disp] if nxt_disp else [[0]*act_max]

    return {
        "obs_tokens":      torch.tensor(obs_ids,  dtype=torch.long, device=device),
        "act_tokens":      torch.tensor(act_ids,  dtype=torch.long, device=device),
        "action_idx":      torch.tensor(tr["action"], dtype=torch.long, device=device),
        "reward":          torch.tensor(tr["reward"],  dtype=torch.float, device=device),
        "next_obs_tokens": torch.tensor(vocab.encode(tr["next_observation"], obs_max),
                                        dtype=torch.long, device=device),
        "next_act_tokens": torch.tensor(nact_ids, dtype=torch.long, device=device),
        "done":            torch.tensor(float(tr["done"]), dtype=torch.float, device=device),
    }


def collate_batch(samples: List[Dict], vocab: Vocab,
                  obs_max: int, act_max: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Collate a list of transition dicts into batched tensors.
    Pads action dimension to max N across the batch.
    """
    encoded = [encode_transition(s, vocab, obs_max, act_max, device) for s in samples]

    # Find max number of admissible actions (for padding)
    max_n  = max(e["act_tokens"].size(0)  for e in encoded)
    max_mn = max(e["next_act_tokens"].size(0) for e in encoded)

    def _pad_acts(tensor, max_n):
        n = tensor.size(0)
        if n < max_n:
            pad = torch.zeros(max_n - n, tensor.size(1), dtype=tensor.dtype, device=device)
            tensor = torch.cat([tensor, pad], dim=0)
        return tensor

    batch = {
        "obs_tokens":      torch.stack([e["obs_tokens"]  for e in encoded]),
        "act_tokens":      torch.stack([_pad_acts(e["act_tokens"],  max_n)  for e in encoded]),
        "action_idx":      torch.stack([e["action_idx"]  for e in encoded]),
        "reward":          torch.stack([e["reward"]       for e in encoded]),
        "next_obs_tokens": torch.stack([e["next_obs_tokens"] for e in encoded]),
        "next_act_tokens": torch.stack([_pad_acts(e["next_act_tokens"], max_mn) for e in encoded]),
        "done":            torch.stack([e["done"]         for e in encoded]),
    }
    return batch


# ---------------------------------------------------------------------------
# Online episode collector
# ---------------------------------------------------------------------------

def collect_online_episode(
    instances: List[InstanceConfig],
    actor,
    vocab: Vocab,
    device: torch.device,
    rng: random.Random,
    gamma: float,
    max_steps: int,
    obs_max: int,
    act_max: int,
    gate_curriculum_prob: float = 0.8,
    gate_epsilon: float = 0.0,
) -> List[Dict]:
    """
    Collect one online episode.

    80% of episodes start from gate-curriculum positions (sc2..sc5),
    20% from the full task start.

    gate_epsilon: probability of forcing the correct gate action at gate nodes
    (epsilon-greedy gate exploration to bootstrap critic Q values when actor
    has near-zero gate-action probability after offline BC pretraining).
    """
    inst = rng.choice(instances)
    env  = SecureCourierEnv(inst, gamma=gamma, max_steps=max_steps)
    env.reset()

    # Determine start point
    if rng.random() < gate_curriculum_prob:
        # Gate curriculum: navigate via oracle to a node 1-4 steps from gate_lobby
        sc_prefix = oracle_shortcut_plan()
        # Nodes 1-4 steps from gate_lobby on shortcut: sc5(1), sc4(2), sc3(3), sc2(4)
        curriculum_nodes = ["sc5", "sc4", "sc3", "sc2"]
        target_node = rng.choice(curriculum_nodes)
        for a in sc_prefix:
            if env.current_node() == target_node:
                break
            adm = env.admissible_structural()
            if a in adm:
                env.execute_structural(a)
            if env.is_done():
                break

    transitions = []
    step_in_episode = 0

    while not env.is_done() and env.step_count() < max_steps:
        state = env.clone_state()
        obs   = env._obs()
        adm_disp = obs["admissible_actions"]
        adm_str  = obs["admissible_structural"]
        if not adm_str:
            break

        # Actor selects action
        obs_ids = torch.tensor([vocab.encode(obs["text"], obs_max)],
                                dtype=torch.long, device=device)
        act_ids = torch.tensor(
            [[vocab.encode(a, act_max) for a in adm_disp]],
            dtype=torch.long, device=device,
        )
        actor.eval()
        with torch.no_grad():
            log_probs = actor(obs_ids, act_ids)
        idx = int(torch.distributions.Categorical(logits=log_probs[0]).sample().item())

        # Gate epsilon-greedy: at gate nodes, override with the correct gate
        # action with probability gate_epsilon to bootstrap critic Q learning.
        # AWAC uses actor log_prob for the chosen action (not behavior policy),
        # so this correctly provides high-advantage gate-completion transitions.
        node = state.node
        if gate_epsilon > 0 and node in _GATE_NODE_TO_GATE_ACTION:
            gate_action = _GATE_NODE_TO_GATE_ACTION[node]
            if gate_action in adm_str and rng.random() < gate_epsilon:
                idx = adm_str.index(gate_action)

        actor.train()
        action_str = adm_str[idx]
        log_p = float(log_probs[0, idx].item())

        obs_t, reward, terminated, truncated, info = env.step(idx)
        next_obs = env._obs()

        transitions.append({
            "transition_id": f"online_{id(inst)}_{step_in_episode}",
            "instance_id": inst.instance_id,
            "instance_seed": inst.instance_seed,
            "episode_id": f"online_ep",
            "t": step_in_episode,
            "observation": obs["text"],
            "admissible_actions": adm_disp,
            "admissible_structural": adm_str,
            "action": idx,
            "action_display": adm_disp[idx],
            "action_structural": action_str,
            "reward": float(reward),
            "next_observation": next_obs["text"],
            "next_admissible_actions": next_obs["admissible_actions"],
            "next_admissible_structural": next_obs["admissible_structural"],
            "done": env.is_done(),
            "behavior_policy_id": "online_actor",
            "behavior_log_prob": log_p,
            "route_label": "online",
        })
        step_in_episode += 1

    return transitions


# ---------------------------------------------------------------------------
# Main Phase 1 runner
# ---------------------------------------------------------------------------

def run_phase1(config_path: str, outdir: str, seed: int,
               warmstart_ckpt: Optional[str] = None,
               warmstart_dataset: Optional[str] = None) -> int:
    start_ts  = datetime.now(timezone.utc)
    run_id    = start_ts.strftime("%Y%m%dT%H%M%SZ")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gamma      = cfg["gamma"]
    max_steps  = cfg["max_steps"]
    batch_size = 64       # pilot batch size (smaller than 256 in config for speed)
    target_tau = cfg["target_tau"]
    awac_lam   = cfg["awac_lambda"]
    weight_clip= cfg["weight_clip"]
    lr         = 3e-4

    embed_dim  = 32
    hidden_dim = 64
    obs_max    = 128
    act_max    = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase1] device={device}  run_id={run_id}  seed={seed}")

    run_dir = Path(outdir) / f"{run_id}_phase1_g1_g2"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Record metadata
    _write_json(run_dir / "env_meta.json", _env_meta(device))
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    import shutil
    shutil.copy(config_path, run_dir / "config.yaml")

    rng    = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ======================================================================
    # 1. Pilot split
    # ======================================================================
    print("[Phase1] Generating pilot split...")
    splits = generate_pilot_split(
        seed_train=cfg["pilot_seed_train"],
        seed_val=cfg["pilot_seed_val"],
        seed_test=cfg["pilot_seed_test"],
        n_train=cfg["pilot_train"],
        n_val=cfg["pilot_val"],
        n_test=cfg["pilot_test"],
    )
    train_insts = splits["train"]
    val_insts   = splits["val"]
    all_insts   = train_insts + val_insts
    reference_instance = train_insts[0]
    print(f"[Phase1] train={len(train_insts)}, val={len(val_insts)}")

    # ======================================================================
    # 2. Offline dataset
    # ======================================================================
    if warmstart_dataset:
        print(f"[Phase1] Warm-start: loading dataset from {warmstart_dataset}")
        from data.collect_offline import load_dataset
        transitions = load_dataset(Path(warmstart_dataset))
        # Symlink into run dir for provenance (use copy if symlink fails)
        dset_path = run_dir / "offline_dataset.jsonl"
        import shutil as _shutil
        _shutil.copy(warmstart_dataset, dset_path)
    else:
        dset_path = run_dir / "offline_dataset.jsonl"
        if dset_path.exists():
            print("[Phase1] Loading existing offline dataset...")
            from data.collect_offline import load_dataset
            transitions = load_dataset(dset_path)
        else:
            print("[Phase1] Collecting offline dataset (~20k transitions)...")
            transitions = collect_offline_dataset(
                train_insts,
                target_transitions=20_000,
                seed=seed + 1,
                gamma=gamma,
                max_steps=max_steps,
            )
            save_dataset(transitions, dset_path)

    print(f"[Phase1] Dataset size: {len(transitions)} transitions")
    _write_json(run_dir / "dataset_stats.json", {
        "n_transitions": len(transitions),
        "route_label_counts": _count_labels(transitions),
    })

    # ======================================================================
    # 3. Build vocab & models
    # ======================================================================
    offline_steps = 3000
    train_logs: List[Dict] = []

    if warmstart_ckpt:
        # Load vocab from the prior checkpoint to preserve token indices exactly.
        import pickle as _pickle
        ws_path = Path(warmstart_ckpt)
        print(f"[Phase1] Warm-start: loading vocab+weights from {ws_path}")
        with open(ws_path / "vocab.pkl", "rb") as _f:
            vocab = _pickle.load(_f)
        print(f"[Phase1] Warm-start vocab size: {vocab.size}")
    else:
        print("[Phase1] Building vocabulary...")
        vocab = build_vocab(transitions, all_insts)
        print(f"[Phase1] Vocab size: {vocab.size}")

    actor         = Actor(vocab.size, embed_dim, hidden_dim).to(device)
    critic        = Critic(vocab.size, embed_dim, hidden_dim).to(device)
    target_critic = critic.make_target().to(device)

    actor_opt  = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    if warmstart_ckpt:
        ws_path = Path(warmstart_ckpt)
        actor.load_state_dict(torch.load(ws_path / "actor.pt", weights_only=True))
        critic.load_state_dict(torch.load(ws_path / "critic.pt", weights_only=True))
        target_critic.load_state_dict(torch.load(ws_path / "target_critic.pt", weights_only=True))
        if (ws_path / "actor_optimizer.pt").exists():
            actor_opt.load_state_dict(torch.load(ws_path / "actor_optimizer.pt", weights_only=True))
        if (ws_path / "critic_optimizer.pt").exists():
            critic_opt.load_state_dict(torch.load(ws_path / "critic_optimizer.pt", weights_only=True))
        print(f"[Phase1] Warm-start: loaded actor/critic states from {ws_path.name}")

    # Build offline replay buffer
    trans_dicts = [t.__dict__ if hasattr(t, '__dict__') else t.to_dict() for t in transitions]
    print("[Phase1] Building replay buffer...")
    offline_buf = ReplayBuffer(capacity=len(trans_dicts) + 1000)
    for tr in trans_dicts:
        offline_buf.add_transition(tr)
    print(f"[Phase1] Offline buffer: {len(offline_buf)} transitions")

    # ======================================================================
    # 4. Offline pretraining (Phase A) — skipped when warm-starting
    # ======================================================================
    if warmstart_ckpt:
        print("[Phase1] Warm-start: skipping offline pretraining.")
        # Save copy as offline_final for provenance / pair-selection
        _save_ckpt(ckpt_dir, "offline_final", actor, critic, target_critic,
                   actor_opt, critic_opt, vocab, offline_steps, seed, "warmstart", save_full=True)
    else:
        print("[Phase1] Starting offline pretraining...")
        log_interval  = 200
        ckpt_interval = 500
        actor.train()
        critic.train()
        for step in range(1, offline_steps + 1):
            samples = offline_buf.sample(batch_size, rng)
            batch   = collate_batch(samples, vocab, obs_max, act_max, device)
            metrics = awac_step(
                actor, critic, target_critic,
                actor_opt, critic_opt, batch,
                gamma=gamma, awac_lambda=awac_lam,
                weight_clip=weight_clip, target_tau=target_tau,
            )
            metrics["step"] = step
            metrics["phase"] = "offline"
            train_logs.append(metrics)
            if step % log_interval == 0:
                print(f"  [offline step {step}/{offline_steps}] "
                      f"critic_loss={metrics['critic_loss']:.4f} "
                      f"actor_loss={metrics['actor_loss']:.4f} "
                      f"q_mean={metrics['q_mean']:.4f}")
            if step % ckpt_interval == 0:
                _save_ckpt(ckpt_dir, f"offline_step{step}", actor, critic, target_critic,
                           actor_opt, critic_opt, vocab, step, seed, "offline")
        # Save final offline checkpoint (full: includes optimizers for potential continuation)
        _save_ckpt(ckpt_dir, "offline_final", actor, critic, target_critic,
                   actor_opt, critic_opt, vocab, offline_steps, seed, "offline", save_full=True)

    # ======================================================================
    # 5. G1 evaluation (old policy gate)
    # ======================================================================
    print("\n[Phase1] === G1 Evaluation (old policy) ===")
    actor.eval()
    g1_metrics = evaluate_checkpoint(
        actor, vocab, val_insts, device,
        n_eval_episodes=100, seed=seed + 100,
        gamma=gamma, max_steps=max_steps,
        obs_max_len=obs_max, act_max_len=act_max,
    )
    g1_metrics["junction_action_probs"] = evaluate_junction_action_probs(
        actor, vocab, reference_instance, device,
        obs_max_len=obs_max, act_max_len=act_max,
    )
    print(f"  full_task_success:                 {g1_metrics['full_task_success']:.3f}  (req >= 0.60)")
    print(f"  safe_route_success:                {g1_metrics['safe_route_success']:.3f}  (req >= 0.80)")
    print(f"  shortcut_arrival_at_gate:          {g1_metrics['shortcut_arrival_at_gate']:.3f}  (req >= 0.80)")
    print(f"  gate_sequence_success:             {g1_metrics['gate_sequence_success']:.3f}  (req <= 0.10)")
    print(f"  forced_shortcut_full_task_success: {g1_metrics['forced_shortcut_full_task_success']:.3f}  (req <= 0.20)")

    g1_pass = _check_g1(g1_metrics)
    g1_metrics["g1_passed"] = g1_pass
    g1_metrics["training_step"] = offline_steps
    g1_metrics["checkpoint"] = "offline_final"
    _write_json(run_dir / "g1_eval.json", g1_metrics)
    print(f"  → G1: {'PASSED' if g1_pass else 'FAILED'}")

    # ======================================================================
    # 6. Online training (Phase B) — gate curriculum
    # ======================================================================
    print("\n[Phase1] === Online training (gate curriculum) ===")
    # Save the old checkpoint before online training
    _save_ckpt(ckpt_dir, "old_policy", actor, critic, target_critic,
               actor_opt, critic_opt, vocab, offline_steps, seed, "pre_online", save_full=True)

    online_buf   = ReplayBuffer(capacity=20_000)
    online_steps = 8000   # run until G2 gate passes (deterministic eval), max budget
    log_interval = 100
    ckpt_interval = 200
    g2_early_stop_step = None  # set when G2 first passes

    # Track gate success over time to find the pair
    online_eval_log: List[Dict] = []
    best_new_ckpt_step = None
    actor.train()
    critic.train()

    for step in range(1, online_steps + 1):
        # Collect one online episode
        online_trans = collect_online_episode(
            train_insts, actor, vocab, device, rng,
            gamma, max_steps, obs_max, act_max,
            gate_curriculum_prob=0.8,
            gate_epsilon=1.0,
        )
        for tr in online_trans:
            online_buf.add_transition(tr)

        if len(online_buf) < batch_size:
            continue

        # 10% offline / 90% online batch mix.
        # Offline shortcut_fail data (25% of offline buffer) carries many
        # wait_at_gate gate_lobby transitions that compete with gate-epsilon
        # scan_badge signal. With gate_epsilon=1.0 the online buffer has no
        # wait_at_gate at gate nodes; reducing offline ratio ensures scan_badge
        # AWAC weight dominates and the argmax flips cleanly.
        off_count    = max(1, int(batch_size * 0.1))
        on_count     = batch_size - off_count
        off_samples  = offline_buf.sample(off_count, rng)
        on_samples   = online_buf.sample(on_count, rng)
        batch = collate_batch(off_samples + on_samples, vocab, obs_max, act_max, device)

        metrics = awac_step(
            actor, critic, target_critic,
            actor_opt, critic_opt, batch,
            gamma=gamma, awac_lambda=awac_lam,
            weight_clip=weight_clip, target_tau=target_tau,
        )
        metrics["step"] = step
        metrics["phase"] = "online"
        train_logs.append(metrics)

        # Periodic evaluation
        if step % log_interval == 0:
            actor.eval()
            quick_metrics = evaluate_checkpoint(
                actor, vocab, val_insts, device,
                n_eval_episodes=40, seed=seed + 200 + step,
                gamma=gamma, max_steps=max_steps,
                obs_max_len=obs_max, act_max_len=act_max,
            )
            actor.train()
            record = {
                "online_step": step,
                "gate_sequence_success": quick_metrics["gate_sequence_success"],
                "full_task_success": quick_metrics["full_task_success"],
                "safe_route_success": quick_metrics["safe_route_success"],
                "forced_shortcut_full_task_success": quick_metrics["forced_shortcut_full_task_success"],
            }
            online_eval_log.append(record)
            print(f"  [online step {step}/{online_steps}] "
                  f"gate={quick_metrics['gate_sequence_success']:.2f} "
                  f"full={quick_metrics['full_task_success']:.2f} "
                  f"safe={quick_metrics['safe_route_success']:.2f}")

        if step % ckpt_interval == 0:
            _save_ckpt(ckpt_dir, f"online_step{step}", actor, critic, target_critic,
                       actor_opt, critic_opt, vocab, offline_steps + step, seed, "online")

        # Early stop: if G2 gate has been above threshold for 2 consecutive evals,
        # continue 200 more steps to confirm stability then stop
        if (g2_early_stop_step is None and len(online_eval_log) >= 2 and
                online_eval_log[-1]["gate_sequence_success"] >= 0.90 and
                online_eval_log[-2]["gate_sequence_success"] >= 0.90 and
                online_eval_log[-1]["forced_shortcut_full_task_success"] >= 0.80 and
                online_eval_log[-1]["safe_route_success"] >= 0.75):
            g2_early_stop_step = step
            print(f"  [G2 criteria met at step {step}! Training 200 more steps to confirm.]")
        if g2_early_stop_step is not None and step >= g2_early_stop_step + 200:
            print(f"  [Early stopping online training at step {step}]")
            break

    actual_online_steps = step
    # Save final online checkpoint (full)
    _save_ckpt(ckpt_dir, "online_final", actor, critic, target_critic,
               actor_opt, critic_opt, vocab, offline_steps + actual_online_steps, seed, "online",
               save_full=True)
    _write_json(run_dir / "online_eval_log.json", online_eval_log)

    # ======================================================================
    # 7. G2 evaluation (new policy gate)
    # ======================================================================
    print("\n[Phase1] === G2 Evaluation (new policy) ===")
    actor.eval()
    g2_metrics = evaluate_checkpoint(
        actor, vocab, val_insts, device,
        n_eval_episodes=100, seed=seed + 300,
        gamma=gamma, max_steps=max_steps,
        obs_max_len=obs_max, act_max_len=act_max,
    )
    g2_metrics["junction_action_probs"] = evaluate_junction_action_probs(
        actor, vocab, reference_instance, device,
        obs_max_len=obs_max, act_max_len=act_max,
    )
    print(f"  gate_sequence_success:             {g2_metrics['gate_sequence_success']:.3f}  (req >= 0.90)")
    print(f"  forced_shortcut_full_task_success: {g2_metrics['forced_shortcut_full_task_success']:.3f}  (req >= 0.80)")
    print(f"  safe_route_success:                {g2_metrics['safe_route_success']:.3f}  (req >= 0.75)")

    g2_pass = _check_g2(g2_metrics, g1_metrics)
    g2_metrics["g2_passed"] = g2_pass
    g2_metrics["checkpoint"] = "online_final"
    _write_json(run_dir / "g2_eval.json", g2_metrics)
    print(f"  → G2 capability metrics: {'PASSED' if g2_pass else 'FAILED'}")

    # ======================================================================
    # 8. Deterministic old/new pair selection (Section 9.4 rule)
    # ======================================================================
    print("\n[Phase1] === Pair selection ===")
    old_ckpt_name, new_ckpt_name, pair_info = _select_pair(
        ckpt_dir, offline_steps, online_eval_log
    )
    print(f"  old = {old_ckpt_name}  new = {new_ckpt_name}")
    _write_json(run_dir / "pair_selection.json", pair_info)

    # ======================================================================
    # 9. True value change table (G2 mechanism check)
    # ======================================================================
    print("\n[Phase1] === True value change table ===")

    # Load old policy (actor only — value table doesn't need critic)
    old_actor_net = Actor(vocab.size, embed_dim, hidden_dim).to(device)
    old_meta, old_vocab = load_actor_only(ckpt_dir / old_ckpt_name, old_actor_net)

    # Load new policy (actor only)
    new_actor_net = Actor(vocab.size, embed_dim, hidden_dim).to(device)
    new_meta, new_vocab = load_actor_only(ckpt_dir / new_ckpt_name, new_actor_net)
    old_actor_net.eval()
    new_actor_net.eval()

    value_table = compute_true_value_table(
        old_actor_net, new_actor_net,
        old_vocab, new_vocab,
        reference_instance, device,
        gamma=gamma, max_steps=max_steps,
        obs_max_len=obs_max, act_max_len=act_max,
    )
    _write_json(run_dir / "true_value_table.json", value_table)

    # Save as CSV for easy inspection
    if value_table:
        csv_path = run_dir / "true_value_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(value_table[0].keys()))
            writer.writeheader()
            writer.writerows(value_table)

    print("\n  Distance table (d = structural actions before gate):")
    print(f"  {'d':>3} {'node':>16} {'action':>20} {'Q_old':>8} {'Q_new':>8} "
          f"{'ΔQ':>8} {'A_old':>8} {'A_new':>8} {'sign?':>6}")
    for row in value_table:
        print(
            f"  {row['distance_to_gate']:>3} "
            f"{row['historical_state_id']:>16} "
            f"{row['historical_action']:>20} "
            f"{row['Q_true_old']:>8.4f} "
            f"{row['Q_true_new']:>8.4f} "
            f"{row['Delta_Q_true']:>8.4f} "
            f"{row['A_true_old']:>8.4f} "
            f"{row['A_true_new']:>8.4f} "
            f"{'YES' if row['advantage_sign_change'] else 'no':>6}"
        )

    # Check: propagation beyond d=1
    delta_qs = {row["distance_to_gate"]: row["Delta_Q_true"] for row in value_table}
    propagation_ok = _check_propagation(delta_qs)
    _write_json(run_dir / "propagation_check.json", {
        "delta_qs": delta_qs,
        "propagation_ok": propagation_ok,
    })
    print(f"\n  Propagation check (ΔQ > 0 at d>=2): {'PASS' if propagation_ok else 'FAIL'}")

    # ======================================================================
    # 10. Summarise
    # ======================================================================
    end_ts = datetime.now(timezone.utc)
    dur = (end_ts - start_ts).total_seconds()

    g2_value_pass = propagation_ok and any(
        row["Delta_Q_true"] > 0.01 for row in value_table
    )
    overall_pass = g1_pass and g2_pass and g2_value_pass

    status = "PASSED" if overall_pass else "FAILED"
    status_record = {
        "status": status,
        "g1_passed": g1_pass,
        "g2_capability_passed": g2_pass,
        "g2_value_change_passed": g2_value_pass,
        "propagation_ok": propagation_ok,
        "run_id": run_id,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "duration_seconds": dur,
    }
    _write_json(run_dir / "status.json", status_record)

    # Save training logs
    _write_json(run_dir / "train_logs.json", train_logs)

    print(f"\n[Phase1] {'='*60}")
    print(f"[Phase1] STATUS: {status}")
    print(f"[Phase1] G1 (old policy):         {'PASS' if g1_pass else 'FAIL'}")
    print(f"[Phase1] G2 capability:            {'PASS' if g2_pass else 'FAIL'}")
    print(f"[Phase1] G2 true value change:     {'PASS' if g2_value_pass else 'FAIL'}")
    print(f"[Phase1] Duration: {dur:.1f}s")
    print(f"[Phase1] Output: {run_dir}")

    _write_report(run_dir, run_id, cfg, seed, g1_metrics, g2_metrics,
                  value_table, delta_qs, propagation_ok,
                  g1_pass, g2_pass, g2_value_pass, online_eval_log,
                  old_ckpt_name, new_ckpt_name)

    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# G1/G2 gate checks
# ---------------------------------------------------------------------------

def _check_g1(m: Dict) -> bool:
    return (
        m["full_task_success"]                >= 0.60 and
        m["safe_route_success"]               >= 0.80 and
        m["shortcut_arrival_at_gate"]         >= 0.80 and
        m["gate_sequence_success"]            <= 0.10 and
        m["forced_shortcut_full_task_success"]<= 0.20
    )


def _check_g2(g2: Dict, g1: Dict) -> bool:
    return (
        g2["gate_sequence_success"]             >= 0.90 and
        g2["forced_shortcut_full_task_success"] >= 0.80 and
        g2["safe_route_success"]                >= 0.75 and
        # Old checkpoint still satisfies G1 (loaded from disk = same as g1_metrics)
        g1["gate_sequence_success"]             <= 0.10  # re-check
    )


def _check_propagation(delta_qs: Dict[int, float]) -> bool:
    """Check that ΔQ > 0 persists beyond d=1 (not boundary-local only)."""
    if not delta_qs:
        return False
    # Require positive ΔQ at d>=2
    high_d = [d for d in delta_qs if d >= 2]
    if not high_d:
        return False
    return any(delta_qs[d] > 1e-4 for d in high_d)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ckpt(ckpt_dir, name, actor, critic, tgt, ao, co, vocab, step, seed, phase,
               save_full: bool = False):
    """Save checkpoint.  Periodic checkpoints (save_full=False) omit optimizers
    and target_critic to save disk space (~450 KB vs ~1.1 MB per checkpoint)."""
    import pickle
    p = Path(ckpt_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(),  p / "actor.pt")
    with open(p / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    if save_full:
        torch.save(critic.state_dict(),         p / "critic.pt")
        torch.save(tgt.state_dict(),             p / "target_critic.pt")
        torch.save(ao.state_dict(),              p / "actor_optimizer.pt")
        torch.save(co.state_dict(),              p / "critic_optimizer.pt")
    import json
    (p / "meta.json").write_text(json.dumps({
        "step": step,
        "training_seed": seed,
        "training_phase": phase,
        "full_save": save_full,
    }, indent=2))


# ---------------------------------------------------------------------------
# Old/new pair selection (Section 9.4 rule)
# ---------------------------------------------------------------------------

def _select_pair(ckpt_dir: Path, offline_steps: int,
                 online_log: List[Dict]) -> Tuple[str, str, Dict]:
    """
    Section 9.4 rule:
      old = checkpoint before gate success first exceeds 0.10
      new = earliest checkpoint after old where gate >= 0.90 and
            forced_shortcut_full_task >= 0.80
    Does NOT use Q-staleness metrics.
    """
    # The offline final checkpoint is always the old policy candidate
    old_name = "offline_final"

    # Find new: first online step where gate >= 0.90 and forced_shortcut >= 0.80
    new_name = "online_final"  # fallback
    for record in online_log:
        step = record["online_step"]
        if (record["gate_sequence_success"] >= 0.90 and
                record["forced_shortcut_full_task_success"] >= 0.80):
            candidate = f"online_step{step}"
            if (ckpt_dir / candidate / "actor.pt").exists():
                new_name = candidate
                break

    info = {
        "old_checkpoint": old_name,
        "new_checkpoint": new_name,
        "selection_rule": (
            "old = offline_final (last checkpoint before online starts); "
            "new = earliest online checkpoint with gate>=0.90 and forced_shortcut>=0.80"
        ),
        "no_q_staleness_used": True,
    }
    return old_name, new_name, info


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_labels(transitions) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tr in transitions:
        label = tr.route_label if hasattr(tr, "route_label") else tr.get("route_label", "?")
        counts[label] = counts.get(label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Phase 1 report writer
# ---------------------------------------------------------------------------

def _write_report(run_dir, run_id, cfg, seed, g1, g2, value_table, delta_qs,
                  prop_ok, g1_pass, g2_cap_pass, g2_val_pass,
                  online_log, old_name, new_name):
    overall = "PASSED" if (g1_pass and g2_cap_pass and g2_val_pass) else "FAILED"

    def _s(ok): return "PASS" if ok else "**FAIL**"

    lines = [
        f"# Phase 1 / G1+G2 Report — {run_id}",
        "",
        f"**Overall status:** `{overall}`  ",
        f"**Run ID:** `{run_id}`  ",
        f"**Training seed:** {seed}  ",
        f"**Old checkpoint:** `{old_name}`  ",
        f"**New checkpoint:** `{new_name}`  ",
        "",
        "---",
        "",
        "## G1 — Old Policy Gate",
        "",
        "| Metric | Value | Threshold | Result |",
        "|---|---|---|---|",
        f"| full_task_success | {g1['full_task_success']:.3f} | >= 0.60 | {_s(g1['full_task_success']>=0.60)} |",
        f"| safe_route_success | {g1['safe_route_success']:.3f} | >= 0.80 | {_s(g1['safe_route_success']>=0.80)} |",
        f"| shortcut_arrival_at_gate | {g1['shortcut_arrival_at_gate']:.3f} | >= 0.80 | {_s(g1['shortcut_arrival_at_gate']>=0.80)} |",
        f"| gate_sequence_success | {g1['gate_sequence_success']:.3f} | <= 0.10 | {_s(g1['gate_sequence_success']<=0.10)} |",
        f"| forced_shortcut_full_task | {g1['forced_shortcut_full_task_success']:.3f} | <= 0.20 | {_s(g1['forced_shortcut_full_task_success']<=0.20)} |",
        "",
        f"**G1 result: `{'PASSED' if g1_pass else 'FAILED'}`**",
        "",
        "### Junction action probabilities (old policy)",
        "",
    ]
    jap = g1.get("junction_action_probs", {})
    for a, p in sorted(jap.items(), key=lambda x: -x[1]):
        lines.append(f"- `{a}`: {p:.4f}")

    lines += [
        "",
        "---",
        "",
        "## G2 — Capability Acquisition",
        "",
        "| Metric | Value | Threshold | Result |",
        "|---|---|---|---|",
        f"| gate_sequence_success | {g2['gate_sequence_success']:.3f} | >= 0.90 | {_s(g2['gate_sequence_success']>=0.90)} |",
        f"| forced_shortcut_full_task | {g2['forced_shortcut_full_task_success']:.3f} | >= 0.80 | {_s(g2['forced_shortcut_full_task_success']>=0.80)} |",
        f"| safe_route_success | {g2['safe_route_success']:.3f} | >= 0.75 | {_s(g2['safe_route_success']>=0.75)} |",
        "",
        f"**G2 capability result: `{'PASSED' if g2_cap_pass else 'FAILED'}`**",
        "",
        "### Junction action probabilities (new policy)",
        "",
    ]
    jap2 = g2.get("junction_action_probs", {})
    for a, p in sorted(jap2.items(), key=lambda x: -x[1]):
        lines.append(f"- `{a}`: {p:.4f}")

    lines += [
        "",
        "---",
        "",
        "## G2 — True Value Change",
        "",
        "Pre-declared: ΔQ_true must be positive at d >= 2 (propagation beyond boundary).",
        "",
        f"**Propagation check: `{'PASSED' if prop_ok else 'FAILED'}`**",
        "",
        "| d | State | Action | Q_old | Q_new | ΔQ | A_old | A_new | Sign? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in value_table:
        lines.append(
            f"| {row['distance_to_gate']} "
            f"| `{row['historical_state_id']}` "
            f"| `{row['historical_action']}` "
            f"| {row['Q_true_old']:.4f} "
            f"| {row['Q_true_new']:.4f} "
            f"| {row['Delta_Q_true']:.4f} "
            f"| {row['A_true_old']:.4f} "
            f"| {row['A_true_new']:.4f} "
            f"| {'YES' if row['advantage_sign_change'] else 'no'} |"
        )

    lines += [
        "",
        "ALFWorld failure-mode check: does ΔQ persist beyond d=1?",
        "",
    ]
    for d in sorted(delta_qs.keys()):
        dq = delta_qs[d]
        lines.append(f"- d={d}: ΔQ = {dq:.4f} ({'positive' if dq > 0 else 'negative/zero'})")

    lines += [
        "",
        "---",
        "",
        "## Online learning curve (gate success vs step)",
        "",
        "| Online step | gate_success | full_task | safe_route | forced_shortcut |",
        "|---|---|---|---|---|",
    ]
    for rec in online_log:
        lines.append(
            f"| {rec['online_step']} "
            f"| {rec['gate_sequence_success']:.3f} "
            f"| {rec['full_task_success']:.3f} "
            f"| {rec['safe_route_success']:.3f} "
            f"| {rec['forced_shortcut_full_task_success']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Pair selection",
        "",
        f"Old: `{old_name}` — last checkpoint before online training (offline_final).",
        f"New: `{new_name}` — earliest online checkpoint with gate>=0.90 and forced_shortcut>=0.80.",
        "No Q-staleness metrics used for selection.",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        f"**Phase 1 / G1+G2 status: `{overall}`**",
        "",
        f"- G1 (old policy gate): `{'PASSED' if g1_pass else 'FAILED'}`",
        f"- G2 capability: `{'PASSED' if g2_cap_pass else 'FAILED'}`",
        f"- G2 true value change + propagation: `{'PASSED' if g2_val_pass else 'FAILED'}`",
        "",
        "Machine-readable evidence:",
        f"- `results/{run_id}_phase1_g1_g2/g1_eval.json`",
        f"- `results/{run_id}_phase1_g1_g2/g2_eval.json`",
        f"- `results/{run_id}_phase1_g1_g2/true_value_table.json`",
        f"- `results/{run_id}_phase1_g1_g2/true_value_table.csv`",
        f"- `results/{run_id}_phase1_g1_g2/status.json`",
        "",
        "**Stop:** Do not proceed to RQ1/RQ2/RQ3 without human review.",
    ]

    (run_dir / "PHASE1_G1_G2_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[Phase1] Report: {run_dir / 'PHASE1_G1_G2_REPORT.md'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/secure_courier_v1.yaml")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--warmstart-ckpt", default=None,
                        help="Path to a prior offline_final checkpoint dir to skip offline pretraining")
    parser.add_argument("--warmstart-dataset", default=None,
                        help="Path to a prior offline_dataset.jsonl to skip data collection")
    args = parser.parse_args()
    sys.exit(run_phase1(args.config, args.outdir, args.seed,
                        warmstart_ckpt=args.warmstart_ckpt,
                        warmstart_dataset=args.warmstart_dataset))
