#!/usr/bin/env python3
"""Run neural-Qold Standard/static/fresh-PDR arms from a frozen branch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parents[1] / 'inputs/loca_neural'
PARENT = HERE.parent / "MiniGridLoCA"
sys.path.insert(0, str(PARENT / "src"))
os.environ.setdefault("MINIGRID_LOCA_PROTOCOL", str(PARENT / "PROTOCOL_LOCK.json"))

from minigrid_loca_core import bellman_residual, build_graph, evaluate_policy, exact_q, generalized_pdr  # noqa: E402
from run_study import (  # noqa: E402
    CHECKPOINTS,
    GAMMA,
    SAC_ALPHA,
    TAU,
    UPDATES,
    Actor,
    Critic,
    actor_objective,
    actor_signal_audit,
    all_state_tensor,
    extract_policy,
    polyak,
    schedules,
)


ARMS = ("A0_STANDARD_TD", "A1_STATIC_NEURAL_QOLD", "A2_FRESH_PDR_NEURAL_QOLD")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_branch(backbone: str, seed: int, evidence: str, device: torch.device):
    path = HERE / "branches" / evidence / backbone / f"seed_{seed}" / "BRANCH.pt"
    raw = torch.load(path, map_location=device, weights_only=False)
    actor = Actor().to(device)
    actor.load_state_dict(raw["actor"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    actor_optimizer.load_state_dict(raw["actor_optimizer"])
    count = 2 if backbone == "SAC" else 1
    critics, targets, optimizers = [], [], []
    for index in range(count):
        critic = Critic().to(device)
        critic.load_state_dict(raw["critics"][index])
        target = Critic().to(device)
        target.load_state_dict(raw["targets"][index])
        optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
        optimizer.load_state_dict(raw["critic_optimizers"][index])
        critics.append(critic)
        targets.append(target)
        optimizers.append(optimizer)
    return path, raw, actor, actor_optimizer, critics, targets, optimizers


def actor_facing_table(critics, all_states: torch.Tensor, backbone: str, graph) -> np.ndarray:
    with torch.no_grad():
        heads = [critic(all_states).cpu().numpy().astype(np.float64) for critic in critics]
    q = np.minimum(heads[0], heads[1]) if backbone == "SAC" else heads[0]
    q[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 0.0
    return q


def run(backbone: str, arm: str, seed: int, evidence: str, device: torch.device, max_cycles: int) -> None:
    cfg_path = HERE / ("PROTOCOL_LOCK.json" if (HERE / "PROTOCOL_LOCK.json").exists() else "PROTOCOL_DRAFT.json")
    cfg = json.loads(cfg_path.read_text())
    if evidence == "confirmatory" and (cfg.get("status") != "LOCKED_BEFORE_CONFIRMATORY_OUTCOMES" or max_cycles != UPDATES):
        raise RuntimeError("Confirmatory runs require the locked protocol and full cycle budget")
    branch_path, raw, actor, actor_optimizer, critics, targets, critic_optimizers = load_branch(backbone, seed, evidence, device)
    frozen_anchor_critics = [copy.deepcopy(critic).eval() for critic in critics]
    for critic in frozen_anchor_critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    graph = build_graph()
    all_states = all_state_tensor(graph, device)
    pi_old = np.asarray(raw["pi_old"], dtype=np.float64)
    alpha = SAC_ALPHA if backbone == "SAC" else 0.0
    neural_qold = actor_facing_table(frozen_anchor_critics, all_states, backbone, graph)
    anchor_residual = bellman_residual(graph, pi_old, neural_qold, graph.reward_a, GAMMA, alpha)
    if anchor_residual > float(cfg["qold_release_gate"]["bellman_residual_max"]):
        raise RuntimeError(f"ANCHOR_GATE_FAILED residual={anchor_residual}")
    schedule = schedules(graph, seed)
    successor = torch.as_tensor(graph.successor, dtype=torch.long, device=device)
    done = torch.as_tensor(graph.done, dtype=torch.float32, device=device)
    reward_b = torch.as_tensor(graph.reward_b, dtype=torch.float32, device=device)
    curves, update_rows, signal_rows, pdr_rows = [], [], [], []
    start = time.perf_counter()

    def policy_np() -> np.ndarray:
        return extract_policy(actor, all_states)

    def record(cycle: int) -> None:
        row = evaluate_policy(graph, policy_np(), GAMMA)
        row.update({"backbone": backbone, "arm": arm, "seed": seed, "cycle": cycle, "evidence": evidence})
        curves.append(row)

    record(0)
    for cycle in range(max_cycles):
        state = torch.as_tensor(schedule["critic_s"][cycle], dtype=torch.long, device=device)
        action = torch.as_tensor(schedule["critic_a"][cycle], dtype=torch.long, device=device)
        next_state = successor[state, action]
        terminal = done[state, action]
        reward = reward_b[state, action]
        with torch.no_grad():
            next_logp = actor.log_policy(next_state)
            next_policy = next_logp.exp()
            target_values = [target(next_state) for target in targets]
            if backbone == "SAC":
                target_q = torch.minimum(target_values[0], target_values[1])
                continuation = (next_policy * (target_q - SAC_ALPHA * next_logp)).sum(dim=1)
            else:
                continuation = (next_policy * target_values[0]).sum(dim=1)
            td_target = reward + GAMMA * (1.0 - terminal) * continuation
        critic_losses = []
        for critic, optimizer in zip(critics, critic_optimizers):
            prediction = critic(state).gather(1, action[:, None]).squeeze(1)
            critic_loss = F.mse_loss(prediction, td_target)
            optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

        actor_state = torch.as_tensor(schedule["actor_s"][cycle], dtype=torch.long, device=device)
        actor_action = torch.as_tensor(schedule["actor_a"][cycle], dtype=torch.long, device=device)
        current_policy = policy_np()
        correction = source = None
        pdr_residual = 0.0
        if arm == "A0_STANDARD_TD":
            values = [critic(actor_state) for critic in critics]
            actor_q = torch.minimum(values[0], values[1]) if backbone == "SAC" else values[0]
        elif arm == "A1_STATIC_NEURAL_QOLD":
            with torch.no_grad():
                values = [critic(actor_state) for critic in frozen_anchor_critics]
                actor_q = torch.minimum(values[0], values[1]) if backbone == "SAC" else values[0]
        elif arm == "A2_FRESH_PDR_NEURAL_QOLD":
            correction, source, pdr_residual = generalized_pdr(graph, pi_old, current_policy, neural_qold, GAMMA, alpha)
            actor_q = torch.as_tensor(neural_qold + correction, dtype=torch.float32, device=device)[actor_state]
        else:
            raise ValueError(arm)

        if cycle + 1 in CHECKPOINTS:
            oracle = exact_q(graph, current_policy, graph.reward_b, GAMMA, alpha)
            oracle_actor_q = torch.as_tensor(oracle, dtype=torch.float32, device=device)[actor_state]
            audit = actor_signal_audit(backbone, actor, actor_state, actor_action, actor_q, oracle_actor_q)
            audit.update({"backbone": backbone, "arm": arm, "seed": seed, "cycle": cycle + 1, "evidence": evidence})
            signal_rows.append(audit)
            if correction is not None:
                nonterminal = graph.nonterminal_states
                pdr_rows.append({
                    "backbone": backbone,
                    "arm": arm,
                    "seed": seed,
                    "cycle": cycle + 1,
                    "anchor_bellman_residual": anchor_residual,
                    "corrected_q_mean_abs_error": float(np.mean(np.abs((neural_qold + correction)[nonterminal] - oracle[nonterminal]))),
                    "corrected_q_max_abs_error": float(np.max(np.abs((neural_qold + correction)[nonterminal] - oracle[nonterminal]))),
                    "pdr_equation_residual": pdr_residual,
                    "source_abs_mean": float(np.mean(np.abs(source[nonterminal]))),
                    "d_abs_mean": float(np.mean(np.abs(correction[nonterminal]))),
                })
        actor_loss, centered, weights = actor_objective(backbone, actor, actor_state, actor_action, actor_q)
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()
        for target, critic in zip(targets, critics):
            polyak(target, critic)
        update_rows.append({
            "backbone": backbone,
            "arm": arm,
            "seed": seed,
            "cycle": cycle + 1,
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(actor_loss.detach().cpu()),
            "advantage_abs_mean": float(centered.abs().mean().detach().cpu()),
            "weight_mean": float(weights.mean().detach().cpu()),
        })
        if cycle + 1 in CHECKPOINTS:
            record(cycle + 1)

    output = HERE / "runs" / evidence / backbone / f"seed_{seed}" / arm
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_parquet(output / "LEARNING_CURVE.parquet", index=False)
    pd.DataFrame(update_rows).to_parquet(output / "UPDATE_DIAGNOSTICS.parquet", index=False)
    pd.DataFrame(signal_rows).to_parquet(output / "ACTOR_SIGNAL_DIAGNOSTICS.parquet", index=False)
    pd.DataFrame(pdr_rows).to_parquet(output / "PDR_DIAGNOSTICS.parquet", index=False)
    status = {
        "status": "COMPLETE",
        "evidence": evidence,
        "backbone": backbone,
        "arm": arm,
        "seed": seed,
        "cycles": max_cycles,
        "closure": 1.0,
        "dropped_policy_mass": 0.0,
        "anchor_bellman_residual": anchor_residual,
        "branch_checkpoint_sha256": sha256(branch_path),
        "initial_actor_hash": hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for _, v in sorted(raw["actor"].items()))).hexdigest(),
        "schedule_hash": hashlib.sha256(b"".join(schedule[key].tobytes() for key in sorted(schedule))).hexdigest(),
        "ordinary_policy_differential_pdr": arm == "A2_FRESH_PDR_NEURAL_QOLD",
        "full_residual_repair_used": False,
        "exact_q_used_for_training": False,
        "wall_seconds": time.perf_counter() - start,
        "peak_cpu_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "protocol_sha256": sha256(cfg_path),
        "runner_sha256": sha256(Path(__file__)),
    }
    (output / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("AWAC", "SAC", "EXPECTED_SARSA_QAC"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--max-cycles", type=int, default=UPDATES)
    args = parser.parse_args()
    device = torch.device(os.environ.get("MINIGRID_LOCA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    run(args.backbone, args.arm, args.seed, args.evidence, device, min(args.max_cycles, UPDATES))


if __name__ == "__main__":
    main()
