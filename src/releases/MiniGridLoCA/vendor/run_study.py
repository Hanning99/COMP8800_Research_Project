#!/usr/bin/env python3
"""Prepare and run the MiniGridLoCA three-backbone confirmatory study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from minigrid_loca_core import (
    bellman_residual,
    build_graph,
    evaluate_policy,
    exact_q,
    generalized_pdr,
    optimal_q,
    task_a_old_policy,
)


ROOT = Path(__file__).resolve().parents[1] / 'inputs/loca_exact'
CONFIG_PATH = Path(os.environ.get("MINIGRID_LOCA_PROTOCOL", ROOT / "PROTOCOL_DRAFT.json")).resolve()
CFG = json.loads(CONFIG_PATH.read_text())
GAMMA = float(CFG["environment"]["discount"])
UPDATES = int(CFG["learner_cycles"])
EARLY = int(CFG["early_auc_end"])
CHECKPOINTS = [int(v) for v in CFG["checkpoints"]]
BATCH = int(CFG["batch_size"])
ACTOR_LR = float(CFG["actor_learning_rate"])
CRITIC_LR = float(CFG["critic_learning_rate"])
TAU = float(CFG["target_polyak_tau"])
AWAC_BETA = float(CFG["awac_beta"])
AWAC_CLIP = float(CFG["awac_weight_clip"])
SAC_ALPHA = float(CFG["sac_alpha"])
BACKBONES = tuple(CFG["backbones"])
ARMS = tuple(CFG["arms"])
ARTIFACT_VERSION = str(CFG.get("artifact_version", "v1"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_dict_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 128)
        self.projection = nn.Sequential(
            nn.Linear(128, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 512), nn.LeakyReLU(0.01),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(states))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.head = nn.Linear(512, 3)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(states))

    def log_policy(self, states: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self(states), dim=-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.head = nn.Linear(512, 3)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(states))


def all_state_tensor(graph, device: torch.device) -> torch.Tensor:
    return torch.arange(graph.n_states, dtype=torch.long, device=device)


def extract_policy(actor: Actor, all_states: torch.Tensor, chunk: int = 256) -> np.ndarray:
    with torch.no_grad():
        values = [F.softmax(actor(all_states[start : start + chunk]), dim=-1).cpu() for start in range(0, len(all_states), chunk)]
    return torch.cat(values).numpy().astype(np.float64)


def task_value(graph, policy: np.ndarray, reward: np.ndarray) -> float:
    q = exact_q(graph, policy, reward, GAMMA)
    return float((policy * q).sum(axis=1)[graph.nonterminal_states].mean())


def set_determinism(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def prepare_common_actor(seed: int, output: Path, device: torch.device) -> dict:
    graph = build_graph()
    all_states = all_state_tensor(graph, device)
    states = torch.as_tensor(graph.nonterminal_states, dtype=torch.long, device=device)
    target_policy_np = task_a_old_policy(graph, GAMMA)
    target_policy = torch.as_tensor(target_policy_np[graph.nonterminal_states], dtype=torch.float32, device=device)
    set_determinism(seed, device)
    actor = Actor().to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    actor_gate_step = None
    actor_error = float("inf")
    actor_max_error = float("inf")
    optimal_q_a = optimal_q(graph, graph.reward_a, GAMMA)
    optimal_return = float(optimal_q_a.max(axis=1)[graph.nonterminal_states].mean())
    for step in range(1, int(CFG["pretraining"]["actor_steps_max"]) + 1):
        logp = actor.log_policy(states)
        loss = -(target_policy * logp).sum(axis=1).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor_optimizer.step()
        if step % 25 == 0:
            policy = extract_policy(actor, all_states)
            absolute_error = np.abs(policy[graph.nonterminal_states] - target_policy_np[graph.nonterminal_states])
            actor_error = float(np.mean(absolute_error))
            actor_max_error = float(np.max(absolute_error))
            ratio = task_value(graph, policy, graph.reward_a) / optimal_return
            if actor_error <= float(CFG["pretraining"]["actor_mean_absolute_probability_error_gate"]) and ratio >= float(CFG["pretraining"]["task_a_policy_optimality_ratio_gate"]):
                actor_gate_step = step
                break
    if actor_gate_step is None:
        raise RuntimeError(f"ACTOR_RELEASE_GATE_FAILED seed={seed} error={actor_error}")
    pi_old = extract_policy(actor, all_states)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "common_actor.pt"
    payload = {
        "actor": actor.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
        "pi_old": pi_old, "seed": seed, "protocol_sha256": sha256(CONFIG_PATH),
    }
    torch.save(payload, checkpoint)
    result = {
        "status": "PASS", "seed": seed, "actor_gate_step": actor_gate_step,
        "actor_mean_absolute_probability_error": actor_error,
        "actor_max_probability_error_diagnostic": actor_max_error,
        "task_a_optimality_ratio": task_value(graph, pi_old, graph.reward_a) / optimal_return,
        "actor_hash": state_dict_hash(actor), "pi_old_hash": hashlib.sha256(pi_old.tobytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_json(output / "COMMON_ACTOR_STATUS.json", result)
    return result


def prepare_branch(backbone: str, seed: int, output: Path, device: torch.device, evidence: str) -> dict:
    graph = build_graph()
    all_states = all_state_tensor(graph, device)
    states = torch.as_tensor(graph.nonterminal_states, dtype=torch.long, device=device)
    common_path = ROOT / "branches" / ARTIFACT_VERSION / evidence / f"seed_{seed}" / "common" / "common_actor.pt"
    common = torch.load(common_path, map_location=device, weights_only=False)
    actor = Actor().to(device)
    actor.load_state_dict(common["actor"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    pi_old = np.asarray(common["pi_old"], dtype=np.float64)
    alpha = SAC_ALPHA if backbone == "SAC" else 0.0
    qold = exact_q(graph, pi_old, graph.reward_a, GAMMA, alpha)
    critic_count = 2 if backbone == "SAC" else 1
    critics = []
    targets = []
    optimizers = []
    critic_steps = []
    critic_errors = []
    qold_t = torch.as_tensor(qold[graph.nonterminal_states], dtype=torch.float32, device=device)
    scale = float(np.mean(np.abs(qold[graph.nonterminal_states])) + 1e-12)
    for index in range(critic_count):
        set_determinism(seed + 10_000 * (index + 1), device)
        critic = Critic().to(device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
        passed = None
        nmae = float("inf")
        for step in range(1, int(CFG["pretraining"]["critic_steps_max"]) + 1):
            predicted = critic(states)
            loss = F.mse_loss(predicted, qold_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 25 == 0:
                with torch.no_grad():
                    nmae = float(torch.mean(torch.abs(critic(states) - qold_t)).cpu()) / scale
                if nmae <= float(CFG["pretraining"]["critic_normalized_mae_gate"]):
                    passed = step
                    break
        if passed is None:
            raise RuntimeError(f"CRITIC_RELEASE_GATE_FAILED backbone={backbone} seed={seed} head={index} nmae={nmae}")
        critics.append(critic)
        targets.append(copy.deepcopy(critic))
        optimizers.append(optimizer)
        critic_steps.append(passed)
        critic_errors.append(nmae)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critics": [module.state_dict() for module in critics],
        "targets": [module.state_dict() for module in targets],
        "critic_optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "pi_old": pi_old,
        "qold": qold,
        "seed": seed,
        "backbone": backbone,
        "protocol_sha256": sha256(CONFIG_PATH),
    }
    checkpoint = output / "branch.pt"
    torch.save(payload, checkpoint)
    result = {
        "status": "PASS",
        "backbone": backbone,
        "seed": seed,
        "common_actor_checkpoint_sha256": sha256(common_path),
        "critic_gate_steps": critic_steps,
        "critic_normalized_mae": critic_errors,
        "qold_bellman_residual": bellman_residual(graph, pi_old, qold, graph.reward_a, GAMMA, alpha),
        "actor_hash": state_dict_hash(actor),
        "critic_hashes": [state_dict_hash(module) for module in critics],
        "pi_old_hash": hashlib.sha256(pi_old.tobytes()).hexdigest(),
        "qold_hash": hashlib.sha256(qold.tobytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_json(output / "BRANCH_STATUS.json", result)
    return result


def schedules(graph, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 300_000)
    states = graph.nonterminal_states
    return {
        "critic_s": rng.choice(states, (UPDATES, BATCH), replace=True).astype(np.int64),
        "critic_a": rng.integers(0, graph.n_actions, (UPDATES, BATCH), dtype=np.int64),
        "actor_s": rng.choice(states, (UPDATES, BATCH), replace=True).astype(np.int64),
        "actor_a": rng.integers(0, graph.n_actions, (UPDATES, BATCH), dtype=np.int64),
    }


def actor_objective(backbone: str, actor: Actor, state_batch: torch.Tensor, actions: torch.Tensor, q: torch.Tensor):
    logp = actor.log_policy(state_batch)
    policy = logp.exp()
    centered = q.detach() - (policy.detach() * q.detach()).sum(axis=1, keepdim=True)
    if backbone == "AWAC":
        weights = torch.clamp(torch.exp(centered / AWAC_BETA), max=AWAC_CLIP)
        loss = -(weights.gather(1, actions[:, None]).squeeze(1) * logp.gather(1, actions[:, None]).squeeze(1)).mean()
    elif backbone == "EXPECTED_SARSA_QAC":
        weights = torch.ones_like(centered)
        loss = -(policy.detach() * logp * centered).sum(axis=1).mean()
    elif backbone == "SAC":
        weights = torch.ones_like(centered)
        loss = (policy * (SAC_ALPHA * logp - q.detach())).sum(axis=1).mean()
    else:
        raise ValueError(backbone)
    return loss, centered, weights


def actor_gradient_vector(loss: torch.Tensor, actor: Actor) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()), retain_graph=True, allow_unused=False)
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])


def actor_signal_audit(
    backbone: str,
    actor: Actor,
    states: torch.Tensor,
    actions: torch.Tensor,
    actor_q: torch.Tensor,
    oracle_q: torch.Tensor,
) -> dict[str, float]:
    candidate_loss, candidate_advantage, _ = actor_objective(backbone, actor, states, actions, actor_q)
    oracle_loss, oracle_advantage, _ = actor_objective(backbone, actor, states, actions, oracle_q)
    candidate_gradient = actor_gradient_vector(candidate_loss, actor)
    oracle_gradient = actor_gradient_vector(oracle_loss, actor)
    denominator = float(torch.linalg.vector_norm(candidate_gradient) * torch.linalg.vector_norm(oracle_gradient))
    cosine = float(torch.dot(candidate_gradient, oracle_gradient) / max(denominator, 1e-30))
    relative_l2 = float(
        torch.linalg.vector_norm(candidate_gradient - oracle_gradient)
        / torch.clamp(torch.linalg.vector_norm(oracle_gradient), min=1e-30)
    )
    return {
        "centered_advantage_max_abs_error": float(torch.max(torch.abs(candidate_advantage - oracle_advantage))),
        "centered_advantage_mean_abs_error": float(torch.mean(torch.abs(candidate_advantage - oracle_advantage))),
        "actor_gradient_cosine_to_exact_current_q": cosine,
        "actor_gradient_relative_l2_to_exact_current_q": relative_l2,
        "actor_gradient_norm": float(torch.linalg.vector_norm(candidate_gradient)),
        "oracle_actor_gradient_norm": float(torch.linalg.vector_norm(oracle_gradient)),
    }


def polyak(target: nn.Module, live: nn.Module) -> None:
    with torch.no_grad():
        for target_parameter, live_parameter in zip(target.parameters(), live.parameters()):
            target_parameter.mul_(1.0 - TAU).add_(live_parameter, alpha=TAU)


def load_branch(backbone: str, seed: int, device: torch.device, evidence: str):
    path = ROOT / "branches" / ARTIFACT_VERSION / evidence / f"seed_{seed}" / backbone / "branch.pt"
    raw = torch.load(path, map_location=device, weights_only=False)
    actor = Actor().to(device)
    actor.load_state_dict(raw["actor"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    actor_optimizer.load_state_dict(raw["actor_optimizer"])
    critic_count = 2 if backbone == "SAC" else 1
    critics, targets, optimizers = [], [], []
    for index in range(critic_count):
        critic = Critic().to(device)
        critic.load_state_dict(raw["critics"][index])
        target = Critic().to(device)
        target.load_state_dict(raw["targets"][index])
        optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
        optimizer.load_state_dict(raw["critic_optimizers"][index])
        critics.append(critic)
        targets.append(target)
        optimizers.append(optimizer)
    return raw, actor, actor_optimizer, critics, targets, optimizers


def run_arm(backbone: str, arm: str, seed: int, output: Path, device: torch.device, evidence: str) -> None:
    graph = build_graph()
    all_states = all_state_tensor(graph, device)
    raw, actor, actor_optimizer, critics, targets, critic_optimizers = load_branch(backbone, seed, device, evidence)
    pi_old = np.asarray(raw["pi_old"], dtype=np.float64)
    qold = np.asarray(raw["qold"], dtype=np.float64)
    alpha = SAC_ALPHA if backbone == "SAC" else 0.0
    schedule = schedules(graph, seed)
    successor = torch.as_tensor(graph.successor, dtype=torch.long, device=device)
    done = torch.as_tensor(graph.done, dtype=torch.float32, device=device)
    reward_b = torch.as_tensor(graph.reward_b, dtype=torch.float32, device=device)
    qold_table = torch.as_tensor(qold, dtype=torch.float32, device=device)
    curves, updates, pdr_rows, actor_signal_rows = [], [], [], []
    max_q_error = 0.0
    max_pdr_residual = 0.0
    start = time.perf_counter()

    def policy_np() -> np.ndarray:
        return extract_policy(actor, all_states)

    def record(cycle: int) -> None:
        metrics = evaluate_policy(graph, policy_np(), GAMMA)
        metrics.update({"backbone": backbone, "arm": arm, "seed": seed, "cycle": cycle, "evidence": evidence})
        curves.append(metrics)

    record(0)
    for cycle in range(UPDATES):
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
                target_min = torch.minimum(target_values[0], target_values[1])
                continuation = (next_policy * (target_min - SAC_ALPHA * next_logp)).sum(axis=1)
            else:
                continuation = (next_policy * target_values[0]).sum(axis=1)
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
        exact_current = None
        if arm == "A0_STANDARD_TD":
            live = [critic(actor_state) for critic in critics]
            actor_q = torch.minimum(live[0], live[1]) if backbone == "SAC" else live[0]
        elif arm == "A1_STATIC_EXACT_QOLD":
            actor_q = qold_table[actor_state]
        elif arm == "A2_FRESH_GENERALIZED_PDR":
            correction, source, residual = generalized_pdr(graph, pi_old, current_policy, qold, GAMMA, alpha)
            exact_current = exact_q(graph, current_policy, graph.reward_b, GAMMA, alpha)
            error = float(np.max(np.abs(qold + correction - exact_current)))
            max_q_error = max(max_q_error, error)
            max_pdr_residual = max(max_pdr_residual, residual)
            actor_q = torch.as_tensor(qold + correction, dtype=torch.float32, device=device)[actor_state]
            if cycle + 1 in CHECKPOINTS:
                pdr_rows.append({
                    "backbone": backbone, "arm": arm, "seed": seed, "cycle": cycle + 1,
                    "corrected_q_max_abs_error": error, "pdr_residual": residual,
                    "source_abs_mean": float(np.mean(np.abs(source))),
                    "d_abs_mean": float(np.mean(np.abs(correction))), "evidence": evidence,
                })
        else:
            raise ValueError(arm)
        if cycle + 1 in CHECKPOINTS:
            if exact_current is None:
                exact_current = exact_q(graph, current_policy, graph.reward_b, GAMMA, alpha)
            oracle_actor_q = torch.as_tensor(exact_current, dtype=torch.float32, device=device)[actor_state]
            signal = actor_signal_audit(backbone, actor, actor_state, actor_action, actor_q, oracle_actor_q)
            signal.update({
                "backbone": backbone,
                "arm": arm,
                "seed": seed,
                "cycle": cycle + 1,
                "evidence": evidence,
            })
            actor_signal_rows.append(signal)
        actor_loss, centered, weights = actor_objective(backbone, actor, actor_state, actor_action, actor_q)
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        gradient_norm = float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in actor.parameters() if parameter.grad is not None)).cpu())
        actor_optimizer.step()
        for target, critic in zip(targets, critics):
            polyak(target, critic)
        updates.append({
            "backbone": backbone, "arm": arm, "seed": seed, "cycle": cycle + 1,
            "critic_loss": float(np.mean(critic_losses)), "actor_loss": float(actor_loss.detach().cpu()),
            "actor_gradient_norm": gradient_norm, "advantage_abs_mean": float(centered.abs().mean().detach().cpu()),
            "weight_mean": float(weights.mean().detach().cpu()), "evidence": evidence,
        })
        if cycle + 1 in CHECKPOINTS:
            record(cycle + 1)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_parquet(output / "LEARNING_CURVE.parquet", index=False)
    pd.DataFrame(updates).to_parquet(output / "UPDATE_DIAGNOSTICS.parquet", index=False)
    pd.DataFrame(pdr_rows).to_parquet(output / "PDR_DIAGNOSTICS.parquet", index=False)
    pd.DataFrame(actor_signal_rows).to_parquet(output / "ACTOR_SIGNAL_DIAGNOSTICS.parquet", index=False)
    n_nonterminal_edges = int(len(graph.nonterminal_states) * graph.n_actions)
    critic_transition_evaluations = int(UPDATES * BATCH)
    pdr_dependency_edge_evaluations = int(UPDATES * 4 * n_nonterminal_edges) if arm == "A2_FRESH_GENERALIZED_PDR" else 0
    status = {
        "status": "COMPLETE", "evidence": evidence, "backbone": backbone, "arm": arm, "seed": seed,
        "cycles": UPDATES, "closure": 1.0, "dropped_policy_mass": 0.0,
        "max_corrected_q_error": max_q_error, "max_pdr_residual": max_pdr_residual,
        "wall_seconds": time.perf_counter() - start,
        "peak_cpu_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "critic_transition_evaluations": critic_transition_evaluations,
        "pdr_dependency_edge_evaluations": pdr_dependency_edge_evaluations,
        "total_transition_dependency_edge_evaluations": critic_transition_evaluations + pdr_dependency_edge_evaluations,
        "actor_neural_examples": int(UPDATES * BATCH),
        "critic_neural_examples": int(UPDATES * BATCH * len(critics)),
        "pdr_dense_linear_solves": int(UPDATES) if arm == "A2_FRESH_GENERALIZED_PDR" else 0,
        "scoring_exact_dp_solves_excluded_from_method_cost": int(UPDATES if arm == "A2_FRESH_GENERALIZED_PDR" else len(CHECKPOINTS)),
        "initial_actor_hash": hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for _, v in sorted(raw["actor"].items()))).hexdigest(),
        "schedule_hash": hashlib.sha256(b"".join(schedule[key].tobytes() for key in sorted(schedule))).hexdigest(),
        "protocol_sha256": sha256(CONFIG_PATH),
    }
    write_json(output / "STATUS.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_actor = sub.add_parser("prepare-actor")
    prepare_actor.add_argument("--seed", type=int, required=True)
    prepare_actor.add_argument("--evidence", choices=["engineering", "confirmatory"], required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--backbone", choices=BACKBONES, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--evidence", choices=["engineering", "confirmatory"], required=True)
    run = sub.add_parser("run")
    run.add_argument("--backbone", choices=BACKBONES, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--evidence", choices=["engineering", "confirmatory"], required=True)
    args = parser.parse_args()
    device = torch.device(os.environ.get("MINIGRID_LOCA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.command == "prepare-actor":
        destination = ROOT / "branches" / ARTIFACT_VERSION / args.evidence / f"seed_{args.seed}" / "common"
        prepare_common_actor(args.seed, destination, device)
    elif args.command == "prepare":
        destination = ROOT / "branches" / ARTIFACT_VERSION / args.evidence / f"seed_{args.seed}" / args.backbone
        prepare_branch(args.backbone, args.seed, destination, device, args.evidence)
    else:
        destination = ROOT / "runs" / ARTIFACT_VERSION / args.evidence / args.backbone / f"seed_{args.seed}" / args.arm
        run_arm(args.backbone, args.arm, args.seed, destination, device, args.evidence)


if __name__ == "__main__":
    main()
