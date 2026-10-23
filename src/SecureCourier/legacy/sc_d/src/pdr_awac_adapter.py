#!/usr/bin/env python3
"""Masked, detached-weight SC-D actor adapter for Standard and fresh PDR."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from reverse_pdr import ClosedDAG, PDRResult, array_hash
from scd_common import ACT_MAX, GRADIENT_CLIP, PARENT, TEMPERATURE, WEIGHT_CLIP, module_hash, tensor_tree_hash

import sys

sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(PARENT / "src/vendor/diy_task_set"))
sys.path.insert(0, str(PARENT / "src/vendor"))
from pdr.masks.action_mask import masked_log_softmax  # noqa: E402
from pdr.projection.advantage import center_under_policy  # noqa: E402


def pair_key(instance_id: str, node: str, step: int, action: str) -> str:
    return f"{instance_id}\x1f{node}\x1f{step:02d}\x1f{action}"


class FactorizedPolicyEvaluator:
    """Exact-for-this-architecture fast policy evaluator, audited against native."""

    def __init__(self, dag: ClosedDAG, vocab, device: torch.device):
        self.dag = dag
        self.device = device
        token_rows = []
        for pair in dag.pairs:
            token_rows.append(vocab.encode(pair["action_display"], ACT_MAX))
        all_tokens = np.asarray(token_rows, dtype=np.int64)
        unique, inverse = np.unique(all_tokens, axis=0, return_inverse=True)
        self.unique_tokens = torch.as_tensor(unique, dtype=torch.long, device=device)
        self.pair_unique_action = torch.as_tensor(inverse, dtype=torch.long, device=device)
        self.groups = []
        for count in sorted({len(x) for x in dag.state_pairs}):
            indices = np.asarray([x for x in dag.state_pairs if len(x) == count])
            self.groups.append((indices, torch.as_tensor(indices, dtype=torch.long, device=device)))

    @torch.inference_mode()
    def evaluate(self, actor) -> np.ndarray:
        actor.eval()
        hidden = int(actor.act_enc.hidden_dim)
        weight = actor.score.weight
        if weight.ndim != 2 or tuple(weight.shape) != (1, hidden * 2):
            raise RuntimeError("ACTOR_FACTORIZATION_ARCHITECTURE_MISMATCH")
        action_h = actor.act_enc(self.unique_tokens[:, None, :]).squeeze(1)
        unique_scores = action_h @ weight[0, hidden:]
        output = np.zeros(len(self.dag.pairs), dtype=np.float64)
        for indices, tensor_indices in self.groups:
            logits = unique_scores[self.pair_unique_action[tensor_indices]]
            probabilities = torch.softmax(logits, dim=-1).double().cpu().numpy()
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            output[indices] = probabilities
        self.dag.validate_policy(output)
        return output


@dataclass(frozen=True)
class RefreshAudit:
    result: PDRResult
    pi_current: np.ndarray
    policy_seconds: float
    solve_seconds: float


class PDRService:
    def __init__(self, graph_path: Path, reference_dir: Path, vocab, device: torch.device):
        self.dag = ClosedDAG(graph_path)
        self.pi_ref = np.load(reference_dir / "pi_ref.npy", allow_pickle=False).astype(np.float64)
        self.qbar = np.load(reference_dir / "qbar.npy", allow_pickle=False).astype(np.float64)
        self.dag.validate_policy(self.pi_ref)
        if self.qbar.shape != (len(self.dag.pairs),) or not np.all(np.isfinite(self.qbar)):
            raise RuntimeError("REFERENCE_QBAR_INVALID")
        self.evaluator = FactorizedPolicyEvaluator(self.dag, vocab, device)

    def refresh(self, actor) -> RefreshAudit:
        started = time.perf_counter()
        pi_current = self.evaluator.evaluate(actor)
        policy_seconds = time.perf_counter() - started
        solve_started = time.perf_counter()
        result = self.dag.solve(self.pi_ref, pi_current, self.qbar)
        solve_seconds = time.perf_counter() - solve_started
        return RefreshAudit(result, pi_current, policy_seconds, solve_seconds)

    def lookup(self, raw_rows: list[dict], q_pdr: np.ndarray, device: torch.device, dtype: torch.dtype):
        if q_pdr.shape != (len(self.dag.pairs),):
            raise RuntimeError("PDR_LOOKUP_VALUE_SHAPE_MISMATCH")
        max_actions = max(len(row["admissible_structural"]) for row in raw_rows)
        values = torch.zeros((len(raw_rows), max_actions), dtype=dtype, device=device)
        mask = torch.zeros((len(raw_rows), max_actions), dtype=torch.bool, device=device)
        covered = 0
        for row_index, row in enumerate(raw_rows):
            state = row.get("full_state") or {}
            node = state.get("node")
            step = int(state.get("step_count"))
            actions = [str(x) for x in row["admissible_structural"]]
            if not 0 <= int(row["action"]) < len(actions):
                raise RuntimeError("BEHAVIOR_ACTION_OUTSIDE_LEGAL_SET")
            q_values = []
            for action in actions:
                key = pair_key(str(row["instance_id"]), str(node), step, action)
                index = self.dag.pair_index.get(key)
                if index is None:
                    raise RuntimeError(f"UNMAPPED_ACTOR_ROW:{key}")
                q_values.append(float(q_pdr[index]))
                covered += 1
            values[row_index, : len(actions)] = torch.tensor(q_values, dtype=dtype, device=device)
            mask[row_index, : len(actions)] = True
        return values, mask, covered


def actor_weight_update(
    *,
    arm: str,
    actor,
    critic,
    optimizer,
    batch: dict[str, torch.Tensor],
    raw_rows: list[dict],
    pdr_service: PDRService | None = None,
    refresh: RefreshAudit | None = None,
) -> dict:
    update_started = time.perf_counter()
    if arm not in {"STANDARD_TD", "FRESH_PDR"}:
        raise ValueError(arm)
    actor.train()
    critic.eval()
    raw_logp = actor(batch["obs_tokens"], batch["act_tokens"])
    logp = masked_log_softmax(raw_logp, batch["action_mask"])
    lookup_coverage = 0
    lookup_seconds = 0.0
    if arm == "STANDARD_TD":
        with torch.no_grad():
            q_values = critic(batch["obs_tokens"], batch["act_tokens"])
    else:
        if pdr_service is None or refresh is None:
            raise RuntimeError("PDR_SERVICE_OR_REFRESH_MISSING")
        lookup_started = time.perf_counter()
        q_values, lookup_mask, lookup_coverage = pdr_service.lookup(
            raw_rows, refresh.result.q_pdr_by_pair, batch["obs_tokens"].device, raw_logp.dtype
        )
        lookup_seconds = time.perf_counter() - lookup_started
        if not torch.equal(lookup_mask, batch["action_mask"]):
            raise RuntimeError("PDR_LOOKUP_MASK_MISMATCH")
    q_values = torch.where(batch["action_mask"], q_values, torch.zeros_like(q_values)).detach()
    advantage = center_under_policy(q_values, logp.exp(), batch["action_mask"]).detach()
    selected_logp = logp.gather(1, batch["action_idx"][:, None]).squeeze(1)
    selected_advantage = advantage.gather(1, batch["action_idx"][:, None]).squeeze(1)
    weights = torch.exp(selected_advantage / TEMPERATURE).clamp(max=WEIGHT_CLIP).detach()
    if weights.requires_grad or advantage.requires_grad or q_values.requires_grad:
        raise RuntimeError("PDR_OR_STANDARD_WEIGHT_NOT_DETACHED")
    loss = -(weights * selected_logp).sum() / selected_logp.shape[0]
    before = module_hash(actor)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = module_hash(actor)
    return {
        "arm": arm,
        "actor_loss": float(loss.detach()),
        "actor_gradient_norm": gradient_norm,
        "actor_loss_denominator": int(selected_logp.shape[0]),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "real_action_slots": int(batch["action_mask"].sum()),
        "padded_action_mass": float(logp.exp()[~batch["action_mask"]].sum()),
        "lookup_coverage": lookup_coverage,
        "lookup_seconds": lookup_seconds,
        "weights_detached": True,
        "actor_hash_before": before,
        "actor_hash_after": after,
        "actor_optimizer_hash_after": tensor_tree_hash(optimizer.state_dict()),
        "pi_current_hash": array_hash(refresh.pi_current) if refresh is not None else None,
        "source_hash": array_hash(refresh.result.source_by_state) if refresh is not None else None,
        "correction_hash": array_hash(refresh.result.correction_by_pair) if refresh is not None else None,
        "q_pdr_hash": array_hash(refresh.result.q_pdr_by_pair) if refresh is not None else None,
        "recurrence_residual_inf": refresh.result.residual_inf if refresh is not None else None,
        "policy_seconds": refresh.policy_seconds if refresh is not None else 0.0,
        "solve_seconds": refresh.solve_seconds if refresh is not None else 0.0,
        "actor_update_seconds": time.perf_counter() - update_started,
    }
