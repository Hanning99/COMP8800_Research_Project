#!/usr/bin/env python3
"""Outcome-free one-batch AWAC interface trace on a legacy checkpoint.

No optimizer step is taken.  The script demonstrates the exact legal-action,
advantage, weight, detachment, loss, and actor-gradient semantics selected for
the new SC-A/SC-B/SC-C implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch


DIY = Path("autoresearch/diy_task_set")
PDR = Path("autoresearch/PDR")
sys.path[:0] = [str(DIY), str(PDR)]

from agents.actor import Actor
from agents.critic import Critic
from pdr.adapters.awac import awac_actor_loss
from pdr.masks.action_mask import masked_log_softmax
from pdr.projection.advantage import center_under_policy
from scripts.run_phase1 import collate_batch


SOURCE_RUN = DIY / "results/20260805T062059Z_phase1_g1_g2"
CHECKPOINT = SOURCE_RUN / "checkpoints/old_policy"
DATASET = SOURCE_RUN / "offline_dataset.jsonl"
AUDIT_SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter))
            .detach()
            .flatten()
            for parameter in model.parameters()
        ]
    ).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with (CHECKPOINT / "vocab.pkl").open("rb") as handle:
        vocab = pickle.load(handle)
    actor = Actor(vocab.size, 32, 64)
    critic = Critic(vocab.size, 32, 64)
    actor.load_state_dict(torch.load(CHECKPOINT / "actor.pt", map_location="cpu", weights_only=True))
    critic.load_state_dict(torch.load(CHECKPOINT / "critic.pt", map_location="cpu", weights_only=True))
    actor.train()
    critic.eval()

    rows = []
    with DATASET.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    indices = random.Random(AUDIT_SEED).choices(range(len(rows)), k=64)
    samples = [rows[index] for index in indices]
    batch = collate_batch(samples, vocab, 128, 32, torch.device("cpu"))

    current_mask = torch.zeros(
        (len(samples), batch["act_tokens"].shape[1]), dtype=torch.bool
    )
    next_mask = torch.zeros(
        (len(samples), batch["next_act_tokens"].shape[1]), dtype=torch.bool
    )
    for row_index, row in enumerate(samples):
        current_mask[row_index, : len(row["admissible_actions"])] = True
        next_mask[row_index, : max(1, len(row["next_admissible_actions"]))] = True

    raw_actor_output = actor(batch["obs_tokens"], batch["act_tokens"])
    legacy_probabilities = raw_actor_output.exp()
    log_probabilities = masked_log_softmax(raw_actor_output, current_mask)
    probabilities = log_probabilities.exp()
    with torch.no_grad():
        q_values = critic(batch["obs_tokens"], batch["act_tokens"])
    advantages = center_under_policy(q_values, probabilities, current_mask)
    loss, diagnostics = awac_actor_loss(
        log_probabilities,
        batch["action_idx"],
        advantages,
        temperature=0.3,
        weight_clip=20.0,
        stop_gradient_weight=True,
        action_mask=current_mask,
    )

    chosen_advantage = advantages.gather(1, batch["action_idx"][:, None]).squeeze(1)
    chosen_log_probability = log_probabilities.gather(
        1, batch["action_idx"][:, None]
    ).squeeze(1)
    chosen_weight = torch.exp(chosen_advantage / 0.3).clamp(0, 20).detach()
    manual_loss = -(chosen_weight * chosen_log_probability).mean()

    actor.zero_grad(set_to_none=True)
    loss.backward()
    gradient = gradient_vector(actor)
    actor.zero_grad(set_to_none=True)

    illegal_mass_per_row = torch.where(
        current_mask,
        torch.zeros_like(legacy_probabilities),
        legacy_probabilities,
    ).sum(-1)
    legal_center_error = torch.abs((probabilities * advantages).sum(-1))
    selected_is_legal = current_mask.gather(
        1, batch["action_idx"][:, None]
    ).squeeze(1)

    result = {
        "status": "M2_ONE_BATCH_TRACE_PASS",
        "fresh_outcome": False,
        "optimizer_step_taken": False,
        "source": {
            "checkpoint": str(CHECKPOINT),
            "actor_sha256": sha256(CHECKPOINT / "actor.pt"),
            "critic_sha256": sha256(CHECKPOINT / "critic.pt"),
            "vocab_sha256": sha256(CHECKPOINT / "vocab.pkl"),
            "dataset": str(DATASET),
            "dataset_sha256": sha256(DATASET),
        },
        "batch": {
            "audit_seed": AUDIT_SEED,
            "size": len(samples),
            "sample_indices": indices,
            "maximum_action_slots": int(current_mask.shape[1]),
            "rows_with_padding": int((~current_mask).any(-1).sum()),
            "all_selected_actions_legal": bool(selected_is_legal.all()),
        },
        "semantics": {
            "critic_count": 1,
            "actor_value_source": "current critic after critic update",
            "td_bootstrap_value_source": "Polyak target critic",
            "centering": "full stochastic current-policy expectation over legal actions",
            "temperature": 0.3,
            "weight_clip": 20.0,
            "stop_gradient_weight": True,
            "actor_loss_reduction": "mean over the fixed batch rows",
            "target_tau": 0.005,
            "gradient_clip": 5.0,
        },
        "parity": {
            "adapter_loss": float(loss.detach()),
            "manual_loss": float(manual_loss.detach()),
            "absolute_loss_error": float(torch.abs(loss.detach() - manual_loss.detach())),
            "maximum_legal_centering_error": float(legal_center_error.max()),
            "clean_illegal_probability_mass": float(probabilities[~current_mask].sum()),
            "legacy_unmasked_illegal_mass_mean": float(illegal_mass_per_row.mean()),
            "legacy_unmasked_illegal_mass_max": float(illegal_mass_per_row.max()),
        },
        "actor_gradient": {
            "norm": float(torch.linalg.vector_norm(gradient)),
            "finite": bool(torch.isfinite(gradient).all()),
            "sha256": tensor_hash(gradient),
        },
        "diagnostics": asdict(diagnostics),
    }
    if result["parity"]["absolute_loss_error"] != 0.0:
        raise RuntimeError("actor loss parity failure")
    if result["parity"]["clean_illegal_probability_mass"] != 0.0:
        raise RuntimeError("illegal probability mass")
    if result["parity"]["maximum_legal_centering_error"] > 1e-6:
        raise RuntimeError("centering failure")
    if not result["actor_gradient"]["finite"]:
        raise RuntimeError("nonfinite actor gradient")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "parity", "actor_gradient")}, indent=2))


if __name__ == "__main__":
    main()
