#!/usr/bin/env python3
"""SC-B actor-frozen ordinary-TD replay repair runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_sc_b_replay import ARMS, base_rows, flags
from scabc_common import (
    GRADIENT_CLIP,
    TEMPERATURE,
    WEIGHT_CLIP,
    clean_collate,
    critic_step,
    exact_instance_metrics,
    learned_state_action_values,
    load_model_bundle,
    module_hash,
    save_model_bundle,
    sha256,
    split_instances,
    tensor_tree_hash,
    update_target,
    write_json,
)


TOTAL_UPDATES = 4_530
EVAL_SCHEDULE = (0, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 3907, 4530)
BATCH_SIZE = 64


def weight(advantage: float) -> float:
    return float(min(WEIGHT_CLIP, math.exp(advantage / TEMPERATURE)))


def replay_view(source: Path, support: Path, arm: str) -> tuple[list[dict], dict]:
    manifest = json.loads((support / "SUPPORT_MANIFEST.json").read_text())
    rows, provenance = base_rows(source)
    expected = manifest["base_provenance"]
    for key in ("offline_sha256", "online_prefix_sha256", "online_prefix_rows"):
        if provenance[key] != expected[key]:
            raise RuntimeError(f"SC-B base replay provenance mismatch: {key}")
    if arm == "FULL_RELEVANT_REPLAY":
        selected = rows
    elif arm == "NO_LOCAL_ANCHOR":
        selected = [row for row in rows if not flags(row)[0]]
    else:
        selected = [row for row in rows if not flags(row)[1]]
    expected_arm = next(item for item in manifest["arms"] if item["arm"] == arm)
    if len(selected) != expected_arm["included_rows"]:
        raise RuntimeError("SC-B replay view count changed")
    return selected, expected_arm


def fixed_exact(actor, vocab, device) -> tuple[dict, dict]:
    instances = split_instances()["train"]
    current = {
        item.instance_id: exact_instance_metrics(actor, vocab, item, device) for item in instances
    }
    return current


def evaluate(actor, critic, vocab, device, old_exact: dict, current_exact: dict) -> dict:
    rows = []
    for instance in split_instances()["train"]:
        learned = learned_state_action_values(actor, critic, vocab, instance, device)
        exact = current_exact[instance.instance_id]
        old = old_exact[instance.instance_id]
        learned_a = learned["A"]["take_shortcut"]
        exact_a = exact["A_shortcut"]
        learned_rank = learned["rank"][0] == "take_shortcut"
        true_flip = old["A_shortcut"] < -0.02 and exact_a > 0.02
        rows.append(
            {
                "instance_id": instance.instance_id,
                "true_flip": true_flip,
                "harmful_stale": true_flip and learned_a <= 0.0,
                "sign_repaired": (not true_flip) or learned_a > 0.0,
                "rank_repaired": (not true_flip) or learned_rank,
                "exact_A_shortcut": exact_a,
                "learned_A_shortcut": learned_a,
                "advantage_error": exact_a - learned_a,
                "exact_Q_shortcut": exact["Q_shortcut"],
                "learned_Q_shortcut": learned["Q"]["take_shortcut"],
                "Q_error": exact["Q_shortcut"] - learned["Q"]["take_shortcut"],
                "exact_weight": weight(exact_a),
                "learned_weight": weight(learned_a),
                "absolute_weight_error": abs(weight(exact_a) - weight(learned_a)),
                "learned_rank_one": learned_rank,
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "macro": {
            key: mean(key)
            for key in (
                "true_flip",
                "harmful_stale",
                "sign_repaired",
                "rank_repaired",
                "exact_A_shortcut",
                "learned_A_shortcut",
                "advantage_error",
                "exact_Q_shortcut",
                "learned_Q_shortcut",
                "Q_error",
                "exact_weight",
                "learned_weight",
                "absolute_weight_error",
                "learned_rank_one",
            )
        },
        "instances": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("--max-updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--branch-manifest", type=Path)
    args = parser.parse_args()
    if args.max_updates not in EVAL_SCHEDULE:
        raise RuntimeError("max-updates must be a locked evaluation boundary")
    if args.mode == "confirmatory":
        for name in ("PREREGISTRATION_LOCK.json", "M7_EXECUTION_AUTHORIZATION.json"):
            if not (ROOT / "preregistration" / name).is_file():
                raise RuntimeError("confirmatory SC-B is locked until M6/M7 authorization")
        if args.branch_manifest is None:
            raise RuntimeError("confirmatory SC-B requires a frozen branch manifest")

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    source = args.source_event.resolve()
    support = args.support.resolve()
    out = args.out.resolve()
    checkpoint_path = source / "capability_checkpoint"
    if args.branch_manifest is not None:
        branch_path = args.branch_manifest.resolve()
        branch = json.loads(branch_path.read_text())
        if branch["source_event"] != str(source):
            raise RuntimeError("SC-B branch/source identity mismatch")
        if branch["support_manifest_sha256"] != sha256(support / "SUPPORT_MANIFEST.json"):
            raise RuntimeError("SC-B support changed after branch materialization")
        position = branch["execution_order"].index(args.arm)
        for prior in branch["execution_order"][:position]:
            if not (branch_path.parent / prior / "RUN_MANIFEST.json").is_file():
                raise RuntimeError(f"frozen SC-B arm order violation: {prior} must complete first")
        checkpoint_path = branch_path.parent / "branches" / args.arm / "branch_checkpoint"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty SC-B arm")
    out.mkdir(parents=True, exist_ok=True)
    replay, arm_support = replay_view(source, support, args.arm)
    actor, critic, target, actor_optimizer, critic_optimizer, vocab, metadata, rng = load_model_bundle(
        checkpoint_path, device
    )
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    initial = {
        "actor_hash": module_hash(actor),
        "critic_hash": module_hash(critic),
        "target_hash": module_hash(target),
        "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
        "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
        "source_checkpoint_manifest_sha256": sha256(source / "capability_checkpoint/manifest.json"),
    }
    old_actor, _, _, _, _, old_vocab, _, _ = load_model_bundle(source / "old/checkpoint", device)
    old_exact = {
        item.instance_id: exact_instance_metrics(old_actor, old_vocab, item, device)
        for item in split_instances()["train"]
    }
    current_exact = fixed_exact(actor, vocab, device)

    config = {
        "arm": args.arm,
        "mode": args.mode,
        "actor_frozen": True,
        "new_environment_collection": False,
        "total_registered_updates": TOTAL_UPDATES,
        "executed_updates": args.max_updates,
        "evaluation_schedule": list(EVAL_SCHEDULE),
        "batch_size": BATCH_SIZE,
        "critic_gradient_clip": GRADIENT_CLIP,
        "same_native_critic_step_as_source": True,
        "support_manifest_sha256": sha256(support / "SUPPORT_MANIFEST.json"),
        "support": arm_support,
    }
    write_json(out / "REPAIR_CONFIG.json", config)
    trace = []
    trace.append({"update": 0, **evaluate(actor, critic, vocab, device, old_exact, current_exact)})
    for update in range(1, args.max_updates + 1):
        batch = clean_collate(rng.choices(replay, k=BATCH_SIZE), vocab, device)
        metrics = critic_step(actor, critic, target, critic_optimizer, batch)
        update_target(target, critic)
        if update in EVAL_SCHEDULE:
            row = {"update": update, **evaluate(actor, critic, vocab, device, old_exact, current_exact)}
            row["last_update_metrics"] = metrics
            trace.append(row)
            save_model_bundle(
                out / f"checkpoints/update_{update:06d}",
                actor,
                critic,
                target,
                actor_optimizer,
                critic_optimizer,
                vocab,
                {
                    "arm": args.arm,
                    "mode": args.mode,
                    "critic_updates": update,
                    "actor_updates": 0,
                    "actor_frozen": True,
                    "source_acquisition_cycle": metadata["acquisition_cycle"],
                    "matching_branch_source": True,
                },
                rng,
                device,
            )
    write_json(out / "repair_trace.json", trace)
    write_json(
        out / "RUN_MANIFEST.json",
        {
            "status": "PILOT_COMPLETE" if args.mode == "pilot" else "SEALED_COMPLETE",
            "fresh_confirmatory_outcome": args.mode == "confirmatory",
            "arm": args.arm,
            "initial": initial,
            "final": {
                "actor_hash": module_hash(actor),
                "critic_hash": module_hash(critic),
                "target_hash": module_hash(target),
                "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
                "critic_updates": args.max_updates,
            },
            "actor_unchanged": module_hash(actor) == initial["actor_hash"],
            "trace_sha256": sha256(out / "repair_trace.json"),
            "config_sha256": sha256(out / "REPAIR_CONFIG.json"),
            "oracle_calls_during_training": 0,
        },
    )


if __name__ == "__main__":
    main()
