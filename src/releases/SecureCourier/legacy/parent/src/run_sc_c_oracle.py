#!/usr/bin/env python3
"""Common SC-C closed-loop learner with a narrowly scoped actor-weight adapter."""

from __future__ import annotations

import argparse
import importlib
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

from build_sc_b_replay import base_rows
from oracle_scoring import exact_rows
from scabc_common import (
    GRADIENT_CLIP,
    TEMPERATURE,
    WEIGHT_CLIP,
    ReplayBuffer,
    actor_values,
    capability_macro,
    clean_collate,
    collect_assisted_episode,
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


ARMS = ("STALE", "CORRECT", "DROP", "PLACEBO")
TOTAL_CYCLES = 2_048
INTERVENTION_END = 1_024
EVAL_INTERVAL = 16
OFFLINE_BATCH = 6
ONLINE_BATCH = 58
ONLINE_CAPACITY = 20_000


def append_jsonl(path: Path, value: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return handle.tell()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def preserve_and_truncate(path: Path, offset: int, archive: Path) -> None:
    if not path.exists() or path.stat().st_size <= offset:
        return
    archive.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as handle:
        handle.seek(offset)
        trailing = handle.read()
    (archive / f"{path.name}.trailing").write_bytes(trailing)
    with path.open("r+b") as handle:
        handle.truncate(offset)


def latest_complete_checkpoint(root: Path) -> Path:
    valid = []
    for path in root.glob("cycle_*"):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if sha256(path / "state.pt") == manifest["state_sha256"]:
                valid.append(path)
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    if not valid:
        raise RuntimeError("no complete SC-C checkpoint available for resume")
    return max(valid, key=lambda path: int(path.name.split("_")[-1]))


def actor_weight_step(
    *,
    arm: str,
    cycle: int,
    event_id: str,
    actor,
    critic,
    optimizer,
    batch,
    raw_rows: list[dict],
    vocab,
    device,
    h_uids: set[str],
    placebo_uids: set[str],
    treatment_log: Path,
    disable_treatment_adapter: bool = False,
) -> dict:
    """Apply exactly one arm-specific change to the native detached weights."""
    actor.train()
    critic.eval()
    logp, _, advantage = actor_values(actor, critic, batch)
    selected_logp = logp.gather(1, batch["action_idx"][:, None]).squeeze(1)
    selected_advantage = advantage.gather(1, batch["action_idx"][:, None]).squeeze(1)
    native_weight = torch.exp(selected_advantage / TEMPERATURE).clamp(max=WEIGHT_CLIP).detach()
    applied_weight = native_weight.clone()
    targeted_uids = h_uids if arm in {"CORRECT", "DROP"} else placebo_uids
    targeted_indices = [
        index for index, row in enumerate(raw_rows) if row["_scb_uid"] in targeted_uids
    ]
    treatment_calls = 0
    if arm == "DROP":
        if targeted_indices:
            applied_weight[targeted_indices] = 0.0
    elif arm in {"CORRECT", "PLACEBO"} and targeted_indices and not disable_treatment_adapter:
        # The treatment module is not imported on STALE or DROP execution paths.
        treatment = importlib.import_module("oracle_treatment")
        targeted_rows = [raw_rows[index] for index in targeted_indices]
        exact = treatment.treatment_weights(
            arm=arm,
            cycle=cycle,
            event_id=event_id,
            actor=actor,
            vocab=vocab,
            device=device,
            rows=targeted_rows,
            log_path=treatment_log,
        )
        for index in targeted_indices:
            applied_weight[index] = exact[raw_rows[index]["_scb_uid"]]
        treatment_calls = 1
    # The denominator remains the original batch size in all four arms.  DROP
    # zeros numerator terms and never resamples replacement rows.
    loss = -(applied_weight * selected_logp).sum() / selected_logp.shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    correction_l1 = float((applied_weight - native_weight).abs().sum())
    return {
        "actor_loss": float(loss.detach()),
        "actor_gradient_norm": gradient_norm,
        "actor_loss_denominator": int(selected_logp.shape[0]),
        "targeted_rows_in_batch": len(targeted_indices),
        "treatment_oracle_calls": treatment_calls,
        "correction_l1": correction_l1,
        "native_weight_mean": float(native_weight.mean()),
        "applied_weight_mean": float(applied_weight.mean()),
        "adapter_disabled": disable_treatment_adapter,
    }


def diagnostic_gradient(
    actor,
    critic,
    vocab,
    device,
    rows: list[dict],
    arm: str = "STALE",
    cycle: int = 0,
    h_uids: set[str] | None = None,
    placebo_uids: set[str] | None = None,
) -> dict:
    """Compute, but never apply, native and exact-current diagnostic gradients."""
    before = module_hash(actor)
    actor_was_training = actor.training
    critic_was_training = critic.training
    actor.train()
    critic.eval()
    batch = clean_collate(rows, vocab, device)
    logp, _, advantage = actor_values(actor, critic, batch)
    selected_logp = logp.gather(1, batch["action_idx"][:, None]).squeeze(1)
    selected_advantage = advantage.gather(1, batch["action_idx"][:, None]).squeeze(1)
    native_weight = torch.exp(selected_advantage / TEMPERATURE).clamp(max=WEIGHT_CLIP).detach()
    scored = exact_rows(actor, vocab, device, rows)
    exact_by_uid = {row["uid"]: row for row in scored}
    exact_weight = torch.tensor(
        [exact_by_uid[row["_scb_uid"]]["weight_exact"] for row in rows],
        dtype=selected_logp.dtype,
        device=device,
    )
    h_uids = h_uids or set()
    placebo_uids = placebo_uids or set()
    applied_weight = native_weight.clone()
    active = cycle <= INTERVENTION_END
    if active and arm == "CORRECT":
        for index, row in enumerate(rows):
            if row["_scb_uid"] in h_uids:
                applied_weight[index] = exact_weight[index]
    elif active and arm == "DROP":
        for index, row in enumerate(rows):
            if row["_scb_uid"] in h_uids:
                applied_weight[index] = 0.0
    elif active and arm == "PLACEBO":
        for index, row in enumerate(rows):
            if row["_scb_uid"] in placebo_uids:
                applied_weight[index] = exact_weight[index]
    native_loss = -(native_weight * selected_logp).mean()
    applied_loss = -(applied_weight * selected_logp).mean()
    exact_loss = -(exact_weight * selected_logp).mean()
    parameters = tuple(actor.parameters())
    native_grad = torch.autograd.grad(native_loss, parameters, retain_graph=True)
    applied_grad = torch.autograd.grad(applied_loss, parameters, retain_graph=True)
    exact_grad = torch.autograd.grad(exact_loss, parameters)
    native_flat = torch.cat([item.detach().reshape(-1) for item in native_grad])
    applied_flat = torch.cat([item.detach().reshape(-1) for item in applied_grad])
    exact_flat = torch.cat([item.detach().reshape(-1) for item in exact_grad])
    native_denominator = float(native_flat.norm() * exact_flat.norm())
    applied_denominator = float(applied_flat.norm() * exact_flat.norm())
    cosine = float(torch.dot(native_flat, exact_flat) / native_denominator) if native_denominator > 0 else None
    applied_cosine = (
        float(torch.dot(applied_flat, exact_flat) / applied_denominator)
        if applied_denominator > 0
        else None
    )
    affected_indices = [
        index for index, row in enumerate(rows) if row["_scb_uid"] in h_uids
    ]
    after = module_hash(actor)
    if before != after:
        raise RuntimeError("exact diagnostic gradient mutated actor")
    actor.train(actor_was_training)
    critic.train(critic_was_training)
    return {
        "native_exact_gradient_cosine": cosine,
        "actor_interface_exact_gradient_cosine": applied_cosine,
        "native_gradient_norm": float(native_flat.norm()),
        "actor_interface_gradient_norm": float(applied_flat.norm()),
        "exact_gradient_norm": float(exact_flat.norm()),
        "mean_absolute_weight_error": float((native_weight - exact_weight).abs().mean()),
        "actor_interface_mean_absolute_weight_error": float(
            (applied_weight - exact_weight).abs().mean()
        ),
        "affected_native_absolute_weight_error": (
            float((native_weight[affected_indices] - exact_weight[affected_indices]).abs().mean())
            if affected_indices
            else None
        ),
        "affected_actor_interface_absolute_weight_error": (
            float((applied_weight[affected_indices] - exact_weight[affected_indices]).abs().mean())
            if affected_indices
            else None
        ),
        "mean_log_weight_error": float(
            (native_weight.clamp_min(1e-12).log() - exact_weight.clamp_min(1e-12).log()).abs().mean()
        ),
        "diagnostic_gradient_applied": False,
        "actor_hash_unchanged": True,
        "batch_size": len(rows),
        "arm": arm,
        "cycle": cycle,
        "intervention_active": active,
    }


def evaluate(
    actor,
    critic,
    vocab,
    device,
    diagnostic_rows: list[dict],
    arm: str,
    cycle: int,
    h_uids: set[str],
    placebo_uids: set[str],
) -> dict:
    actor_hash_before = module_hash(actor)
    critic_hash_before = module_hash(critic)
    instances = []
    for instance in split_instances()["train"]:
        exact = exact_instance_metrics(actor, vocab, instance, device)
        learned = learned_state_action_values(actor, critic, vocab, instance, device)
        exact_a = exact["A_shortcut"]
        learned_a = learned["A"]["take_shortcut"]
        exact_weight = min(WEIGHT_CLIP, math.exp(exact_a / TEMPERATURE))
        learned_weight = min(WEIGHT_CLIP, math.exp(learned_a / TEMPERATURE))
        learned_rank = learned["rank"][0] == "take_shortcut"
        instances.append(
            {
                "instance_id": instance.instance_id,
                "J_exact": exact["J_exact"],
                "p_shortcut": exact["p_shortcut"],
                "full_start_shortcut_route_probability": exact["full_start_shortcut_route_probability"],
                "full_start_safe_route_probability": exact["full_start_safe_route_probability"],
                "exact_A_shortcut": exact_a,
                "learned_A_shortcut": learned_a,
                "exact_weight_shortcut": exact_weight,
                "learned_weight_shortcut": learned_weight,
                "absolute_weight_error_shortcut": abs(exact_weight - learned_weight),
                "harmful_stale": exact_a > 0.02 and learned_a <= 0.0,
                "rank_agreement": exact["shortcut_rank_one"] == learned_rank,
                "learned_rank_one": learned_rank,
                "shortcut_log_odds_vs_safe": math.log(
                    max(exact["p_shortcut"], 1e-12) / max(exact["p_safe"], 1e-12)
                ),
            }
        )
    capability = capability_macro(actor, vocab, split_instances()["train"], device)
    gradient = diagnostic_gradient(
        actor,
        critic,
        vocab,
        device,
        diagnostic_rows,
        arm=arm,
        cycle=cycle,
        h_uids=h_uids,
        placebo_uids=placebo_uids,
    )
    if actor_hash_before != module_hash(actor) or critic_hash_before != module_hash(critic):
        raise RuntimeError("SC-C scoring mutated training state")

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in instances]))

    return {
        "macro": {
            key: mean(key)
            for key in (
                "J_exact",
                "p_shortcut",
                "full_start_shortcut_route_probability",
                "full_start_safe_route_probability",
                "exact_A_shortcut",
                "learned_A_shortcut",
                "exact_weight_shortcut",
                "learned_weight_shortcut",
                "absolute_weight_error_shortcut",
                "harmful_stale",
                "rank_agreement",
                "learned_rank_one",
                "shortcut_log_odds_vs_safe",
            )
        },
        "capability": capability["macro"],
        "diagnostic_gradient": gradient,
        "instances": instances,
        "scoring_mutated_training_state": False,
    }


def load_registered_rows(source: Path) -> tuple[list[dict], list[dict], dict[str, dict]]:
    rows, _ = base_rows(source)
    offline = [row for row in rows if row["_scb_origin"] == "offline"]
    online = [row for row in rows if row["_scb_origin"] == "online"]
    return offline, online, {row["_scb_uid"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("--max-cycles", type=int, default=TOTAL_CYCLES)
    parser.add_argument("--disable-treatment-adapter", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-cycle", type=int)
    args = parser.parse_args()
    if args.max_cycles < 0 or args.max_cycles % EVAL_INTERVAL != 0 or args.max_cycles > TOTAL_CYCLES:
        raise RuntimeError("max-cycles must be a registered evaluation boundary")
    if args.disable_treatment_adapter and args.mode != "pilot":
        raise RuntimeError("adapter-disable is an interface pilot only")
    if args.pause_after_cycle is not None and args.pause_after_cycle % EVAL_INTERVAL != 0:
        raise RuntimeError("pause-after-cycle must be a registered checkpoint boundary")
    if args.mode == "confirmatory":
        for name in ("PREREGISTRATION_LOCK.json", "M7_EXECUTION_AUTHORIZATION.json"):
            if not (ROOT / "preregistration" / name).is_file():
                raise RuntimeError("confirmatory SC-C is locked until M6/M7 authorization")

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    event_root = args.event_root.resolve()
    panels = args.panels.resolve()
    branch_manifest = json.loads((event_root / "BRANCH_MANIFEST.json").read_text())
    if sha256(panels / "H_PANEL.json") != branch_manifest["H_panel_sha256"]:
        raise RuntimeError("H panel changed after branch materialization")
    if sha256(panels / "PLACEBO_PANEL.json") != branch_manifest["placebo_panel_sha256"]:
        raise RuntimeError("placebo panel changed after branch materialization")
    order = branch_manifest["execution_order"]
    position = order.index(args.arm)
    for prior in order[:position]:
        if not (event_root / prior / "RUN_MANIFEST.json").is_file():
            raise RuntimeError(f"frozen arm order violation: {prior} must complete first")
    arm_root = event_root / args.arm
    if (arm_root / "RUN_MANIFEST.json").exists():
        raise RuntimeError("refusing to overwrite a completed SC-C branch")
    if not args.resume and (arm_root / "training_log.jsonl").exists():
        raise RuntimeError("existing SC-C trace requires explicit --resume")
    source = Path(branch_manifest["source_event"])
    offline_rows, initial_online_rows, row_by_uid = load_registered_rows(source)
    h_panel = json.loads((panels / "H_PANEL.json").read_text())
    placebo_panel = json.loads((panels / "PLACEBO_PANEL.json").read_text())
    h_uids = {row["uid"] for row in h_panel["items"]}
    placebo_uids = {row["uid"] for row in placebo_panel["items"]}
    diagnostic_uids = json.loads((panels / "DIAGNOSTIC_BATCH.json").read_text())
    diagnostic_rows = [row_by_uid[uid] for uid in diagnostic_uids]
    branch_checkpoint_manifest = json.loads((arm_root / "branch_checkpoint/manifest.json").read_text())
    initial = {
        key: branch_checkpoint_manifest[key]
        for key in (
            "actor_hash",
            "critic_hash",
            "target_hash",
            "actor_optimizer_hash",
            "critic_optimizer_hash",
        )
    }
    offline = ReplayBuffer(len(offline_rows) + 1)
    offline.extend(offline_rows)
    config = {
        "arm": args.arm,
        "mode": args.mode,
        "total_registered_cycles": TOTAL_CYCLES,
        "intervention_cycles": [0, INTERVENTION_END],
        "evaluation_interval": EVAL_INTERVAL,
        "offline_online_batch": [OFFLINE_BATCH, ONLINE_BATCH],
        "continuation_recipe": "same assisted acquisition collector and clean AWAC update order as source",
        "update_order": ["collect", "insert", "critic", "arm actor weight", "actor", "Polyak", "evaluate"],
        "actor_loss_denominator": OFFLINE_BATCH + ONLINE_BATCH,
        "drop_resamples_rows": False,
        "disable_treatment_adapter": args.disable_treatment_adapter,
        "fully_resumable_at_evaluation_boundaries": True,
    }
    stream_path = arm_root / "continuation_stream.jsonl"
    training_path = arm_root / "training_log.jsonl"
    treatment_log = arm_root / "treatment_oracle_calls.jsonl"
    evaluation_path = arm_root / "evaluation_log.jsonl"
    config_path = arm_root / "RUN_CONFIG.json"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("SC-C config changed since checkpoint")
        checkpoint = latest_complete_checkpoint(arm_root / "checkpoints")
        actor, critic, target, actor_optimizer, critic_optimizer, vocab, resume_meta, rng = load_model_bundle(
            checkpoint, device
        )
        if resume_meta["arm"] != args.arm or resume_meta["mode"] != args.mode:
            raise RuntimeError("SC-C resume identity mismatch")
        interruption = arm_root / "interrupted_attempts" / f"after_cycle_{resume_meta['continuation_cycle']:06d}"
        for path, key in (
            (stream_path, "continuation_stream_byte_offset"),
            (training_path, "training_log_byte_offset"),
            (evaluation_path, "evaluation_log_byte_offset"),
            (treatment_log, "treatment_log_byte_offset"),
        ):
            preserve_and_truncate(path, int(resume_meta[key]), interruption)
        continuation_rows = read_jsonl(stream_path)
        if len(continuation_rows) != int(resume_meta["continuation_stream_rows"]):
            raise RuntimeError("SC-C continuation row count does not match checkpoint")
        online = ReplayBuffer(ONLINE_CAPACITY)
        online.extend(initial_online_rows)
        online.extend(continuation_rows)
        if len(online) != int(resume_meta["online_replay_size"]):
            raise RuntimeError("SC-C online replay size does not match checkpoint")
        start_cycle = int(resume_meta["continuation_cycle"])
        total_treatment_calls = int(resume_meta["total_treatment_calls"])
        total_targeted_rows = int(resume_meta["total_targeted_rows"])
        continuation_stream_rows = int(resume_meta["continuation_stream_rows"])
    else:
        write_json(config_path, config)
        actor, critic, target, actor_optimizer, critic_optimizer, vocab, source_meta, rng = load_model_bundle(
            arm_root / "branch_checkpoint", device
        )
        observed_initial = {
            "actor_hash": module_hash(actor),
            "critic_hash": module_hash(critic),
            "target_hash": module_hash(target),
            "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
            "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
        }
        if observed_initial != initial:
            raise RuntimeError("loaded branch state does not match manifest")
        online = ReplayBuffer(ONLINE_CAPACITY)
        online.extend(initial_online_rows)
        start_cycle = 0
        total_treatment_calls = 0
        total_targeted_rows = 0
        continuation_stream_rows = 0
        append_jsonl(
            evaluation_path,
            {
                "cycle": 0,
                **evaluate(
                    actor,
                    critic,
                    vocab,
                    device,
                    diagnostic_rows,
                    args.arm,
                    0,
                    h_uids,
                    placebo_uids,
                ),
            },
        )
    source_meta = json.loads((source / "capability_checkpoint/manifest.json").read_text())
    stream_offset = stream_path.stat().st_size if stream_path.exists() else 0
    training_offset = training_path.stat().st_size if training_path.exists() else 0
    evaluation_offset = evaluation_path.stat().st_size
    treatment_offset = treatment_log.stat().st_size if treatment_log.exists() else 0

    for cycle in range(start_cycle + 1, args.max_cycles + 1):
        collected, episode = collect_assisted_episode(
            actor,
            vocab,
            split_instances()["train"],
            device,
            rng,
            cycle,
            curriculum_probability=0.8,
            gate_epsilon=1.0,
        )
        for index, row in enumerate(collected):
            row["episode_id"] = f"continuation_c{cycle}"
            row["transition_id"] = f"continuation_c{cycle}_t{index}"
            row["_scb_uid"] = f"continuation:{cycle:06d}:{index:03d}"
            row["_scb_origin"] = "continuation"
            online.add(row)
            stream_offset = append_jsonl(stream_path, row)
            continuation_stream_rows += 1
        raw_batch = offline.sample(OFFLINE_BATCH, rng) + online.sample(ONLINE_BATCH, rng)
        batch = clean_collate(raw_batch, vocab, device)
        critic_metrics = critic_step(actor, critic, target, critic_optimizer, batch)
        active_arm = args.arm if cycle <= INTERVENTION_END else "STALE"
        actor_metrics = actor_weight_step(
            arm=active_arm,
            cycle=cycle,
            event_id=f"seed_{branch_manifest['source_seed']}",
            actor=actor,
            critic=critic,
            optimizer=actor_optimizer,
            batch=batch,
            raw_rows=raw_batch,
            vocab=vocab,
            device=device,
            h_uids=h_uids,
            placebo_uids=placebo_uids,
            treatment_log=treatment_log,
            disable_treatment_adapter=args.disable_treatment_adapter,
        )
        update_target(target, critic)
        total_treatment_calls += actor_metrics["treatment_oracle_calls"]
        total_targeted_rows += actor_metrics["targeted_rows_in_batch"]
        training_offset = append_jsonl(
            training_path,
            {
                "cycle": cycle,
                "episode": episode,
                "online_replay_size": len(online),
                **critic_metrics,
                **actor_metrics,
            },
        )
        treatment_offset = treatment_log.stat().st_size if treatment_log.exists() else 0
        if cycle % EVAL_INTERVAL == 0:
            evaluation_offset = append_jsonl(
                evaluation_path,
                {
                    "cycle": cycle,
                    **evaluate(
                        actor,
                        critic,
                        vocab,
                        device,
                        diagnostic_rows,
                        args.arm,
                        cycle,
                        h_uids,
                        placebo_uids,
                    ),
                },
            )
            save_model_bundle(
                arm_root / f"checkpoints/cycle_{cycle:06d}",
                actor,
                critic,
                target,
                actor_optimizer,
                critic_optimizer,
                vocab,
                {
                    "arm": args.arm,
                    "mode": args.mode,
                    "continuation_cycle": cycle,
                    "actor_updates": cycle,
                    "critic_updates": cycle,
                    "online_replay_size": len(online),
                    "continuation_stream_rows": continuation_stream_rows,
                    "continuation_stream_byte_offset": stream_offset,
                    "training_log_byte_offset": training_offset,
                    "evaluation_log_byte_offset": evaluation_offset,
                    "treatment_log_byte_offset": treatment_offset,
                    "total_treatment_calls": total_treatment_calls,
                    "total_targeted_rows": total_targeted_rows,
                    "source_acquisition_cycle": source_meta["acquisition_cycle"],
                    "branch_manifest_hash": branch_manifest["manifest_hash"],
                    "matching_actor_critic": True,
                },
                rng,
                device,
            )
            if args.pause_after_cycle is not None and cycle >= args.pause_after_cycle:
                write_json(
                    arm_root / "STATUS.json",
                    {
                        "status": "PILOT_PAUSED_AT_REGISTERED_CHECKPOINT",
                        "arm": args.arm,
                        "cycle": cycle,
                        "resume_required": True,
                    },
                )
                return
    evaluation_trace = read_jsonl(evaluation_path)
    write_json(arm_root / "evaluation_trace.json", evaluation_trace)
    oracle_log_lines = 0
    if treatment_log.exists():
        with treatment_log.open() as handle:
            oracle_log_lines = sum(1 for line in handle if line.strip())
    unauthorized = args.arm in {"STALE", "DROP"}
    if unauthorized and oracle_log_lines != 0:
        raise RuntimeError("unauthorized arm reached treatment oracle")
    write_json(
        arm_root / "RUN_MANIFEST.json",
        {
            "status": "PILOT_COMPLETE" if args.mode == "pilot" else "SEALED_COMPLETE",
            "fresh_confirmatory_outcome": args.mode == "confirmatory",
            "arm": args.arm,
            "initial": initial,
            "final": {
                "actor_hash": module_hash(actor),
                "critic_hash": module_hash(critic),
                "target_hash": module_hash(target),
                "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
                "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
                "cycles": args.max_cycles,
            },
            "H_panel_size": len(h_uids),
            "placebo_panel_size": len(placebo_uids),
            "total_targeted_rows_in_actor_batches": total_targeted_rows,
            "treatment_oracle_calls": oracle_log_lines,
            "treatment_call_counter_matches_log": total_treatment_calls == oracle_log_lines,
            "unauthorized_treatment_oracle_calls": oracle_log_lines if unauthorized else 0,
            "evaluation_trace_sha256": sha256(arm_root / "evaluation_trace.json"),
            "training_log_sha256": sha256(training_path) if training_path.exists() else None,
            "continuation_stream_sha256": sha256(stream_path) if stream_path.exists() else None,
            "config_sha256": sha256(arm_root / "RUN_CONFIG.json"),
            "resumed": args.resume,
            "fully_resumable": True,
        },
    )


if __name__ == "__main__":
    main()
