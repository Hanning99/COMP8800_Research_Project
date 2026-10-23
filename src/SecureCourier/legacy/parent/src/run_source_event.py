#!/usr/bin/env python3
"""Generate one natural matching actor-critic SecureCourier source event.

The first capability checkpoint is selected solely from deterministic capability
metrics.  Exact DP is called only after that selection to evaluate structural
eligibility.  Learned critic staleness is never an eligibility input.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/vendor/diy_task_set"))

from agents.actor import Actor
from agents.critic import Critic
from data.collect_offline import collect_offline_dataset, load_dataset, save_dataset

from scabc_common import (
    ACT_MAX,
    EMBED_DIM,
    GAMMA,
    HIDDEN_DIM,
    LR,
    MAX_STEPS,
    OBS_MAX,
    ReplayBuffer,
    build_vocab,
    capability_macro,
    clean,
    clean_awac_step,
    clean_collate,
    collect_assisted_episode,
    load_model_bundle,
    module_hash,
    now_utc,
    save_model_bundle,
    seed_everything,
    sha256,
    split_instances,
    structural_eligibility,
    tensor_tree_hash,
    write_json,
)


OFFLINE_TRANSITIONS = 20_000
OFFLINE_UPDATES = 3_000
BATCH_SIZE = 64
OFFLINE_BATCH = 6
ONLINE_BATCH = 58
ONLINE_CAPACITY = 20_000
SOURCE_MAX_CYCLES = 8_000
CHECKPOINT_INTERVAL = 16
CAPABILITY_EVAL_INTERVAL = 16
CAPABILITY_PERSISTENCE = 3
CURRICULUM_PROBABILITY = 0.8
GATE_EPSILON = 1.0


def append_jsonl(path: Path, value: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), sort_keys=True, allow_nan=False) + "\n"
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


def source_config() -> dict:
    return {
        "offline_transitions": OFFLINE_TRANSITIONS,
        "offline_updates": OFFLINE_UPDATES,
        "batch_size": BATCH_SIZE,
        "offline_batch": OFFLINE_BATCH,
        "online_batch": ONLINE_BATCH,
        "online_capacity": ONLINE_CAPACITY,
        "source_max_cycles": SOURCE_MAX_CYCLES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "capability_eval_interval": CAPABILITY_EVAL_INTERVAL,
        "capability_persistence": CAPABILITY_PERSISTENCE,
        "capability": {
            "gate_sequence_success": ">=0.90",
            "forced_shortcut_full_task_success": ">=0.80",
            "safe_route_success": ">=0.90",
            "evaluation": "one deterministic greedy evaluation per each of 16 train surfaces",
            "selection": "first checkpoint in first run of three consecutive passing evaluations",
        },
        "old_behavior": {
            "safe_route_success": ">=0.90",
            "gate_sequence_success": "<=0.10",
            "forced_shortcut_full_task_success": "<=0.20",
        },
        "curriculum_probability": CURRICULUM_PROBABILITY,
        "curriculum_nodes": ["sc5", "sc4", "sc3", "sc2"],
        "gate_epsilon": GATE_EPSILON,
        "optimizer": "Adam",
        "actor_lr": LR,
        "critic_lr": LR,
        "gamma": GAMMA,
        "max_steps": MAX_STEPS,
        "obs_max": OBS_MAX,
        "act_max": ACT_MAX,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "clean_awac_semantics": "M2_AWAC_INTERFACE_LOCK.md",
    }


def old_behavior_pass(metrics: dict) -> tuple[bool, dict]:
    macro = metrics["macro"]
    gates = {
        "safe_route_success_ge_0.90": macro["safe_route_success"] >= 0.90,
        "gate_sequence_success_le_0.10": macro["gate_sequence_success"] <= 0.10,
        "forced_shortcut_success_le_0.20": macro["forced_shortcut_full_task_success"] <= 0.20,
    }
    return all(gates.values()), gates


def capability_pass(metrics: dict) -> tuple[bool, dict]:
    macro = metrics["macro"]
    gates = {
        "gate_sequence_success_ge_0.90": macro["gate_sequence_success"] >= 0.90,
        "forced_shortcut_success_ge_0.80": macro["forced_shortcut_full_task_success"] >= 0.80,
        "safe_route_success_ge_0.90": macro["safe_route_success"] >= 0.90,
    }
    return all(gates.values()), gates


def dataset_rows(path: Path) -> list[dict]:
    return [item.to_dict() for item in load_dataset(path)]


def write_cycle_rows(path: Path, rows: list[dict]) -> int:
    offset = 0
    for row in rows:
        offset = append_jsonl(path, row)
    return offset


def complete_checkpoint(path: Path) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        return (
            (path / "state.pt").is_file()
            and (path / "vocab.pkl").is_file()
            and sha256(path / "state.pt") == manifest["state_sha256"]
            and sha256(path / "vocab.pkl") == manifest["vocab_sha256"]
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def latest_checkpoint(checkpoint_root: Path) -> Path:
    valid = [path for path in checkpoint_root.glob("cycle_*") if complete_checkpoint(path)]
    if not valid:
        raise RuntimeError("no complete acquisition checkpoint for resume")
    return max(valid, key=lambda path: int(path.name.split("_")[-1]))


def preserve_and_truncate(path: Path, offset: int, archive: Path) -> None:
    if not path.exists() or path.stat().st_size <= offset:
        return
    archive.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source:
        source.seek(offset)
        trailing = source.read()
    (archive / f"{path.name}.trailing").write_bytes(trailing)
    with path.open("r+b") as handle:
        handle.truncate(offset)


def checkpoint_metadata(
    seed: int,
    mode: str,
    cycle: int,
    critic_updates: int,
    actor_updates: int,
    stream_rows: int,
    stream_offset: int,
    training_offset: int,
    capability_offset: int,
    dataset_hash: str,
) -> dict:
    return {
        "seed": seed,
        "mode": mode,
        "acquisition_cycle": cycle,
        "critic_updates": critic_updates,
        "actor_updates": actor_updates,
        "online_stream_rows": stream_rows,
        "online_stream_byte_offset": stream_offset,
        "training_log_byte_offset": training_offset,
        "capability_log_byte_offset": capability_offset,
        "offline_dataset_sha256": dataset_hash,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "matching_actor_critic": True,
        "fully_resumable_with_stream_prefix": True,
        "created_at_utc": now_utc(),
    }


def prepare_new(out: Path, seed: int, mode: str, device: torch.device):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty source-event directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "SOURCE_CONFIG.json", source_config())
    splits = split_instances()
    train = splits["train"]
    all_vocab_instances = splits["train"] + splits["val"]
    rng = seed_everything(seed, device)

    dataset_path = out / "offline_dataset.jsonl"
    transitions = collect_offline_dataset(
        train,
        target_transitions=OFFLINE_TRANSITIONS,
        seed=seed + 1,
        gamma=GAMMA,
        max_steps=MAX_STEPS,
    )
    save_dataset(transitions, dataset_path)
    rows = [item.to_dict() for item in transitions]
    dataset_hash = sha256(dataset_path)
    vocab = build_vocab(transitions, all_vocab_instances)
    actor = Actor(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    critic = Critic(vocab.size, EMBED_DIM, HIDDEN_DIM).to(device)
    target = critic.make_target().to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
    offline = ReplayBuffer(len(rows) + 1)
    offline.extend(rows)

    offline_log = []
    for update in range(1, OFFLINE_UPDATES + 1):
        batch = clean_collate(offline.sample(BATCH_SIZE, rng), vocab, device)
        metrics = clean_awac_step(
            actor, critic, target, actor_optimizer, critic_optimizer, batch
        )
        if update == 1 or update % 100 == 0 or update == OFFLINE_UPDATES:
            offline_log.append({"offline_update": update, **metrics})
    write_json(out / "old/offline_training_log.json", offline_log)
    old_metrics = capability_macro(actor, vocab, train, device)
    old_pass, old_gates = old_behavior_pass(old_metrics)
    write_json(
        out / "old/behavior.json",
        {"passed": old_pass, "gates": old_gates, **old_metrics},
    )
    save_model_bundle(
        out / "old/checkpoint",
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        {
            "seed": seed,
            "mode": mode,
            "phase": "old_after_offline",
            "offline_updates": OFFLINE_UPDATES,
            "offline_dataset_sha256": dataset_hash,
            "matching_actor_critic": True,
        },
        rng,
        device,
    )
    if not old_pass:
        write_json(
            out / "STATUS.json",
            {
                "status": "PILOT_OLD_BEHAVIOR_FAILED" if mode == "pilot" else "PREARM_REJECT_OLD_BEHAVIOR",
                "seed": seed,
                "mode": mode,
                "fresh_confirmatory_outcome": False,
                "old_gates": old_gates,
            },
        )
        return None

    checkpoint_root = out / "acquisition_trace/checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        checkpoint_root / "cycle_000000",
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        checkpoint_metadata(seed, mode, 0, 0, 0, 0, 0, 0, 0, dataset_hash),
        rng,
        device,
    )
    return (
        splits,
        rows,
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        rng,
        0,
        0,
        0,
        ReplayBuffer(ONLINE_CAPACITY),
        [],
    )


def resume_existing(out: Path, seed: int, mode: str, device: torch.device):
    config = json.loads((out / "SOURCE_CONFIG.json").read_text())
    if config != source_config():
        raise RuntimeError("source config changed since checkpoint")
    checkpoint = latest_checkpoint(out / "acquisition_trace/checkpoints")
    (
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        metadata,
        rng,
    ) = load_model_bundle(checkpoint, device)
    if metadata["seed"] != seed or metadata["mode"] != mode:
        raise RuntimeError("resume identity mismatch")
    dataset_path = out / "offline_dataset.jsonl"
    if sha256(dataset_path) != metadata["offline_dataset_sha256"]:
        raise RuntimeError("offline dataset changed")
    interruption = out / "interrupted_attempts" / now_utc().replace(":", "-")
    stream_path = out / "acquisition_trace/online_stream.jsonl"
    training_path = out / "acquisition_trace/training_log.jsonl"
    capability_path = out / "acquisition_trace/capability_log.jsonl"
    preserve_and_truncate(stream_path, metadata["online_stream_byte_offset"], interruption)
    preserve_and_truncate(training_path, metadata["training_log_byte_offset"], interruption)
    preserve_and_truncate(capability_path, metadata["capability_log_byte_offset"], interruption)
    stream = read_jsonl(stream_path)
    if len(stream) != metadata["online_stream_rows"]:
        raise RuntimeError("stream row count does not match resume checkpoint")
    online = ReplayBuffer(ONLINE_CAPACITY)
    online.extend(stream)
    capability_log = read_jsonl(capability_path)
    return (
        split_instances(),
        dataset_rows(dataset_path),
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        rng,
        int(metadata["acquisition_cycle"]),
        int(metadata["critic_updates"]),
        int(metadata["actor_updates"]),
        online,
        capability_log,
    )


def finalize_selected(
    out: Path,
    seed: int,
    mode: str,
    selected_cycle: int,
    confirmation_cycle: int,
    device: torch.device,
) -> None:
    source_checkpoint = out / f"acquisition_trace/checkpoints/cycle_{selected_cycle:06d}"
    capability_checkpoint = out / "capability_checkpoint"
    if capability_checkpoint.exists():
        raise RuntimeError("capability checkpoint already materialized")
    shutil.copytree(source_checkpoint, capability_checkpoint)
    old = load_model_bundle(out / "old/checkpoint", device)
    current = load_model_bundle(capability_checkpoint, device)
    old_actor, old_vocab = old[0], old[5]
    current_actor, current_vocab = current[0], current[5]
    if sha256(out / "old/checkpoint/vocab.pkl") != sha256(capability_checkpoint / "vocab.pkl"):
        raise RuntimeError("vocabulary changed across source event")
    eligibility = structural_eligibility(
        old_actor,
        current_actor,
        current_vocab,
        split_instances()["train"],
        device,
    )
    eligibility.update(
        {
            "seed": seed,
            "mode": mode,
            "selected_cycle": selected_cycle,
            "stability_confirmation_cycle": confirmation_cycle,
            "checkpoint_selection_used_exact_DP": False,
            "checkpoint_selection_used_learned_critic": False,
            "exact_DP_called_only_after_checkpoint_selection": True,
        }
    )
    write_json(out / "eligibility.json", eligibility)
    status = "PILOT_ELIGIBLE" if mode == "pilot" else "ACCEPTED"
    if eligibility["status"] != "ELIGIBLE":
        status = "PILOT_STRUCTURALLY_INELIGIBLE" if mode == "pilot" else "PREARM_REJECT_STRUCTURAL_ELIGIBILITY"
    checkpoint_files = sorted(
        path
        for path in [out / "old/checkpoint", capability_checkpoint]
        for path in path.iterdir()
        if path.is_file()
    )
    write_json(
        out / "source_manifest.json",
        {
            "seed": seed,
            "mode": mode,
            "status": status,
            "matching_actor_critic_source": True,
            "selected_cycle": selected_cycle,
            "stability_confirmation_cycle": confirmation_cycle,
            "files": [
                {"path": str(path.relative_to(out)), "size": path.stat().st_size, "sha256": sha256(path)}
                for path in checkpoint_files
            ],
            "source_config_sha256": sha256(out / "SOURCE_CONFIG.json"),
            "offline_dataset_sha256": sha256(out / "offline_dataset.jsonl"),
            "eligibility_sha256": sha256(out / "eligibility.json"),
        },
    )
    write_json(
        out / "STATUS.json",
        {
            "status": status,
            "seed": seed,
            "mode": mode,
            "fresh_confirmatory_outcome": mode == "confirmatory",
            "source_event_complete": True,
            "selected_cycle": selected_cycle,
            "stability_confirmation_cycle": confirmation_cycle,
            "eligibility": eligibility["status"],
        },
    )


def run(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ledger_path = ROOT / "manifests/FRESH_SEED_LEDGER.json"
    ledger = json.loads(ledger_path.read_text())
    confirmatory_seeds = [int(row["seed"]) for row in ledger["confirmatory"]]
    reserve_seeds = [int(row["seed"]) for row in ledger["ordered_reserve"]]
    registered_seeds = set(confirmatory_seeds + reserve_seeds)
    if args.mode == "pilot" and args.seed in registered_seeds:
        raise RuntimeError("pilot seed overlaps frozen confirmatory/reserve ledger")
    if args.mode == "confirmatory":
        lock = ROOT / "preregistration/PREREGISTRATION_LOCK.json"
        authorization = ROOT / "preregistration/M7_EXECUTION_AUTHORIZATION.json"
        if not lock.is_file() or not authorization.is_file():
            raise RuntimeError("confirmatory source acquisition is locked until M6 and M7 authorization")
        if args.seed not in registered_seeds:
            raise RuntimeError("confirmatory seed is absent from the frozen ledger")
    if args.resume:
        state = resume_existing(out, args.seed, args.mode, device)
    else:
        state = prepare_new(out, args.seed, args.mode, device)
    if state is None:
        return
    (
        splits,
        offline_rows,
        actor,
        critic,
        target,
        actor_optimizer,
        critic_optimizer,
        vocab,
        rng,
        start_cycle,
        critic_updates,
        actor_updates,
        online,
        capability_log,
    ) = state
    train = splits["train"]
    offline = ReplayBuffer(len(offline_rows) + 1)
    offline.extend(offline_rows)
    stream_path = out / "acquisition_trace/online_stream.jsonl"
    training_path = out / "acquisition_trace/training_log.jsonl"
    capability_path = out / "acquisition_trace/capability_log.jsonl"
    stream_offset = stream_path.stat().st_size if stream_path.exists() else 0
    training_offset = training_path.stat().st_size if training_path.exists() else 0
    capability_offset = capability_path.stat().st_size if capability_path.exists() else 0
    dataset_hash = sha256(out / "offline_dataset.jsonl")
    maximum = min(args.max_acquisition_cycles, SOURCE_MAX_CYCLES)

    for cycle in range(start_cycle + 1, maximum + 1):
        rows, episode = collect_assisted_episode(
            actor,
            vocab,
            train,
            device,
            rng,
            cycle,
            curriculum_probability=CURRICULUM_PROBABILITY,
            gate_epsilon=GATE_EPSILON,
        )
        online.extend(rows)
        stream_offset = write_cycle_rows(stream_path, rows)
        metrics = {}
        if len(online) >= BATCH_SIZE:
            samples = offline.sample(OFFLINE_BATCH, rng) + online.sample(ONLINE_BATCH, rng)
            batch = clean_collate(samples, vocab, device)
            metrics = clean_awac_step(
                actor, critic, target, actor_optimizer, critic_optimizer, batch
            )
            critic_updates += 1
            actor_updates += 1
        training_offset = append_jsonl(
            training_path,
            {
                **episode,
                "critic_updates": critic_updates,
                "actor_updates": actor_updates,
                "online_replay_size": len(online),
                **metrics,
            },
        )

        if cycle % CAPABILITY_EVAL_INTERVAL == 0:
            capability = capability_macro(actor, vocab, train, device)
            passed, gates = capability_pass(capability)
            evaluation = {
                "cycle": cycle,
                "passed": passed,
                "gates": gates,
                **capability,
                "selection_inputs": "capability metrics only",
            }
            capability_offset = append_jsonl(capability_path, evaluation)
            capability_log.append(evaluation)
            checkpoint = out / f"acquisition_trace/checkpoints/cycle_{cycle:06d}"
            save_model_bundle(
                checkpoint,
                actor,
                critic,
                target,
                actor_optimizer,
                critic_optimizer,
                vocab,
                checkpoint_metadata(
                    args.seed,
                    args.mode,
                    cycle,
                    critic_updates,
                    actor_updates,
                    len(online),
                    stream_offset,
                    training_offset,
                    capability_offset,
                    dataset_hash,
                ),
                rng,
                device,
            )
            recent = capability_log[-CAPABILITY_PERSISTENCE:]
            if len(recent) == CAPABILITY_PERSISTENCE and all(row["passed"] for row in recent):
                selected_cycle = int(recent[0]["cycle"])
                finalize_selected(out, args.seed, args.mode, selected_cycle, cycle, device)
                print(json.dumps(json.loads((out / "STATUS.json").read_text()), indent=2))
                return
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "mode": args.mode,
                        "cycle": cycle,
                        "capability_pass": passed,
                        "gate": capability["macro"]["gate_sequence_success"],
                        "forced_shortcut": capability["macro"]["forced_shortcut_full_task_success"],
                        "safe": capability["macro"]["safe_route_success"],
                        "critic_updates": critic_updates,
                    }
                ),
                flush=True,
            )
        if args.stop_after_cycle is not None and cycle >= args.stop_after_cycle:
            write_json(
                out / "STATUS.json",
                {
                    "status": "PILOT_PAUSED_AT_REGISTERED_CHECKPOINT",
                    "seed": args.seed,
                    "mode": args.mode,
                    "cycle": cycle,
                    "resume_required": True,
                },
            )
            return

    write_json(
        out / "STATUS.json",
        {
            "status": "PILOT_MAX_CYCLE_REACHED" if args.mode == "pilot" else "PREARM_REJECT_NO_STABLE_CAPABILITY",
            "seed": args.seed,
            "mode": args.mode,
            "max_cycle": maximum,
            "fresh_confirmatory_outcome": args.mode == "confirmatory",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-acquisition-cycles", type=int, default=SOURCE_MAX_CYCLES)
    parser.add_argument("--stop-after-cycle", type=int)
    args = parser.parse_args()
    if args.stop_after_cycle is not None and args.stop_after_cycle % CHECKPOINT_INTERVAL != 0:
        raise RuntimeError("stop-after-cycle must be a registered checkpoint boundary")
    run(args)


if __name__ == "__main__":
    main()
