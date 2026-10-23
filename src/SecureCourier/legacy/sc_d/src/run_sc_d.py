#!/usr/bin/env python3
"""Outcome-locked SC-D runner; confirmatory mode remains disabled until D4."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from pdr_awac_adapter import PDRService, actor_weight_update
from scd_common import clean, module_hash, sha256, tensor_tree_hash
from training_core import (
    OFFLINE_BATCH,
    ONLINE_BATCH,
    ONLINE_CAPACITY,
    ReplayBuffer,
    append_jsonl,
    base_rows,
    clean_collate,
    collect_assisted_episode,
    critic_step,
    load_model_bundle,
    read_jsonl,
    save_model_bundle,
    split_instances,
    update_target,
)


ARMS = ("STANDARD_TD", "FRESH_PDR")


def pdr_intervention_active(arm: str, cycle: int, intervention_end: int) -> bool:
    """Single authoritative treatment-window predicate, including cycle 1025 restoration."""
    if cycle < 1 or intervention_end < 0:
        raise RuntimeError("INVALID_INTERVENTION_BOUNDARY")
    return arm == "FRESH_PDR" and cycle <= intervention_end


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def truncate_diagnostic_after_cycle(path: Path, cycle: int, archive: Path) -> None:
    """Restore a diagnostic log without putting diagnostic offsets in training state."""
    if not path.exists():
        return
    raw_lines = path.read_bytes().splitlines(keepends=True)
    keep = 0
    for raw in raw_lines:
        if int(json.loads(raw)["cycle"]) <= cycle:
            keep += len(raw)
        else:
            break
    preserve_and_truncate(path, keep, archive)


def latest_checkpoint(root: Path) -> Path:
    candidates = []
    for path in root.glob("cycle_*"):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if sha256(path / "state.pt") == manifest["state_sha256"]:
                candidates.append(path)
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    if not candidates:
        raise RuntimeError("NO_COMPLETE_SC_D_CHECKPOINT")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[1]))


def score_actor(actor, vocab, device) -> dict:
    if os.environ.get("SC_D_FORBID_SCORER_IMPORT") == "1":
        raise RuntimeError("SCORER_IMPORT_FORBIDDEN")
    actor_mode = actor.training
    torch_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    scoring = importlib.import_module("scabc_common")
    rows = [scoring.exact_instance_metrics(actor, vocab, instance, device) for instance in split_instances()["train"]]
    actor.train(actor_mode)
    torch.set_rng_state(torch_state)
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    return {
        "J_exact": float(np.mean([row["J_exact"] for row in rows])),
        "p_shortcut": float(np.mean([row["p_shortcut"] for row in rows])),
        "shortcut_log_odds_vs_safe": float(
            np.mean([np.log(max(row["p_shortcut"], 1e-12) / max(row["p_safe"], 1e-12)) for row in rows])
        ),
        "scoring_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--intervention-end", type=int, default=1024)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=16)
    parser.add_argument("--disable-scoring", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-cycle", type=int)
    args = parser.parse_args()
    if args.mode == "confirmatory":
        from lock_sc_d import verify_lock
        lock = verify_lock(Path(__file__).resolve().parent.parent, args.source_event.resolve().name)
        from scd_common import PARENT
        if args.source_event.resolve().parent != PARENT / 'source_events':
            raise RuntimeError('OFFICIAL_SOURCE_PATH_MISMATCH')
        if (args.max_cycles,args.intervention_end,args.checkpoint_interval,args.evaluation_interval,args.disable_scoring)!=(2048,1024,16,16,True):
            raise RuntimeError('OFFICIAL_CONFIG_MISMATCH')
        if args.arm != 'FRESH_PDR': raise RuntimeError('OFFICIAL_ARM_MISMATCH')
    # A pilot label must not authorize updates to the confirmatory cohort.
    from scd_common import PARENT
    pilot_root = (PARENT / "pilots").resolve()
    source_path = args.source_event.resolve()
    if args.mode == 'pilot' and source_path.parent != pilot_root:
        raise RuntimeError("PILOT_SOURCE_MUST_BE_DISJOINT_PARENT_PILOT")
    reference_manifest = json.loads((args.reference / "REFERENCE_MANIFEST.json").read_text())
    if Path(reference_manifest["reference_path"]).resolve() != source_path / "old/checkpoint":
        raise RuntimeError("PDR_REFERENCE_SOURCE_EVENT_MISMATCH")
    if sha256(args.graph) != reference_manifest['graph_sha256']:
        raise RuntimeError('PDR_REFERENCE_GRAPH_MISMATCH')
    for name in ('pi_ref','qbar'):
        if sha256(args.reference / (name+'.npy')) != reference_manifest[name+'_file_sha256']:
            raise RuntimeError('PDR_REFERENCE_ARRAY_HASH_MISMATCH')
    if args.max_cycles < 0 or args.checkpoint_interval <= 0 or args.evaluation_interval <= 0:
        raise RuntimeError("INVALID_CYCLE_CONFIGURATION")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    source = args.source_event.resolve()
    branch = source / "capability_checkpoint"
    output = args.output.resolve()
    config = {
        "schema": "SC-D pilot runner v2",
        "source_event": str(source),
        "reference": str(args.reference.resolve()),
        "graph": str(args.graph.resolve()),
        "arm": args.arm,
        "mode": args.mode,
        "max_cycles": args.max_cycles,
        "intervention_end": args.intervention_end,
        "checkpoint_interval": args.checkpoint_interval,
        "evaluation_interval": args.evaluation_interval,
        "disable_scoring": args.disable_scoring,
        "offline_online_batch": [OFFLINE_BATCH, ONLINE_BATCH],
        "update_order": ["collect", "insert", "critic", "fresh PDR if active", "actor", "Polyak", "score"],
    }
    config_path = output / "RUN_CONFIG.json"
    stream_path = output / "continuation_stream.jsonl"
    training_path = output / "training_log.jsonl"
    compute_path = output / "pdr_compute_log.jsonl"
    scoring_path = output / "scoring_log.jsonl"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("RESUME_CONFIG_MISMATCH")
        checkpoint = latest_checkpoint(output / "checkpoints")
        actor, critic, target, actor_optimizer, critic_optimizer, vocab, metadata, rng = load_model_bundle(checkpoint, device)
        start_cycle = int(metadata["cycle"])
        interruption = output / "interrupted_attempts" / f"after_{start_cycle:06d}"
        for path, key in (
            (stream_path, "stream_offset"),
            (training_path, "training_offset"),
        ):
            preserve_and_truncate(path, int(metadata.get(key, 0)), interruption)
        truncate_diagnostic_after_cycle(compute_path, start_cycle, interruption)
        truncate_diagnostic_after_cycle(scoring_path, start_cycle, interruption)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("REFUSING_TO_OVERWRITE_SC_D_OUTPUT")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
        actor, critic, target, actor_optimizer, critic_optimizer, vocab, metadata, rng = load_model_bundle(branch, device)
        start_cycle = 0
    rows = base_rows(source)
    offline_rows = [row for row in rows if row["_scb_origin"] == "offline"]
    initial_online_rows = [row for row in rows if row["_scb_origin"] == "online"]
    continuation_rows = read_jsonl(stream_path) if stream_path.exists() else []
    offline = ReplayBuffer(len(offline_rows) + 1)
    offline.extend(offline_rows)
    online = ReplayBuffer(ONLINE_CAPACITY)
    online.extend(initial_online_rows)
    online.extend(continuation_rows)
    initial = {
        "actor_hash": module_hash(actor) if start_cycle == 0 else None,
        "critic_hash": module_hash(critic) if start_cycle == 0 else None,
        "target_hash": module_hash(target) if start_cycle == 0 else None,
        "branch_state_sha256": sha256(branch / "state.pt"),
    }
    service = PDRService(args.graph, args.reference, vocab, device) if args.arm == "FRESH_PDR" else None
    if not args.resume and not args.disable_scoring:
        append_jsonl(scoring_path, {"cycle": 0, **score_actor(actor, vocab, device)})
    for cycle in range(start_cycle + 1, args.max_cycles + 1):
        cycle_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        collected, episode = collect_assisted_episode(actor, vocab, split_instances()["train"], device, rng, cycle)
        for index, row in enumerate(collected):
            row["episode_id"] = f"continuation_c{cycle}"
            row["transition_id"] = f"continuation_c{cycle}_t{index}"
            row["_scb_uid"] = f"continuation:{cycle:06d}:{index:03d}"
            row["_scb_origin"] = "continuation"
            online.add(row)
            append_jsonl(stream_path, row)
        raw_batch = offline.sample(OFFLINE_BATCH, rng) + online.sample(ONLINE_BATCH, rng)
        batch = clean_collate(raw_batch, vocab, device)
        critic_metrics = critic_step(actor, critic, target, critic_optimizer, batch)
        if cycle == 1:
            write_json(output / 'first_update_batch.json', raw_batch)
            save_model_bundle(output / 'first_pre_actor', actor, critic, target,
                              actor_optimizer, critic_optimizer, vocab,
                              {'cycle': 1, 'phase': 'after_critic_before_actor', 'matching_actor_critic': True}, rng, device)
        active_pdr = pdr_intervention_active(args.arm, cycle, args.intervention_end)
        refresh = service.refresh(actor) if active_pdr else None
        effective_arm = "FRESH_PDR" if active_pdr else "STANDARD_TD"
        actor_metrics = actor_weight_update(
            arm=effective_arm,
            actor=actor,
            critic=critic,
            optimizer=actor_optimizer,
            batch=batch,
            raw_rows=raw_batch,
            pdr_service=service,
            refresh=refresh,
        )
        update_target(target, critic)
        compute = {
            "cycle": cycle,
            "event_id": source.name,
            "replicate_id": "replicate_00",
            "arm": args.arm,
            "effective_arm": effective_arm,
            "pre_actor_update_hash": actor_metrics["actor_hash_before"],
            "pi_current_hash": actor_metrics["pi_current_hash"],
            "source_hash": actor_metrics["source_hash"],
            "correction_hash": actor_metrics["correction_hash"],
            "delivered_value_hash": actor_metrics["q_pdr_hash"],
            "full_domain_actor_evaluations": len(service.dag.pairs) if active_pdr else 0,
            "lookup_coverage": actor_metrics["lookup_coverage"],
            "recurrence_residual_inf": actor_metrics["recurrence_residual_inf"],
            "policy_seconds": actor_metrics.pop("policy_seconds"),
            "solve_seconds": actor_metrics.pop("solve_seconds"),
            "lookup_seconds": actor_metrics.pop("lookup_seconds"),
            "actor_update_seconds": actor_metrics.pop("actor_update_seconds"),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        compute["total_cycle_seconds"] = time.perf_counter() - cycle_started
        append_jsonl(compute_path, compute)
        append_jsonl(training_path, {"cycle": cycle, "episode": episode, "online_replay_size": len(online), **critic_metrics, **actor_metrics})
        if not args.disable_scoring and cycle % args.evaluation_interval == 0:
            append_jsonl(scoring_path, {"cycle": cycle, **score_actor(actor, vocab, device)})
        should_checkpoint = cycle in (1,1025) or cycle % args.checkpoint_interval == 0 or cycle == args.max_cycles or (
            args.pause_after_cycle is not None and cycle >= args.pause_after_cycle
        )
        if should_checkpoint:
            save_model_bundle(
                output / f"checkpoints/cycle_{cycle:06d}",
                actor,
                critic,
                target,
                actor_optimizer,
                critic_optimizer,
                vocab,
                {
                    "schema": "SC-D resumable checkpoint v2",
                    "cycle": cycle,
                    "arm": args.arm,
                    "matching_actor_critic": True,
                    "online_replay_size": len(online),
                    "continuation_rows": len(read_jsonl(stream_path)),
                    "stream_offset": stream_path.stat().st_size if stream_path.exists() else 0,
                    "training_offset": training_path.stat().st_size if training_path.exists() else 0,
                },
                rng,
                device,
            )
        if args.pause_after_cycle is not None and cycle >= args.pause_after_cycle:
            write_json(output / "STATUS.json", {"status": "PILOT_PAUSED", "cycle": cycle, "resume_required": True})
            return
    manifest = {
        "schema": "SC-D pilot run manifest v2",
        "status": "SEALED_COMPLETE" if args.mode == 'confirmatory' else "PILOT_COMPLETE",
        "official_pdr_outcome": args.mode == 'confirmatory',
        "prelock_deviation_disclosed": args.mode == 'confirmatory',
        "lock_sha256": sha256(Path(__file__).resolve().parent.parent/'preregistration/SC_D_PREREGISTRATION_LOCK.json') if args.mode == 'confirmatory' else None,
        "arm": args.arm,
        "initial": initial,
        "final": {
            "cycle": args.max_cycles,
            "actor_hash": module_hash(actor),
            "critic_hash": module_hash(critic),
            "target_hash": module_hash(target),
            "actor_optimizer_hash": tensor_tree_hash(actor_optimizer.state_dict()),
            "critic_optimizer_hash": tensor_tree_hash(critic_optimizer.state_dict()),
        },
        "training_log_sha256": sha256(training_path),
        "continuation_stream_sha256": sha256(stream_path),
        "compute_log_sha256": sha256(compute_path),
        "scoring_log_sha256": sha256(scoring_path) if scoring_path.exists() else None,
        "resumed": args.resume,
    }
    write_json(output / "RUN_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
