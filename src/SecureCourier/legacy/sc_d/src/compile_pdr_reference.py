#!/usr/bin/env python3
"""Compile matched frozen pi_ref and learned single-critic Qbar arrays."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reverse_pdr import ClosedDAG, array_hash
from scd_common import ACT_MAX, OBS_MAX, PARENT, sha256, write_pretty_json


sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(PARENT / "src/vendor/diy_task_set"))
from scabc_common import load_model_bundle  # noqa: E402


@torch.inference_mode()
def evaluate_actor_critic(dag: ClosedDAG, actor, critic, vocab, device: torch.device, batch_size: int = 512):
    actor.eval()
    critic.eval()
    pi = np.zeros(len(dag.pairs), dtype=np.float64)
    qbar = np.zeros(len(dag.pairs), dtype=np.float64)
    groups: dict[int, list[int]] = {}
    for sid, state in enumerate(dag.states):
        groups.setdefault(len(state["structural_actions"]), []).append(sid)
    for count, state_ids in sorted(groups.items()):
        for start in range(0, len(state_ids), batch_size):
            chunk = state_ids[start : start + batch_size]
            obs = torch.tensor(
                [vocab.encode(dag.states[sid]["observation"], OBS_MAX) for sid in chunk],
                dtype=torch.long,
                device=device,
            )
            actions = torch.tensor(
                [
                    [vocab.encode(text, ACT_MAX) for text in dag.states[sid]["display_actions"]]
                    for sid in chunk
                ],
                dtype=torch.long,
                device=device,
            )
            probabilities = actor(obs, actions).exp().double().cpu().numpy()
            values = critic(obs, actions).double().cpu().numpy()
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for row, sid in enumerate(chunk):
                indices = dag.state_pairs[sid]
                if len(indices) != count:
                    raise RuntimeError("GROUP_COUNT_MISMATCH")
                pi[indices] = probabilities[row]
                qbar[indices] = values[row]
    dag.validate_policy(pi)
    if not np.all(np.isfinite(qbar)):
        raise RuntimeError("QBAR_NONFINITE")
    return pi, qbar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    dag = ClosedDAG(args.graph)
    actor, critic, _, _, _, vocab, metadata, _ = load_model_bundle(args.reference, device)
    if not metadata.get("matching_actor_critic"):
        raise RuntimeError("REFERENCE_NOT_MATCHED_ACTOR_CRITIC")
    if metadata.get("phase") != "old_after_offline":
        raise RuntimeError("REFERENCE_PHASE_MISMATCH")
    pi_ref, qbar = evaluate_actor_critic(dag, actor, critic, vocab, device)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "pi_ref.npy", pi_ref, allow_pickle=False)
    np.save(args.output / "qbar.npy", qbar, allow_pickle=False)
    manifest = {
        "schema": "SC-D frozen matched PDR reference v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(args.reference.resolve()),
        "reference_files": {
            name: sha256(args.reference / name) for name in ("manifest.json", "state.pt", "vocab.pkl")
        },
        "reference_phase": metadata["phase"],
        "matching_actor_critic": True,
        "graph_path": str(args.graph.resolve()),
        "graph_sha256": dag.graph_sha256,
        "pair_count": len(dag.pairs),
        "pi_ref_array_hash": array_hash(pi_ref),
        "qbar_array_hash": array_hash(qbar),
        "pi_ref_file_sha256": sha256(args.output / "pi_ref.npy"),
        "qbar_file_sha256": sha256(args.output / "qbar.npy"),
        "policy_row_max_abs_sum_error": float(
            np.max(np.abs(np.bincount(dag.pair_state, weights=pi_ref, minlength=len(dag.states)) - 1.0))
        ),
        "finite": bool(np.all(np.isfinite(pi_ref)) and np.all(np.isfinite(qbar))),
    }
    write_pretty_json(args.output / "REFERENCE_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()

