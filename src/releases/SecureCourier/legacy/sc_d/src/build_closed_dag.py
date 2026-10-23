#!/usr/bin/env python3
"""Build the frozen simulator-derived SecureCourier state-action closed DAG."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from scd_common import GAMMA, MAX_STEPS, PARENT, canonical_hash, sha256, write_canonical_json, write_pretty_json


VENDOR = PARENT / "src/vendor/diy_task_set"
PARENT_SRC = PARENT / "src"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(PARENT_SRC))

from scabc_common import split_instances  # noqa: E402
from secure_courier.env import SecureCourierEnv  # noqa: E402
from secure_courier.state import EnvState  # noqa: E402
from secure_courier.task_graph import GRAPH, step_transition  # noqa: E402


CURRICULUM_NODES = ("sc5", "sc4", "sc3", "sc2")


def state_key(instance_id: str, node: str, step: int) -> str:
    return f"{instance_id}\x1f{node}\x1f{step:02d}"


def pair_key(instance_id: str, node: str, step: int, action: str) -> str:
    return f"{instance_id}\x1f{node}\x1f{step:02d}\x1f{action}"


def reachable_states(instance_id: str) -> set[tuple[str, int]]:
    # The full parent start is sufficient to generate every state reachable from
    # its curriculum starts too; list the latter separately in the audit.
    queue: deque[tuple[str, int]] = deque([("depot", 0)])
    seen: set[tuple[str, int]] = set()
    while queue:
        node, step = queue.popleft()
        if (node, step) in seen or step >= MAX_STEPS or not GRAPH.get(node):
            continue
        seen.add((node, step))
        for action, _ in GRAPH[node]:
            next_node, _, terminal = step_transition(node, action)
            if not terminal and step + 1 < MAX_STEPS:
                queue.append((next_node, step + 1))
    return seen


def build_graph() -> dict:
    train = split_instances()["train"]
    if len(train) != 16:
        raise RuntimeError(f"EXPECTED_16_PRIMARY_SURFACES:{len(train)}")
    states: list[dict] = []
    pairs: list[dict] = []
    for instance in sorted(train, key=lambda x: x.instance_id):
        for node, step in sorted(reachable_states(instance.instance_id), key=lambda x: (x[1], x[0])):
            env = SecureCourierEnv(instance, gamma=GAMMA, max_steps=MAX_STEPS)
            env.restore_state(
                EnvState(
                    node=node,
                    step_count=step,
                    done=False,
                    terminal_success=False,
                    instance_id=instance.instance_id,
                )
            )
            observation = env._obs()
            structural = [str(x) for x in observation["admissible_structural"]]
            display = [str(x) for x in observation["admissible_actions"]]
            expected = [str(action) for action, _ in GRAPH[node]]
            if structural != expected:
                raise RuntimeError(f"ACTION_ORDER_MISMATCH:{instance.instance_id}:{node}:{step}")
            skey = state_key(instance.instance_id, node, step)
            states.append(
                {
                    "key": skey,
                    "instance_id": instance.instance_id,
                    "instance_seed": int(instance.instance_seed),
                    "node": node,
                    "step": step,
                    "observation": observation["text"],
                    "structural_actions": structural,
                    "display_actions": display,
                }
            )
            for action, action_display in zip(structural, display):
                next_node, reward, structural_terminal = step_transition(node, action)
                horizon_terminal = step + 1 >= MAX_STEPS
                continuation = not structural_terminal and not horizon_terminal
                successor = state_key(instance.instance_id, next_node, step + 1) if continuation else None
                pairs.append(
                    {
                        "key": pair_key(instance.instance_id, node, step, action),
                        "state_key": skey,
                        "instance_id": instance.instance_id,
                        "node": node,
                        "step": step,
                        "action": action,
                        "action_display": action_display,
                        "successor_state_key": successor,
                        "reward": float(reward),
                        "structural_terminal": bool(structural_terminal),
                        "horizon_terminal": bool(horizon_terminal),
                        "continuation": bool(continuation),
                        "transition_probability": 1.0,
                    }
                )
    state_keys = {row["key"] for row in states}
    pair_keys = [row["key"] for row in pairs]
    if len(state_keys) != len(states) or len(set(pair_keys)) != len(pairs):
        raise RuntimeError("DUPLICATE_STATE_OR_PAIR")
    missing_successors = [row["key"] for row in pairs if row["continuation"] and row["successor_state_key"] not in state_keys]
    if missing_successors:
        raise RuntimeError(f"CLOSURE_FAILURE:{missing_successors[:3]}")
    return {
        "schema": "SC_D_SIMULATOR_CLOSED_DAG_V2",
        "gamma": GAMMA,
        "max_steps": MAX_STEPS,
        "surface_split": "train",
        "primary_surface_count": len(train),
        "start_states": ["depot@0"],
        "curriculum_nodes_covered_by_full_closure": list(CURRICULUM_NODES),
        "construction": "forward closure from every primary full start under every legal action",
        "states": states,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    graph = build_graph()
    write_canonical_json(args.output, graph)
    state_steps = [row["step"] for row in graph["states"]]
    pair_counts: dict[str, int] = {}
    for state in graph["states"]:
        key = str(len(state["structural_actions"]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    manifest = {
        "schema": "SC-D simulator closed DAG manifest v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_path": "registered_v2/manifests/support_graph.json",
        "graph_sha256": sha256(args.output),
        "canonical_content_sha256": canonical_hash(graph),
        "state_count": len(graph["states"]),
        "pair_count": len(graph["pairs"]),
        "continuing_edge_count": sum(row["continuation"] for row in graph["pairs"]),
        "terminal_or_horizon_pair_count": sum(not row["continuation"] for row in graph["pairs"]),
        "time_min": min(state_steps),
        "time_max": max(state_steps),
        "action_count_state_histogram": pair_counts,
        "deterministic_unit_probability": all(row["transition_probability"] == 1.0 for row in graph["pairs"]),
        "strictly_time_layered": all(
            not row["continuation"] or int(row["successor_state_key"].rsplit("\x1f", 1)[1]) == row["step"] + 1
            for row in graph["pairs"]
        ),
        "closure_pass": True,
        "dropped_policy_mass": 0.0,
        "primary_surface_count": graph["primary_surface_count"],
        "validation_or_test_surfaces_included": False,
        "environment_files": {
            str(path.relative_to(PARENT)): sha256(path)
            for path in sorted((PARENT / "src/vendor/diy_task_set/secure_courier").glob("*.py"))
        },
    }
    write_pretty_json(args.manifest, manifest)


if __name__ == "__main__":
    main()
