"""Scoring-only exact interface; its outputs must never enter optimizer updates."""

from __future__ import annotations

import math
from typing import Any, Sequence

from scabc_common import (
    TEMPERATURE,
    WEIGHT_CLIP,
    exact_policy_solution,
    split_instances,
)


def exact_rows(actor, vocab, device, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    instances = {item.instance_id: item for item in split_instances()["train"]}
    solutions = {}
    result = []
    for row in rows:
        instance_id = row["instance_id"]
        if instance_id not in instances:
            raise RuntimeError(f"row is outside registered training surfaces: {instance_id}")
        if instance_id not in solutions:
            solutions[instance_id] = exact_policy_solution(actor, vocab, instances[instance_id], device)
        state = row.get("full_state") or {}
        node = state["node"]
        step = int(state["step_count"])
        action = row["action_structural"]
        solution = solutions[instance_id]
        q = float(solution["Q"][(node, step, action)])
        value = float(solution["V"][(node, step)])
        advantage = q - value
        result.append(
            {
                "uid": row["_scb_uid"],
                "instance_id": instance_id,
                "node": node,
                "step": step,
                "action_structural": action,
                "Q_exact": q,
                "V_exact": value,
                "A_exact": advantage,
                "weight_exact": float(min(WEIGHT_CLIP, math.exp(advantage / TEMPERATURE))),
            }
        )
    return result
