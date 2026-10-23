"""Oracle-treatment interface, importable only by CORRECT and PLACEBO arms."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from oracle_scoring import exact_rows
from scabc_common import canonical_hash, module_hash


AUTHORIZED_ARMS = {"CORRECT", "PLACEBO"}


def treatment_weights(
    *,
    arm: str,
    cycle: int,
    event_id: str,
    actor,
    vocab,
    device,
    rows: Sequence[dict[str, Any]],
    log_path: Path,
) -> dict[str, float]:
    if arm not in AUTHORIZED_ARMS:
        raise PermissionError(f"oracle treatment is forbidden for arm {arm}")
    before = module_hash(actor)
    exact = exact_rows(actor, vocab, device, rows)
    after = module_hash(actor)
    if before != after:
        raise RuntimeError("oracle treatment mutated actor state")
    weights = {row["uid"]: row["weight_exact"] for row in exact}
    record = {
        "event_id": event_id,
        "arm": arm,
        "cycle": cycle,
        "item_ids": sorted(weights),
        "input_policy_hash": before,
        "output_weight_hash": canonical_hash(weights),
        "call_item_count": len(weights),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True) + "\n"
    with log_path.open("a") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return weights
