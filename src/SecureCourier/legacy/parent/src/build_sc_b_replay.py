#!/usr/bin/env python3
"""Build auditable SC-B replay views without duplicating the base replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import canonical_hash, sha256, write_json


ARMS = ("FULL_RELEVANT_REPLAY", "NO_LOCAL_ANCHOR", "NO_DOWNSTREAM_SUPPORT")
DOWNSTREAM_NODES = {
    "sc1",
    "sc2",
    "sc3",
    "sc4",
    "sc5",
    "gate_lobby",
    "gate_badged",
    "gate_coded",
    "gate_passed",
}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def prefix_sha256(path: Path, byte_count: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = byte_count
        while remaining:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                raise RuntimeError("online stream shorter than selected checkpoint prefix")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def row_uid(origin: str, index: int, row: dict) -> str:
    return f"{origin}:{index:06d}:{row.get('transition_id', 'missing')}"


def base_rows(source: Path) -> tuple[list[dict], dict]:
    selected = json.loads((source / "capability_checkpoint/manifest.json").read_text())
    online_count = int(selected["online_stream_rows"])
    online_bytes = int(selected["online_stream_byte_offset"])
    offline_path = source / "offline_dataset.jsonl"
    online_path = source / "acquisition_trace/online_stream.jsonl"
    offline = read_jsonl(offline_path)
    online = read_jsonl(online_path, limit=online_count)
    if len(online) != online_count:
        raise RuntimeError("online row count does not match capability checkpoint")
    rows = []
    for origin, values in (("offline", offline), ("online", online)):
        for index, row in enumerate(values):
            item = dict(row)
            item["_scb_uid"] = row_uid(origin, index, row)
            item["_scb_origin"] = origin
            rows.append(item)
    provenance = {
        "offline_path": str(offline_path),
        "offline_rows": len(offline),
        "offline_sha256": sha256(offline_path),
        "online_path": str(online_path),
        "online_prefix_rows": online_count,
        "online_prefix_bytes": online_bytes,
        "online_prefix_sha256": prefix_sha256(online_path, online_bytes),
        "selected_checkpoint_manifest_sha256": sha256(source / "capability_checkpoint/manifest.json"),
    }
    return rows, provenance


def flags(row: dict) -> tuple[bool, bool]:
    state = row.get("full_state") or {}
    node = state.get("node")
    local = node == "junction" and row.get("action_structural") == "take_shortcut"
    downstream = node in DOWNSTREAM_NODES
    return local, downstream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_event.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty SC-B replay directory")
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = base_rows(source)
    index_rows = []
    local_ids = []
    downstream_ids = []
    for row in rows:
        local, downstream = flags(row)
        if local:
            local_ids.append(row["_scb_uid"])
        if downstream:
            downstream_ids.append(row["_scb_uid"])
        index_rows.append(
            {
                "uid": row["_scb_uid"],
                "origin": row["_scb_origin"],
                "instance_id": row.get("instance_id"),
                "node": (row.get("full_state") or {}).get("node"),
                "step": (row.get("full_state") or {}).get("step_count"),
                "action_structural": row.get("action_structural"),
                "local_junction_shortcut_anchor": local,
                "downstream_bellman_support": downstream,
            }
        )
    index_path = out / "BASE_ROW_INDEX.jsonl"
    with index_path.open("w") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(out / "REMOVED_LOCAL_ANCHOR_IDS.json", local_ids)
    write_json(out / "REMOVED_DOWNSTREAM_SUPPORT_IDS.json", downstream_ids)

    arm_rows = []
    for arm in ARMS:
        if arm == "FULL_RELEVANT_REPLAY":
            removed = []
        elif arm == "NO_LOCAL_ANCHOR":
            removed = local_ids
        else:
            removed = downstream_ids
        included = len(rows) - len(set(removed))
        arm_rows.append(
            {
                "arm": arm,
                "base_rows": len(rows),
                "included_rows": included,
                "removed_rows": len(removed),
                "removed_ids_sha256": canonical_hash(removed),
                "local_anchor_coverage": len(local_ids) if arm != "NO_LOCAL_ANCHOR" else 0,
                "downstream_support_coverage": len(downstream_ids) if arm != "NO_DOWNSTREAM_SUPPORT" else 0,
                "sampling": "uniform with replacement over included rows",
                "sampling_probability_per_included_row": 1.0 / included,
            }
        )
    manifest = {
        "schema": "SC-B replay support manifest v1",
        "source_event": str(source),
        "matching_capability_checkpoint": True,
        "base_provenance": provenance,
        "base_rows": len(rows),
        "base_row_index_sha256": sha256(index_path),
        "support_definitions": {
            "local_anchor": "full_state.node == junction AND action_structural == take_shortcut",
            "downstream_support_nodes": sorted(DOWNSTREAM_NODES),
            "downstream_boundary": "remove shortcut-exclusive state rows from sc1 through gate_passed; junction anchor and shared delivery/terminal-reward rows remain",
            "counterfactual_closure_generated": False,
        },
        "arms": arm_rows,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(out / "SUPPORT_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
