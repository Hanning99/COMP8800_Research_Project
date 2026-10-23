#!/usr/bin/env python3
"""Freeze the harmful-stale and matched placebo panels before SC-C arms run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_sc_b_replay import base_rows
from oracle_scoring import exact_rows
from scabc_common import (
    TEMPERATURE,
    WEIGHT_CLIP,
    canonical_hash,
    learned_state_action_values,
    load_model_bundle,
    sha256,
    split_instances,
    write_json,
)


def learned_row(actor, critic, vocab, device, row: dict) -> dict:
    instances = {item.instance_id: item for item in split_instances()["train"]}
    state = row["full_state"]
    values = learned_state_action_values(
        actor,
        critic,
        vocab,
        instances[row["instance_id"]],
        device,
        node=state["node"],
        step=int(state["step_count"]),
    )
    action = row["action_structural"]
    advantage = values["A"][action]
    return {
        "A_learned": advantage,
        "Q_learned": values["Q"][action],
        "weight_learned": float(min(WEIGHT_CLIP, math.exp(advantage / TEMPERATURE))),
        "learned_rank_one": values["rank"][0] == action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch

    source = args.source_event.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty SC-C panel directory")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows, provenance = base_rows(source)
    current_actor, current_critic, _, _, _, vocab, _, _ = load_model_bundle(
        source / "capability_checkpoint", device
    )
    old_actor, _, _, _, _, old_vocab, _, _ = load_model_bundle(source / "old/checkpoint", device)

    # The primary affected decision is the historical junction shortcut.  The
    # row IDs are frozen before any arm outcome exists.
    candidate_h = [
        row
        for row in rows
        if row.get("full_state", {}).get("node") == "junction"
        and row.get("action_structural") == "take_shortcut"
    ]
    current_exact = {row["uid"]: row for row in exact_rows(current_actor, vocab, device, candidate_h)}
    old_exact = {row["uid"]: row for row in exact_rows(old_actor, old_vocab, device, candidate_h)}
    h_panel = []
    for row in candidate_h:
        learned = learned_row(current_actor, current_critic, vocab, device, row)
        current = current_exact[row["_scb_uid"]]
        old = old_exact[row["_scb_uid"]]
        true_flip = old["A_exact"] < -0.02 and current["A_exact"] > 0.02
        harmful = true_flip and learned["A_learned"] <= 0.0
        if harmful:
            h_panel.append(
                {
                    "uid": row["_scb_uid"],
                    "instance_id": row["instance_id"],
                    "origin": row["_scb_origin"],
                    "node": "junction",
                    "step": int(row["full_state"]["step_count"]),
                    "action_structural": "take_shortcut",
                    "old_A_exact": old["A_exact"],
                    "current_A_exact": current["A_exact"],
                    "current_weight_exact": current["weight_exact"],
                    **learned,
                }
            )

    # Match within the same junction and, whenever possible, the same instance.
    placebo_candidates = [
        row
        for row in rows
        if row.get("full_state", {}).get("node") == "junction"
        and row.get("action_structural") != "take_shortcut"
    ]
    candidate_exact = {
        row["uid"]: row for row in exact_rows(current_actor, vocab, device, placebo_candidates)
    }
    candidate_records = []
    for row in placebo_candidates:
        learned = learned_row(current_actor, current_critic, vocab, device, row)
        exact = candidate_exact[row["_scb_uid"]]
        candidate_records.append(
            {
                "row": row,
                "exact": exact,
                "learned": learned,
                "non_stale": not (exact["A_exact"] > 0.02 and learned["A_learned"] <= 0.0),
            }
        )
    available = [item for item in candidate_records if item["non_stale"]]
    matched = []
    for affected in sorted(h_panel, key=lambda item: item["uid"]):
        if not available:
            raise RuntimeError("insufficient one-to-one non-stale placebo candidates")
        choice = min(
            available,
            key=lambda item: (
                item["row"]["instance_id"] != affected["instance_id"],
                abs(int(item["row"]["full_state"]["step_count"]) - affected["step"]),
                abs(math.log(item["learned"]["weight_learned"]) - math.log(affected["weight_learned"])),
                item["row"]["_scb_uid"],
            ),
        )
        available.remove(choice)
        row = choice["row"]
        matched.append(
            {
                "affected_uid": affected["uid"],
                "uid": row["_scb_uid"],
                "instance_id": row["instance_id"],
                "origin": row["_scb_origin"],
                "node": "junction",
                "step": int(row["full_state"]["step_count"]),
                "action_structural": row["action_structural"],
                "current_A_exact": choice["exact"]["A_exact"],
                "current_weight_exact": choice["exact"]["weight_exact"],
                **choice["learned"],
                "same_instance": row["instance_id"] == affected["instance_id"],
                "same_route_position": True,
                "same_action_identity": row["action_structural"] == affected["action_structural"],
            }
        )

    diagnostic_uids = sorted([item["uid"] for item in h_panel[:32]] + [item["uid"] for item in matched[:32]])
    h_result = {
        "schema": "SC-C harmful-stale panel v1",
        "event_acceptance_depends_on_panel_nonempty": False,
        "definition": "old exact A<-0.02 AND current exact A>0.02 AND matching learned A<=0",
        "source_checkpoint_sha256": sha256(source / "capability_checkpoint/manifest.json"),
        "base_provenance": provenance,
        "candidate_junction_shortcut_rows": len(candidate_h),
        "panel_size": len(h_panel),
        "items": h_panel,
    }
    h_result["panel_hash"] = canonical_hash(h_result)
    placebo_result = {
        "schema": "SC-C one-to-one placebo panel v1",
        "matching_priority": ["same instance", "same junction step", "learned log-weight", "stable uid"],
        "panel_size": len(matched),
        "affected_panel_hash": h_result["panel_hash"],
        "items": matched,
    }
    placebo_result["panel_hash"] = canonical_hash(placebo_result)
    write_json(out / "H_PANEL.json", h_result)
    write_json(out / "PLACEBO_PANEL.json", placebo_result)
    write_json(out / "DIAGNOSTIC_BATCH.json", diagnostic_uids)


if __name__ == "__main__":
    main()
