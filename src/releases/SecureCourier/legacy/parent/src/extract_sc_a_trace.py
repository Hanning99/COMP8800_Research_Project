#!/usr/bin/env python3
"""Extract SC-A measurements from synchronized matching actor--critic checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import (
    TEMPERATURE,
    WEIGHT_CLIP,
    exact_instance_metrics,
    learned_state_action_values,
    load_model_bundle,
    module_hash,
    sha256,
    split_instances,
    write_json,
)


PERSISTENCE_CHECKPOINTS = 2
POLICY_ADOPTION_PROBABILITY = 0.50


def awac_weight(advantage: float) -> float:
    return float(min(WEIGHT_CLIP, math.exp(advantage / TEMPERATURE)))


def first_persistent(rows: list[dict], predicate) -> tuple[int | None, bool]:
    for index in range(len(rows) - PERSISTENCE_CHECKPOINTS + 1):
        window = rows[index : index + PERSISTENCE_CHECKPOINTS]
        if all(predicate(row) for row in window):
            return int(window[0]["cycle"]), False
    return None, True


def checkpoint_row(
    checkpoint: Path,
    cycle: int,
    device: torch.device,
    old_exact: dict[str, dict],
    old_learned: dict[str, dict],
) -> dict:
    actor, critic, _, _, _, vocab, metadata, _ = load_model_bundle(checkpoint, device)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if int(metadata.get("acquisition_cycle", cycle)) != cycle:
        raise RuntimeError(f"counter mismatch at {checkpoint}")
    if manifest["actor_hash"] != module_hash(actor) or manifest["critic_hash"] != module_hash(critic):
        raise RuntimeError(f"checkpoint hash mismatch at {checkpoint}")

    instances = []
    for instance in split_instances()["train"]:
        exact = exact_instance_metrics(actor, vocab, instance, device)
        learned = learned_state_action_values(actor, critic, vocab, instance, device)
        exact_a = exact["A_shortcut"]
        learned_a = learned["A"]["take_shortcut"]
        exact_rank = exact["shortcut_rank_one"]
        learned_rank = learned["rank"][0] == "take_shortcut"
        old_e = old_exact[instance.instance_id]
        old_l = old_learned[instance.instance_id]
        delta_exact = exact["Q_shortcut"] - old_e["Q_shortcut"]
        delta_learned = learned["Q"]["take_shortcut"] - old_l["Q"]["take_shortcut"]
        true_flip = old_e["A_shortcut"] < -0.02 and exact_a > 0.02
        p_shortcut = learned["policy"]["take_shortcut"]
        p_safe = learned["policy"]["take_safe_route"]
        instances.append(
            {
                "instance_id": instance.instance_id,
                "exact_Q_shortcut": exact["Q_shortcut"],
                "exact_A_shortcut": exact_a,
                "learned_Q_shortcut": learned["Q"]["take_shortcut"],
                "learned_A_shortcut": learned_a,
                "exact_rank_one": exact_rank,
                "learned_rank_one": learned_rank,
                "rank_agreement": exact_rank == learned_rank,
                "exact_weight": awac_weight(exact_a),
                "learned_weight": awac_weight(learned_a),
                "log_weight_error": math.log(awac_weight(exact_a)) - math.log(awac_weight(learned_a)),
                "advantage_error": exact_a - learned_a,
                "p_shortcut": p_shortcut,
                "p_safe": p_safe,
                "shortcut_log_odds_vs_safe": math.log(max(p_shortcut, 1e-12) / max(p_safe, 1e-12)),
                "full_start_shortcut_route_probability": exact["full_start_shortcut_route_probability"],
                "full_start_safe_route_probability": exact["full_start_safe_route_probability"],
                "J_exact": exact["J_exact"],
                "true_flip": true_flip,
                "harmful_stale": true_flip and learned_a <= 0.0,
                "fraction_true_value_change_tracked": delta_learned / delta_exact if abs(delta_exact) > 1e-12 else None,
            }
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in instances if row[key] is not None]
        return float(np.mean(values)) if values else None

    macro = {
        key: mean(key)
        for key in (
            "exact_Q_shortcut",
            "exact_A_shortcut",
            "learned_Q_shortcut",
            "learned_A_shortcut",
            "exact_weight",
            "learned_weight",
            "log_weight_error",
            "advantage_error",
            "p_shortcut",
            "p_safe",
            "shortcut_log_odds_vs_safe",
            "full_start_shortcut_route_probability",
            "full_start_safe_route_probability",
            "J_exact",
            "fraction_true_value_change_tracked",
        )
    }
    macro.update(
        {
            "true_flip_prevalence": mean("true_flip"),
            "harmful_stale_prevalence": mean("harmful_stale"),
            "exact_rank_one_prevalence": mean("exact_rank_one"),
            "learned_rank_one_prevalence": mean("learned_rank_one"),
            "rank_agreement": mean("rank_agreement"),
        }
    )
    return {
        "cycle": cycle,
        "critic_updates": int(metadata.get("critic_updates", 0)),
        "actor_updates": int(metadata.get("actor_updates", 0)),
        "matching_actor_critic": bool(manifest["matching_actor_critic"]),
        "actor_hash": manifest["actor_hash"],
        "critic_hash": manifest["critic_hash"],
        "checkpoint_manifest_sha256": sha256(checkpoint / "manifest.json"),
        "macro": macro,
        "instances": instances,
    }


def trapezoid_exposure(rows: list[dict]) -> float:
    total = 0.0
    for left, right in zip(rows, rows[1:]):
        x0, x1 = left["cycle"], right["cycle"]
        y0 = max(0.0, left["macro"]["advantage_error"])
        y1 = max(0.0, right["macro"]["advantage_error"])
        total += (x1 - x0) * (y0 + y1) / 2.0
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    source = args.source_event.resolve()
    status = json.loads((source / "STATUS.json").read_text())
    selected_cycle = int(status["selected_cycle"])

    old_actor, old_critic, _, _, _, old_vocab, _, _ = load_model_bundle(source / "old/checkpoint", device)
    old_exact = {
        instance.instance_id: exact_instance_metrics(old_actor, old_vocab, instance, device)
        for instance in split_instances()["train"]
    }
    old_learned = {
        instance.instance_id: learned_state_action_values(old_actor, old_critic, old_vocab, instance, device)
        for instance in split_instances()["train"]
    }

    checkpoints = sorted((source / "acquisition_trace/checkpoints").glob("cycle_*"))
    rows = []
    for checkpoint in checkpoints:
        cycle = int(checkpoint.name.split("_")[-1])
        rows.append(checkpoint_row(checkpoint, cycle, device, old_exact, old_learned))

    tau_true, true_censored = first_persistent(
        rows,
        lambda row: row["macro"]["exact_A_shortcut"] > 0.02
        and row["macro"]["true_flip_prevalence"] >= 0.90,
    )
    tau_q, q_censored = first_persistent(rows, lambda row: row["macro"]["learned_A_shortcut"] > 0.0)
    tau_pi, pi_censored = first_persistent(
        rows, lambda row: row["macro"]["p_shortcut"] >= POLICY_ADOPTION_PROBABILITY
    )
    selected = next(row for row in rows if row["cycle"] == selected_cycle)
    result = {
        "source_seed": status["seed"],
        "mode": status["mode"],
        "fresh_confirmatory_outcome": False if status["mode"] == "pilot" else True,
        "tau_capability": selected_cycle,
        "event_definitions": {
            "persistence_checkpoints": PERSISTENCE_CHECKPOINTS,
            "tau_true": "first of two consecutive checkpoints with macro exact A>0.02 and true-flip prevalence>=0.90",
            "tau_Q": "first of two consecutive checkpoints with macro learned actor-facing A>0",
            "tau_pi": "first of two consecutive checkpoints with mean junction shortcut probability>=0.50",
            "right_censoring": "no qualifying persistent run by final recorded checkpoint",
        },
        "events": {
            "tau_true": tau_true,
            "tau_true_censored": true_censored,
            "tau_Q": tau_q,
            "tau_Q_censored": q_censored,
            "tau_pi": tau_pi,
            "tau_pi_censored": pi_censored,
            "L_Q": None if tau_true is None or tau_q is None else tau_q - tau_true,
            "L_pi": None if tau_true is None or tau_pi is None else tau_pi - tau_true,
            "observation_end_cycle": rows[-1]["cycle"],
        },
        "tau_capability_metrics": selected["macro"],
        "integrated_stale_exposure_prebranch": trapezoid_exposure(rows),
        "all_checkpoints_matching": all(row["matching_actor_critic"] for row in rows),
        "checkpoints": rows,
    }
    write_json(args.out, result)


if __name__ == "__main__":
    main()
