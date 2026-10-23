#!/usr/bin/env python3
"""Summarize the nonconfirmatory SC-A/SC-B interface pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_sc_b_replay import ARMS
from scabc_common import canonical_hash, sha256, write_json


def auc(trace: list[dict], key: str) -> float:
    total = 0.0
    for left, right in zip(trace, trace[1:]):
        total += (right["update"] - left["update"]) * (
            left["macro"][key] + right["macro"][key]
        ) / 2.0
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    sc_a = json.loads((pilot / "sc_a_prebranch.json").read_text())
    repair_root = pilot / "sc_b_full_pilot_v2"
    arms = []
    initials = []
    for arm in ARMS:
        trace_path = repair_root / arm / "repair_trace.json"
        manifest_path = repair_root / arm / "RUN_MANIFEST.json"
        trace = json.loads(trace_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        initials.append(manifest["initial"])
        repaired = next(
            (
                row["update"]
                for row in trace
                if row["macro"]["harmful_stale"] == 0.0
                and row["macro"]["rank_repaired"] == 1.0
            ),
            None,
        )
        arms.append(
            {
                "arm": arm,
                "repair_time": repaired,
                "right_censored": repaired is None,
                "harmful_stale_auc": auc(trace, "harmful_stale"),
                "endpoint": trace[-1]["macro"],
                "actor_unchanged": manifest["actor_unchanged"],
                "critic_updates": manifest["final"]["critic_updates"],
                "oracle_calls_during_training": manifest["oracle_calls_during_training"],
                "trace_sha256": sha256(trace_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
    result = {
        "status": "NONCONFIRMATORY_M4_PILOT_PASS",
        "source_seed": 0,
        "fresh_confirmatory_outcome": False,
        "sc_a": {
            "tau_true": sc_a["events"]["tau_true"],
            "tau_capability": sc_a["tau_capability"],
            "tau_Q": sc_a["events"]["tau_Q"],
            "tau_Q_right_censored_prebranch": sc_a["events"]["tau_Q_censored"],
            "tau_pi": sc_a["events"]["tau_pi"],
            "tau_pi_right_censored_prebranch": sc_a["events"]["tau_pi_censored"],
            "capability_checkpoint": sc_a["tau_capability_metrics"],
            "all_checkpoints_matching": sc_a["all_checkpoints_matching"],
            "trace_sha256": sha256(pilot / "sc_a_prebranch.json"),
        },
        "sc_b": {
            "branch_initial_states_equal": all(item == initials[0] for item in initials),
            "arms": arms,
        },
    }
    result["summary_hash"] = canonical_hash(result)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
