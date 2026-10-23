#!/usr/bin/env python3
"""Summarize the nonconfirmatory 32-cycle SC-C interface pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_sc_c_oracle import ARMS
from scabc_common import canonical_hash, sha256, write_json


def normalized_auc(trace: list[dict], key: str) -> float:
    total = 0.0
    for left, right in zip(trace, trace[1:]):
        total += (right["cycle"] - left["cycle"]) * (
            left["macro"][key] + right["macro"][key]
        ) / 2.0
    return total / (trace[-1]["cycle"] - trace[0]["cycle"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    event = args.event_root.resolve()
    source = args.source_event.resolve()
    rows = {}
    initials = []
    for arm in ARMS:
        manifest = json.loads((event / arm / "RUN_MANIFEST.json").read_text())
        trace = json.loads((event / arm / "evaluation_trace.json").read_text())
        initials.append(manifest["initial"])
        rows[arm] = {
            "EarlyAUC_32": normalized_auc(trace, "J_exact"),
            "shortcut_route_AUC_32": normalized_auc(trace, "full_start_shortcut_route_probability"),
            "endpoint": trace[-1],
            "targeted_rows": manifest["total_targeted_rows_in_actor_batches"],
            "treatment_calls": manifest["treatment_oracle_calls"],
            "unauthorized_calls": manifest["unauthorized_treatment_oracle_calls"],
            "trace_sha256": sha256(event / arm / "evaluation_trace.json"),
        }
    contrasts = {
        "CORRECT_minus_STALE": rows["CORRECT"]["EarlyAUC_32"] - rows["STALE"]["EarlyAUC_32"],
        "CORRECT_minus_DROP": rows["CORRECT"]["EarlyAUC_32"] - rows["DROP"]["EarlyAUC_32"],
        "CORRECT_minus_PLACEBO": rows["CORRECT"]["EarlyAUC_32"] - rows["PLACEBO"]["EarlyAUC_32"],
        "DROP_minus_STALE": rows["DROP"]["EarlyAUC_32"] - rows["STALE"]["EarlyAUC_32"],
    }
    stale_final = json.loads((event / "STALE/RUN_MANIFEST.json").read_text())["final"]
    natural = json.loads(
        (source / "acquisition_trace/checkpoints/cycle_000128/manifest.json").read_text()
    )
    keys = ("actor_hash", "critic_hash", "target_hash", "actor_optimizer_hash", "critic_optimizer_hash")
    result = {
        "status": "NONCONFIRMATORY_M5_32_CYCLE_INTERFACE_PILOT_PASS",
        "fresh_confirmatory_outcome": False,
        "branch_initial_states_equal": all(item == initials[0] for item in initials),
        "stale_matches_unbranched_natural_cycle_128": all(stale_final[key] == natural[key] for key in keys),
        "arms": rows,
        "early_auc_contrasts_descriptive_only": contrasts,
        "dose_note": "32-cycle target-row counts are small and unequal; registered full-window results must report dose and dose-adjusted sensitivity",
    }
    result["summary_hash"] = canonical_hash(result)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
