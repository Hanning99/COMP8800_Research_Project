#!/usr/bin/env python3
"""Fail-closed, outcome-free D3 adapter/closed-loop audit for SC-D."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pdr_awac_adapter import PDRService
from run_sc_d import pdr_intervention_active
from scd_common import PARENT, module_hash, sha256
from training_core import base_rows, clean_collate, load_model_bundle

sys.path.insert(0, str(PARENT / "src/vendor/diy_task_set"))
sys.path.insert(0, str(PARENT / "src/vendor"))
from pdr.masks.action_mask import masked_log_softmax  # noqa: E402


BASE = Path(__file__).resolve().parent.parent
PILOTS = BASE / "pilots"
ARCHIVED = PARENT / "sc_c_oracle/confirmatory_000/STALE"


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def same_bytes(left: Path, right: Path) -> bool:
    return left.read_bytes() == right.read_bytes()


def same_prefix(short: Path, long: Path) -> bool:
    payload = short.read_bytes()
    return long.read_bytes()[: len(payload)] == payload


def hashes(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def final_hashes(root: Path) -> dict:
    return read(root / "RUN_MANIFEST.json")["final"]


def selected_equal(left: dict, right: dict, keys: list[str]) -> bool:
    return all(left[key] == right[key] for key in keys)


def archive_parity() -> dict:
    replay = PILOTS / "d1_archive_parity_event000_confirmatory/STALE"
    checkpoint_files = [
        "branch_checkpoint/state.pt",
        "branch_checkpoint/vocab.pkl",
        "branch_checkpoint/manifest.json",
        "checkpoints/cycle_000016/state.pt",
        "checkpoints/cycle_000016/vocab.pkl",
        "checkpoints/cycle_000016/manifest.json",
    ]
    log_files = ["continuation_stream.jsonl", "training_log.jsonl", "evaluation_log.jsonl"]
    checkpoint_matches = {name: same_bytes(replay / name, ARCHIVED / name) for name in checkpoint_files}
    log_matches = {name: same_prefix(replay / name, ARCHIVED / name) for name in log_files}
    replay_trace = read(replay / "evaluation_trace.json")
    archive_trace = read(ARCHIVED / "evaluation_trace.json")
    trace_match = replay_trace == archive_trace[: len(replay_trace)]
    return {
        "event_id": "confirmatory_000",
        "cycles_replayed": 16,
        "stack": "parent run_sc_c_oracle.py, confirmatory mode, CUDA GPU0",
        "checkpoint_byte_matches": checkpoint_matches,
        "jsonl_byte_prefix_matches": log_matches,
        "evaluation_trace_value_prefix_match": trace_match,
        "replay_files": {name: hashes(replay / name) for name in checkpoint_files + log_files},
        "pass": all(checkpoint_matches.values()) and all(log_matches.values()) and trace_match,
    }


def standard_adapter_parity() -> dict:
    adapter = PILOTS / "d3v4_standard_16"
    parent = PILOTS / "d3_parent_standard_16/STALE"
    observed = final_hashes(adapter)
    expected = read(parent / "checkpoints/cycle_000016/manifest.json")
    hash_keys = ["actor_hash", "critic_hash", "target_hash", "actor_optimizer_hash", "critic_optimizer_hash"]
    state_hash_matches = {key: observed[key] == expected[key] for key in hash_keys}
    ours = rows(adapter / "training_log.jsonl")
    theirs = rows(parent / "training_log.jsonl")
    scalar_keys = [
        "cycle",
        "episode",
        "online_replay_size",
        "critic_loss",
        "critic_gradient_norm",
        "target_q_mean",
        "selected_q_mean",
        "actor_loss",
        "actor_gradient_norm",
        "actor_loss_denominator",
    ]
    scalar_match = len(ours) == len(theirs) == 16 and all(
        selected_equal(a, b, scalar_keys) for a, b in zip(ours, theirs)
    )
    weight_match = all(a["weight_mean"] == b["applied_weight_mean"] for a, b in zip(ours, theirs))
    stream_match = same_bytes(adapter / "continuation_stream.jsonl", parent / "continuation_stream.jsonl")
    return {
        "cycles": 16,
        "device": "cpu",
        "state_hash_matches": state_hash_matches,
        "all_shared_training_scalars_exact": scalar_match,
        "actor_weight_mean_exact": weight_match,
        "continuation_stream_byte_identical": stream_match,
        "pass": all(state_hash_matches.values()) and scalar_match and weight_match and stream_match,
    }


def first_treatment_difference() -> dict:
    standard = PILOTS / "d3v4_first_standard"
    pdr = PILOTS / "d3v4_first_pdr_no_scorer"
    std_manifest = read(standard / "RUN_MANIFEST.json")
    pdr_manifest = read(pdr / "RUN_MANIFEST.json")
    std_log = rows(standard / "training_log.jsonl")[0]
    pdr_log = rows(pdr / "training_log.jsonl")[0]
    common_initial = ["actor_hash", "critic_hash", "target_hash", "branch_state_sha256"]
    critic_keys = ["critic_hash", "target_hash", "critic_optimizer_hash"]
    critic_scalars = ["critic_loss", "critic_gradient_norm", "target_q_mean", "selected_q_mean"]
    return {
        "same_initial_training_state": selected_equal(std_manifest["initial"], pdr_manifest["initial"], common_initial),
        "same_pre_treatment_action_stream": same_bytes(
            standard / "continuation_stream.jsonl", pdr / "continuation_stream.jsonl"
        ),
        "same_critic_update": selected_equal(std_manifest["final"], pdr_manifest["final"], critic_keys)
        and selected_equal(std_log, pdr_log, critic_scalars),
        "same_pre_actor_hash": std_log["actor_hash_before"] == pdr_log["actor_hash_before"],
        "actor_diverges_only_at_adapter": std_manifest["final"]["actor_hash"] != pdr_manifest["final"]["actor_hash"]
        and std_manifest["final"]["actor_optimizer_hash"] != pdr_manifest["final"]["actor_optimizer_hash"],
        "fresh_pdr_materialized": all(
            pdr_log[key] is not None for key in ("pi_current_hash", "source_hash", "correction_hash", "q_pdr_hash")
        ),
        "pdr_recurrence_residual_inf": pdr_log["recurrence_residual_inf"],
        "pass": False,
    }


def masking_detachment_and_lookup() -> dict:
    source = PARENT / "pilots/m3_resume_a"
    branch = source / "capability_checkpoint"
    actor, _, _, _, _, vocab, _, _ = load_model_bundle(branch, torch.device("cpu"))
    replay = base_rows(source)
    batch = clean_collate(replay[:64], vocab, torch.device("cpu"))
    with torch.inference_mode():
        raw = actor(batch["obs_tokens"], batch["act_tokens"])
        legal_logp = masked_log_softmax(raw, batch["action_mask"])
        padded_mass = float(legal_logp.exp()[~batch["action_mask"]].sum())
    service = PDRService(BASE / "manifests/support_graph.json", BASE / "pilots/d2_reference", vocab, torch.device("cpu"))
    corrupt = copy.deepcopy(replay[0])
    corrupt["full_state"]["node"] = "__INTENTIONALLY_UNMAPPED__"
    failure = None
    try:
        service.lookup([corrupt], np.zeros(len(service.dag.pairs)), torch.device("cpu"), torch.float32)
    except RuntimeError as exc:
        failure = str(exc)
    pdr_logs = rows(PILOTS / "d3v4_pdr_16/training_log.jsonl")
    checks = {
        "real_batch_contains_padding": int((~batch["action_mask"]).sum()) > 0,
        "padded_probability_mass_exact_zero": padded_mass == 0.0,
        "runtime_padded_mass_exact_zero": all(row["padded_action_mass"] == 0.0 for row in pdr_logs),
        "weights_detached_every_update": all(row["weights_detached"] is True for row in pdr_logs),
        "loss_denominator_is_batch_size": all(row["actor_loss_denominator"] == 64 for row in pdr_logs),
        "all_real_legal_actions_mapped": all(row["lookup_coverage"] == row["real_action_slots"] for row in pdr_logs),
        "unmapped_row_fails_closed": failure is not None and failure.startswith("UNMAPPED_ACTOR_ROW:"),
    }
    return {
        "padded_slots_in_real_batch": int((~batch["action_mask"]).sum()),
        "padded_probability_mass": padded_mass,
        "unmapped_failure": failure,
        "checks": checks,
        "pass": all(checks.values()),
    }


def firewall() -> dict:
    disabled = PILOTS / "d3v4_firewall_disabled"
    enabled = PILOTS / "d3v4_firewall_scoring"
    required_equal = [
        "training_log.jsonl",
        "continuation_stream.jsonl",
        "checkpoints/cycle_000002/state.pt",
        "checkpoints/cycle_000002/vocab.pkl",
        "checkpoints/cycle_000002/manifest.json",
    ]
    byte_matches = {name: same_bytes(disabled / name, enabled / name) for name in required_equal}
    scoring_rows = rows(enabled / "scoring_log.jsonl")
    source_text = "\n".join(
        (BASE / "src" / name).read_text() for name in ("pdr_awac_adapter.py", "reverse_pdr.py")
    )
    no_scorer_run = read(PILOTS / "d3v4_first_pdr_no_scorer/RUN_MANIFEST.json")
    config = read(PILOTS / "d3v4_first_pdr_no_scorer/RUN_CONFIG.json")
    import_probe = subprocess.run(
        [sys.executable, "-c", "import sys; import run_sc_d; assert 'scabc_common' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": str(BASE / "src")},
        capture_output=True,
        text=True,
    )
    checks = {
        "training_artifacts_byte_identical": all(byte_matches.values()),
        "scoring_rows_labeled_scoring_only": bool(scoring_rows) and all(row.get("scoring_only") is True for row in scoring_rows),
        "scorer_forbidden_run_completed": no_scorer_run["status"] == "PILOT_COMPLETE",
        "training_import_graph_excludes_scorer": import_probe.returncode == 0,
        "pdr_modules_do_not_import_scorer": "scabc_common" not in source_text and "exact_instance_metrics" not in source_text,
        "pdr_routing_excludes_H_e": "H_e" not in source_text,
        "pdr_manifest_has_no_exact_q_input": not any("exact" in str(key).lower() for key in config),
    }
    return {"byte_matches": byte_matches, "checks": checks, "pass": all(checks.values())}


def resume() -> dict:
    full = PILOTS / "d3v4_resume_full"
    split = PILOTS / "d3v4_resume_split"
    equal_files = [
        "continuation_stream.jsonl",
        "training_log.jsonl",
        "checkpoints/cycle_000002/state.pt",
        "checkpoints/cycle_000002/manifest.json",
        "checkpoints/cycle_000004/state.pt",
        "checkpoints/cycle_000004/manifest.json",
    ]
    matches = {name: same_bytes(full / name, split / name) for name in equal_files}
    final_keys = ["actor_hash", "critic_hash", "target_hash", "actor_optimizer_hash", "critic_optimizer_hash"]
    final_match = selected_equal(final_hashes(full), final_hashes(split), final_keys)
    return {"byte_matches": matches, "final_training_hashes_match": final_match, "pass": all(matches.values()) and final_match}


def restoration() -> dict:
    standard = PILOTS / "d3v4_first_standard"
    inactive = PILOTS / "d3v4_restore_pdr_inactive"
    final_keys = ["actor_hash", "critic_hash", "target_hash", "actor_optimizer_hash", "critic_optimizer_hash"]
    standard_effective = rows(inactive / "pdr_compute_log.jsonl")[0]["effective_arm"] == "STANDARD_TD"
    checks = {
        "cycle_1024_is_pdr": pdr_intervention_active("FRESH_PDR", 1024, 1024),
        "cycle_1025_is_standard": not pdr_intervention_active("FRESH_PDR", 1025, 1024),
        "inactive_fresh_pdr_uses_standard_adapter": standard_effective,
        "fallback_training_log_byte_identical": same_bytes(
            standard / "training_log.jsonl", inactive / "training_log.jsonl"
        ),
        "fallback_action_stream_byte_identical": same_bytes(
            standard / "continuation_stream.jsonl", inactive / "continuation_stream.jsonl"
        ),
        "fallback_final_training_hashes_identical": selected_equal(
            final_hashes(standard), final_hashes(inactive), final_keys
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def closed_loop_and_lock() -> dict:
    pilot = PILOTS / "d3v4_pdr_16"
    manifest = read(pilot / "RUN_MANIFEST.json")
    training = rows(pilot / "training_log.jsonl")
    compute = rows(pilot / "pdr_compute_log.jsonl")
    compute_fields = {
        "cycle", "event_id", "replicate_id", "arm", "effective_arm", "pre_actor_update_hash",
        "pi_current_hash", "source_hash", "correction_hash", "delivered_value_hash",
        "full_domain_actor_evaluations", "lookup_coverage", "recurrence_residual_inf",
        "policy_seconds", "solve_seconds", "lookup_seconds", "actor_update_seconds",
        "peak_memory_bytes", "total_cycle_seconds",
    }
    with tempfile.TemporaryDirectory(prefix="scd_confirmatory_lock_") as temporary:
        result = subprocess.run(
            [
                sys.executable, str(BASE / "src/run_sc_d.py"),
                "--source-event", str(PARENT / "pilots/m3_resume_a"),
                "--reference", str(BASE / "pilots/d2_reference"),
                "--graph", str(BASE / "manifests/support_graph.json"),
                "--arm", "FRESH_PDR", "--output", str(Path(temporary) / "forbidden"),
                "--mode", "confirmatory", "--max-cycles", "1",
            ],
            env={**os.environ, "PYTHONPATH": str(BASE / "src")},
            capture_output=True,
            text=True,
        )
    checks = {
        "pilot_complete": manifest["status"] == "PILOT_COMPLETE",
        "pilot_explicitly_nonofficial": manifest["official_pdr_outcome"] is False,
        "sixteen_closed_loop_updates": len(training) == len(compute) == 16,
        "pdr_active_every_pilot_cycle": all(row["effective_arm"] == "FRESH_PDR" for row in compute),
        "compute_counter_schema_complete": all(compute_fields <= set(row) for row in compute),
        "full_graph_evaluated_every_refresh": all(row["full_domain_actor_evaluations"] == 14576 for row in compute),
        "confirmatory_mode_still_hard_locked": result.returncode != 0 and "CONFIRMATORY_SC_D_LOCKED_UNTIL_D4" in result.stderr,
    }
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    first = first_treatment_difference()
    first["pass"] = all(value for key, value in first.items() if key not in {"pass", "pdr_recurrence_residual_inf"}) and first["pdr_recurrence_residual_inf"] <= 1e-10
    sections = {
        "archive_standard_parity": archive_parity(),
        "standard_adapter_parity": standard_adapter_parity(),
        "first_treatment_difference": first,
        "mask_detachment_lookup": masking_detachment_and_lookup(),
        "scoring_firewall": firewall(),
        "resume_determinism": resume(),
        "cycle_1025_restoration": restoration(),
        "closed_loop_pilot_and_lock": closed_loop_and_lock(),
    }
    passed = all(section["pass"] for section in sections.values())
    payload = {
        "schema": "SC-D D3 closed-loop pilot audit v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_pdr_outcomes_generated": False,
        "pilot_use_restriction": "engineering diagnosis only; no pilot effect estimate inspected or reported",
        **sections,
        "code_hashes": {
            name: sha256(BASE / "src" / name)
            for name in ("training_core.py", "pdr_awac_adapter.py", "run_sc_d.py", "audit_d3.py")
        },
        "status": "D3_CLOSED_LOOP_PILOT_PASS" if passed else "D3_CLOSED_LOOP_PILOT_FAIL",
        "pass": passed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# SC-D D3 AWAC adapter and closed-loop pilot audit",
        "",
        f"- Status: `{payload['status']}`",
        "- Official PDR outcomes generated: `false`",
        "- Pilot outcome use: engineering diagnosis only; no pilot effect estimate was inspected or reported.",
        "",
        "## Gate results",
        "",
    ]
    labels = {
        "archive_standard_parity": "Archived Standard same-stack byte parity",
        "standard_adapter_parity": "SC-D Standard adapter vs parent Standard",
        "first_treatment_difference": "First treatment difference occurs at actor adapter",
        "mask_detachment_lookup": "Mask, padding, detachment, and fail-closed lookup",
        "scoring_firewall": "Exact-current scoring firewall",
        "resume_determinism": "Pause/resume byte determinism",
        "cycle_1025_restoration": "Cycle-1025 Standard-signal restoration",
        "closed_loop_pilot_and_lock": "16-cycle closed loop and pre-D4 confirmatory lock",
    }
    for key, label in labels.items():
        lines.append(f"- {label}: **{'PASS' if sections[key]['pass'] else 'FAIL'}**")
    lines.extend([
        "",
        "The archived comparator may be reused because the parent runner reproduced its cycle-16 state, manifests, action stream, training values, and evaluation values exactly on the locked CUDA stack. The pilot establishes interface operation only and is not scientific SC-D evidence.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
