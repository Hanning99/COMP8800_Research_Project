#!/usr/bin/env python3
"""Run the outcome-free SC-D graph/formulation/solver audit."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_closed_dag import build_graph
from compile_pdr_reference import evaluate_actor_critic
from pdr_awac_adapter import FactorizedPolicyEvaluator
from reverse_pdr import ClosedDAG, array_hash, reverse_solve_arrays
from scd_common import LEGACY, PARENT, canonical_hash, read_json, sha256, write_pretty_json


sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(PARENT / "src/vendor/diy_task_set"))
from scabc_common import load_model_bundle  # noqa: E402
from build_sc_b_replay import base_rows  # noqa: E402


def legacy_equivalence() -> dict:
    sys.path.insert(0, str(LEGACY))
    from pdr_advanced.pipeline import CompiledPDR, load_actor  # noqa: E402

    pipeline = CompiledPDR(LEGACY)
    actor_path = sorted((LEGACY / "actor_checkpoints").glob("seed_*/PDR_AWAC/actor_u0016.pt"))[0]
    actor = load_actor(LEGACY, actor_path, device="cpu")
    pi = pipeline.native_policy(actor, device="cpu")
    expected = pipeline.solve(pi)
    arrays = pipeline.arrays
    state_step = np.asarray([json.loads(x)[2] for x in arrays["state_json"].tolist()], dtype=np.int16)
    observed = reverse_solve_arrays(
        pair_state=arrays["pair_state"],
        pair_step=arrays["pair_step"],
        successor=arrays["successor_state"],
        continuation=arrays["continuation"],
        state_step=state_step,
        pi_ref=arrays["pi_old"],
        pi_current=pi,
        qbar=arrays["qtheta"],
    )
    return {
        "actor_path": str(actor_path),
        "actor_sha256": sha256(actor_path),
        "max_abs_source_error": float(np.max(np.abs(observed.source_by_state - expected.b_state))),
        "max_abs_correction_error": float(np.max(np.abs(observed.correction_by_pair - expected.d))),
        "max_abs_q_pdr_error": float(np.max(np.abs(observed.q_pdr_by_pair - expected.q_pdr))),
        "observed_residual_inf": observed.residual_inf,
        "pass": bool(
            np.array_equal(observed.source_by_state, expected.b_state)
            and np.array_equal(observed.correction_by_pair, expected.d)
            and np.array_equal(observed.q_pdr_by_pair, expected.q_pdr)
        ),
    }


def replay_mapping(dag: ClosedDAG) -> dict:
    mapped = 0
    per_event = []
    missing = []
    for index in range(50):
        event_id = f"confirmatory_{index:03d}"
        rows, _ = base_rows(PARENT / "source_events" / event_id)
        event_mapped = 0
        for row in rows:
            state = row.get("full_state") or {}
            key = f"{row['instance_id']}\x1f{state.get('node')}\x1f{int(state.get('step_count')):02d}\x1f{row['action_structural']}"
            if key not in dag.pair_index:
                missing.append({"event_id": event_id, "uid": row.get("_scb_uid"), "key": key})
            else:
                event_mapped += 1
        mapped += event_mapped
        per_event.append({"event_id": event_id, "rows": len(rows), "mapped": event_mapped})
    return {
        "total_rows": sum(row["rows"] for row in per_event),
        "mapped_rows": mapped,
        "unmapped_rows": len(missing),
        "first_unmapped": missing[:20],
        "per_event": per_event,
        "pass": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--graph-manifest", type=Path, required=True)
    parser.add_argument("--pilot-reference", type=Path, required=True)
    parser.add_argument("--pilot-current", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    legacy_reuse_manifest_path = Path(__file__).resolve().parent.parent / "manifests/legacy_reuse_manifest.json"
    dag = ClosedDAG(args.graph)
    pi_ref = np.load(args.pilot_reference / "pi_ref.npy", allow_pickle=False)
    qbar = np.load(args.pilot_reference / "qbar.npy", allow_pickle=False)
    actor, critic, _, _, _, vocab, _, _ = load_model_bundle(args.pilot_current, torch.device("cpu"))
    pi_current, _ = evaluate_actor_critic(dag, actor, critic, vocab, torch.device("cpu"))
    fast_pi_current = FactorizedPolicyEvaluator(dag, vocab, torch.device("cpu")).evaluate(actor)
    fast_policy_error = float(np.max(np.abs(fast_pi_current - pi_current)))
    fast_policy_row_error = float(
        np.max(np.abs(np.bincount(dag.pair_state, weights=fast_pi_current, minlength=len(dag.states)) - 1.0))
    )
    solve_start = time.perf_counter()
    first = dag.solve(pi_ref, pi_current, qbar)
    solve_seconds = time.perf_counter() - solve_start
    second = dag.solve(pi_ref, pi_current, qbar)
    zero = dag.solve(pi_ref, pi_ref.copy(), qbar)
    graph_again = build_graph()
    graph_manifest = read_json(args.graph_manifest)
    legacy = legacy_equivalence()
    mapping = replay_mapping(dag)
    checks = {
        "graph_canonical_determinism": canonical_hash(graph_again) == graph_manifest["canonical_content_sha256"],
        "closure": bool(graph_manifest["closure_pass"]),
        "unit_probability": bool(graph_manifest["deterministic_unit_probability"]),
        "strict_time_layering": bool(graph_manifest["strictly_time_layered"]),
        "zero_dropped_policy_mass": graph_manifest["dropped_policy_mass"] == 0.0,
        "policy_finite": bool(np.all(np.isfinite(pi_ref)) and np.all(np.isfinite(pi_current))),
        "qbar_finite": bool(np.all(np.isfinite(qbar))),
        "source_finite": bool(np.all(np.isfinite(first.source_by_state))),
        "correction_finite": bool(np.all(np.isfinite(first.correction_by_pair))),
        "q_pdr_finite": bool(np.all(np.isfinite(first.q_pdr_by_pair))),
        "recurrence_residual_le_1e_10": first.residual_inf <= 1e-10,
        "terminal_correction_exact_zero": bool(np.max(np.abs(first.correction_by_pair[~dag.continuation])) == 0.0),
        "zero_source_correction_le_1e_12": bool(np.max(np.abs(zero.correction_by_pair)) <= 1e-12),
        "repeat_bitwise_source": bool(np.array_equal(first.source_by_state, second.source_by_state)),
        "repeat_bitwise_correction": bool(np.array_equal(first.correction_by_pair, second.correction_by_pair)),
        "legacy_bitwise_equivalence": bool(legacy["pass"]),
        "all_parent_actor_rows_map": bool(mapping["pass"]),
        "legacy_reuse_manifest_present": legacy_reuse_manifest_path.is_file(),
        "accelerated_policy_matches_native_le_5e_7": fast_policy_error <= 5e-7,
        "accelerated_policy_rows_normalized_le_2e_12": fast_policy_row_error <= 2e-12,
    }
    payload = {
        "schema": "SC-D D2 PDR interface audit v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_pdr_outcomes_generated": False,
        "inputs": {
            "graph": str(args.graph.resolve()),
            "graph_sha256": sha256(args.graph),
            "pilot_reference": str(args.pilot_reference.resolve()),
            "pilot_current": str(args.pilot_current.resolve()),
        },
        "graph": graph_manifest,
        "pilot": {
            "pi_ref_hash": array_hash(pi_ref),
            "pi_current_hash": array_hash(pi_current),
            "qbar_hash": array_hash(qbar),
            "source_hash": array_hash(first.source_by_state),
            "correction_hash": array_hash(first.correction_by_pair),
            "q_pdr_hash": array_hash(first.q_pdr_by_pair),
            "residual_inf": first.residual_inf,
            "max_abs_correction": float(np.max(np.abs(first.correction_by_pair))),
            "solve_seconds_cpu": solve_seconds,
            "used_exact_current_values": False,
        },
        "legacy_equivalence": legacy,
        "legacy_reuse_manifest": {
            "path": str(legacy_reuse_manifest_path),
            "sha256": sha256(legacy_reuse_manifest_path),
        },
        "actor_policy_evaluator": {
            "native_hash": array_hash(pi_current),
            "accelerated_hash": array_hash(fast_pi_current),
            "max_abs_error": fast_policy_error,
            "threshold": 5e-7,
            "post_normalization_row_max_abs_sum_error": fast_policy_row_error,
            "pass": fast_policy_error <= 5e-7 and fast_policy_row_error <= 2e-12,
        },
        "parent_replay_mapping": mapping,
        "checks": checks,
        "code_hashes": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ("scd_common.py", "build_closed_dag.py", "compile_pdr_reference.py", "reverse_pdr.py", "pdr_awac_adapter.py", "audit_d2.py")
        },
        "wall_seconds": time.perf_counter() - started,
        "status": "D2_PDR_INTERFACE_AUDIT_PASS" if all(checks.values()) else "D2_PDR_INTERFACE_AUDIT_FAIL",
        "pass": all(checks.values()),
    }
    write_pretty_json(args.output_json, payload)
    lines = [
        "# SC-D D2 graph, formulation, and solver audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Graph: `{graph_manifest['state_count']}` states, `{graph_manifest['pair_count']}` legal state-action pairs",
        f"- Replay-row coverage: `{mapping['mapped_rows']}/{mapping['total_rows']}`",
        f"- Recurrence residual infinity norm: `{first.residual_inf:.3e}`",
        f"- Zero-policy-change correction max norm: `{np.max(np.abs(zero.correction_by_pair)):.3e}`",
        f"- Legacy correction max absolute error: `{legacy['max_abs_correction_error']:.3e}`",
        f"- Accelerated-vs-native full-domain policy max error: `{fast_policy_error:.3e}`",
        f"- CPU reverse solve time: `{solve_seconds:.6f}` seconds",
        "- Official PDR outcomes generated: `false`",
        "",
        "Every registered D2 closure, finiteness, terminal, zero-source, recurrence, deterministic-repeat, legacy-equivalence, and actor-row mapping check passed." if all(checks.values()) else "At least one fail-closed D2 check failed; see JSON.",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
