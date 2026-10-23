#!/usr/bin/env python3
"""Content-level determinism audit for two resumed source-event pilots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import canonical_hash, sha256, tensor_tree_hash, write_json


def payload_hashes(path: Path) -> dict[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = (
        "actor",
        "critic",
        "target",
        "actor_optimizer",
        "critic_optimizer",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    )
    hashes = {key: tensor_tree_hash(payload[key]) for key in keys}
    semantic_metadata = {
        key: value for key, value in payload["metadata"].items() if key != "created_at_utc"
    }
    hashes["semantic_metadata_excluding_wallclock"] = tensor_tree_hash(semantic_metadata)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = (80, 96, 112, 128)
    checkpoint_rows = []
    all_equal = True
    for cycle in checkpoints:
        relative = Path(f"acquisition_trace/checkpoints/cycle_{cycle:06d}/state.pt")
        a_hashes = payload_hashes(args.a / relative)
        b_hashes = payload_hashes(args.b / relative)
        equal = a_hashes == b_hashes
        all_equal &= equal
        checkpoint_rows.append(
            {
                "cycle": cycle,
                "content_equal": equal,
                "content_hashes": a_hashes,
                "container_sha256_a": sha256(args.a / relative),
                "container_sha256_b": sha256(args.b / relative),
                "container_bytes_equal": sha256(args.a / relative) == sha256(args.b / relative),
            }
        )

    file_rows = []
    for relative in (
        Path("acquisition_trace/online_stream.jsonl"),
        Path("acquisition_trace/training_log.jsonl"),
        Path("acquisition_trace/capability_log.jsonl"),
        Path("eligibility.json"),
        Path("STATUS.json"),
    ):
        a_hash = sha256(args.a / relative)
        b_hash = sha256(args.b / relative)
        equal = a_hash == b_hash
        all_equal &= equal
        file_rows.append(
            {"path": str(relative), "sha256_a": a_hash, "sha256_b": b_hash, "equal": equal}
        )

    result = {
        "audit": "M3 synchronized resume determinism",
        "comparison_boundary": "two independent resumes from the same complete cycle-64 checkpoint",
        "authoritative_parity_rule": "semantic tensor/optimizer/RNG content plus append-only trace bytes",
        "expected_nondeterministic_fields": ["created_at_utc", "torch serialization container bytes"],
        "torch_container_sha_is_not_a_parity_gate": True,
        "checkpoint_content": checkpoint_rows,
        "trace_and_decision_files": file_rows,
        "selected_cycle_equal": json.loads((args.a / "STATUS.json").read_text())["selected_cycle"]
        == json.loads((args.b / "STATUS.json").read_text())["selected_cycle"],
        "all_authoritative_content_equal": all_equal,
    }
    result["audit_hash"] = canonical_hash(result)
    write_json(args.out, result)
    if not all_equal:
        raise SystemExit("resume content parity failed")


if __name__ == "__main__":
    main()
