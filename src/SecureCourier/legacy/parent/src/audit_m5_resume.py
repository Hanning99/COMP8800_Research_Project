#!/usr/bin/env python3
"""Content-level SC-C uninterrupted versus resumed trajectory audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import canonical_hash, sha256, tensor_tree_hash, write_json


def semantic_payload(path: Path) -> dict[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {
        key: tensor_tree_hash(payload[key])
        for key in (
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
    }
    metadata = {key: value for key, value in payload["metadata"].items() if key != "created_at_utc"}
    result["semantic_metadata_excluding_wallclock"] = tensor_tree_hash(metadata)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a = args.continuous.resolve()
    b = args.resumed.resolve()
    relative_checkpoint = Path("checkpoints/cycle_000032/state.pt")
    payload_a = semantic_payload(a / relative_checkpoint)
    payload_b = semantic_payload(b / relative_checkpoint)
    files = []
    all_equal = payload_a == payload_b
    for relative in (
        Path("RUN_CONFIG.json"),
        Path("continuation_stream.jsonl"),
        Path("training_log.jsonl"),
        Path("evaluation_log.jsonl"),
        Path("evaluation_trace.json"),
        Path("treatment_oracle_calls.jsonl"),
    ):
        equal = sha256(a / relative) == sha256(b / relative)
        all_equal &= equal
        files.append(
            {
                "path": str(relative),
                "continuous_sha256": sha256(a / relative),
                "resumed_sha256": sha256(b / relative),
                "equal": equal,
            }
        )
    result = {
        "audit": "M5 SC-C uninterrupted 32 cycles versus pause-at-16 plus resume-to-32",
        "arm": "PLACEBO",
        "semantic_checkpoint_content_equal": payload_a == payload_b,
        "checkpoint_content_hashes": payload_a,
        "torch_container_bytes_equal": sha256(a / relative_checkpoint) == sha256(b / relative_checkpoint),
        "files": files,
        "all_authoritative_content_equal": all_equal,
        "expected_nondeterministic_fields": ["created_at_utc", "torch serialization container bytes"],
    }
    result["audit_hash"] = canonical_hash(result)
    write_json(args.out, result)
    if not all_equal:
        raise SystemExit("SC-C resume parity failed")


if __name__ == "__main__":
    main()
