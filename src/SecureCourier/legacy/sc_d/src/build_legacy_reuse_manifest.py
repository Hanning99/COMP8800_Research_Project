#!/usr/bin/env python3
"""Hash the exact legacy artifacts touched by the D2 equivalence check."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scd_common import LEGACY, sha256


def record(path: Path, role: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actor = sorted((LEGACY / "actor_checkpoints").glob("seed_*/PDR_AWAC/actor_u0016.pt"))[0]
    files = [
        (LEGACY / "PROVENANCE_LOCK.json", "legacy provenance lock"),
        (LEGACY / "pdr_advanced/pipeline.py", "independent legacy native-policy and sparse-solver implementation"),
        (LEGACY / "advanced_artifacts/compiled_fast_inputs.npz", "frozen legacy compiled recurrence inputs"),
        (LEGACY / "advanced_artifacts/COMPILE_AUDIT.json", "legacy compiler audit"),
        (actor, "disjoint legacy actor used for equivalence"),
        (LEGACY / "frozen_inputs/policies/new_online_final/vocab.pkl", "legacy actor vocabulary"),
        (LEGACY / "implementation/close_loop_source/securecourier_env/agents/actor.py", "legacy actor architecture"),
        (LEGACY / "implementation/close_loop_source/securecourier_env/agents/critic.py", "module imported by legacy loader"),
        (LEGACY / "implementation/close_loop_source/securecourier_env/secure_courier/env.py", "module imported by legacy loader"),
        (LEGACY / "implementation/close_loop_source/securecourier_env/secure_courier/generator.py", "module imported by legacy loader"),
        (LEGACY / "implementation/close_loop_source/securecourier_env/secure_courier/state.py", "module imported by legacy loader"),
    ]
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"LEGACY_REUSE_FILE_MISSING:{missing}")
    payload = {
        "schema": "SC-D legacy engineering reuse manifest v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "D2 equivalence audit only; no legacy outcome enters SC-D",
        "artifacts": [record(path, role) for path, role in files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
