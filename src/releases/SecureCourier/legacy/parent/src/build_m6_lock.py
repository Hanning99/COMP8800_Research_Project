#!/usr/bin/env python3
"""Create the content-addressed M6 preregistration lock before fresh outcomes."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import canonical_hash, now_utc, sha256, write_json


def included_files():
    explicit = [
        "PLAN.md",
        "M0_LEGACY_AUDIT.md",
        "M1_ENVIRONMENT_LOCK.md",
        "M2_AWAC_INTERFACE_LOCK.md",
        "M3_SOURCE_EVENT_PROTOCOL.md",
        "M3_SOURCE_EVENT_AUDIT.md",
        "M4_DISSOCIATION_AND_REPAIR_AUDIT.md",
        "M5_CAUSAL_INTERVENTION_AUDIT.md",
        "M5_POWER_SENSITIVITY.md",
        "M6_PREREGISTRATION.md",
        "configs/secure_courier_v1.yaml",
        "configs/scabc_v1.json",
        "manifests/FRESH_SEED_LEDGER.json",
        "manifests/PROVENANCE_HASHES.json",
        "manifests/M5_POWER_SENSITIVITY.json",
    ]
    files = [ROOT / item for item in explicit]
    for directory in ("src", "analysis", "tests"):
        files.extend(sorted((ROOT / directory).rglob("*.py")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"M6 lock inputs missing: {missing}")
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_events = list((ROOT / "source_events").rglob("STATUS.json"))
    if source_events:
        raise RuntimeError("refusing M6 lock after any fresh source-event status exists")
    files = [
        {
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in included_files()
    ]
    lock = {
        "status": "M6_PREREGISTRATION_LOCKED_BEFORE_FRESH_OUTCOMES",
        "created_at_utc": now_utc(),
        "fresh_source_status_files_at_lock": 0,
        "scientific_tree": files,
        "scientific_tree_hash": canonical_hash(files),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "confirmatory_events": 50,
        "ordered_reserves": 20,
        "fresh_pdr_in_primary_verdict": False,
        "analysis_entrypoint": "analysis/analyze_locked.py",
        "execution_entrypoint": "src/launch_m7.py",
    }
    lock["lock_hash"] = canonical_hash(lock)
    write_json(args.out, lock)


if __name__ == "__main__":
    main()
