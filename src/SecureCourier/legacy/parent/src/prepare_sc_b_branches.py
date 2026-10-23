#!/usr/bin/env python3
"""Materialize and hash all actor-frozen SC-B branches before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_sc_b_replay import ARMS
from scabc_common import canonical_hash, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_event.resolve()
    support = args.support.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty SC-B branch directory")
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((source / "STATUS.json").read_text())
    seed = int(status["seed"])
    order = sorted(
        ARMS,
        key=lambda arm: hashlib.sha256(f"SC-B:{seed}:{arm}".encode()).hexdigest(),
    )
    checkpoint = source / "capability_checkpoint"
    names = ("state.pt", "vocab.pkl", "manifest.json")
    expected = {name: sha256(checkpoint / name) for name in names}
    branches = []
    for arm in ARMS:
        target = out / "branches" / arm / "branch_checkpoint"
        shutil.copytree(checkpoint, target)
        observed = {name: sha256(target / name) for name in names}
        if observed != expected:
            raise RuntimeError(f"SC-B branch copy mismatch: {arm}")
        branches.append({"arm": arm, "checkpoint_hashes": observed})
    manifest = {
        "schema": "SC-B branch materialization v1",
        "source_event": str(source),
        "source_seed": seed,
        "all_branches_materialized_before_training": True,
        "randomization_rule": "ascending sha256('SC-B:<source_seed>:<arm>')",
        "execution_order": order,
        "source_checkpoint_hashes": expected,
        "support_manifest_sha256": sha256(support / "SUPPORT_MANIFEST.json"),
        "branches": branches,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(out / "BRANCH_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
