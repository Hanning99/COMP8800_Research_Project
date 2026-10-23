#!/usr/bin/env python3
"""Materialize all SC-C branches and freeze arm order before any arm training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scabc_common import canonical_hash, sha256, write_json


ARMS = ("STALE", "CORRECT", "DROP", "PLACEBO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-event", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_event.resolve()
    panels = args.panels.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty SC-C event")
    out.mkdir(parents=True, exist_ok=True)
    source_status = json.loads((source / "STATUS.json").read_text())
    source_seed = int(source_status["seed"])
    order = sorted(
        ARMS,
        key=lambda arm: hashlib.sha256(f"SC-C:{source_seed}:{arm}".encode()).hexdigest(),
    )
    source_checkpoint = source / "capability_checkpoint"
    file_names = ("state.pt", "vocab.pkl", "manifest.json")
    expected = {name: sha256(source_checkpoint / name) for name in file_names}
    branches = []
    for arm in ARMS:
        branch = out / arm / "branch_checkpoint"
        shutil.copytree(source_checkpoint, branch)
        observed = {name: sha256(branch / name) for name in file_names}
        if observed != expected:
            raise RuntimeError(f"branch copy differs for {arm}")
        branches.append({"arm": arm, "checkpoint_hashes": observed})
    manifest = {
        "schema": "SC-C branch materialization v1",
        "source_event": str(source),
        "source_seed": source_seed,
        "source_selected_cycle": source_status["selected_cycle"],
        "all_branches_materialized_before_training": True,
        "randomization_rule": "ascending sha256('SC-C:<source_seed>:<arm>')",
        "execution_order": order,
        "source_checkpoint_hashes": expected,
        "H_panel_sha256": sha256(panels / "H_PANEL.json"),
        "placebo_panel_sha256": sha256(panels / "PLACEBO_PANEL.json"),
        "diagnostic_batch_sha256": sha256(panels / "DIAGNOSTIC_BATCH.json"),
        "branches": branches,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(out / "BRANCH_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
