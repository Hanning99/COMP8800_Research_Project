#!/usr/bin/env python3
"""Fail-closed per-seed M7 source and downstream execution driver."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from scabc_common import canonical_hash, now_utc, sha256, write_json


def run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("COMMAND_JSON=" + json.dumps(command) + "\n" + completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"subprocess failed ({completed.returncode}); see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ledger-position", type=int, required=True)
    parser.add_argument("--stage", choices=("source", "downstream", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    event_id = args.attempt_id
    source = root / "source_events" / event_id
    logs = root / "logs" / event_id
    python = sys.executable

    if args.stage in {"source", "all"}:
        terminal = False
        if (source / "STATUS.json").is_file():
            terminal = json.loads((source / "STATUS.json").read_text())["status"] in {
                "ACCEPTED",
                "PREARM_REJECT_OLD_BEHAVIOR",
                "PREARM_REJECT_NO_STABLE_CAPABILITY",
                "PREARM_REJECT_STRUCTURAL_ELIGIBILITY",
            }
        if not terminal:
            source_command = [
                python,
                str(CODE_ROOT / "src/run_source_event.py"),
                "--seed",
                str(args.seed),
                "--out",
                str(source),
                "--device",
                args.device,
                "--mode",
                "confirmatory",
            ]
            if source.exists():
                source_command.append("--resume")
            run_logged(source_command, logs / "01_source.log")
        status = json.loads((source / "STATUS.json").read_text())
        status["ledger_position"] = args.ledger_position
        status["attempt_id"] = event_id
        write_json(source / "STATUS.json", status)
        if status["status"] != "ACCEPTED":
            return

    if args.stage in {"downstream", "all"}:
        status = json.loads((source / "STATUS.json").read_text())
        if status["status"] != "ACCEPTED" or int(status["seed"]) != args.seed:
            raise RuntimeError("downstream requested for a nonaccepted or mismatched source event")
        sca_root = root / "sc_a_dissociation" / event_id
        sca_root.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                python,
                str(CODE_ROOT / "src/extract_sc_a_trace.py"),
                "--source-event",
                str(source),
                "--out",
                str(sca_root / "sc_a_trace.json"),
                "--device",
                args.device,
            ],
            logs / "02_sc_a.log",
        )

        support = root / "manifests/sc_b_support" / event_id
        run_logged(
            [python, str(CODE_ROOT / "src/build_sc_b_replay.py"), "--source-event", str(source), "--out", str(support)],
            logs / "03_sc_b_support.log",
        )
        scb = root / "sc_b_repair" / event_id
        run_logged(
            [
                python,
                str(CODE_ROOT / "src/prepare_sc_b_branches.py"),
                "--source-event",
                str(source),
                "--support",
                str(support),
                "--out",
                str(scb),
            ],
            logs / "04_sc_b_branches.log",
        )
        scb_manifest = scb / "BRANCH_MANIFEST.json"
        for order_index, arm in enumerate(json.loads(scb_manifest.read_text())["execution_order"]):
            run_logged(
                [
                    python,
                    str(CODE_ROOT / "src/run_sc_b_repair.py"),
                    "--source-event",
                    str(source),
                    "--support",
                    str(support),
                    "--arm",
                    arm,
                    "--out",
                    str(scb / arm),
                    "--device",
                    args.device,
                    "--mode",
                    "confirmatory",
                    "--branch-manifest",
                    str(scb_manifest),
                ],
                logs / f"05_sc_b_{order_index}_{arm}.log",
            )

        panels = root / "manifests/sc_c_panels" / event_id
        run_logged(
            [
                python,
                str(CODE_ROOT / "src/build_sc_c_panels.py"),
                "--source-event",
                str(source),
                "--out",
                str(panels),
                "--device",
                args.device,
            ],
            logs / "06_sc_c_panels.log",
        )
        scc = root / "sc_c_oracle" / event_id
        run_logged(
            [
                python,
                str(CODE_ROOT / "src/prepare_sc_c_branches.py"),
                "--source-event",
                str(source),
                "--panels",
                str(panels),
                "--out",
                str(scc),
            ],
            logs / "07_sc_c_branches.log",
        )
        scc_manifest = scc / "BRANCH_MANIFEST.json"
        for order_index, arm in enumerate(json.loads(scc_manifest.read_text())["execution_order"]):
            run_logged(
                [
                    python,
                    str(CODE_ROOT / "src/run_sc_c_oracle.py"),
                    "--event-root",
                    str(scc),
                    "--panels",
                    str(panels),
                    "--arm",
                    arm,
                    "--device",
                    args.device,
                    "--mode",
                    "confirmatory",
                ],
                logs / f"08_sc_c_{order_index}_{arm}.log",
            )
        critical = [
            source / "source_manifest.json",
            sca_root / "sc_a_trace.json",
            scb / "BRANCH_MANIFEST.json",
            scc / "BRANCH_MANIFEST.json",
        ] + [scb / arm / "RUN_MANIFEST.json" for arm in ("FULL_RELEVANT_REPLAY", "NO_LOCAL_ANCHOR", "NO_DOWNSTREAM_SUPPORT")] + [
            scc / arm / "RUN_MANIFEST.json" for arm in ("STALE", "CORRECT", "DROP", "PLACEBO")
        ]
        seal = {
            "status": "SEALED_COMPLETE",
            "attempt_id": event_id,
            "seed": args.seed,
            "ledger_position": args.ledger_position,
            "sealed_at_utc": now_utc(),
            "critical_files": [
                {"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in critical
            ],
        }
        seal["seal_hash"] = canonical_hash(seal)
        write_json(root / "manifests/event_seals" / f"{event_id}.json", seal)


if __name__ == "__main__":
    main()
