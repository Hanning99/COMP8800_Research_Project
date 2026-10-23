#!/usr/bin/env python3
"""Eight-GPU, integrity-only M7 coordinator with ordered reserve replacement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from scabc_common import now_utc, write_json


def command(root: Path, attempt_id: str, row: dict, stage: str):
    return [
        sys.executable,
        str(CODE_ROOT / "src/run_m7_event.py"),
        "--experiment-root",
        str(root),
        "--attempt-id",
        attempt_id,
        "--seed",
        str(row["seed"]),
        "--ledger-position",
        str(row["ledger_position"]),
        "--stage",
        stage,
        "--device",
        "cuda:0",
    ]


def run_pool(root: Path, jobs: list[tuple[str, dict]], stage: str, gpus: list[int]):
    pending = list(jobs)
    active = {}
    completed = []
    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        for gpu in free:
            if not pending:
                break
            attempt_id, row = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(command(root, attempt_id, row, stage), env=env)
            active[gpu] = (process, attempt_id, row, now_utc())
        time.sleep(2)
        for gpu, (process, attempt_id, row, started) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            completed.append(
                {
                    "attempt_id": attempt_id,
                    "seed": row["seed"],
                    "ledger_position": row["ledger_position"],
                    "stage": stage,
                    "gpu": gpu,
                    "started_at_utc": started,
                    "finished_at_utc": now_utc(),
                    "exit_code": code,
                }
            )
            del active[gpu]
            write_json(
                root / "manifests/M7_PROGRESS.json",
                {
                    "updated_at_utc": now_utc(),
                    "completed_jobs": completed,
                    "active": [
                        {"gpu": g, "attempt_id": item[1], "stage": stage}
                        for g, item in active.items()
                    ],
                    "pending_count": len(pending),
                    "performance_by_arm_exposed": False,
                },
            )
            if code != 0:
                raise RuntimeError(f"M7 {stage} job failed: {attempt_id} on GPU {gpu}")
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--phase", choices=("source", "downstream", "all"), default="all")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    gpus = [int(item) for item in args.gpus.split(",")]
    ledger = json.loads((CODE_ROOT / "manifests/FRESH_SEED_LEDGER.json").read_text())
    initial = [(f"confirmatory_{row['ledger_position']:03d}", row) for row in ledger["confirmatory"]]
    reserve = [(f"reserve_{row['ledger_position']:03d}", row) for row in ledger["ordered_reserve"]]

    if args.phase in {"source", "all"}:
        run_pool(root, initial, "source", gpus)
        accepted = []
        rejected = []
        for attempt_id, row in initial:
            status = json.loads((root / "source_events" / attempt_id / "STATUS.json").read_text())
            (accepted if status["status"] == "ACCEPTED" else rejected).append((attempt_id, row))
        reserve_index = 0
        while len(accepted) < 50:
            need = min(50 - len(accepted), len(gpus), len(reserve) - reserve_index)
            if need <= 0:
                raise RuntimeError("ordered reserve ledger exhausted before 50 accepted events")
            batch = reserve[reserve_index : reserve_index + need]
            reserve_index += need
            run_pool(root, batch, "source", gpus)
            for attempt_id, row in batch:
                status = json.loads((root / "source_events" / attempt_id / "STATUS.json").read_text())
                (accepted if status["status"] == "ACCEPTED" else rejected).append((attempt_id, row))
        selection = {
            "accepted_target": 50,
            "accepted": [
                {"attempt_id": attempt_id, **row} for attempt_id, row in accepted[:50]
            ],
            "rejected": [
                {
                    "attempt_id": attempt_id,
                    **row,
                    "status": json.loads(
                        (root / "source_events" / attempt_id / "STATUS.json").read_text()
                    )["status"],
                }
                for attempt_id, row in rejected
            ],
            "unused_reserves": [
                {"attempt_id": attempt_id, **row} for attempt_id, row in reserve[reserve_index:]
            ],
            "selection_used_arm_outcomes": False,
        }
        write_json(root / "manifests/M7_ACCEPTED_EVENT_SET.json", selection)

    if args.phase in {"downstream", "all"}:
        selection = json.loads((root / "manifests/M7_ACCEPTED_EVENT_SET.json").read_text())
        jobs = [
            (
                row["attempt_id"],
                {"seed": row["seed"], "ledger_position": row["ledger_position"]},
            )
            for row in selection["accepted"]
            if not (root / "manifests/event_seals" / f"{row['attempt_id']}.json").is_file()
        ]
        run_pool(root, jobs, "downstream", gpus)


if __name__ == "__main__":
    main()
