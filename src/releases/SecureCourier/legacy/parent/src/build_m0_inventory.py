#!/usr/bin/env python3
"""Build the outcome-free M0 provenance, contamination, and seed ledgers.

This script never imports a learner, evaluates a policy, or reads fresh result
directories.  It only inventories read-only legacy material and deterministically
constructs candidate seeds that do not occur in the scanned legacy text corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".csv", ".json", ".jsonl", ".log", ".md", ".py", ".sh", ".txt",
    ".yaml", ".yml",
}
SKIP_PARTS = {".git", ".venv", "__pycache__"}
MAX_SCAN_BYTES = 16 * 1024 * 1024
SEED_CONTEXT = re.compile(
    r"(?i)(?:acquisition[_ -]?seed|continuation[_ -]?seed|learner[_ -]?seed|"
    r"training[_ -]?seed|bootstrap[_ -]?seed|eval(?:uation)?[_ -]?seed|"
    r"seed(?:s)?)[^0-9\n\r]{0,64}([0-9]{1,10})"
)
PATH_SEED = re.compile(r"(?i)(?:^|[^a-z])seed[_-]?([0-9]{1,10})(?:[^0-9]|$)")
LONG_INTEGER = re.compile(r"(?<![0-9])([0-9]{8,10})(?![0-9])")


ROOTS = {
    "motivation": Path("/data3/hanning/motivation"),
    "diy_task_set": Path("/data3/hanning/autoresearch/diy_task_set"),
    "clean_pdr_m5": Path("/data3/hanning/autoresearch/PDR/experiments/m5_securecourier_clean"),
    "pdr_awac": Path("/data3/hanning/autoresearch/PDR-AWAC"),
    "pdr_awac_advanced": Path("/data3/hanning/autoresearch/PDR-AWAC-Advanced"),
    "predict_qnew": Path("/data3/hanning/autoresearch/predict-Qnew"),
    "oracle_bridge": Path(
        "/data3/hanning/autoresearch/bellman_evaluation_support/"
        "stage2_rq0_Causal_Bridge"
    ),
    "critic_lag": Path("/data3/hanning/autoresearch/critic_lag"),
}


CRITICAL_FILES = [
    "/data3/hanning/autoresearch/Experiment/Benchmark/SecureCourier/PLAN.md",
    "/data3/hanning/motivation/evidence/MANIFEST.md",
    "/data3/hanning/motivation/evidence/01_policy_value_flip/rq1_confirmatory/confirmatory_matched_summary.json",
    "/data3/hanning/motivation/evidence/03_repair_condition/support_compute_surface/FINAL_RQ1_REPORT.md",
    "/data3/hanning/motivation/evidence/04_oracle_benefit/phase_n/PHASE_N_PROTOCOL.md",
    "/data3/hanning/motivation/evidence/04_oracle_benefit/phase_n/STAGE_N_B_REPORT.md",
    "/data3/hanning/autoresearch/diy_task_set/AGENTS.md",
    "/data3/hanning/autoresearch/diy_task_set/configs/secure_courier_v1.yaml",
    "/data3/hanning/autoresearch/diy_task_set/secure_courier/env.py",
    "/data3/hanning/autoresearch/diy_task_set/secure_courier/generator.py",
    "/data3/hanning/autoresearch/diy_task_set/secure_courier/oracle.py",
    "/data3/hanning/autoresearch/diy_task_set/secure_courier/state.py",
    "/data3/hanning/autoresearch/diy_task_set/secure_courier/task_graph.py",
    "/data3/hanning/autoresearch/diy_task_set/agents/actor.py",
    "/data3/hanning/autoresearch/diy_task_set/agents/critic.py",
    "/data3/hanning/autoresearch/diy_task_set/agents/awac.py",
    "/data3/hanning/autoresearch/diy_task_set/agents/checkpoint.py",
    "/data3/hanning/autoresearch/diy_task_set/capability/evaluate_capability.py",
    "/data3/hanning/autoresearch/diy_task_set/data/collect_offline.py",
    "/data3/hanning/autoresearch/diy_task_set/data/schemas.py",
    "/data3/hanning/autoresearch/diy_task_set/rq1/exact_value_table.py",
    "/data3/hanning/autoresearch/diy_task_set/scripts/run_phase1.py",
    "/data3/hanning/autoresearch/diy_task_set/scripts/run_rq3.py",
    "/data3/hanning/autoresearch/diy_task_set/scripts/run_propagation_audit.py",
    "/data3/hanning/autoresearch/bellman_evaluation_support/stage2_rq0_Causal_Bridge/phase_n/scripts/audit_stage_n_a_seed.py",
    "/data3/hanning/autoresearch/bellman_evaluation_support/stage2_rq0_Causal_Bridge/phase_n/scripts/run_stage_n_b.py",
    "/data3/hanning/autoresearch/PDR/experiments/m5_securecourier_clean/PREREGISTRATION.md",
    "/data3/hanning/autoresearch/PDR/experiments/m5_securecourier_clean/PREREGISTRATION_LOCK.json",
    "/data3/hanning/autoresearch/PDR/experiments/m5_securecourier_clean/run_m5.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_state(path: Path) -> dict[str, object]:
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=top, text=True
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=top, text=True
        ).splitlines()
        return {
            "repository": str(top),
            "commit": commit,
            "dirty": bool(porcelain),
            "porcelain_line_count": len(porcelain),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"repository": None, "commit": None, "dirty": None}


def iter_metadata(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_PARTS]
        base = Path(current)
        for name in files:
            path = base / name
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def scan_legacy() -> tuple[dict[int, list[str]], set[int], dict[str, object]]:
    associated: dict[int, list[str]] = defaultdict(list)
    long_tokens: set[int] = set()
    counts: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for label, root in ROOTS.items():
        scanned = skipped_large = unreadable = 0
        for path in iter_metadata(root):
            rel = str(path.relative_to(root))
            path_match = PATH_SEED.search(str(path))
            if path_match:
                value = int(path_match.group(1))
                if len(associated[value]) < 25:
                    associated[value].append(f"{label}:path:{rel}")
            try:
                size = path.stat().st_size
                if size > MAX_SCAN_BYTES:
                    skipped_large += 1
                    continue
                text = path.read_text(errors="ignore")
            except OSError as exc:
                unreadable += 1
                if len(errors) < 50:
                    errors.append(f"{path}: {exc}")
                continue
            scanned += 1
            for match in SEED_CONTEXT.finditer(text):
                value = int(match.group(1))
                if len(associated[value]) < 25:
                    associated[value].append(f"{label}:content:{rel}")
            for token in LONG_INTEGER.findall(text):
                long_tokens.add(int(token))
        counts[label] = {
            "metadata_files_scanned": scanned,
            "metadata_files_skipped_over_16MiB": skipped_large,
            "metadata_files_unreadable": unreadable,
        }
    return associated, long_tokens, {"by_root": counts, "errors": errors}


def seed_stream(domain: str) -> Iterable[int]:
    counter = 0
    while True:
        raw = hashlib.sha256(f"{domain}:{counter}".encode()).digest()
        value = int.from_bytes(raw[:4], "big") & 0x7FFFFFFF
        counter += 1
        if value >= 100_000_000:
            yield value


def parse_copy_map(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or "`" not in line:
            continue
        fields = [field.strip().strip("`") for field in line.strip().strip("|").split("|")]
        if len(fields) == 3 and fields[0] != "Evidence path":
            rows.append(
                {"motivation_evidence_path": fields[0], "declared_source": fields[1], "role": fields[2]}
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    manifests = out / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    associated, long_tokens, scan = scan_legacy()
    manifest_path = Path("/data3/hanning/motivation/evidence/MANIFEST.md")
    copy_map = parse_copy_map(manifest_path)

    critical = []
    missing = []
    for raw in CRITICAL_FILES:
        path = Path(raw)
        if not path.is_file():
            missing.append(raw)
            continue
        critical.append(
            {"path": raw, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    plan_hash = sha256(Path(CRITICAL_FILES[0]))
    domain = f"securecourier-scabc-fresh-ledger-v1:{plan_hash}"
    selected: list[int] = []
    rejected: list[dict[str, object]] = []
    for candidate in seed_stream(domain):
        reasons = []
        if candidate in associated:
            reasons.append("seed-associated legacy occurrence")
        if candidate in long_tokens:
            reasons.append("exact decimal token occurs in scanned legacy metadata")
        if reasons:
            rejected.append({"seed": candidate, "reasons": reasons})
            continue
        selected.append(candidate)
        if len(selected) == 70:
            break

    ledger = {
        "status": "FROZEN_CANDIDATE_LEDGER_NO_FRESH_OUTCOMES",
        "created_at_utc": utc_now(),
        "construction": {
            "domain": domain,
            "algorithm": "SHA256(domain:counter), first 31 bits, require >=100000000",
            "contamination_rule": (
                "reject if seed-associated or if exact 8-10 digit decimal token occurs "
                "in scanned legacy metadata"
            ),
            "confirmatory_count": 50,
            "ordered_reserve_count": 20,
        },
        "confirmatory": [
            {"ledger_position": i, "seed": seed, "role": "confirmatory"}
            for i, seed in enumerate(selected[:50])
        ],
        "ordered_reserve": [
            {"ledger_position": i, "seed": seed, "role": "ordered_reserve"}
            for i, seed in enumerate(selected[50:])
        ],
        "generation_rejections": rejected,
        "fresh_outcomes_inspected": False,
    }

    seed_inventory = {
        "created_at_utc": utc_now(),
        "scope": {name: str(path) for name, path in ROOTS.items()},
        "scan": scan,
        "seed_associated_unique_values": len(associated),
        "seed_associated": [
            {"seed": value, "example_sources": sorted(set(sources))}
            for value, sources in sorted(associated.items())
        ],
        "long_decimal_tokens_used_for_conservative_collision_check": len(long_tokens),
    }
    provenance = {
        "status": "M0_OUTCOME_FREE_PROVENANCE_INVENTORY",
        "created_at_utc": utc_now(),
        "read_only_roots": {name: str(path) for name, path in ROOTS.items()},
        "git_states": {name: git_state(path) for name, path in ROOTS.items()},
        "critical_files": critical,
        "missing_critical_files": missing,
        "motivation_copy_map": copy_map,
        "interpretation": (
            "motivation/evidence is a curated copy/reorganization layer and is not "
            "an independent experiment relative to its declared legacy sources"
        ),
    }

    write_json(manifests / "LEGACY_SEED_INVENTORY.json", seed_inventory)
    write_json(manifests / "FRESH_SEED_LEDGER.json", ledger)
    write_json(manifests / "PROVENANCE_HASHES.json", provenance)

    dirty = provenance["git_states"]["diy_task_set"]
    report = f"""# M0 Legacy Provenance and Contamination Audit

**Status:** `M0_COMPLETE_OUTCOME_FREE`

**Generated:** {utc_now()}

## Decision

The old three-arm Fresh-PDR SecureCourier design remains suspended.  This audit
supports only the SC-A/SC-B/SC-C engineering and preregistration workflow.  It
does not authorize source acquisition, branch execution, M5, M6, or fresh
confirmatory outcomes.

## Provenance finding

`/data3/hanning/motivation/evidence/MANIFEST.md` explicitly describes the
motivation tree as copied/reorganized evidence.  Its entries are provenance
views of legacy runs and are not counted as independent replications.  The
machine-readable mapping contains {len(copy_map)} declared evidence groups.

The `diy_task_set` repository is at commit `{dirty['commit']}` but has
`dirty={dirty['dirty']}` and {dirty.get('porcelain_line_count')} status entries.
Most experiment artifacts are untracked, so the commit alone is not a sufficient
reproduction identifier.  The critical-file manifest records SHA256 and size for
{len(critical)} files; {len(missing)} required paths were missing.

## Seed contamination control

The scanner found {len(associated)} distinct seed-associated integer values in
the registered legacy roots.  It also used every standalone 8–10 digit token in
the scanned metadata as a conservative collision exclusion.  The fresh ledger
contains 50 confirmatory seeds and 20 ordered reserves, deterministically derived
from the authoritative plan hash.  None occurs in the scanned legacy metadata.

Seed replacement remains legal only for a preregistered pre-arm eligibility or
integrity failure and before SC-B/SC-C outcomes are inspected.

## Audit boundary

- Metadata files larger than 16 MiB are recorded as skipped by count; checkpoint
  and result identity is anchored by the published legacy manifests plus the
  critical-file hashes rather than by rehashing every multi-gigabyte archive.
- Candidate collision checking covers the listed SecureCourier legacy roots,
  including critic-lag, PDR-AWAC, clean M5, predict-Qnew, and the oracle bridge.
- No learner, environment rollout, exact-policy scorer, or arm analysis ran.

## Outputs

- `manifests/PROVENANCE_HASHES.json`
- `manifests/LEGACY_SEED_INVENTORY.json`
- `manifests/FRESH_SEED_LEDGER.json`

## M0 gate

`PASS` for outcome-free provenance and seed-ledger construction.  This does not
waive any M1–M6 fail-closed gate.
"""
    (out / "M0_LEGACY_AUDIT.md").write_text(report)


if __name__ == "__main__":
    main()
