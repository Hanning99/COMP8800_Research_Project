"""Shared deterministic utilities for the preregistered SecureCourier SC-D bridge."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


PARENT = Path("autoresearch/SecureCourier-Mechanism-20260902")
LEGACY = Path("autoresearch/PDR-AWAC-Advanced")
GAMMA = 0.97
MAX_STEPS = 40
OBS_MAX = 128
ACT_MAX = 32
TEMPERATURE = 0.3
WEIGHT_CLIP = 20.0
GRADIENT_CLIP = 5.0
TARGET_TAU = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_pretty_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def clean(value: Any) -> Any:
    if torch.is_tensor(value):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def tensor_tree_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return tensor_tree_hash(module.state_dict())
