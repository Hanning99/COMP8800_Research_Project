#!/usr/bin/env python3
"""Authoritative float64 policy-difference source and reverse-time PDR solve."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scd_common import GAMMA, MAX_STEPS, read_json, sha256


@dataclass(frozen=True)
class PDRResult:
    source_by_state: np.ndarray
    correction_by_pair: np.ndarray
    q_pdr_by_pair: np.ndarray
    residual_by_pair: np.ndarray

    @property
    def residual_inf(self) -> float:
        return float(np.max(np.abs(self.residual_by_pair))) if self.residual_by_pair.size else 0.0


def reverse_solve_arrays(
    *,
    pair_state: np.ndarray,
    pair_step: np.ndarray,
    successor: np.ndarray,
    continuation: np.ndarray,
    state_step: np.ndarray,
    pi_ref: np.ndarray,
    pi_current: np.ndarray,
    qbar: np.ndarray,
    gamma: float = GAMMA,
    max_steps: int = MAX_STEPS,
) -> PDRResult:
    """Pure array implementation used by the authoritative path and audits."""
    pair_state = np.asarray(pair_state, dtype=np.int32)
    pair_step = np.asarray(pair_step, dtype=np.int16)
    successor = np.asarray(successor, dtype=np.int32)
    continuation = np.asarray(continuation, dtype=np.bool_)
    state_step = np.asarray(state_step, dtype=np.int16)
    pi_ref = np.asarray(pi_ref, dtype=np.float64)
    pi_current = np.asarray(pi_current, dtype=np.float64)
    qbar = np.asarray(qbar, dtype=np.float64)
    pair_count = len(pair_state)
    state_count = len(state_step)
    for array in (pair_step, successor, continuation, pi_ref, pi_current, qbar):
        if len(array) != pair_count:
            raise RuntimeError("PDR_ARRAY_LENGTH_MISMATCH")
    if not all(np.all(np.isfinite(x)) for x in (pi_ref, pi_current, qbar)):
        raise RuntimeError("PDR_INPUT_NONFINITE")
    for policy in (pi_ref, pi_current):
        sums = np.bincount(pair_state, weights=policy, minlength=state_count)
        if np.any(policy < 0.0) or float(np.max(np.abs(sums - 1.0))) > 2e-12:
            raise RuntimeError("PDR_POLICY_INVALID")
    source = np.bincount(
        pair_state,
        weights=(pi_current - pi_ref) * qbar,
        minlength=state_count,
    ).astype(np.float64)
    correction = np.zeros(pair_count, dtype=np.float64)
    value_d = np.zeros(state_count, dtype=np.float64)
    for step in range(max_steps - 1, -1, -1):
        indices = np.flatnonzero(pair_step == step)
        live = indices[continuation[indices]]
        next_states = successor[live]
        correction[live] = gamma * (source[next_states] + value_d[next_states])
        states = np.flatnonzero(state_step == step)
        if len(states):
            value_d[states] = np.bincount(
                pair_state[indices],
                weights=pi_current[indices] * correction[indices],
                minlength=state_count,
            )[states]
    value_d_all = np.bincount(pair_state, weights=pi_current * correction, minlength=state_count)
    target = np.zeros(pair_count, dtype=np.float64)
    live = np.flatnonzero(continuation)
    target[live] = gamma * (source[successor[live]] + value_d_all[successor[live]])
    residual = correction - target
    q_pdr = qbar + correction
    return PDRResult(source, correction, q_pdr, residual)


class ClosedDAG:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path.resolve()
        self.graph_sha256 = sha256(self.graph_path)
        graph = read_json(self.graph_path)
        if graph.get("schema") != "SC_D_SIMULATOR_CLOSED_DAG_V2":
            raise RuntimeError("GRAPH_SCHEMA_MISMATCH")
        self.states = graph["states"]
        self.pairs = graph["pairs"]
        self.state_index = {row["key"]: i for i, row in enumerate(self.states)}
        self.pair_index = {row["key"]: i for i, row in enumerate(self.pairs)}
        if len(self.state_index) != len(self.states) or len(self.pair_index) != len(self.pairs):
            raise RuntimeError("DUPLICATE_GRAPH_KEYS")
        self.pair_state = np.asarray([self.state_index[row["state_key"]] for row in self.pairs], dtype=np.int32)
        self.pair_step = np.asarray([row["step"] for row in self.pairs], dtype=np.int16)
        self.continuation = np.asarray([row["continuation"] for row in self.pairs], dtype=np.bool_)
        self.successor = np.asarray(
            [self.state_index[row["successor_state_key"]] if row["continuation"] else -1 for row in self.pairs],
            dtype=np.int32,
        )
        self.state_pairs: list[np.ndarray] = []
        for sid in range(len(self.states)):
            indices = np.flatnonzero(self.pair_state == sid).astype(np.int32)
            if len(indices) != len(self.states[sid]["structural_actions"]):
                raise RuntimeError("STATE_ACTION_COUNT_MISMATCH")
            self.state_pairs.append(indices)
        self.step_pairs = [np.flatnonzero(self.pair_step == step).astype(np.int32) for step in range(MAX_STEPS)]
        self.step_states = [
            np.asarray([i for i, row in enumerate(self.states) if row["step"] == step], dtype=np.int32)
            for step in range(MAX_STEPS)
        ]
        self._validate_topology()
        self.state_step = np.asarray([row["step"] for row in self.states], dtype=np.int16)

    def _validate_topology(self) -> None:
        if np.any(self.successor[self.continuation] < 0):
            raise RuntimeError("CONTINUATION_WITHOUT_SUCCESSOR")
        state_steps = np.asarray([row["step"] for row in self.states], dtype=np.int16)
        if np.any(state_steps[self.successor[self.continuation]] != self.pair_step[self.continuation] + 1):
            raise RuntimeError("GRAPH_NOT_TIME_LAYERED")

    def validate_policy(self, policy: np.ndarray) -> None:
        if policy.shape != (len(self.pairs),) or not np.all(np.isfinite(policy)):
            raise RuntimeError("POLICY_SHAPE_OR_FINITENESS_FAILURE")
        if np.any(policy < 0.0):
            raise RuntimeError("NEGATIVE_POLICY_MASS")
        sums = np.bincount(self.pair_state, weights=policy, minlength=len(self.states))
        if float(np.max(np.abs(sums - 1.0))) > 2e-12:
            raise RuntimeError("POLICY_NORMALIZATION_FAILURE")

    def solve(self, pi_ref: np.ndarray, pi_current: np.ndarray, qbar: np.ndarray) -> PDRResult:
        for array in (pi_ref, pi_current, qbar):
            if array.shape != (len(self.pairs),) or not np.all(np.isfinite(array)):
                raise RuntimeError("PDR_INPUT_INVALID")
        self.validate_policy(pi_ref)
        self.validate_policy(pi_current)
        result = reverse_solve_arrays(
            pair_state=self.pair_state,
            pair_step=self.pair_step,
            successor=self.successor,
            continuation=self.continuation,
            state_step=self.state_step,
            pi_ref=pi_ref,
            pi_current=pi_current,
            qbar=qbar,
        )
        source = result.source_by_state
        correction = result.correction_by_pair
        residual = result.residual_by_pair
        q_pdr = result.q_pdr_by_pair
        if not all(np.all(np.isfinite(x)) for x in (source, correction, residual, q_pdr)):
            raise RuntimeError("PDR_OUTPUT_NONFINITE")
        if float(np.max(np.abs(correction[~self.continuation]))) != 0.0:
            raise RuntimeError("TERMINAL_CORRECTION_NONZERO")
        if float(np.max(np.abs(residual))) > 1e-10:
            raise RuntimeError("PDR_RECURRENCE_RESIDUAL_FAILURE")
        return PDRResult(source, correction, q_pdr, residual)

    def residual(self, pi_current: np.ndarray, source: np.ndarray, correction: np.ndarray) -> np.ndarray:
        value_d = np.bincount(
            self.pair_state,
            weights=pi_current * correction,
            minlength=len(self.states),
        )
        target = np.zeros(len(self.pairs), dtype=np.float64)
        live = np.flatnonzero(self.continuation)
        target[live] = GAMMA * (source[self.successor[live]] + value_d[self.successor[live]])
        return correction - target


def array_hash(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode())
    digest.update(json.dumps(list(normalized.shape)).encode())
    digest.update(normalized.tobytes())
    return digest.hexdigest()
