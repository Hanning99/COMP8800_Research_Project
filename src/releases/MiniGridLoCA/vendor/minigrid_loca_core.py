#!/usr/bin/env python3
"""Exact MiniGridLoCA graph, RGB observations, DP, and generalized PDR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GRID_SIZE = 8
N_DIRECTIONS = 4
N_ACTIONS = 3
LEFT, RIGHT, FORWARD = range(3)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
T1 = (0, 0)
T2 = (7, 7)
T1_ZONE = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
REWARDS = {"A": {1: 4.0, 2: 2.0}, "B": {1: 1.0, 2: 2.0}}


@dataclass(frozen=True)
class Graph:
    states: tuple[tuple[int, int, int], ...]
    state_index: dict[tuple[int, int, int], int]
    successor: np.ndarray
    done: np.ndarray
    terminal_id: np.ndarray
    reward_a: np.ndarray
    reward_b: np.ndarray
    nonterminal_states: np.ndarray
    images_uint8: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return N_ACTIONS


class MiniGridLoCAEnv:
    """Small exact interface mirroring the published MiniGridLoCA dynamics."""

    def __init__(self, phase: Literal["A", "B"] = "A", max_steps: int = 100):
        self.phase = phase
        self.max_steps = int(max_steps)
        self.position = (0, 0)
        self.direction = 0
        self.step_count = 0

    def restore(self, state: tuple[int, int, int, int]) -> None:
        x, y, direction, step_count = state
        assert 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE
        assert 0 <= direction < N_DIRECTIONS and 0 <= step_count < self.max_steps
        self.position = (int(x), int(y))
        self.direction = int(direction)
        self.step_count = int(step_count)

    def exact_state(self) -> tuple[int, int, int, int]:
        return (*self.position, self.direction, self.step_count)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert action in (LEFT, RIGHT, FORWARD)
        x, y = self.position
        direction = self.direction
        if action == LEFT:
            direction = (direction - 1) % N_DIRECTIONS
        elif action == RIGHT:
            direction = (direction + 1) % N_DIRECTIONS
        else:
            dx, dy = DIRECTIONS[direction]
            candidate = (min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1))
            if not (self.position in T1_ZONE and candidate not in T1_ZONE):
                x, y = candidate
        self.position = (x, y)
        self.direction = direction
        self.step_count += 1
        terminal = 1 if self.position == T1 else 2 if self.position == T2 else 0
        reward = REWARDS[self.phase].get(terminal, 0.0)
        terminated = terminal != 0
        truncated = self.step_count >= self.max_steps and not terminated
        return render_state(x, y, direction), float(reward), terminated, truncated, {"terminal_id": terminal}


def _agent_mask(direction: int) -> tuple[tuple[int, int], ...]:
    # A compact asymmetric triangle inside an 8x8 cell. Direction 0 points right.
    right = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (5, 4), (6, 3))
    points = np.asarray(right, dtype=np.int64)
    for _ in range(direction):
        points = np.stack([points[:, 1], 7 - points[:, 0]], axis=1)
    return tuple((int(a), int(b)) for a, b in points)


def render_state(x: int, y: int, direction: int) -> np.ndarray:
    cell = 8
    image = np.full((GRID_SIZE * cell, GRID_SIZE * cell, 3), 190, dtype=np.uint8)
    for boundary in range(0, GRID_SIZE * cell, cell):
        image[boundary, :, :] = 45
        image[:, boundary, :] = 45
    image[-1, :, :] = 45
    image[:, -1, :] = 45
    for tx, ty in (T1, T2):
        y0, x0 = ty * cell + 1, tx * cell + 1
        image[y0 : y0 + cell - 1, x0 : x0 + cell - 1] = (25, 190, 45)
    y0, x0 = y * cell, x * cell
    for px, py in _agent_mask(direction):
        image[y0 + py, x0 + px] = (220, 30, 30)
    return image


def build_graph() -> Graph:
    states = tuple((x, y, d) for y in range(GRID_SIZE) for x in range(GRID_SIZE) for d in range(N_DIRECTIONS))
    index = {state: sid for sid, state in enumerate(states)}
    n = len(states)
    successor = np.zeros((n, N_ACTIONS), dtype=np.int64)
    done = np.zeros((n, N_ACTIONS), dtype=bool)
    terminal_id = np.zeros((n, N_ACTIONS), dtype=np.int8)
    for sid, (x, y, direction) in enumerate(states):
        if (x, y) in (T1, T2):
            successor[sid] = sid
            done[sid] = True
            terminal_id[sid] = 1 if (x, y) == T1 else 2
            continue
        for action in range(N_ACTIONS):
            nx, ny, nd = x, y, direction
            if action == LEFT:
                nd = (direction - 1) % N_DIRECTIONS
            elif action == RIGHT:
                nd = (direction + 1) % N_DIRECTIONS
            else:
                dx, dy = DIRECTIONS[direction]
                candidate = (min(max(x + dx, 0), GRID_SIZE - 1), min(max(y + dy, 0), GRID_SIZE - 1))
                if not ((x, y) in T1_ZONE and candidate not in T1_ZONE):
                    nx, ny = candidate
            successor[sid, action] = index[(nx, ny, nd)]
            if (nx, ny) == T1:
                done[sid, action] = True
                terminal_id[sid, action] = 1
            elif (nx, ny) == T2:
                done[sid, action] = True
                terminal_id[sid, action] = 2
    reward_a = np.where(terminal_id == 1, 4.0, np.where(terminal_id == 2, 2.0, 0.0))
    reward_b = np.where(terminal_id == 1, 1.0, np.where(terminal_id == 2, 2.0, 0.0))
    terminal_positions = {T1, T2}
    nonterminal = np.asarray([sid for sid, (x, y, _) in enumerate(states) if (x, y) not in terminal_positions], dtype=np.int64)
    images = np.stack([render_state(*state) for state in states])
    return Graph(states, index, successor, done, terminal_id, reward_a, reward_b, nonterminal, images)


def transition_matrix(graph: Graph, policy: np.ndarray) -> np.ndarray:
    matrix = np.zeros((graph.n_states, graph.n_states), dtype=np.float64)
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                matrix[s, graph.successor[s, a]] += policy[s, a]
    return matrix


def exact_q(graph: Graph, policy: np.ndarray, reward: np.ndarray, gamma: float, alpha: float = 0.0) -> np.ndarray:
    transition = transition_matrix(graph, policy)
    log_policy = np.log(np.clip(policy, 1e-15, 1.0))
    entropy = -(policy * log_policy).sum(axis=1)
    expected_reward = (policy * reward).sum(axis=1)
    rhs = expected_reward + gamma * transition @ (alpha * entropy)
    expected_q = np.linalg.solve(np.eye(graph.n_states) - gamma * transition, rhs)
    q = reward.astype(np.float64).copy()
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                ns = graph.successor[s, a]
                q[s, a] += gamma * (expected_q[ns] + alpha * entropy[ns])
    q[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 0.0
    return q


def generalized_pdr(
    graph: Graph,
    pi_old: np.ndarray,
    pi_new: np.ndarray,
    q_old_task_a: np.ndarray,
    gamma: float,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    log_old = np.log(np.clip(pi_old, 1e-15, 1.0))
    log_new = np.log(np.clip(pi_new, 1e-15, 1.0))
    old_aggregate = (pi_old * (q_old_task_a - alpha * log_old)).sum(axis=1)
    new_aggregate = (pi_new * (q_old_task_a - alpha * log_new)).sum(axis=1)
    source = graph.reward_b.astype(np.float64) - graph.reward_a.astype(np.float64)
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                ns = graph.successor[s, a]
                source[s, a] += gamma * (new_aggregate[ns] - old_aggregate[ns])
    source[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 0.0
    transition = transition_matrix(graph, pi_new)
    state_rhs = (pi_new * source).sum(axis=1)
    delta_expected_q = np.linalg.solve(np.eye(graph.n_states) - gamma * transition, state_rhs)
    correction = source.copy()
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                correction[s, a] += gamma * delta_expected_q[graph.successor[s, a]]
    propagated = np.zeros_like(correction)
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                ns = graph.successor[s, a]
                propagated[s, a] = gamma * (pi_new[ns] * correction[ns]).sum()
    residual = correction - source - propagated
    return correction, source, float(np.max(np.abs(residual)))


def bellman_residual(graph: Graph, policy: np.ndarray, q: np.ndarray, reward: np.ndarray, gamma: float, alpha: float = 0.0) -> float:
    log_policy = np.log(np.clip(policy, 1e-15, 1.0))
    aggregate = (policy * (q - alpha * log_policy)).sum(axis=1)
    target = reward.astype(np.float64).copy()
    for s in graph.nonterminal_states:
        for a in range(graph.n_actions):
            if not graph.done[s, a]:
                target[s, a] += gamma * aggregate[graph.successor[s, a]]
    target[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 0.0
    return float(np.max(np.abs(q - target)))


def optimal_q(graph: Graph, reward: np.ndarray, gamma: float, tolerance: float = 1e-14) -> np.ndarray:
    value = np.zeros(graph.n_states, dtype=np.float64)
    for _ in range(100_000):
        q = reward.astype(np.float64).copy()
        for s in graph.nonterminal_states:
            for a in range(graph.n_actions):
                if not graph.done[s, a]:
                    q[s, a] += gamma * value[graph.successor[s, a]]
        q[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 0.0
        new_value = q.max(axis=1)
        if np.max(np.abs(new_value - value)) <= tolerance:
            return q
        value = new_value
    raise RuntimeError("value iteration did not converge")


def task_a_old_policy(graph: Graph, gamma: float, best_probability: float = 0.92) -> np.ndarray:
    q_star = optimal_q(graph, graph.reward_a, gamma)
    policy = np.full((graph.n_states, graph.n_actions), (1.0 - best_probability) / (graph.n_actions - 1), dtype=np.float64)
    best = np.argmax(q_star, axis=1)
    policy[np.arange(graph.n_states), best] = best_probability
    policy[np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)] = 1.0 / graph.n_actions
    return policy


def finite_horizon_metrics(graph: Graph, policy: np.ndarray, gamma: float, horizon: int = 100):
    value = np.zeros(graph.n_states, dtype=np.float64)
    p1 = np.zeros(graph.n_states, dtype=np.float64)
    p2 = np.zeros(graph.n_states, dtype=np.float64)
    continuation_mask = (~graph.done).astype(np.float64)
    for _ in range(horizon):
        next_value = value[graph.successor]
        next_p1 = p1[graph.successor]
        next_p2 = p2[graph.successor]
        value = (policy * (graph.reward_b + gamma * continuation_mask * next_value)).sum(axis=1)
        p1 = (policy * ((graph.terminal_id == 1) + continuation_mask * next_p1)).sum(axis=1)
        p2 = (policy * ((graph.terminal_id == 2) + continuation_mask * next_p2)).sum(axis=1)
        terminal_states = np.setdiff1d(np.arange(graph.n_states), graph.nonterminal_states)
        value[terminal_states] = 0.0
        p1[terminal_states] = 0.0
        p2[terminal_states] = 0.0
    return value, p1, p2


def evaluate_policy(graph: Graph, policy: np.ndarray, gamma: float) -> dict[str, float]:
    value, p1, p2 = finite_horizon_metrics(graph, policy, gamma, horizon=100)
    states = graph.nonterminal_states
    return {
        "task_b_exact_discounted_return": float(value[states].mean()),
        "t1_probability": float(p1[states].mean()),
        "t2_probability": float(p2[states].mean()),
    }
