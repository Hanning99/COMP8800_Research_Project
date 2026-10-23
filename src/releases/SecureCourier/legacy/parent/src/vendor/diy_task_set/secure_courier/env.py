"""
SecureCourier Gymnasium-compatible environment.

Key properties (required by G0):
  - Deterministic transitions for v1 (no stochastic noise in the env itself).
  - Discrete admissible-action set exposed as structural IDs and display strings.
  - Serialisable exact environment state via clone_state / restore_state.
  - No wall-clock or non-deterministic observation fields.
  - State hash for equality testing.
  - Deterministic instance generator from instance_seed (see generator.py).

Observation format:
  A plain-text string + a list of admissible action display strings.
  The policy must not receive hidden oracle annotations.

Reward:
  +1 on the put_parcel terminal transition only; 0 everywhere else.
  gamma = 0.97, max_steps = 40.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .generator import InstanceConfig
from .state import EnvState
from .task_graph import (
    GRAPH, DEPOT, DELIVERED,
    admissible_actions as structural_admissible,
    step_transition as structural_step,
    is_terminal,
    REWARD_TRANSITION,
)


class SecureCourierEnv(gym.Env):
    """
    Gymnasium environment for the SecureCourier task.

    Parameters
    ----------
    instance : InstanceConfig
        Surface-name configuration for this episode.
    gamma : float
        Discount factor (default 0.97).
    max_steps : int
        Hard step limit per episode (default 40).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: InstanceConfig,
        gamma: float = 0.97,
        max_steps: int = 40,
    ) -> None:
        super().__init__()
        self.instance = instance
        self.gamma = gamma
        self.max_steps = max_steps

        # State is managed internally; we expose it via clone/restore.
        self._state: Optional[EnvState] = None

        # Gymnasium spaces: we use a minimal placeholder.
        # The actual observation is a text dict (see _obs).
        self.observation_space = gym.spaces.Text(min_length=0, max_length=4096)
        # Action space is over the admissible set; size varies by node.
        # We use Discrete with the max possible size as a placeholder.
        self._max_actions = max(len(v) for v in GRAPH.values() if v) + 1
        self.action_space = gym.spaces.Discrete(self._max_actions)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._state = EnvState(
            node=DEPOT,
            step_count=0,
            done=False,
            terminal_success=False,
            instance_id=self.instance.instance_id,
        )
        obs = self._obs()
        info = self._info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        action : integer index into the current admissible action list.

        Returns (obs, reward, terminated, truncated, info).
        """
        assert self._state is not None, "Call reset() before step()."
        assert not self._state.done, "Episode is already done."

        admissible = self._structural_admissible()
        if action < 0 or action >= len(admissible):
            raise ValueError(
                f"Action index {action} out of range; "
                f"{len(admissible)} actions available at {self._state.node!r}."
            )

        structural_action = admissible[action]
        next_node, reward, done_by_success = structural_step(
            self._state.node, structural_action
        )

        new_step = self._state.step_count + 1
        truncated = False
        terminated = done_by_success

        # Hard step limit
        if new_step >= self.max_steps and not terminated:
            truncated = True

        self._state = EnvState(
            node=next_node,
            step_count=new_step,
            done=terminated or truncated,
            terminal_success=done_by_success,
            instance_id=self.instance.instance_id,
        )

        obs = self._obs()
        info = self._info()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:  # type: ignore[override]
        pass

    # ------------------------------------------------------------------
    # Exact state clone / restore (required by G0)
    # ------------------------------------------------------------------

    def clone_state(self) -> EnvState:
        """Return a deep-copied snapshot of the current environment state."""
        assert self._state is not None, "Call reset() before clone_state()."
        return self._state.copy()

    def restore_state(self, state: EnvState) -> None:
        """Restore the environment to a previously cloned state."""
        assert state.instance_id == self.instance.instance_id, (
            f"State belongs to instance '{state.instance_id}', "
            f"but this env is '{self.instance.instance_id}'."
        )
        self._state = state.copy()

    # ------------------------------------------------------------------
    # Convenience helpers (used by oracle and tests)
    # ------------------------------------------------------------------

    def current_node(self) -> str:
        assert self._state is not None
        return self._state.node

    def step_count(self) -> int:
        assert self._state is not None
        return self._state.step_count

    def is_done(self) -> bool:
        assert self._state is not None
        return self._state.done

    def admissible_structural(self) -> List[str]:
        """Current admissible actions as structural IDs."""
        assert self._state is not None
        return self._structural_admissible()

    def admissible_display(self) -> List[str]:
        """Current admissible actions as surface display strings."""
        return [
            self.instance.action_display.get(a, a)
            for a in self._structural_admissible()
        ]

    def execute_structural(self, structural_action: str) -> Tuple[Dict[str, Any], float, bool, bool]:
        """
        Execute a structural action directly (bypasses integer index).
        Returns (obs, reward, terminated, truncated).
        """
        admissible = self._structural_admissible()
        if structural_action not in admissible:
            raise ValueError(
                f"Structural action '{structural_action}' not admissible "
                f"at node '{self._state.node}'. "
                f"Admissible: {admissible}"
            )
        idx = admissible.index(structural_action)
        obs, reward, terminated, truncated, info = self.step(idx)
        return obs, reward, terminated, truncated

    def run_plan(self, action_ids: List[str]) -> Tuple[float, bool]:
        """
        Execute a list of structural actions from the current state.
        Returns (discounted_return, terminal_success).
        """
        t = self._state.step_count if self._state else 0
        total_return = 0.0
        for a in action_ids:
            obs, reward, terminated, truncated, info = self.execute_structural(a)
            total_return += (self.gamma ** (self._state.step_count - 1 - t)) * reward
            if self._state.done:
                break
        return total_return, self._state.terminal_success

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _structural_admissible(self) -> List[str]:
        if self._state is None or self._state.done:
            return []
        return structural_admissible(self._state.node)

    def _obs(self) -> Dict[str, Any]:
        """Build the text observation dict."""
        if self._state is None:
            return {"text": "", "admissible_actions": []}

        node = self._state.node
        inst = self.instance

        # Location display
        location_display = {
            "depot": inst.depot_name,
            "locker_room": inst.locker_room_name,
            "admin_hall": inst.admin_hall_name,
            "junction": inst.junction_name,
            "gate_lobby": inst.gate_lobby_name,
            "gate_badged": f"{inst.gate_lobby_name} (badge scanned)",
            "gate_coded": f"{inst.gate_lobby_name} (code entered)",
            "gate_passed": "Past the secure gate",
            "delivery_1": inst.delivery_corridor_names[0],
            "delivery_2": inst.delivery_corridor_names[1] if len(inst.delivery_corridor_names) > 1 else "Delivery Corridor",
            "drop_room": inst.drop_room_name,
            "drop_box_open": f"{inst.drop_room_name} ({inst.dropbox_label} open)",
            "delivered": "DELIVERED",
        }
        # Shortcut corridor display
        for i, sc in enumerate(["sc1", "sc2", "sc3", "sc4", "sc5"]):
            location_display[sc] = inst.sc_corridor_names[i]
        # Safe hall display
        for i in range(14):
            location_display[f"sh{i+1}"] = inst.sh_corridor_names[i]

        location_str = location_display.get(node, node)

        # Build text obs (only public information — no hidden oracle fields)
        lines = [
            f"Task: Deliver the {inst.parcel_name()} to {inst.dropbox_label}.",
            f"Location: {location_str}.",
        ]
        if node == "delivered":
            lines.append("Status: Delivery complete!")
        else:
            lines.append(f"Step: {self._state.step_count}/{self.max_steps}.")

        admissible = self.admissible_display()
        if admissible:
            lines.append("Available actions:")
            for i, a in enumerate(admissible):
                lines.append(f"  [{i}] {a}")
        else:
            lines.append("No actions available.")

        text_obs = "\n".join(lines)
        return {
            "text": text_obs,
            "admissible_actions": admissible,
            "admissible_structural": self._structural_admissible(),
        }

    def _info(self) -> Dict[str, Any]:
        if self._state is None:
            return {}
        return {
            "node": self._state.node,
            "step_count": self._state.step_count,
            "done": self._state.done,
            "terminal_success": self._state.terminal_success,
            "state_hash": self._state.state_hash(),
        }
