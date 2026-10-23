"""Transition schema for the offline dataset (Section 7 of EXPERIMENT_PLAN)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Transition:
    # IDs
    transition_id: str
    instance_id: str
    instance_seed: int
    episode_id: str
    t: int

    # State
    full_state: Dict[str, Any]       # EnvState.to_dict()
    observation: str                  # text observation
    admissible_actions: List[str]     # display strings
    admissible_structural: List[str]  # structural IDs

    # Action taken
    action: int                       # index into admissible_actions
    action_display: str
    action_structural: str

    # Transition result
    reward: float
    next_state: Dict[str, Any]
    next_observation: str
    next_admissible_actions: List[str]
    next_admissible_structural: List[str]
    done: bool

    # Metadata
    behavior_policy_id: str
    behavior_log_prob: float          # log P(action | state) under behavior
    route_label: str                  # "safe", "shortcut_fail", "exploration", etc.
    historical_return: Optional[float] = None  # filled in post-hoc if known

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Transition":
        return cls(**d)
