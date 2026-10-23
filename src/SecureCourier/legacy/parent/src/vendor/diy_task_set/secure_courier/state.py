"""
Environment state for SecureCourier.

EnvState is an immutable snapshot of everything needed to reproduce future
environment behavior.  It is used for clone_state / restore_state and for
the DP oracle.

The instance surface names (room names, parcel label, etc.) are stored in the
InstanceConfig (see generator.py) and are NOT part of the state, because they
do not change within an episode.  However, instance_id is included so that a
state snapshot can be validated against the correct instance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict


@dataclass
class EnvState:
    """Fully serialisable snapshot of a single environment episode."""

    node: str               # current structural node
    step_count: int         # number of steps taken so far (0 at reset)
    done: bool              # episode has ended
    terminal_success: bool  # episode ended with successful delivery
    instance_id: str        # identifies which instance this belongs to

    # ---------- serialisation ----------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvState":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    # ---------- hashing (for clone/restore verification) ----------

    def state_hash(self) -> str:
        """Deterministic SHA-256 of the serialised state."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    # ---------- copy ----------

    def copy(self) -> "EnvState":
        return EnvState(
            node=self.node,
            step_count=self.step_count,
            done=self.done,
            terminal_success=self.terminal_success,
            instance_id=self.instance_id,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvState):
            return NotImplemented
        return self.state_hash() == other.state_hash()

    def __repr__(self) -> str:
        return (
            f"EnvState(node={self.node!r}, step={self.step_count}, "
            f"done={self.done}, success={self.terminal_success}, "
            f"inst={self.instance_id!r})"
        )
