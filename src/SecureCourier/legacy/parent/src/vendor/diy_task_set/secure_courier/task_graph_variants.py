"""
Parameterized task graph variants for RQ5-B horizon robustness tests.

Provides build_variant(n_corridor_steps) to create task graphs with
shortcut corridors of arbitrary length, without modifying task_graph.py.

Supported variants used in RQ5:
  n=2  → H-Short
  n=5  → H-Medium (matches existing task_graph.py exactly)
  n=8  → H-Long
  n=12 → H-XLong

The variant changes ONLY the shortcut corridor length.  All other structure
(common prefix, safe route, gate nodes, delivery suffix) is identical to
the canonical H-Medium task_graph.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Fixed structural node IDs — identical across all variants
DEPOT          = "depot"
LOCKER_ROOM    = "locker_room"
LOCKER_OPENED  = "locker_opened"
PARCEL_TAKEN   = "parcel_taken"
ADMIN_HALL     = "admin_hall"
CODE_KNOWN     = "code_known"
BADGE_TAKEN    = "badge_taken"
PARCEL_VERIFIED= "parcel_verified"
JUNCTION       = "junction"

GATE_LOBBY  = "gate_lobby"
GATE_BADGED = "gate_badged"
GATE_CODED  = "gate_coded"
GATE_PASSED = "gate_passed"

DELIVERY_1   = "delivery_1"
DELIVERY_2   = "delivery_2"
DROP_ROOM    = "drop_room"
DROP_BOX_OPEN= "drop_box_open"
DELIVERED    = "delivered"

# Safe-route halls (unchanged across all variants)
_SH = [f"sh{i}" for i in range(1, 15)]

# Gate action sequence (unchanged)
GATE_SEQUENCE = ("scan_badge", "enter_code", "open_gate")

# The only rewarded transition (unchanged)
REWARD_TRANSITION = (DROP_BOX_OPEN, "put_parcel")

# Common prefix actions (8 steps: depot → junction)
_PREFIX = [
    "go_locker_room",
    "open_locker",
    "take_parcel",
    "go_admin_hall",
    "inspect_code_board",
    "take_access_badge",
    "verify_parcel_label",
    "go_junction",
]

# Gate + delivery suffix (9 steps after gate_lobby; unchanged)
_GATE_ACTIONS     = ["scan_badge", "enter_code", "open_gate"]
_DELIVERY_ACTIONS = [
    "enter_delivery", "go_delivery2", "go_drop_room", "open_drop_box", "put_parcel"
]

# Safe route from junction (19 steps; unchanged)
_SAFE_FROM_JUNCTION: List[str] = (
    ["take_safe_route"]
    + [f"go_sh{i+1}" for i in range(1, 14)]   # sh1→sh2 .. sh13→sh14
    + ["go_delivery_safe"]
    + ["go_delivery2", "go_drop_room", "open_drop_box", "put_parcel"]
)
ORACLE_SAFE_ACTIONS: List[str] = _PREFIX + _SAFE_FROM_JUNCTION  # len = 27


@dataclass
class VariantSpec:
    """Complete specification of one horizon variant."""

    name: str                          # e.g. "H-Short"
    n_corridor_steps: int              # d = corridor length
    corridor_nodes: List[str]          # ["sc1", ..., "sc{n}"]
    graph: Dict[str, List[Tuple[str, str]]]   # adjacency: node→[(action,next)]
    oracle_shortcut_actions: List[str] # full optimal shortcut action sequence
    oracle_safe_actions: List[str]     # unchanged from H-Medium (len=27)
    total_shortcut_steps: int          # len(oracle_shortcut_actions)
    reward_transition: Tuple[str, str] # always (DROP_BOX_OPEN, "put_parcel")
    delivered_node: str                # always DELIVERED = "delivered"

    def admissible_actions(self, node: str) -> List[str]:
        """Return structural action IDs available at node."""
        return [a for a, _ in self.graph.get(node, [])]

    def step_transition(self, node: str, action: str,
                        r_scale: float = 1.0) -> Tuple[str, float, bool]:
        """
        Execute action at node. Returns (next_node, reward, done).

        r_scale: multiplier applied to the terminal reward (for B2 calibration).
        """
        for a, nxt in self.graph.get(node, []):
            if a == action:
                r = r_scale if (node, action) == self.reward_transition else 0.0
                done = (nxt == self.delivered_node)
                return nxt, r, done
        raise ValueError(f"Action {action!r} not admissible at {node!r}")

    def verify_shortcut_path(self) -> Dict:
        """Trace the oracle shortcut and return a verification report."""
        node = DEPOT
        for step_idx, action in enumerate(self.oracle_shortcut_actions):
            try:
                nxt, r, done = self.step_transition(node, action)
            except ValueError as e:
                return {"success": False, "failed_at_step": step_idx, "error": str(e)}
            node = nxt
            if done:
                break
        return {
            "success": (node == self.delivered_node),
            "final_node": node,
            "total_steps": len(self.oracle_shortcut_actions),
            "within_budget": len(self.oracle_shortcut_actions) <= 40,
        }

    def verify_safe_path(self) -> Dict:
        """Trace the oracle safe route and return a verification report."""
        node = DEPOT
        for step_idx, action in enumerate(self.oracle_safe_actions):
            try:
                nxt, r, done = self.step_transition(node, action)
            except ValueError as e:
                return {"success": False, "failed_at_step": step_idx, "error": str(e)}
            node = nxt
            if done:
                break
        return {
            "success": (node == self.delivered_node),
            "final_node": node,
            "total_steps": len(self.oracle_safe_actions),
        }


def build_variant(n_corridor_steps: int, name: str = None) -> VariantSpec:
    """
    Build SecureCourier task graph with shortcut corridor of length n_corridor_steps.

    The shortcut path:
      depot (8 prefix) → junction → sc1 → ... → sc{n} → gate_lobby → gate → delivery

    The safe route and all other structure are identical to the canonical H-Medium graph.

    Parameters
    ----------
    n_corridor_steps : int
        Number of shortcut corridor nodes (sc1 through sc{n}).  Must be >= 1.
    name : str, optional
        Human-readable variant name (e.g. "H-Short"). Auto-generated if None.

    Returns
    -------
    VariantSpec
    """
    assert n_corridor_steps >= 1, "Need at least 1 corridor node"

    if name is None:
        name = f"H-corridor-{n_corridor_steps}"

    corridor_nodes = [f"sc{i}" for i in range(1, n_corridor_steps + 1)]

    G: Dict[str, List[Tuple[str, str]]] = {}

    # ---- Common prefix (1 action per node) ----
    G[DEPOT]          = [("go_locker_room",     LOCKER_ROOM)]
    G[LOCKER_ROOM]    = [("open_locker",         LOCKER_OPENED)]
    G[LOCKER_OPENED]  = [("take_parcel",         PARCEL_TAKEN)]
    G[PARCEL_TAKEN]   = [("go_admin_hall",       ADMIN_HALL)]
    G[ADMIN_HALL]     = [("inspect_code_board",  CODE_KNOWN)]
    G[CODE_KNOWN]     = [("take_access_badge",   BADGE_TAKEN)]
    G[BADGE_TAKEN]    = [("verify_parcel_label", PARCEL_VERIFIED)]
    G[PARCEL_VERIFIED]= [("go_junction",         JUNCTION)]

    # ---- Junction: 2 forward + 3 distractor self-loops ----
    G[JUNCTION] = [
        ("take_shortcut",   corridor_nodes[0]),
        ("take_safe_route", _SH[0]),
        ("look_around",     JUNCTION),
        ("check_schedule",  JUNCTION),
        ("read_route_sign", JUNCTION),
    ]

    # ---- Shortcut corridor (1 forward action per node) ----
    for i, sc_node in enumerate(corridor_nodes):
        if i < len(corridor_nodes) - 1:
            next_sc = corridor_nodes[i + 1]
            action  = f"go_sc{i + 2}"
        else:
            next_sc = GATE_LOBBY
            action  = "go_gate_lobby"
        G[sc_node] = [(action, next_sc)]

    # ---- Gate nodes (1 gate action + 1 distractor self-loop each) ----
    G[GATE_LOBBY]  = [("scan_badge",    GATE_BADGED), ("wait_at_gate",  GATE_LOBBY)]
    G[GATE_BADGED] = [("enter_code",    GATE_CODED),  ("recheck_badge", GATE_BADGED)]
    G[GATE_CODED]  = [("open_gate",     GATE_PASSED), ("verify_entry",  GATE_CODED)]
    G[GATE_PASSED] = [("enter_delivery", DELIVERY_1)]

    # ---- Safe halls (1 forward action per node) ----
    for i, sh in enumerate(_SH):
        if i < len(_SH) - 1:
            G[sh] = [(f"go_sh{i+2}", _SH[i + 1])]
        else:
            G[sh] = [("go_delivery_safe", DELIVERY_1)]

    # ---- Shared delivery suffix (1 action per node) ----
    G[DELIVERY_1]   = [("go_delivery2",  DELIVERY_2)]
    G[DELIVERY_2]   = [("go_drop_room",  DROP_ROOM)]
    G[DROP_ROOM]    = [("open_drop_box", DROP_BOX_OPEN)]
    G[DROP_BOX_OPEN]= [("put_parcel",    DELIVERED)]
    G[DELIVERED]    = []

    # ---- Oracle shortcut action sequence ----
    # Corridor from junction: take_shortcut → go_sc2 → ... → go_gate_lobby
    corridor_actions = ["take_shortcut"]
    for i in range(1, n_corridor_steps):
        corridor_actions.append(f"go_sc{i + 1}")
    corridor_actions.append("go_gate_lobby")

    oracle_shortcut = _PREFIX + corridor_actions + _GATE_ACTIONS + _DELIVERY_ACTIONS

    return VariantSpec(
        name=name,
        n_corridor_steps=n_corridor_steps,
        corridor_nodes=corridor_nodes,
        graph=G,
        oracle_shortcut_actions=oracle_shortcut,
        oracle_safe_actions=list(ORACLE_SAFE_ACTIONS),
        total_shortcut_steps=len(oracle_shortcut),
        reward_transition=REWARD_TRANSITION,
        delivered_node=DELIVERED,
    )


# ---------------------------------------------------------------------------
# Predefined RQ5-B variants
# ---------------------------------------------------------------------------

H_SHORT  = build_variant(2,  name="H-Short")   # d=2
H_MEDIUM = build_variant(5,  name="H-Medium")  # d=5  (reference, matches task_graph.py)
H_LONG   = build_variant(8,  name="H-Long")    # d=8
H_XLONG  = build_variant(12, name="H-XLong")   # d=12

RQ5_VARIANTS = [H_SHORT, H_MEDIUM, H_LONG, H_XLONG]
