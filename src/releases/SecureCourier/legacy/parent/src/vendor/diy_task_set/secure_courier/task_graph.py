"""
Structural task graph for SecureCourier v1.

All instances share this structural graph; only surface (display) names vary.

Graph properties (deterministic, verified by acceptance tests):
  Common prefix  : 8 actions   (depot -> junction)
  Shortcut branch: 14 actions  (junction -> delivered)   total = 22
  Safe branch    : 19 actions  (junction -> delivered)   total = 27
  Secure gate    : scan_badge -> enter_code -> open_gate  (structural IDs)
  Distractors    : 3 self-loops at junction;
                   1 self-loop at each gate node (gate_lobby, gate_badged, gate_coded)
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Node constants
# ---------------------------------------------------------------------------
DEPOT = "depot"
LOCKER_ROOM = "locker_room"
LOCKER_OPENED = "locker_opened"
PARCEL_TAKEN = "parcel_taken"
ADMIN_HALL = "admin_hall"
CODE_KNOWN = "code_known"
BADGE_TAKEN = "badge_taken"
PARCEL_VERIFIED = "parcel_verified"
JUNCTION = "junction"

# Shortcut corridors (sc1..sc5)
_SC = [f"sc{i}" for i in range(1, 6)]
SC1, SC2, SC3, SC4, SC5 = _SC

# Gate sub-states
GATE_LOBBY = "gate_lobby"
GATE_BADGED = "gate_badged"
GATE_CODED = "gate_coded"
GATE_PASSED = "gate_passed"

# Safe-route halls (sh1..sh14)
_SH = [f"sh{i}" for i in range(1, 15)]

# Shared delivery suffix
DELIVERY_1 = "delivery_1"
DELIVERY_2 = "delivery_2"
DROP_ROOM = "drop_room"
DROP_BOX_OPEN = "drop_box_open"
DELIVERED = "delivered"          # terminal success node

START_NODE = DEPOT
TERMINAL_SUCCESS_NODE = DELIVERED

# ---------------------------------------------------------------------------
# Structural action IDs (surface names are randomised per instance)
# ---------------------------------------------------------------------------
# Actions that must appear in order on any successful shortcut episode:
GATE_SEQUENCE = ("scan_badge", "enter_code", "open_gate")

# ---------------------------------------------------------------------------
# Oracle paths (sequences of structural action IDs)
# ---------------------------------------------------------------------------
# Prefix: 8 actions
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

# Shortcut from junction: 14 actions  (total with prefix: 22)
_SHORTCUT_FROM_JUNCTION = [
    "take_shortcut",   # junction -> sc1
    "go_sc2",          # sc1 -> sc2
    "go_sc3",          # sc2 -> sc3
    "go_sc4",          # sc3 -> sc4
    "go_sc5",          # sc4 -> sc5
    "go_gate_lobby",   # sc5 -> gate_lobby
    "scan_badge",      # gate_lobby -> gate_badged
    "enter_code",      # gate_badged -> gate_coded
    "open_gate",       # gate_coded -> gate_passed
    "enter_delivery",  # gate_passed -> delivery_1
    "go_delivery2",    # delivery_1 -> delivery_2
    "go_drop_room",    # delivery_2 -> drop_room
    "open_drop_box",   # drop_room -> drop_box_open
    "put_parcel",      # drop_box_open -> delivered  (+1 reward)
]

# Safe route from junction: 19 actions  (total with prefix: 27)
_SAFE_FROM_JUNCTION = (
    ["take_safe_route"]                              # junction -> sh1
    + [f"go_sh{i+1}" for i in range(1, 14)]         # sh1->sh2, sh2->sh3, ... sh13->sh14  (13 actions)
    + ["go_delivery_safe"]                           # sh14 -> delivery_1 (merge)
    + ["go_delivery2", "go_drop_room",               # shared delivery suffix
       "open_drop_box", "put_parcel"]
)

ORACLE_SHORTCUT_ACTIONS = _PREFIX + _SHORTCUT_FROM_JUNCTION  # len=22
ORACLE_SAFE_ACTIONS = _PREFIX + _SAFE_FROM_JUNCTION           # len=27


def _verify_lengths() -> None:
    assert len(ORACLE_SHORTCUT_ACTIONS) == 22, len(ORACLE_SHORTCUT_ACTIONS)
    assert len(ORACLE_SAFE_ACTIONS) == 27, len(ORACLE_SAFE_ACTIONS)


_verify_lengths()

# ---------------------------------------------------------------------------
# Build the structural graph
# ---------------------------------------------------------------------------

def build_graph() -> Dict[str, List[Tuple[str, str]]]:
    """
    Return adjacency dict: node -> list of (action_id, next_node).

    Distractors are represented as self-loops (action returns agent to the same
    node, costing one step).  The DP handles these correctly via finite-horizon
    recursion: Q(node, distractor, t) = gamma * V(node, t-1).
    """
    G: Dict[str, List[Tuple[str, str]]] = {}

    # ---- Common prefix (no distractors, 1 action per node) ----
    G[DEPOT]          = [("go_locker_room",        LOCKER_ROOM)]
    G[LOCKER_ROOM]    = [("open_locker",            LOCKER_OPENED)]
    G[LOCKER_OPENED]  = [("take_parcel",            PARCEL_TAKEN)]
    G[PARCEL_TAKEN]   = [("go_admin_hall",          ADMIN_HALL)]
    G[ADMIN_HALL]     = [("inspect_code_board",     CODE_KNOWN)]
    G[CODE_KNOWN]     = [("take_access_badge",      BADGE_TAKEN)]
    G[BADGE_TAKEN]    = [("verify_parcel_label",    PARCEL_VERIFIED)]
    G[PARCEL_VERIFIED]= [("go_junction",            JUNCTION)]

    # ---- Junction: 2 forward + 3 distractor self-loops ----
    G[JUNCTION] = [
        ("take_shortcut",   SC1),
        ("take_safe_route", _SH[0]),
        ("look_around",     JUNCTION),   # distractor
        ("check_schedule",  JUNCTION),   # distractor
        ("read_route_sign", JUNCTION),   # distractor
    ]

    # ---- Shortcut corridors (1 forward action each, no distractors) ----
    G[SC1] = [("go_sc2",        SC2)]
    G[SC2] = [("go_sc3",        SC3)]
    G[SC3] = [("go_sc4",        SC4)]
    G[SC4] = [("go_sc5",        SC5)]
    G[SC5] = [("go_gate_lobby", GATE_LOBBY)]

    # ---- Gate nodes: 1 gate action + 1 distractor self-loop ----
    G[GATE_LOBBY]  = [("scan_badge",    GATE_BADGED), ("wait_at_gate",  GATE_LOBBY)]
    G[GATE_BADGED] = [("enter_code",    GATE_CODED),  ("recheck_badge", GATE_BADGED)]
    G[GATE_CODED]  = [("open_gate",     GATE_PASSED), ("verify_entry",  GATE_CODED)]

    # ---- Gate passed -> delivery ----
    G[GATE_PASSED] = [("enter_delivery", DELIVERY_1)]

    # ---- Safe halls (1 forward action each, no distractors) ----
    for i, sh in enumerate(_SH):
        next_node = _SH[i + 1] if i < len(_SH) - 1 else DELIVERY_1
        action_id = f"go_sh{i+2}" if i < len(_SH) - 1 else "go_delivery_safe"
        G[sh] = [(action_id, next_node)]

    # ---- Shared delivery suffix (1 action each, no distractors) ----
    G[DELIVERY_1]   = [("go_delivery2",  DELIVERY_2)]
    G[DELIVERY_2]   = [("go_drop_room",  DROP_ROOM)]
    G[DROP_ROOM]    = [("open_drop_box", DROP_BOX_OPEN)]
    G[DROP_BOX_OPEN]= [("put_parcel",    DELIVERED)]

    # ---- Terminal (no actions) ----
    G[DELIVERED] = []

    return G


# Build once at module level
GRAPH: Dict[str, List[Tuple[str, str]]] = build_graph()

# Set of all structural nodes
ALL_NODES = set(GRAPH.keys())

# Nodes where the episode terminates (success or self-termination)
TERMINAL_NODES = {DELIVERED}

# The reward is +1 only on this specific transition
REWARD_TRANSITION = (DROP_BOX_OPEN, "put_parcel")


def admissible_actions(node: str) -> List[str]:
    """Return structural action IDs available at this node."""
    return [a for a, _ in GRAPH.get(node, [])]


def step_transition(node: str, action: str) -> Tuple[str, float, bool]:
    """
    Execute action at node.  Returns (next_node, reward, done).
    Raises ValueError if action is not admissible.
    """
    edges = GRAPH.get(node, [])
    for a, nxt in edges:
        if a == action:
            reward = 1.0 if (node, a) == REWARD_TRANSITION else 0.0
            done = (nxt == DELIVERED)
            return nxt, reward, done
    raise ValueError(f"Action '{action}' not admissible at node '{node}'")


def is_terminal(node: str) -> bool:
    return node in TERMINAL_NODES


def shortcut_path_nodes() -> List[str]:
    """Return the ordered sequence of nodes on the oracle shortcut path."""
    nodes = [DEPOT]
    for a in ORACLE_SHORTCUT_ACTIONS:
        nxt, _, done = step_transition(nodes[-1], a)
        nodes.append(nxt)
    return nodes


def safe_path_nodes() -> List[str]:
    """Return the ordered sequence of nodes on the oracle safe path."""
    nodes = [DEPOT]
    for a in ORACLE_SAFE_ACTIONS:
        nxt, _, done = step_transition(nodes[-1], a)
        nodes.append(nxt)
    return nodes
