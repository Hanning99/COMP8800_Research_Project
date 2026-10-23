"""
Instance generator for SecureCourier v1.

Each instance is a deterministic seed-driven assignment of surface names to
the fixed structural graph.  The structural graph (node count, action count,
route lengths) is identical across all instances; only display strings differ.

Randomised surface factors (Section 3.2):
  - Room / corridor display names
  - Parcel name / colour
  - Badge name / colour
  - Access code token
  - Drop-box label
  - Order of distractor actions at distractor nodes
  - Surface strings for all structural actions

Fixed structural factors:
  - Common-prefix length: 8 actions
  - Shortcut optimal length: 22 actions total
  - Safe optimal length: 27 actions total
  - Position and identity of the secure-gate sequence
  - Terminal reward definition
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Name pools for surface randomisation
# ---------------------------------------------------------------------------

_PARCEL_COLOURS = ["blue", "red", "green", "yellow", "orange", "purple", "grey"]
_PARCEL_ITEMS   = ["parcel", "package", "box", "crate", "container"]

_BADGE_COLOURS  = ["silver", "gold", "green", "black", "white", "copper"]
_BADGE_TYPES    = ["badge", "card", "pass", "token", "tag"]

_CODE_PREFIXES  = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
_CODE_DIGITS    = "0123456789"

_LOCKER_ROOM_NAMES = [
    "Storage Room", "Equipment Locker", "Supply Closet", "Prep Room",
    "Load-Out Bay", "Staging Area", "Warehouse Bay",
]
_ADMIN_HALL_NAMES = [
    "Admin Corridor", "Control Passage", "Management Wing",
    "Operations Hall", "Command Corridor",
]
_JUNCTION_NAMES = [
    "Route Junction", "Crossroads", "Split Point", "Decision Point",
    "Fork Junction", "Branch Point",
]
_GATE_LOBBY_NAMES = [
    "Gate Lobby", "Security Anteroom", "Access Foyer",
    "Checkpoint Lobby", "Secure Vestibule",
]
_DROP_ROOM_NAMES = [
    "Drop Room", "Delivery Room", "Handover Bay",
    "Parcel Hub", "Dispatch Room",
]

_SC_CORRIDOR_NAMES = [
    "Secure Passage", "Express Corridor", "Priority Hall",
    "Fast Lane", "Direct Route", "Alpha Corridor", "Beta Hall",
    "Gamma Passage", "Delta Way", "Echo Route",
]

_SH_CORRIDOR_NAMES = [
    "Main Corridor", "Standard Hall", "Regular Passage",
    "Service Route", "Public Way", "North Wing", "South Wing",
    "East Passage", "West Hall", "Central Aisle",
    "Auxiliary Corridor", "Secondary Hall", "Alternate Route",
    "Overflow Passage", "Side Wing",
]

_DELIVERY_CORRIDOR_NAMES = [
    "Delivery Passage", "Final Corridor", "Handover Hall",
    "Terminal Wing", "Last Mile Hall",
]

_DEPOT_NAMES = [
    "Depot", "Distribution Centre", "Sorting Hub",
    "Dispatch Point", "Origin Station",
]

_DROPBOX_LABELS = [
    "DropBox-A", "DropBox-B", "DropBox-C", "Collection Point-1",
    "Drop Station Alpha", "Parcel Terminal", "Delivery Hub-7",
]

_DISTRACTOR_ACTIONS_JUNCTION = [
    ["look_around", "check_schedule", "read_route_sign"],
    ["glance_at_map", "check_watch", "inspect_notice_board"],
    ["adjust_load", "rest_briefly", "review_instructions"],
    ["check_delivery_note", "consult_route_card", "pause_for_breath"],
]

_DISTRACTOR_ACTIONS_GATE = [
    ("wait_at_gate", "recheck_badge", "verify_entry"),
    ("stand_by_gate", "inspect_panel", "check_display"),
    ("hold_position", "scan_surroundings", "reread_code"),
    ("wait_briefly", "double_check_badge", "confirm_code"),
]


# ---------------------------------------------------------------------------
# InstanceConfig
# ---------------------------------------------------------------------------

@dataclass
class InstanceConfig:
    """
    Surface name assignment for one SecureCourier instance.

    All fields are pure display strings; they do not affect transitions.
    The structural graph is the same for every instance.
    """

    instance_id: str
    instance_seed: int
    split: str  # "train", "val", or "test"

    # Parcel / item
    parcel_colour: str
    parcel_item: str

    # Badge / access credential
    badge_colour: str
    badge_type: str

    # Access code
    access_code: str

    # Drop box
    dropbox_label: str

    # Location display names
    depot_name: str
    locker_room_name: str
    admin_hall_name: str
    junction_name: str
    gate_lobby_name: str
    drop_room_name: str

    # Shortcut corridor display names (5 names for sc1..sc5)
    sc_corridor_names: List[str]

    # Safe-route hall display names (14 names for sh1..sh14)
    sh_corridor_names: List[str]

    # Delivery corridor display names (2 names for delivery_1, delivery_2)
    delivery_corridor_names: List[str]

    # Structural-action-ID -> display string mapping
    action_display: Dict[str, str] = field(default_factory=dict)

    # Distractor structural IDs at junction and gate nodes
    junction_distractor_ids: List[str] = field(default_factory=list)
    gate_lobby_distractor_id: str = ""
    gate_badged_distractor_id: str = ""
    gate_coded_distractor_id: str = ""

    def parcel_name(self) -> str:
        return f"{self.parcel_colour} {self.parcel_item}"

    def badge_name(self) -> str:
        return f"{self.badge_colour} {self.badge_type}"


def generate_instance(instance_seed: int, split: str = "train",
                      index: int = 0) -> InstanceConfig:
    """
    Generate one InstanceConfig from an integer seed.

    The seed fully determines all surface randomisation.
    The structural graph is identical regardless of seed.
    """
    rng = random.Random(instance_seed)

    instance_id = f"{split}_{index:04d}_s{instance_seed}"

    parcel_colour = rng.choice(_PARCEL_COLOURS)
    parcel_item   = rng.choice(_PARCEL_ITEMS)
    badge_colour  = rng.choice(_BADGE_COLOURS)
    badge_type    = rng.choice(_BADGE_TYPES)

    # Access code: 2 letters + 2 digits
    code = (
        rng.choice(_CODE_PREFIXES)
        + rng.choice(_CODE_PREFIXES)
        + rng.choice(_CODE_DIGITS)
        + rng.choice(_CODE_DIGITS)
    )

    dropbox_label = rng.choice(_DROPBOX_LABELS)
    depot_name    = rng.choice(_DEPOT_NAMES)
    locker_room   = rng.choice(_LOCKER_ROOM_NAMES)
    admin_hall    = rng.choice(_ADMIN_HALL_NAMES)
    junction_name = rng.choice(_JUNCTION_NAMES)
    gate_lobby    = rng.choice(_GATE_LOBBY_NAMES)
    drop_room     = rng.choice(_DROP_ROOM_NAMES)

    sc_pool = list(_SC_CORRIDOR_NAMES)
    rng.shuffle(sc_pool)
    sc_names = sc_pool[:5]

    sh_pool = list(_SH_CORRIDOR_NAMES)
    rng.shuffle(sh_pool)
    sh_names = sh_pool[:14]

    del_pool = list(_DELIVERY_CORRIDOR_NAMES)
    rng.shuffle(del_pool)
    del_names = del_pool[:2]

    # Distractor display strings (choose from pools, shuffle per instance)
    dist_j_pool = list(rng.choice(_DISTRACTOR_ACTIONS_JUNCTION))
    rng.shuffle(dist_j_pool)
    dist_g_pool = list(rng.choice(_DISTRACTOR_ACTIONS_GATE))

    # Build action display map (structural_action_id -> surface string)
    adisplay: Dict[str, str] = {}

    # Common prefix actions
    adisplay["go_locker_room"]       = f"go to {locker_room}"
    adisplay["open_locker"]          = "open locker"
    adisplay["take_parcel"]          = f"take {parcel_colour} {parcel_item}"
    adisplay["go_admin_hall"]        = f"go to {admin_hall}"
    adisplay["inspect_code_board"]   = "inspect code board"
    adisplay["take_access_badge"]    = f"take {badge_colour} {badge_type}"
    adisplay["verify_parcel_label"]  = f"verify {parcel_colour} {parcel_item} label"
    adisplay["go_junction"]          = f"go to {junction_name}"

    # Junction forward actions
    adisplay["take_shortcut"]        = f"take shortcut via {sc_names[0]}"
    adisplay["take_safe_route"]      = f"take safe route via {sh_names[0]}"

    # Junction distractors (structural IDs are fixed; display varies)
    _j_dist_ids = ["look_around", "check_schedule", "read_route_sign"]
    _j_dist_disp = dist_j_pool[:3]
    for sid, sdisp in zip(_j_dist_ids, _j_dist_disp):
        adisplay[sid] = sdisp.replace("_", " ")

    # Shortcut corridors
    adisplay["go_sc2"] = f"go through {sc_names[1]}"
    adisplay["go_sc3"] = f"go through {sc_names[2]}"
    adisplay["go_sc4"] = f"go through {sc_names[3]}"
    adisplay["go_sc5"] = f"go through {sc_names[4]}"
    adisplay["go_gate_lobby"] = f"go to {gate_lobby}"

    # Gate actions
    adisplay["scan_badge"]   = f"scan {badge_colour} {badge_type} at reader"
    adisplay["enter_code"]   = f"enter code {code} on keypad"
    adisplay["open_gate"]    = "open secure gate"

    # Gate distractors
    adisplay["wait_at_gate"]  = dist_g_pool[0].replace("_", " ")
    adisplay["recheck_badge"] = dist_g_pool[1].replace("_", " ")
    adisplay["verify_entry"]  = dist_g_pool[2].replace("_", " ")

    # Gate passed -> delivery
    adisplay["enter_delivery"] = f"enter {del_names[0]}"

    # Safe halls
    for i in range(13):
        adisplay[f"go_sh{i+2}"] = f"go through {sh_names[i+1]}"
    adisplay["go_delivery_safe"] = f"enter {del_names[0]}"

    # Shared delivery suffix
    adisplay["go_delivery2"]  = f"go through {del_names[1]}"
    adisplay["go_drop_room"]  = f"go to {drop_room}"
    adisplay["open_drop_box"] = f"open {dropbox_label}"
    adisplay["put_parcel"]    = f"put {parcel_colour} {parcel_item} in {dropbox_label}"

    return InstanceConfig(
        instance_id=instance_id,
        instance_seed=instance_seed,
        split=split,
        parcel_colour=parcel_colour,
        parcel_item=parcel_item,
        badge_colour=badge_colour,
        badge_type=badge_type,
        access_code=code,
        dropbox_label=dropbox_label,
        depot_name=depot_name,
        locker_room_name=locker_room,
        admin_hall_name=admin_hall,
        junction_name=junction_name,
        gate_lobby_name=gate_lobby,
        drop_room_name=drop_room,
        sc_corridor_names=sc_names,
        sh_corridor_names=sh_names,
        delivery_corridor_names=del_names,
        action_display=adisplay,
        junction_distractor_ids=_j_dist_ids,
        gate_lobby_distractor_id="wait_at_gate",
        gate_badged_distractor_id="recheck_badge",
        gate_coded_distractor_id="verify_entry",
    )


def generate_pilot_split(
    seed_train: int = 1000,
    seed_val:   int = 2000,
    seed_test:  int = 3000,
    n_train: int = 16,
    n_val:   int = 8,
    n_test:  int = 8,
) -> Dict[str, List[InstanceConfig]]:
    """
    Generate the 16/8/8 pilot split.

    Each instance gets its own seed derived from the split seed + index.
    """
    splits: Dict[str, List[InstanceConfig]] = {}

    for split_name, base_seed, n in [
        ("train", seed_train, n_train),
        ("val",   seed_val,   n_val),
        ("test",  seed_test,  n_test),
    ]:
        instances = []
        for i in range(n):
            inst_seed = base_seed + i
            instances.append(generate_instance(inst_seed, split=split_name, index=i))
        splits[split_name] = instances

    return splits
