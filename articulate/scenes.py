"""Scenes: a whole small room written as code, not one object.

A scene is a Model like any other, so it writes the same URDF, MJCF and glTF
and runs through the same gate. The first scene is one room with a hinged door
and a sliding drawer:

    room    the static root: floor, four walls with a doorway in the front wall,
            and a low console against the back wall (all merged into one link)
    door    revolute `door_hinge` about Z, swinging into the room
    drawer  prismatic `drawer_slide`, the console's drawer, sliding toward the room

Frame, as in the object families: origin on the floor under the middle of the
room, X across the width, Y back (the front wall is at -Y), Z up. The door and
the drawer keep a clearance gap from everything they are not jointed to, so
nothing touches at rest and, by plain geometry, the joints should sweep clear through
their whole range. That is the design intent, not a measurement: the gate's own sweep
is what decides it.

Scenes are kept out of PILOT and FAMILIES on purpose: the pilot and the
generated catalog are pinned, and a scene is a different kind of thing.
"""
from __future__ import annotations

from typing import Callable, Optional

from .breakers import inverted_limits, missing_collision, zero_mass
from .families import GAP, HANDLE_R, drawer_shapes
from .model import Box, Cylinder, Model

FLOOR = 0.08          # floor slab thickness
WALL = 0.08           # wall thickness
DOOR_W = 0.9          # the doorway: clear width
DOOR_H = 2.0          # the doorway: clear height above the floor
DOOR_T = 0.04         # door leaf thickness
HINGE_R = 0.02        # hinge barrel radius: half the leaf thickness, so the corner stays on it
DOOR_OPEN = 1.5       # door_hinge upper limit, radians: 86 degrees, short of a quarter turn so the leaf's
                      # hinge corner never leans past the jamb
DOORWAY_FROM_CORNER = 0.3   # inner face of the left wall to the left jamb
CONSOLE_W, CONSOLE_H, CONSOLE_D = 0.8, 0.75, 0.45
CONSOLE_X = 0.6       # console centre, along the back wall
BACK_CLEARANCE = 0.02 # console to back wall


def _console_shapes(cx: float, cy: float, z0: float) -> list:
    """A low console: a top and four legs, standing on z0, centred at (cx, cy)."""
    w, h, d = CONSOLE_W, CONSOLE_H, CONSOLE_D
    tt, leg = 0.025, 0.04
    leg_h = h - tt
    lx, ly = w / 2 - leg / 2 - 0.02, d / 2 - leg / 2 - 0.02
    return [Box((w, d, tt), (cx, cy, z0 + h - tt / 2)),
            *[Box((leg, leg, leg_h), (cx + sx * lx, cy + sy * ly, z0 + leg_h / 2)) for sx in (-1, 1) for sy in (-1, 1)]]


def room_door_drawer(width: float, depth: float, height: float) -> Model:
    m = Model("room_door_drawer",
              description="One small room with a hinged door that swings into it and a console with a sliding drawer.",
              params=dict(width=width, depth=depth, height=height))
    wall_h = height - FLOOR
    zc = FLOOR + wall_h / 2
    dx0 = -width / 2 + WALL + DOORWAY_FROM_CORNER        # left jamb
    dx1 = dx0 + DOOR_W                                   # right jamb
    fy = -depth / 2 + WALL / 2                           # front wall mid-plane (door sits in it)
    # No two shapes of the room overlap, so the mass is each volume counted once.
    room = [
        Box((width, depth, FLOOR), (0, 0, FLOOR / 2)),
        Box((width, WALL, wall_h), (0, depth / 2 - WALL / 2, zc)),
        Box((WALL, depth - 2 * WALL, wall_h), (-width / 2 + WALL / 2, 0, zc)),
        Box((WALL, depth - 2 * WALL, wall_h), (width / 2 - WALL / 2, 0, zc)),
        Box((dx0 + width / 2, WALL, wall_h), ((dx0 - width / 2) / 2, fy, zc)),                       # front wall, left of the doorway
        Box((width / 2 - dx1, WALL, wall_h), ((dx1 + width / 2) / 2, fy, zc)),                       # front wall, right of the doorway
        Box((DOOR_W, WALL, height - FLOOR - DOOR_H), ((dx0 + dx1) / 2, fy, FLOOR + DOOR_H + (height - FLOOR - DOOR_H) / 2)),  # above the doorway
    ]
    cy = depth / 2 - WALL - BACK_CLEARANCE - CONSOLE_D / 2
    room += _console_shapes(CONSOLE_X, cy, FLOOR)
    m.link("room", *room)

    # The door: hinge barrel on the jamb side, leaf beside it, handle on the room side.
    leaf_w = DOOR_W - 2 * HINGE_R - 2 * GAP
    leaf_h = DOOR_H - 2 * GAP
    m.link("door",
           Cylinder(HINGE_R, leaf_h, (0, 0, leaf_h / 2), axis="z", material="wood"),
           Box((leaf_w, DOOR_T, leaf_h), (HINGE_R + leaf_w / 2, 0, leaf_h / 2)),
           Cylinder(HANDLE_R, 0.12, (HINGE_R + leaf_w - 0.06, DOOR_T / 2 + HANDLE_R, leaf_h / 2), axis="z"))
    m.joint("door_hinge", "revolute", "room", "door", (dx0 + HINGE_R + GAP, fy, FLOOR + GAP), "z", 0.0, DOOR_OPEN)

    # The drawer under the console top, the same drawer as the desk family's.
    top = 0.025
    dw = min(0.5, CONSOLE_W - 2 * (0.04 + 0.02) - 0.1)
    dh = 0.09
    dd = CONSOLE_D - 0.12
    m.link("drawer", *drawer_shapes(dw, dh, dd))
    m.joint("drawer_slide", "prismatic", "room", "drawer",
            (CONSOLE_X, cy - CONSOLE_D / 2 + 0.03, FLOOR + CONSOLE_H - top - GAP - dh), (0, -1, 0), 0.0, round(0.7 * dd, 4))
    return m


# The scenes. Same shape as PILOT: id -> (function, parameters), so build(id, SCENES) works.
SCENES = {
    "room-door-drawer-01": (room_door_drawer, dict(width=2.4, depth=2.0, height=2.4)),
}


def door_into_console(m: Model) -> Optional[Model]:
    """A console pushed into the door's swing. At rest nothing touches, so only
    sweeping the hinge through its range finds it: the mistake a static check
    cannot see. Applies to scenes with a joint named door_hinge."""
    hinge = next((j for j in m.joints if j.name == "door_hinge"), None)
    if hinge is None:
        return None
    b = m.copy()
    # 0.42 m along X and Y from the hinge: 0.6 m from the hinge, a quarter turn into the swing.
    at = (hinge.origin[0] + 0.42, hinge.origin[1] + 0.42, FLOOR + 0.3)
    b.get_link(hinge.parent).shapes.append(Box((0.4, 0.3, 0.6), at))
    return b


# The mistakes a scene ships broken with: three the objects have, plus the one only a sweep finds.
SCENE_BREAKERS: dict[str, Callable[[Model], Optional[Model]]] = {
    "zero_mass": zero_mass,
    "inverted_limits": inverted_limits,
    "missing_collision": missing_collision,
    "door_into_console": door_into_console,
}

# Broken variants a contract cannot be expected to fail: the sweep is what catches them.
# name -> the joint whose sweep must collide.
SWEEP_CAUGHT = {"door_into_console": "door_hinge"}
