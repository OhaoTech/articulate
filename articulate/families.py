"""Object families. Each is a function of a few plain parameters that returns a
Model. Frame: origin at the floor under the middle of the object, X across the
width, Y back (front face at -Y), Z up. Moving parts keep a clearance gap from
everything they are not jointed to, so they never touch at rest.
"""
from __future__ import annotations

from .model import Box, Cylinder, Model

T = 0.018        # panel thickness
GAP = 0.003      # clearance between a moving part and its neighbours
HANDLE_R = 0.006


def carcass(m: Model, w: float, h: float, d: float, top: bool = True, back: bool = True) -> None:
    """A box of panels standing on the floor: sides, bottom, optional top and back."""
    shapes = [
        Box((T, d, h), (-w / 2 + T / 2, 0, h / 2)),
        Box((T, d, h), (w / 2 - T / 2, 0, h / 2)),
        Box((w - 2 * T, d, T), (0, 0, T / 2)),
    ]
    if top:
        shapes.append(Box((w - 2 * T, d, T), (0, 0, h - T / 2)))
    if back:
        inner_h = h - T * (2 if top else 1)
        shapes.append(Box((w - 2 * T, T, inner_h), (0, d / 2 - T / 2, T + inner_h / 2)))
    m.link("carcass", *shapes)


def drawer_shapes(w: float, h: float, d: float) -> list:
    """An open drawer: front, bottom, two sides, back, and a handle. Frame at the
    middle of the front face's bottom edge; the drawer extends toward +Y."""
    t = 0.012
    return [
        Box((w, t, h), (0, t / 2, h / 2)),                                     # front
        Box((w - 2 * t, d - 2 * t, t), (0, d / 2, t / 2)),                     # bottom
        Box((t, d - t, h * 0.8), (-w / 2 + t / 2, t + (d - t) / 2, h * 0.4)),  # left side
        Box((t, d - t, h * 0.8), (w / 2 - t / 2, t + (d - t) / 2, h * 0.4)),   # right side
        Box((w - 2 * t, t, h * 0.8), (0, d - t / 2, h * 0.4)),                 # back
        Cylinder(HANDLE_R, min(0.12, w * 0.4), (0, -HANDLE_R, h / 2), axis="x"),
    ]


def drawer_cabinet(width: float, height: float, depth: float, drawers: int) -> Model:
    m = Model(f"drawer_cabinet_{drawers}", description=f"A {drawers}-drawer cabinet. Each drawer slides out along its runner.",
              params=dict(width=width, height=height, depth=depth, drawers=drawers))
    carcass(m, width, height, depth)
    inner_w = width - 2 * T
    inner_h = height - 2 * T
    inner_d = depth - T - GAP
    slot = inner_h / drawers
    for i in range(drawers):
        name = f"drawer_{i + 1}"
        dh = slot - 2 * GAP
        m.link(name, *drawer_shapes(inner_w - 2 * GAP, dh, inner_d - GAP))
        z = T + i * slot + GAP
        m.joint(f"{name}_slide", "prismatic", "carcass", name, (0, -depth / 2, z), (0, -1, 0), 0.0, round(0.75 * inner_d, 4))
    return m


def door_cabinet(width: float, height: float, depth: float, doors: int) -> Model:
    m = Model(f"door_cabinet_{doors}", description=f"A cabinet with {doors} hinged door{'s' if doors > 1 else ''} that swing outward.",
              params=dict(width=width, height=height, depth=depth, doors=doors))
    carcass(m, width, height, depth)
    dt = 0.018
    dh = height - 2 * GAP
    y = -depth / 2 - GAP   # hinge line, just in front of the carcass
    if doors == 1:
        dw = width - 2 * GAP
        m.link("door", Box((dw, dt, dh), (dw / 2, -dt / 2, dh / 2)),
               Cylinder(HANDLE_R, min(0.2, dh * 0.3), (dw - 0.04, -dt - HANDLE_R, dh / 2), axis="z"))
        m.joint("door_hinge", "revolute", "carcass", "door", (-width / 2, y, GAP), (0, 0, -1), 0.0, 1.6)
    else:
        dw = width / 2 - 2 * GAP
        m.link("door_left", Box((dw, dt, dh), (dw / 2, -dt / 2, dh / 2)),
               Cylinder(HANDLE_R, min(0.2, dh * 0.3), (dw - 0.04, -dt - HANDLE_R, dh / 2), axis="z"))
        m.link("door_right", Box((dw, dt, dh), (-dw / 2, -dt / 2, dh / 2)),
               Cylinder(HANDLE_R, min(0.2, dh * 0.3), (-dw + 0.04, -dt - HANDLE_R, dh / 2), axis="z"))
        m.joint("door_left_hinge", "revolute", "carcass", "door_left", (-width / 2, y, GAP), (0, 0, -1), 0.0, 1.6)
        m.joint("door_right_hinge", "revolute", "carcass", "door_right", (width / 2, y, GAP), (0, 0, 1), 0.0, 1.6)
    return m


def chest(width: float, height: float, depth: float) -> Model:
    m = Model("chest", description="A chest with a lid hinged along its back edge.",
              params=dict(width=width, height=height, depth=depth))
    carcass(m, width, height, depth, top=False)
    lt = 0.02
    m.link("lid", Box((width, depth, lt), (0, -depth / 2, lt / 2)),
           Cylinder(HANDLE_R, 0.1, (0, -depth - HANDLE_R, lt / 2), axis="x"))
    m.joint("lid_hinge", "revolute", "carcass", "lid", (0, depth / 2, height + GAP), (-1, 0, 0), 0.0, 1.9)
    return m


def sliding_cabinet(width: float, height: float, depth: float) -> Model:
    m = Model("sliding_cabinet", description="A cabinet with two sliding doors on staggered tracks; they pass each other.",
              params=dict(width=width, height=height, depth=depth))
    carcass(m, width, height, depth)
    dt = 0.012
    overlap = 0.02
    dw = width / 2 + overlap / 2 - GAP
    dh = height - 2 * T - 2 * GAP
    travel = round(width / 2 - overlap - 0.03, 4)
    y_back = -depth / 2 - GAP - dt / 2
    y_front = y_back - dt - GAP
    m.link("door_back", Box((dw, dt, dh), (0, 0, dh / 2)))
    m.link("door_front", Box((dw, dt, dh), (0, 0, dh / 2)))
    x = width / 2 - GAP - dw / 2
    m.joint("door_back_slide", "prismatic", "carcass", "door_back", (-x, y_back, T + GAP), (1, 0, 0), 0.0, travel)
    m.joint("door_front_slide", "prismatic", "carcass", "door_front", (x, y_front, T + GAP), (-1, 0, 0), 0.0, travel)
    return m


def desk(width: float, height: float, depth: float) -> Model:
    m = Model("desk", description="A desk on four legs with one drawer under the top.",
              params=dict(width=width, height=height, depth=depth))
    tt, leg = 0.025, 0.04
    leg_h = height - tt
    lx, ly = width / 2 - leg / 2 - 0.02, depth / 2 - leg / 2 - 0.02
    m.link("frame",
           Box((width, depth, tt), (0, 0, height - tt / 2)),
           *[Box((leg, leg, leg_h), (sx * lx, sy * ly, leg_h / 2)) for sx in (-1, 1) for sy in (-1, 1)])
    dw = min(0.5, width - 2 * (leg + 0.02) - 0.1)
    dh = 0.09
    dd = depth - 0.12
    m.link("drawer", *drawer_shapes(dw, dh, dd))
    m.joint("drawer_slide", "prismatic", "frame", "drawer", (0, -depth / 2 + 0.03, height - tt - GAP - dh), (0, -1, 0), 0.0, round(0.7 * dd, 4))
    return m


# The pilot set: 5 families x 4 sizes = 20 objects.
PILOT = {
    "drawer-cabinet-01": (drawer_cabinet, dict(width=0.45, height=0.6, depth=0.45, drawers=2)),
    "drawer-cabinet-02": (drawer_cabinet, dict(width=0.6, height=0.9, depth=0.5, drawers=3)),
    "drawer-cabinet-03": (drawer_cabinet, dict(width=0.8, height=1.1, depth=0.5, drawers=4)),
    "drawer-cabinet-04": (drawer_cabinet, dict(width=1.2, height=0.75, depth=0.55, drawers=3)),
    "door-cabinet-01": (door_cabinet, dict(width=0.4, height=0.7, depth=0.35, doors=1)),
    "door-cabinet-02": (door_cabinet, dict(width=0.8, height=0.9, depth=0.45, doors=2)),
    "door-cabinet-03": (door_cabinet, dict(width=1.0, height=1.8, depth=0.55, doors=2)),
    "door-cabinet-04": (door_cabinet, dict(width=0.5, height=1.6, depth=0.5, doors=1)),
    "chest-01": (chest, dict(width=0.5, height=0.35, depth=0.35)),
    "chest-02": (chest, dict(width=0.8, height=0.45, depth=0.45)),
    "chest-03": (chest, dict(width=1.1, height=0.5, depth=0.55)),
    "chest-04": (chest, dict(width=0.3, height=0.2, depth=0.22)),
    "sliding-cabinet-01": (sliding_cabinet, dict(width=0.9, height=0.8, depth=0.4)),
    "sliding-cabinet-02": (sliding_cabinet, dict(width=1.2, height=0.9, depth=0.45)),
    "sliding-cabinet-03": (sliding_cabinet, dict(width=1.6, height=0.85, depth=0.5)),
    "sliding-cabinet-04": (sliding_cabinet, dict(width=1.0, height=1.9, depth=0.6)),
    "desk-01": (desk, dict(width=1.0, height=0.74, depth=0.55)),
    "desk-02": (desk, dict(width=1.2, height=0.75, depth=0.6)),
    "desk-03": (desk, dict(width=1.4, height=0.76, depth=0.7)),
    "desk-04": (desk, dict(width=0.8, height=0.72, depth=0.5)),
}


def build(object_id: str, catalog: dict | None = None) -> Model:
    """Build one object by id, from the pilot set or from a generated catalog."""
    fn, params = (catalog or PILOT)[object_id]
    m = fn(**params)
    m.name = object_id
    return m
