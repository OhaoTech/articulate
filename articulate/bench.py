"""The benchmark: plain requests an agent answers with a URDF, scored two ways.

1. Spec: is it the object that was asked for? Overall size within tolerance,
   standing on the floor, the right number of moving joints of the right kind
   moving the right way, with a useful range.
2. Gate: does it pass `articulated.sim-ready@1` (simcheck in MuJoCo)? That part
   runs in gate.py's checker; this module only does the spec.

A task passes when both hold. Tasks are drawn from a different seed than the
dataset (seed 1 vs 0), so no task is a copy of a dataset object.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from . import families as f
from .generate import catalog

BENCH_SEED = 1
FRAME = ("Use metres. Put the origin on the floor under the middle of the object: "
         "X across the width, Y toward the back (the front faces -Y), Z up.")


def _cm(x: float) -> str:
    return f"{round(x * 100)} cm"


def _size(p: dict) -> str:
    return f"{_cm(p['width'])} wide, {_cm(p['height'])} tall and {_cm(p['depth'])} deep"


def _joints(fn, p: dict) -> tuple[str, list[dict]]:
    """What the request says about the moving parts, and the spec it implies."""
    if fn is f.drawer_cabinet:
        n = p["drawers"]
        return (f"It has {n} drawers stacked one above the other that slide out toward the front.",
                [{"type": "prismatic", "axis": "y", "count": n, "min_range": 0.2}])
    if fn is f.door_cabinet:
        n = p["doors"]
        what = "one door" if n == 1 else "two doors that meet in the middle"
        return (f"It has {what}, hinged on a vertical edge, swinging open outward.",
                [{"type": "revolute", "axis": "z", "count": n, "min_range": 1.2}])
    if fn is f.chest:
        return ("Its lid is hinged along the back edge and swings up to open.",
                [{"type": "revolute", "axis": "x", "count": 1, "min_range": 1.2}])
    if fn is f.sliding_cabinet:
        return ("It has two sliding doors on separate front tracks that slide sideways past each other.",
                [{"type": "prismatic", "axis": "x", "count": 2, "min_range": 0.2}])
    if fn is f.desk:
        return ("It stands on four legs and has one drawer under the top that slides out toward the front.",
                [{"type": "prismatic", "axis": "y", "count": 1, "min_range": 0.2}])
    raise ValueError(fn)


NOUN = {f.drawer_cabinet: "a drawer cabinet", f.door_cabinet: "a cabinet with hinged doors",
        f.chest: "a chest with a lid", f.sliding_cabinet: "a cabinet with sliding doors", f.desk: "a desk"}


def tasks(per_family: int, seed: int = BENCH_SEED) -> list[dict]:
    out = []
    for oid, (fn, p) in catalog(per_family, seed).items():
        moving, joints = _joints(fn, p)
        out.append({
            "id": f"bench-{oid}",
            "prompt": f"Write a URDF for {NOUN[fn]} {_size(p)}. {moving} {FRAME} "
                      "Give every link real mass, inertia and collision geometry.",
            "spec": {"size": [p["width"], p["depth"], p["height"]], "joints": joints},
            "reference": {"family": fn.__name__, "params": p},
        })
    return out


# ---- reading a URDF ---------------------------------------------------------

def _vec(s: str | None, n: int = 3) -> list[float]:
    if not s:
        return [0.0] * n
    parts = [float(x) for x in s.split()]
    if len(parts) != n:
        raise ValueError(f"expected {n} numbers, got {len(parts)}: {s!r}")
    return parts


def _rot(rpy) -> list[list[float]]:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _apply(R, t, v):
    return [sum(R[i][k] * v[k] for k in range(3)) + t[i] for i in range(3)]


def _half_extents(geom: ET.Element):
    if (b := geom.find("box")) is not None:
        return [x / 2 for x in _vec(b.get("size"))]
    if (c := geom.find("cylinder")) is not None:
        r, l = float(c.get("radius")), float(c.get("length"))
        return [r, r, l / 2]
    if (s := geom.find("sphere")) is not None:
        r = float(s.get("radius"))
        return [r, r, r]
    return None   # a mesh: size unknown without loading it


def extent(urdf: str):
    """World-space bounds of every visual and collision primitive at rest (joints at 0).
    Returns (lo, hi, unknown) where unknown counts shapes it could not size (meshes)."""
    root = ET.fromstring(urdf)
    links = {lk.get("name"): lk for lk in root.findall("link")}
    joints = root.findall("joint")
    for j in joints:
        for end in ("parent", "child"):
            name = j.find(end).get("link")
            if name not in links:
                raise ValueError(f"joint {j.get('name')!r} names {end} link {name!r}, which is not defined")
    children = {j.find("child").get("link") for j in joints}
    base = [n for n in links if n not in children]
    if len(base) != 1:
        raise ValueError(f"expected one root link, found {len(base)}")
    pose = {base[0]: ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])}
    todo = [base[0]]
    while todo:
        p = todo.pop()
        R, t = pose[p]
        for j in joints:
            if j.find("parent").get("link") != p:
                continue
            o = j.find("origin")
            jr = _rot(_vec(o.get("rpy")) if o is not None else [0, 0, 0])
            jt = _vec(o.get("xyz")) if o is not None else [0.0] * 3
            c = j.find("child").get("link")
            pose[c] = (_mul(R, jr), _apply(R, t, jt))
            todo.append(c)
    lo, hi, unknown = [math.inf] * 3, [-math.inf] * 3, 0
    for name, lk in links.items():
        if name not in pose:
            continue
        R, t = pose[name]
        for el in lk.findall("visual") + lk.findall("collision"):
            g = el.find("geometry")
            h = _half_extents(g) if g is not None else None
            if h is None:
                unknown += 1
                continue
            o = el.find("origin")
            er = _rot(_vec(o.get("rpy")) if o is not None else [0, 0, 0])
            et = _vec(o.get("xyz")) if o is not None else [0.0] * 3
            RR, tt = _mul(R, er), _apply(R, t, et)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        w = _apply(RR, tt, [sx * h[0], sy * h[1], sz * h[2]])
                        for k in range(3):
                            lo[k], hi[k] = min(lo[k], w[k]), max(hi[k], w[k])
    return lo, hi, unknown


AXES = {"x": 0, "y": 1, "z": 2}


def check_spec(task: dict, urdf: str) -> list[dict]:
    """Every way the URDF is not the object the task asked for. Empty means it matches."""
    out = []
    try:
        root = ET.fromstring(urdf)
        lo, hi, unknown = extent(urdf)
    except (ET.ParseError, ValueError, AttributeError, TypeError, IndexError) as e:
        return [{"rule": "spec.parse", "got": str(e)[:200]}]
    spec = task["spec"]
    if unknown:
        out.append({"rule": "spec.primitives", "got": f"{unknown} shapes are meshes; size can't be checked"})
    else:
        for k, (axis, want) in enumerate(zip("xyz", [spec["size"][0], spec["size"][1], spec["size"][2]])):
            got = hi[k] - lo[k]
            tol = max(0.05, 0.1 * want)
            if abs(got - want) > tol:
                out.append({"rule": "spec.size", "subject": axis, "expected": round(want, 3), "got": round(got, 3)})
        if abs(lo[2]) > 0.02:
            out.append({"rule": "spec.on_floor", "expected": 0.0, "got": round(lo[2], 3)})
    try:
        return out + _joint_rules(root, spec)
    except (ValueError, AttributeError, TypeError, IndexError) as e:
        # A joint axis or limit that is not numbers (or not 3 of them) is a parse failure,
        # not a crash of the whole scoring run.
        return out + [{"rule": "spec.parse", "got": str(e)[:200]}]


def _joint_rules(root, spec: dict) -> list[dict]:
    """The joint count, axis and range rules of check_spec. Raises on an unreadable joint."""
    out = []
    moving = [j for j in root.findall("joint") if j.get("type") not in ("fixed", None)]
    for want in spec["joints"]:
        hits = []
        for j in moving:
            if j.get("type") != want["type"] and not (want["type"] == "revolute" and j.get("type") == "continuous"):
                continue
            a = _vec(j.find("axis").get("xyz")) if j.find("axis") is not None else [1.0, 0.0, 0.0]
            n = math.sqrt(sum(x * x for x in a)) or 1.0
            if abs(a[AXES[want["axis"]]] / n) < 0.95:
                continue
            hits.append(j)
        if len(hits) != want["count"]:
            out.append({"rule": "spec.joints", "subject": f"{want['type']} along {want['axis']}",
                        "expected": want["count"], "got": len(hits)})
        for j in hits:
            lim = j.find("limit")
            if j.get("type") == "continuous":
                continue
            rng = float(lim.get("upper", 0)) - float(lim.get("lower", 0)) if lim is not None else 0.0
            if rng < want["min_range"]:
                out.append({"rule": "spec.range", "subject": j.get("name"),
                            "expected": f">= {want['min_range']}", "got": round(rng, 3)})
    extra = len(moving) - sum(w["count"] for w in spec["joints"])
    if extra > 0:
        out.append({"rule": "spec.joints", "subject": "other moving joints", "expected": 0, "got": extra})
    return out
