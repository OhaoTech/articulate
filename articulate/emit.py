"""Write a Model as URDF, MJCF or glTF 2.0. Deterministic: the same model
always gives the same bytes, so a file's hash names the object."""
from __future__ import annotations

import base64
import json
import math
import struct
from xml.sax.saxutils import quoteattr

from .model import RGBA, Box, Model, Shape


def _f(v: float) -> str:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _v(vs) -> str:
    return " ".join(_f(v) for v in vs)


# Rotation that turns a z-aligned cylinder onto the named axis (URDF rpy).
CYL_RPY = {"z": (0.0, 0.0, 0.0), "x": (0.0, math.pi / 2, 0.0), "y": (math.pi / 2, 0.0, 0.0)}
# The same as MJCF quaternions (w x y z).
_H = math.sqrt(0.5)
CYL_QUAT = {"z": (1.0, 0.0, 0.0, 0.0), "x": (_H, 0.0, _H, 0.0), "y": (_H, _H, 0.0, 0.0)}


def _materials(model: Model) -> list[str]:
    seen: list[str] = []
    for lk in model.links:
        for s in lk.shapes:
            if s.material not in seen:
                seen.append(s.material)
    return seen


# ── URDF ─────────────────────────────────────────────────────────────

def _urdf_geom(s: Shape) -> tuple[str, str]:
    if isinstance(s, Box):
        return f'<origin xyz="{_v(s.at)}" rpy="0 0 0"/>', f'<box size="{_v(s.size)}"/>'
    return (f'<origin xyz="{_v(s.at)}" rpy="{_v(CYL_RPY[s.axis])}"/>',
            f'<cylinder radius="{_f(s.radius)}" length="{_f(s.length)}"/>')


def to_urdf(model: Model) -> str:
    out = ['<?xml version="1.0"?>', f'<robot name={quoteattr(model.name)}>']
    for m in _materials(model):
        out.append(f'  <material name="{m}"><color rgba="{_v(RGBA[m])}"/></material>')
    for lk in model.links:
        it = lk.inertial()
        out.append(f'  <link name={quoteattr(lk.name)}>')
        out.append('    <inertial>')
        out.append(f'      <origin xyz="{_v(it.com)}" rpy="0 0 0"/>')
        out.append(f'      <mass value="{_f(it.mass)}"/>')
        ixx, iyy, izz = it.diag
        out.append(f'      <inertia ixx="{ixx:.6e}" ixy="0" ixz="0" iyy="{iyy:.6e}" iyz="0" izz="{izz:.6e}"/>')
        out.append('    </inertial>')
        for s in lk.shapes:
            origin, geom = _urdf_geom(s)
            out.append(f'    <visual>{origin}<geometry>{geom}</geometry><material name="{s.material}"/></visual>')
            if lk.collision:
                out.append(f'    <collision>{origin}<geometry>{geom}</geometry></collision>')
        out.append('  </link>')
    for j in model.joints:
        out.append(f'  <joint name={quoteattr(j.name)} type="{j.kind}">')
        out.append(f'    <parent link={quoteattr(j.parent)}/>')
        out.append(f'    <child link={quoteattr(j.child)}/>')
        out.append(f'    <origin xyz="{_v(j.origin)}" rpy="0 0 0"/>')
        out.append(f'    <axis xyz="{_v(j.axis)}"/>')
        out.append(f'    <limit lower="{_f(j.lower)}" upper="{_f(j.upper)}" effort="{_f(j.effort)}" velocity="{_f(j.velocity)}"/>')
        out.append(f'    <dynamics damping="{_f(j.damping)}"/>')
        out.append('  </joint>')
    out.append('</robot>')
    return "\n".join(out) + "\n"


# ── MJCF ─────────────────────────────────────────────────────────────

def _mjcf_geom(s: Shape, contype: bool) -> str:
    col = "" if contype else ' contype="0" conaffinity="0"'
    if isinstance(s, Box):
        half = tuple(v / 2 for v in s.size)
        return f'<geom type="box" size="{_v(half)}" pos="{_v(s.at)}" material="{s.material}" mass="0"{col}/>'
    return (f'<geom type="cylinder" size="{_f(s.radius)} {_f(s.length / 2)}" pos="{_v(s.at)}" '
            f'quat="{_v(CYL_QUAT[s.axis])}" material="{s.material}" mass="0"{col}/>')


def to_mjcf(model: Model) -> str:
    out = [f'<mujoco model={quoteattr(model.name)}>',
           '  <compiler angle="radian" inertiafromgeom="false"/>',
           '  <option gravity="0 0 -9.81"/>',
           '  <asset>']
    for m in _materials(model):
        out.append(f'    <material name="{m}" rgba="{_v(RGBA[m])}"/>')
    out.append('  </asset>')
    out.append('  <worldbody>')

    def body(link_name: str, pos, joint, depth: int):
        pad = "  " * depth
        lk = model.get_link(link_name)
        it = lk.inertial()
        out.append(f'{pad}<body name={quoteattr(link_name)} pos="{_v(pos)}">')
        if joint is None:
            out.append(f'{pad}  <freejoint name={quoteattr(link_name + "_free")}/>')
        else:
            kind = "hinge" if joint.kind == "revolute" else "slide"
            out.append(f'{pad}  <joint name={quoteattr(joint.name)} type="{kind}" axis="{_v(joint.axis)}" '
                       f'range="{_f(joint.lower)} {_f(joint.upper)}" limited="true" damping="{_f(joint.damping)}"/>')
        out.append(f'{pad}  <inertial pos="{_v(it.com)}" mass="{_f(it.mass)}" diaginertia="{it.diag[0]:.6e} {it.diag[1]:.6e} {it.diag[2]:.6e}"/>')
        for s in lk.shapes:
            out.append(f'{pad}  {_mjcf_geom(s, lk.collision)}')
        for j in model.children_of(link_name):
            body(j.child, j.origin, j, depth + 1)
        out.append(f'{pad}</body>')

    body(model.root().name, (0.0, 0.0, 0.0), None, 2)
    out.append('  </worldbody>')
    out.append('</mujoco>')
    return "\n".join(out) + "\n"


# ── glTF 2.0 (one embedded buffer) ───────────────────────────────────

def _box_mesh(size, at):
    hx, hy, hz = (v / 2 for v in size)
    pos, nrm, idx = [], [], []
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    for n, quad in faces:
        base = len(pos)
        for sx, sy, sz in quad:
            pos.append((at[0] + sx * hx, at[1] + sy * hy, at[2] + sz * hz))
            nrm.append(n)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nrm, idx


def _cyl_mesh(r, h, at, axis, seg=24):
    # Build along z, then send local (a, b, length) onto output axes perm[0..2].
    # Cyclic permutations only, so triangle winding (and outward normals) survive.
    perm = {"z": (0, 1, 2), "x": (1, 2, 0), "y": (2, 0, 1)}[axis]

    def place(p):
        q = [0.0, 0.0, 0.0]
        for i, k in enumerate(perm):
            q[k] = p[i]
        return q

    pos, nrm, idx = [], [], []
    for i in range(seg + 1):
        a = 2 * math.pi * i / seg
        c, s = math.cos(a), math.sin(a)
        for z in (-h / 2, h / 2):
            p = place((r * c, r * s, z))
            pos.append(tuple(at[k] + p[k] for k in range(3)))
            nrm.append(tuple(place((c, s, 0.0))))
    for i in range(seg):
        a, b = 2 * i, 2 * i + 2
        idx += [a, b, a + 1, a + 1, b, b + 1]
    for zsign in (-1, 1):
        centre = len(pos)
        p = place((0.0, 0.0, zsign * h / 2))
        pos.append(tuple(at[k] + p[k] for k in range(3)))
        nrm.append(tuple(place((0.0, 0.0, float(zsign)))))
        ring = len(pos)
        for i in range(seg + 1):
            a = 2 * math.pi * i / seg
            p = place((r * math.cos(a), r * math.sin(a), zsign * h / 2))
            pos.append(tuple(at[k] + p[k] for k in range(3)))
            nrm.append(tuple(place((0.0, 0.0, float(zsign)))))
        for i in range(seg):
            if zsign > 0:
                idx += [centre, ring + i, ring + i + 1]
            else:
                idx += [centre, ring + i + 1, ring + i]
    return pos, nrm, idx


def to_gltf(model: Model) -> str:
    """glTF 2.0, Y up as the spec requires: the root node turns our Z-up frame."""
    mats = _materials(model)
    blob = bytearray()
    accessors, views, meshes, nodes = [], [], [], []

    def add(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    link_node: dict[str, int] = {}
    for lk in model.links:
        prims = []
        for s in lk.shapes:
            if isinstance(s, Box):
                pos, nrm, idx = _box_mesh(s.size, s.at)
            else:
                pos, nrm, idx = _cyl_mesh(s.radius, s.length, s.at, s.axis)
            pv = add(b"".join(struct.pack("<3f", *p) for p in pos), 34962)
            nv = add(b"".join(struct.pack("<3f", *n) for n in nrm), 34962)
            iv = add(b"".join(struct.pack("<I", i) for i in idx), 34963)
            lo = [min(p[k] for p in pos) for k in range(3)]
            hi = [max(p[k] for p in pos) for k in range(3)]
            accessors.append({"bufferView": pv, "componentType": 5126, "count": len(pos), "type": "VEC3", "min": lo, "max": hi})
            accessors.append({"bufferView": nv, "componentType": 5126, "count": len(nrm), "type": "VEC3"})
            accessors.append({"bufferView": iv, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
            n = len(accessors)
            prims.append({"attributes": {"POSITION": n - 3, "NORMAL": n - 2}, "indices": n - 1, "material": mats.index(s.material)})
        meshes.append({"name": lk.name, "primitives": prims})
        nodes.append({"name": lk.name, "mesh": len(meshes) - 1})
        link_node[lk.name] = len(nodes) - 1

    for j in model.joints:
        nodes[link_node[j.child]]["translation"] = list(j.origin)
        nodes[link_node[j.parent]].setdefault("children", []).append(link_node[j.child])
        nodes[link_node[j.child]]["extras"] = {"joint": {"name": j.name, "type": j.kind, "axis": list(j.axis),
                                                         "lower": j.lower, "upper": j.upper}}
    root = len(nodes)
    nodes.append({"name": model.name, "rotation": [-_H, 0.0, 0.0, _H], "children": [link_node[model.root().name]]})

    doc = {
        "asset": {"version": "2.0", "generator": "niua-articulate"},
        "scene": 0,
        "scenes": [{"nodes": [root]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [{"name": m, "pbrMetallicRoughness": {"baseColorFactor": list(RGBA[m]),
                                                            "metallicFactor": 1.0 if m == "metal" else 0.0,
                                                            "roughnessFactor": 0.4 if m == "metal" else 0.8},
                       **({"alphaMode": "BLEND"} if RGBA[m][3] < 1 else {})} for m in mats],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
    }
    return json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n"
