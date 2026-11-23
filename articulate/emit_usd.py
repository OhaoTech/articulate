"""Write a Model as OpenUSD (.usda) with UsdPhysics schemas.

Deterministic: the same model always produces identical bytes.
Units are metres (metersPerUnit = 1) and Z is up (upAxis = "Z").
Every rigid body sits directly under the articulation root (no nested
RigidBodyAPI prims, which PhysX/UsdPhysics disallows). Joints reference
parent/child bodies via physics:body0 and physics:body1, with revolute
limits in degrees and prismatic limits in metres per the UsdPhysics spec.
"""
from __future__ import annotations

import math
import re

from .model import RGBA, Box, Cylinder, Joint, Model, Shape, Vec


def _f(v: float) -> str:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _fe(v: float) -> str:
    if v == 0.0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e6:
        return f"{v:.6e}"
    return _f(v)


def _v3(vs: Vec) -> str:
    return f"({_f(vs[0])}, {_f(vs[1])}, {_f(vs[2])})"


def _v3e(vs: Vec) -> str:
    return f"({_fe(vs[0])}, {_fe(vs[1])}, {_fe(vs[2])})"


def usd_ident(name: str) -> str:
    """Turn an arbitrary model/link/joint id into a valid USD prim identifier."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = "_" + clean
    return clean


def _materials(model: Model) -> list[str]:
    seen: list[str] = []
    for lk in model.links:
        for s in lk.shapes:
            if s.material not in seen:
                seen.append(s.material)
    return seen


def _world_offsets(model: Model) -> dict[str, Vec]:
    root = model.root().name
    offsets: dict[str, Vec] = {root: (0.0, 0.0, 0.0)}
    todo = [root]
    while todo:
        p = todo.pop()
        for j in model.children_of(p):
            offsets[j.child] = (
                offsets[p][0] + j.origin[0],
                offsets[p][1] + j.origin[1],
                offsets[p][2] + j.origin[2],
            )
            todo.append(j.child)
    return offsets


def _joint_axis_and_rot(axis: Vec) -> tuple[str, tuple[float, float, float, float]]:
    """Map an axis vector to a UsdPhysics axis token ("X"|"Y"|"Z") plus a
    (w, x, y, z) quaternion for localRot0/localRot1 when the axis is negated.
    """
    ax, ay, az = axis
    if abs(ax) >= abs(ay) and abs(ax) >= abs(az):
        return ("X", (1.0, 0.0, 0.0, 0.0) if ax >= 0 else (0.0, 0.0, 0.0, 1.0))
    if abs(ay) >= abs(ax) and abs(ay) >= abs(az):
        return ("Y", (1.0, 0.0, 0.0, 0.0) if ay >= 0 else (0.0, 0.0, 0.0, 1.0))
    return ("Z", (1.0, 0.0, 0.0, 0.0) if az >= 0 else (0.0, 1.0, 0.0, 0.0))


def _q4(q: tuple[float, float, float, float]) -> str:
    return f"({_f(q[0])}, {_f(q[1])}, {_f(q[2])}, {_f(q[3])})"


def _emit_shape(s: Shape, idx: int, root_prim: str, collision: bool) -> list[str]:
    pad = "        "
    mat_path = f"/{root_prim}/Looks/mat_{usd_ident(s.material)}"
    apis = ['"MaterialBindingAPI"']
    if collision:
        apis.insert(0, '"PhysicsCollisionAPI"')
    api_block = f' (\n{pad}    prepend apiSchemas = [{", ".join(apis)}]\n{pad})'

    lines: list[str] = []
    if isinstance(s, Box):
        lines.append(f'{pad}def Cube "shape_{idx}"{api_block}')
        lines.append(f"{pad}{{")
        lines.append(f"{pad}    double size = 1")
        lines.append(f"{pad}    double3 xformOp:translate = {_v3(s.at)}")
        lines.append(f"{pad}    float3 xformOp:scale = {_v3(s.size)}")
        lines.append(f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
        lines.append(f"{pad}    rel material:binding = <{mat_path}>")
        lines.append(f"{pad}}}")
    else:
        assert isinstance(s, Cylinder)
        lines.append(f'{pad}def Cylinder "shape_{idx}"{api_block}')
        lines.append(f"{pad}{{")
        lines.append(f'{pad}    uniform token axis = "{s.axis.upper()}"')
        lines.append(f"{pad}    double radius = {_f(s.radius)}")
        lines.append(f"{pad}    double height = {_f(s.length)}")
        lines.append(f"{pad}    double3 xformOp:translate = {_v3(s.at)}")
        lines.append(f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate"]')
        lines.append(f"{pad}    rel material:binding = <{mat_path}>")
        lines.append(f"{pad}}}")
    return lines


def _emit_joint(j: Joint, root_prim: str) -> list[str]:
    pad = "        "
    jname = usd_ident(j.name)
    body0 = f"/{root_prim}/{usd_ident(j.parent)}"
    body1 = f"/{root_prim}/{usd_ident(j.child)}"
    axis_tok, rot = _joint_axis_and_rot(j.axis)

    if j.kind == "revolute":
        usd_type = "PhysicsRevoluteJoint"
        drive_type = "angular"
        lo = math.degrees(j.lower)
        hi = math.degrees(j.upper)
    elif j.kind == "prismatic":
        usd_type = "PhysicsPrismaticJoint"
        drive_type = "linear"
        lo = j.lower
        hi = j.upper
    else:
        raise ValueError(f"unsupported joint kind for OpenUSD: {j.kind}")

    return [
        f'{pad}def {usd_type} "{jname}" (',
        f'{pad}    prepend apiSchemas = ["PhysicsDriveAPI:{drive_type}"]',
        f"{pad})",
        f"{pad}{{",
        f"{pad}    rel physics:body0 = <{body0}>",
        f"{pad}    rel physics:body1 = <{body1}>",
        f'{pad}    uniform token physics:axis = "{axis_tok}"',
        f"{pad}    point3f physics:localPos0 = {_v3(j.origin)}",
        f"{pad}    point3f physics:localPos1 = (0, 0, 0)",
        f"{pad}    quatf physics:localRot0 = {_q4(rot)}",
        f"{pad}    quatf physics:localRot1 = {_q4(rot)}",
        f"{pad}    float physics:lowerLimit = {_f(lo)}",
        f"{pad}    float physics:upperLimit = {_f(hi)}",
        f"{pad}    float drive:{drive_type}:damping = {_f(j.damping)}",
        f"{pad}    float drive:{drive_type}:maxForce = {_f(j.effort)}",
        f"{pad}    float drive:{drive_type}:stiffness = 0",
        f"{pad}}}",
    ]


def to_usda(model: Model) -> str:
    """Emit a deterministic OpenUSD ASCII (.usda) stage with UsdPhysics schemas."""
    root_prim = usd_ident(model.name)
    offsets = _world_offsets(model)
    mats = _materials(model)

    out: list[str] = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_prim}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "Generated by NIUA Articulate (simulation only, no hardware tested)."',
        ")",
        "",
        f'def Xform "{root_prim}" (',
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI"]',
        ")",
        "{",
        '    def Scope "Looks"',
        "    {",
    ]

    for m in mats:
        r, g, b, a = RGBA[m]
        mname = f"mat_{usd_ident(m)}"
        metallic = "1" if m == "metal" else "0"
        roughness = "0.4" if m == "metal" else "0.8"
        out += [
            f'        def Material "{mname}"',
            "        {",
            f"            token outputs:surface.connect = </{root_prim}/Looks/{mname}/Shader.outputs:surface>",
            '            def Shader "Shader"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = ({_f(r)}, {_f(g)}, {_f(b)})",
            f"                float inputs:metallic = {metallic}",
            f"                float inputs:opacity = {_f(a)}",
            f"                float inputs:roughness = {roughness}",
            "                token outputs:surface",
            "            }",
            "        }",
        ]
    out.append("    }")

    for lk in model.links:
        lname = usd_ident(lk.name)
        it = lk.inertial()
        off = offsets[lk.name]
        out += [
            "",
            f'    def Xform "{lname}" (',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            "    )",
            "    {",
            f"        double3 xformOp:translate = {_v3(off)}",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            f"        float physics:mass = {_f(it.mass)}",
            f"        point3f physics:centerOfMass = {_v3(it.com)}",
            f"        float3 physics:diagonalInertia = {_v3e(it.diag)}",
            "        quatf physics:principalAxes = (1, 0, 0, 0)",
        ]
        for idx, s in enumerate(lk.shapes):
            out += _emit_shape(s, idx, root_prim, lk.collision)
        out.append("    }")

    if model.joints:
        out += ["", '    def Scope "Joints"', "    {"]
        for j in model.joints:
            out += _emit_joint(j, root_prim)
        out.append("    }")

    out.append("}")
    return "\n".join(out) + "\n"
