"""Bridge 3D meshes (GLB/OBJ/STL/PLY) -> sim-ready OpenUSD (.usda) rigid bodies.

Loads any triangular mesh with trimesh, computes real bounding-box extents,
centre of mass, mass, and physically plausible diagonal inertia, and emits a
self-contained OpenUSD (.usda) stage with UsdPhysics.RigidBodyAPI,
UsdPhysics.MassAPI, and UsdPhysics.CollisionAPI that passes `usd.sim-ready@1`.

Simulation only, no hardware tested.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .emit_usd import _f, _fe, _v3, _v3e, usd_ident
from .model import DENSITY


def _load_single_mesh(path: str, scale: float = 1.0):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required for wrap-mesh. Install with: pip install 'niua-articulate[sim]'") from exc

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{path}: scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"{path}: not a non-empty triangle mesh")
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(scale))
    return mesh


def wrap_mesh_to_usda(
    mesh_path: str,
    name: Optional[str] = None,
    mass_kg: Optional[float] = None,
    density: float = DENSITY["wood"],
    scale: float = 1.0,
) -> str:
    """Convert a 3D mesh file into a self-contained sim-ready OpenUSD (.usda) string."""
    mesh = _load_single_mesh(mesh_path, scale=scale)
    root = usd_ident(name or Path(mesh_path).stem)

    bounds = mesh.bounds  # (2, 3)
    ext = [max(float(bounds[1][k] - bounds[0][k]), 1e-4) for k in range(3)]
    centroid = tuple(float((bounds[0][k] + bounds[1][k]) / 2.0) for k in range(3))

    # Prefer watertight mesh volume; fall back to 0.5 * bounding-box volume for open/generated meshes.
    vol = abs(float(mesh.volume)) if mesh.is_volume and abs(float(mesh.volume)) > 1e-9 else (0.5 * ext[0] * ext[1] * ext[2])
    mass = float(mass_kg) if mass_kg is not None else max(0.05, vol * float(density))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"mass must be finite and > 0, got {mass}")

    # Box-equivalent principal moments guarantee positive-definite, triangle-inequality-safe inertia
    # even on non-watertight AI-generated meshes.
    sx, sy, sz = ext
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0

    verts = ", ".join(f"({_f(float(v[0]))}, {_f(float(v[1]))}, {_f(float(v[2]))})" for v in mesh.vertices)
    counts = ", ".join(["3"] * len(mesh.faces))
    indices = ", ".join(str(int(i)) for f in mesh.faces for i in f)
    lo_s = f"({_f(float(bounds[0][0]))}, {_f(float(bounds[0][1]))}, {_f(float(bounds[0][2]))})"
    hi_s = f"({_f(float(bounds[1][0]))}, {_f(float(bounds[1][1]))}, {_f(float(bounds[1][2]))})"

    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "Wrapped by NIUA Articulate (simulation only, no hardware tested)."',
        ")",
        "",
        f'def Xform "{root}" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        ")",
        "{",
        f"    float physics:mass = {_f(mass)}",
        f"    point3f physics:centerOfMass = {_v3(centroid)}",
        f"    float3 physics:diagonalInertia = {_v3e((ixx, iyy, izz))}",
        "    quatf physics:principalAxes = (1, 0, 0, 0)",
        "",
        '    def Mesh "geometry" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        "    )",
        "    {",
        f"        float3[] extent = [{lo_s}, {hi_s}]",
        '        uniform token physics:approximation = "convexHull"',
        f"        int[] faceVertexCounts = [{counts}]",
        f"        int[] faceVertexIndices = [{indices}]",
        f"        point3f[] points = [{verts}]",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"
