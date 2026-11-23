"""usd.sim-ready@1: CPU-only OpenUSD physics gate validator powered by usd-core (pxr).

Checks an OpenUSD stage (.usda / .usd / .usdc) for real simulator-readiness
issues that break Isaac Sim, Omniverse, and MuJoCo-USD imports:
  - usd.parse                    Stage opens cleanly with a valid defaultPrim
  - usd.units                    Explicit metersPerUnit (positive, metres) & upAxis
  - usd.articulation_root        ArticulationRootAPI present; no nested RigidBodyAPI
  - usd.inertial.mass_positive   Every RigidBodyAPI has finite positive mass
  - usd.inertial.inertia_valid   Diagonal inertia positive, triangle-inequality & scale-plausible
  - usd.collision.present        Every RigidBodyAPI with geometry has PhysicsCollisionAPI
  - usd.joints.targets           Joints connect two distinct existing RigidBodyAPI prims, acyclic
  - usd.joints.limits            Revolute/prismatic limits finite and lower < upper
  - usd.geometry.rest_overlap    No interpenetrating collision bounds between distinct rigid bodies at rest

Simulation only, no hardware tested.
"""
from __future__ import annotations

import math
from typing import Any

CONTRACT = "usd.sim-ready@1"

# Tolerance for rest-pose AABB overlap (metres).
OVERLAP_TOL_M = 1e-4
# Reject placeholder inertia far below a solid sphere/box of the body's bbox scale,
# or far above m * d^2 (catches both tiny_inertia and millimetre_scale).
MAX_GYRATION_RATIO = 1.05
MIN_INERTIA_RATIO = 1e-3


def _fail(rule: str, subject: str, expected: str, got: str, fix: str, severity: str = "error") -> dict[str, str]:
    return {
        "rule": rule,
        "severity": severity,
        "subject": subject,
        "expected": expected,
        "got": got,
        "fix": fix,
    }


def _finite_vec(v) -> bool:
    return v is not None and all(math.isfinite(float(x)) for x in v)


def _aabb_overlap(a_lo, a_hi, b_lo, b_hi, tol: float = OVERLAP_TOL_M) -> bool:
    return all(a_lo[k] < b_hi[k] - tol and b_lo[k] < a_hi[k] - tol for k in range(3))


def check_usd(path: str) -> dict[str, Any]:
    """Run usd.sim-ready@1 on a USD file path. Requires `usd-core` (pxr)."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "usd-core (pxr) is required to run usd.sim-ready@1. Install with: pip install usd-core"
        ) from exc

    failures: list[dict[str, str]] = []
    try:
        stage = Usd.Stage.Open(path)
    except Exception as exc:
        failures.append(_fail(
            "usd.parse",
            path,
            "readable OpenUSD stage",
            f"Usd.Stage.Open raised {type(exc).__name__}: {str(exc).splitlines()[0]}",
            "Fix the OpenUSD syntax or layer error so Usd.Stage.Open succeeds.",
        ))
        return {"contract": CONTRACT, "verdict": "FAIL", "failures": failures, "metrics": {}}
    if stage is None:
        failures.append(_fail(
            "usd.parse",
            path,
            "readable OpenUSD stage",
            "Usd.Stage.Open returned None",
            "Ensure the file is valid .usda/.usdc/.usd with no syntax errors.",
        ))
        return {"contract": CONTRACT, "verdict": "FAIL", "failures": failures, "metrics": {}}

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        failures.append(_fail(
            "usd.parse",
            path,
            "valid defaultPrim metadata",
            "missing or invalid defaultPrim",
            "Author `defaultPrim` on the stage root layer.",
        ))

    # ── Stage units & up-axis ────────────────────────────────────────
    if not stage.HasAuthoredMetadata("metersPerUnit"):
        failures.append(_fail(
            "usd.units",
            "stage",
            "authored metersPerUnit = 1.0",
            "metersPerUnit not authored",
            "Set stage metadata `metersPerUnit = 1`.",
        ))
        mpu = 1.0
    else:
        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(mpu) or mpu <= 0:
            failures.append(_fail(
                "usd.units",
                "stage",
                "positive finite metersPerUnit",
                f"metersPerUnit = {mpu}",
                "Set `metersPerUnit = 1` (metres).",
            ))
            mpu = 1.0

    if not stage.HasAuthoredMetadata("upAxis"):
        failures.append(_fail(
            "usd.units",
            "stage",
            'authored upAxis ("Z" or "Y")',
            "upAxis not authored",
            'Set stage metadata `upAxis = "Z"`.',
        ))

    # ── Discover rigid bodies, articulation roots, joints ────────────
    rigid_bodies: dict[str, Any] = {}
    articulation_roots: list[str] = []
    joints: list[Any] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies[str(prim.GetPath())] = prim
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)

    if not rigid_bodies:
        failures.append(_fail(
            "usd.rigid_bodies",
            "stage",
            "at least one PhysicsRigidBodyAPI prim",
            "0 rigid bodies",
            "Apply UsdPhysics.RigidBodyAPI to each link prim.",
        ))

    if joints and not articulation_roots:
        failures.append(_fail(
            "usd.articulation_root",
            "stage",
            "PhysicsArticulationRootAPI on articulated root",
            "missing ArticulationRootAPI",
            "Apply PhysicsArticulationRootAPI to the root Xform prim.",
        ))

    # No nested RigidBodyAPI prims (PhysX rejects nested rigid bodies).
    for rb_path, prim in rigid_bodies.items():
        anc = prim.GetParent()
        while anc and anc.IsValid() and not anc.IsPseudoRoot():
            if str(anc.GetPath()) in rigid_bodies:
                failures.append(_fail(
                    "usd.articulation_root",
                    rb_path,
                    "sibling rigid bodies (no nested RigidBodyAPI)",
                    f"nested under rigid body {anc.GetPath()}",
                    "Place all RigidBodyAPI link prims as siblings under the articulation root and connect them with PhysicsJoints.",
                ))
                break
            anc = anc.GetParent()

    # ── Per-body mass, inertia, collision, and world AABBs ───────────
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
    )
    body_colliders: dict[str, list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]] = {}

    for rb_path, prim in rigid_bodies.items():
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        diag_attr = mass_api.GetDiagonalInertiaAttr()

        mass = float(mass_attr.Get()) if mass_attr and mass_attr.HasAuthoredValue() else None
        if mass is None or not math.isfinite(mass) or mass <= 0.0:
            failures.append(_fail(
                "usd.inertial.mass_positive",
                rb_path,
                "finite physics:mass > 0",
                f"mass = {mass}",
                "Author a positive physical `float physics:mass` on PhysicsMassAPI.",
            ))

        # Collect visual & collision descendants of this rigid body.
        has_gprim = False
        colliders: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
        for desc in Usd.PrimRange(prim):
            if desc.IsA(UsdGeom.Gprim):
                has_gprim = True
                if desc.HasAPI(UsdPhysics.CollisionAPI):
                    rng = bbox_cache.ComputeWorldBound(desc).ComputeAlignedRange()
                    if not rng.IsEmpty():
                        lo = tuple(float(v) * mpu for v in rng.GetMin())
                        hi = tuple(float(v) * mpu for v in rng.GetMax())
                        colliders.append((str(desc.GetPath()), lo, hi))

        body_colliders[rb_path] = colliders
        if has_gprim and not colliders:
            failures.append(_fail(
                "usd.collision.present",
                rb_path,
                "at least one PhysicsCollisionAPI prim on body with geometry",
                "0 colliders",
                "Apply PhysicsCollisionAPI to the link's collision geometry prims.",
            ))

        # Diagonal inertia validation + physical plausibility vs body bounding box.
        diag = diag_attr.Get() if diag_attr and diag_attr.HasAuthoredValue() else None
        if diag is None or not _finite_vec(diag):
            failures.append(_fail(
                "usd.inertial.inertia_valid",
                rb_path,
                "finite positive physics:diagonalInertia (ixx, iyy, izz)",
                f"diagonalInertia = {diag}",
                "Author `float3 physics:diagonalInertia` computed from the body's geometry and density.",
            ))
        else:
            ix, iy, iz = float(diag[0]), float(diag[1]), float(diag[2])
            if min(ix, iy, iz) <= 0.0:
                failures.append(_fail(
                    "usd.inertial.inertia_valid",
                    rb_path,
                    "all principal moments > 0",
                    f"({ix:.3e}, {iy:.3e}, {iz:.3e})",
                    "Compute positive principal moments of inertia from shape geometry.",
                ))
            elif ix + iy < iz - 1e-9 or iy + iz < ix - 1e-9 or ix + iz < iy - 1e-9:
                failures.append(_fail(
                    "usd.inertial.inertia_valid",
                    rb_path,
                    "triangle inequality (ixx + iyy >= izz, etc.)",
                    f"({ix:.3e}, {iy:.3e}, {iz:.3e})",
                    "Recompute diagonal inertia from real shape volumes; moments violate triangle inequality.",
                ))
            elif mass is not None and mass > 0.0 and colliders:
                lo_all = [min(c[1][k] for c in colliders) for k in range(3)]
                hi_all = [max(c[2][k] for c in colliders) for k in range(3)]
                ext = [hi_all[k] - lo_all[k] for k in range(3)]
                diag_len = math.sqrt(sum(e * e for e in ext))
                max_plausible = mass * (diag_len * MAX_GYRATION_RATIO) ** 2
                # A solid body filling even a fraction of its bounding box has inertia
                # scaling with m * (min_non_thin_extent)^2; compare largest moment against
                # the body's two largest extents to catch placeholder 1e-6 and mm-scale errors.
                ext_sorted = sorted(ext)
                span_sq = ext_sorted[1] ** 2 + ext_sorted[2] ** 2
                min_plausible = mass * span_sq / 12.0 * MIN_INERTIA_RATIO
                if max(ix, iy, iz) > max_plausible and diag_len > 0:
                    failures.append(_fail(
                        "usd.inertial.inertia_valid",
                        rb_path,
                        f"max(I) <= {max_plausible:.4e} kg*m^2 (bounding diagonal {diag_len:.3f} m)",
                        f"max(I) = {max(ix, iy, iz):.4e}",
                        "Recompute diagonalInertia to match the body's real mass and geometry size.",
                    ))
                elif max(ix, iy, iz) < min_plausible and span_sq > 0:
                    failures.append(_fail(
                        "usd.inertial.inertia_valid",
                        rb_path,
                        f"max(I) >= {min_plausible:.4e} kg*m^2 for mass {mass:.3f} kg and extents {[round(e, 3) for e in ext]} m",
                        f"max(I) = {max(ix, iy, iz):.4e} (placeholder inertia or unit-scale mismatch)",
                        "Check metersPerUnit / geometry scale (e.g. mm exported as m) and recompute diagonalInertia from geometry.",
                    ))

    # ── Joints: targets, limits, acyclicity ──────────────────────────
    children_seen: set[str] = set()
    edges: dict[str, list[str]] = {}
    for jp in joints:
        jpath = str(jp.GetPath())
        j = UsdPhysics.Joint(jp)
        b0 = [str(p) for p in j.GetBody0Rel().GetTargets()]
        b1 = [str(p) for p in j.GetBody1Rel().GetTargets()]
        if len(b0) != 1 or len(b1) != 1 or b0[0] not in rigid_bodies or b1[0] not in rigid_bodies or b0[0] == b1[0]:
            failures.append(_fail(
                "usd.joints.targets",
                jpath,
                "distinct valid PhysicsRigidBodyAPI targets on body0 and body1",
                f"body0={b0}, body1={b1}",
                "Point physics:body0 (parent) and physics:body1 (child) to two distinct RigidBodyAPI prims.",
            ))
        else:
            if b1[0] in children_seen:
                failures.append(_fail(
                    "usd.joints.targets",
                    jpath,
                    "tree topology (each child body has one parent joint)",
                    f"{b1[0]} already has a parent joint",
                    "Remove duplicate parent joint or split kinematic loop.",
                ))
            children_seen.add(b1[0])
            edges.setdefault(b0[0], []).append(b1[0])

        if jp.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(jp)
            lo = rj.GetLowerLimitAttr().Get()
            hi = rj.GetUpperLimitAttr().Get()
            if lo is None or hi is None or not (math.isfinite(float(lo)) and math.isfinite(float(hi))) or float(lo) >= float(hi):
                failures.append(_fail(
                    "usd.joints.limits",
                    jpath,
                    "finite lowerLimit < upperLimit (degrees)",
                    f"lower={lo}, upper={hi}",
                    "Set physics:lowerLimit < physics:upperLimit in degrees.",
                ))
        elif jp.IsA(UsdPhysics.PrismaticJoint):
            pj = UsdPhysics.PrismaticJoint(jp)
            lo = pj.GetLowerLimitAttr().Get()
            hi = pj.GetUpperLimitAttr().Get()
            if lo is None or hi is None or not (math.isfinite(float(lo)) and math.isfinite(float(hi))) or float(lo) >= float(hi):
                failures.append(_fail(
                    "usd.joints.limits",
                    jpath,
                    "finite lowerLimit < upperLimit (metres)",
                    f"lower={lo}, upper={hi}",
                    "Set physics:lowerLimit < physics:upperLimit in metres.",
                ))

    # Cycle detection in joint graph.
    visited: dict[str, int] = {}

    def _has_cycle(node: str) -> bool:
        visited[node] = 1
        for nxt in edges.get(node, []):
            st = visited.get(nxt, 0)
            if st == 1 or (st == 0 and _has_cycle(nxt)):
                return True
        visited[node] = 2
        return False

    for rb_path in rigid_bodies:
        if visited.get(rb_path, 0) == 0 and _has_cycle(rb_path):
            failures.append(_fail(
                "usd.joints.targets",
                rb_path,
                "acyclic articulation tree",
                "cycle detected in PhysicsJoint graph",
                "Break the cycle so the articulation forms a tree.",
            ))
            break

    # ── Rest-pose collision overlap between distinct rigid bodies ────
    rb_list = list(body_colliders.items())
    for i, (pa, ca) in enumerate(rb_list):
        for pb, cb in rb_list[i + 1:]:
            hit = next(
                ((na, nb) for na, alo, ahi in ca for nb, blo, bhi in cb if _aabb_overlap(alo, ahi, blo, bhi)),
                None,
            )
            if hit is not None:
                failures.append(_fail(
                    "usd.geometry.rest_overlap",
                    f"{pa} <-> {pb}",
                    f"no collision AABB overlap at rest (tol {OVERLAP_TOL_M} m)",
                    f"collider {hit[0]} overlaps {hit[1]} at rest",
                    "Separate the overlapping rigid body parts so they clear each other at rest.",
                ))

    verdict = "FAIL" if any(f["severity"] == "error" for f in failures) else "PASS"
    return {
        "contract": CONTRACT,
        "verdict": verdict,
        "failures": failures,
        "metrics": {
            "meters_per_unit": mpu,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "rigid_bodies": len(rigid_bodies),
            "joints": len(joints),
            "articulation_roots": len(articulation_roots),
        },
    }
