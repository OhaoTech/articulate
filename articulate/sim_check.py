"""Standalone open Python physics referee for URDF, MJCF, and OpenUSD assets.

Runs completely offline on CPU without any private gateway binary:
  - `.usda` / `.usd` / `.usdc` -> `usd.sim-ready@1` (usd-core / pxr)
  - `.urdf` / `.xml` / `.zip`  -> `articulated.sim-ready@1` (simcheck + MuJoCo)
    plus optional kinematic joint sweep (`urdf.kinematic-sweep@1`, MuJoCo CPU).

Simulation only, no hardware tested.
"""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .usd_check import check_usd

ARTICULATED_CONTRACT = "articulated.sim-ready@1"
SWEEP_TOLERANCE_M = 0.001
DEFAULT_SWEEP_SAMPLES = 48


def check_articulated(path: str, kind: str = "auto", timeout_s: float = 40.0) -> dict[str, Any]:
    """Run articulated.sim-ready@1 on a URDF or MJCF file via simcheck (MuJoCo CPU)."""
    try:
        import simcheck
    except ImportError as exc:
        raise RuntimeError(
            "simcheck + mujoco are required for URDF/MJCF physics checks. "
            "Install with: pip install 'niua-articulate[sim]'"
        ) from exc

    try:
        rep = simcheck.check(path, engines="mujoco", kind=kind, thresholds={"engine_timeout": timeout_s})
        data = rep.to_dict()
    except Exception as exc:
        return {
            "contract": ARTICULATED_CONTRACT,
            "verdict": "FAIL",
            "failures": [{
                "rule": "simcheck.parse",
                "severity": "error",
                "subject": path,
                "expected": "readable URDF or MJCF file",
                "got": f"{type(exc).__name__}: {str(exc)[:240]}",
                "fix": "Ensure the asset is valid URDF/MJCF with positive masses and valid geometry.",
            }],
            "metrics": {},
        }

    failures: list[dict[str, str]] = []
    for res in data.get("results", []):
        for f in res.get("findings", []):
            sev = str(f.get("severity", "error")).lower()
            if sev not in ("error", "warning"):
                continue
            failures.append({
                "rule": f"simcheck.{f.get('check', res.get('id', 'check'))}",
                "severity": sev,
                "subject": str(f.get("target") or os.path.basename(path)),
                "expected": "physics-valid articulated model",
                "got": str(f.get("message", ""))[:300],
                "fix": "Fix the reported geometry, inertial, joint, or collision defect and re-check.",
            })

    has_error = any(f["severity"] == "error" for f in failures) or str(data.get("verdict", "")).upper() != "PASS"
    if has_error and not any(f["severity"] == "error" for f in failures):
        failures.append({
            "rule": "simcheck.no_errors",
            "severity": "error",
            "subject": os.path.basename(path),
            "expected": "simcheck verdict PASS",
            "got": f"verdict={data.get('verdict')}, complete={rep.complete}",
            "fix": "Inspect simcheck findings and fix all physics errors.",
        })

    return {
        "contract": ARTICULATED_CONTRACT,
        "verdict": "FAIL" if has_error else "PASS",
        "failures": failures,
        "metrics": {
            "simcheck": data.get("simcheck_version"),
            "engines": data.get("engines"),
            "score": data.get("score"),
            "complete": bool(rep.complete),
        },
    }


def sweep_urdf(path: str, samples: int = DEFAULT_SWEEP_SAMPLES) -> dict[str, Any]:
    """Kinematic joint range-of-motion collision sweep in MuJoCo (CPU, no gravity/actuators)."""
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "mujoco and numpy are required for --sweep. Install with: pip install 'niua-articulate[sim]'"
        ) from exc

    p = Path(path).resolve()
    tree = ET.parse(p)
    robot = tree.getroot()
    for old in robot.findall("mujoco"):
        robot.remove(old)
    ext = ET.SubElement(robot, "mujoco")
    ET.SubElement(ext, "compiler", {"strippath": "false", "discardvisual": "true", "fusestatic": "false"})

    with tempfile.TemporaryDirectory(prefix="niua-sweep-") as tmp:
        patched = Path(tmp) / "model.urdf"
        tree.write(patched, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(patched))

    data = mujoco.MjData(model)

    def deepest_contact() -> tuple[float, tuple[str, str] | None]:
        mujoco.mj_forward(model, data)
        worst = 0.0
        pair: tuple[str, str] | None = None
        for k in range(data.ncon):
            c = data.contact[k]
            pen = -float(c.dist)
            if pen > worst:
                b1 = int(model.geom_bodyid[c.geom1])
                b2 = int(model.geom_bodyid[c.geom2])
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
                worst = pen
                pair = tuple(sorted((n1, n2)))  # type: ignore[assignment]
        return round(worst, 6), pair

    hinge_t = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide_t = int(mujoco.mjtJoint.mjJNT_SLIDE)

    base_qpos = np.array(model.qpos0, dtype=float)
    for jid in range(model.njnt):
        if bool(model.jnt_limited[jid]):
            adr = int(model.jnt_qposadr[jid])
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            base_qpos[adr] = float(np.clip(0.0, lo, hi))

    data.qpos[:] = base_qpos
    neutral_pen, neutral_pair = deepest_contact()

    joints_out = []
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        if jtype not in (hinge_t, slide_t) or not bool(model.jnt_limited[jid]):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        adr = int(model.jnt_qposadr[jid])
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        qs = np.linspace(lo, hi, max(2, samples))
        max_pen = 0.0
        worst_info = None
        for q in qs:
            data.qpos[:] = base_qpos
            data.qpos[adr] = float(q)
            pen, pair = deepest_contact()
            if pen > max_pen:
                max_pen = pen
                worst_info = {"bodies": list(pair), "at": round(float(q), 4)} if pair else None
        joints_out.append({
            "name": name,
            "type": "revolute" if jtype == hinge_t else "prismatic",
            "unit": "rad" if jtype == hinge_t else "m",
            "lower": round(lo, 6),
            "upper": round(hi, 6),
            "max_penetration_m": max_pen,
            "collides": max_pen > SWEEP_TOLERANCE_M,
            "worst": worst_info,
        })

    return {
        "ok": True,
        "mujoco": mujoco.__version__,
        "mode": "kinematic sweep, collision geometry only, no gravity or actuators",
        "tolerance_m": SWEEP_TOLERANCE_M,
        "neutral": {"max_penetration_m": neutral_pen, "bodies": list(neutral_pair) if neutral_pair else None},
        "joints": joints_out,
    }


def check_any(path: str, sweep: bool = False) -> dict[str, Any]:
    """Dispatch any supported file (.usda/.usd/.usdc/.urdf/.xml) to its physics contract."""
    low = path.lower()
    if low.endswith((".usda", ".usd", ".usdc")):
        return check_usd(path)
    if low.endswith((".urdf", ".xml")):
        res = check_articulated(path)
        if sweep and low.endswith(".urdf") and res["verdict"] == "PASS":
            sw = sweep_urdf(path)
            res["sweep"] = sw
            for j in sw.get("joints", []):
                if j["collides"]:
                    res["verdict"] = "FAIL"
                    res["failures"].append({
                        "rule": "sweep.collision_free",
                        "severity": "error",
                        "subject": j["name"],
                        "expected": f"max penetration <= {SWEEP_TOLERANCE_M} m",
                        "got": f"max penetration {j['max_penetration_m']} m ({j['worst']})",
                        "fix": f"Adjust geometry or joint limits on {j['name']} to eliminate self-collision.",
                    })
        return res
    raise ValueError(f"unsupported file format: {path!r} (expected .usda, .usd, .usdc, .urdf, or .xml)")
