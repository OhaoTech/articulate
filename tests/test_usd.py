"""OpenUSD (.usda) emission and usd.sim-ready@1 gate tests.

Pure-Python emitter tests run everywhere (no usd-core needed).
Stage-loading and usd.sim-ready@1 physics-gate tests run automatically
whenever `usd-core` (pxr) is installed (CI / Modal CPU).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import articulate as a  # noqa: E402
from articulate.cli import main as cli_main  # noqa: E402

PILOT_IDS = list(a.PILOT)
SCENE_ID = "room-door-drawer-01"


# ── Pure-Python OpenUSD emitter tests (always run) ───────────────────

def test_usda_header_and_determinism():
    for oid in (PILOT_IDS[0], "drawer-cabinet-02", "door-cabinet-01"):
        m = a.build(oid)
        u1 = a.to_usda(m)
        u2 = a.to_usda(m)
        assert u1 == u2
        assert u1.startswith("#usda 1.0\n")
        assert 'metersPerUnit = 1' in u1
        assert 'upAxis = "Z"' in u1
        assert f'defaultPrim = "{a.usd_ident(oid)}"' in u1
        assert '"PhysicsArticulationRootAPI"' in u1


def test_scene_usda_sanitizes_hyphenated_root_and_has_degrees_and_metres():
    m = a.build(SCENE_ID, a.SCENES)
    u = a.to_usda(m)
    assert 'defaultPrim = "room_door_drawer_01"' in u
    assert 'def PhysicsRevoluteJoint "door_hinge"' in u
    assert 'def PhysicsPrismaticJoint "drawer_slide"' in u
    # Revolute upper limit must be in degrees (~85.94 deg for 1.5 rad), not 1.5.
    door_deg = math.degrees(m.joints[0].upper)
    assert f"float physics:upperLimit = {door_deg:.6f}".rstrip("0").rstrip(".") in u
    # Drawer axis is (0, -1, 0): axis="Y" with 180 deg Z rotation quat (0, 0, 0, 1).
    assert 'uniform token physics:axis = "Y"' in u
    assert "quatf physics:localRot0 = (0, 0, 0, 1)" in u


# ── Real usd-core (pxr) stage + usd.sim-ready@1 gate tests ───────────

def _write_tmp(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return str(p)


@pytest.mark.parametrize("oid", PILOT_IDS)
def test_pilot_usda_loads_and_passes_usd_gate(tmp_path, oid):
    pxr_usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdPhysics

    m = a.build(oid)
    path = _write_tmp(tmp_path, f"{oid}.usda", a.to_usda(m))
    stage = pxr_usd.Stage.Open(path)
    assert stage is not None
    assert stage.GetDefaultPrim().GetName() == a.usd_ident(oid)

    # Check exact rigid body & joint count parity with Model.
    rbs = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(rbs) == len(m.links)
    assert len(joints) == len(m.joints)

    res = a.check_usd(path)
    assert res["verdict"] == "PASS", (oid, res["failures"])
    assert res["metrics"]["rigid_bodies"] == len(m.links)
    assert res["metrics"]["joints"] == len(m.joints)


def test_scene_usda_loads_and_passes_usd_gate(tmp_path):
    pytest.importorskip("pxr.Usd")
    m = a.build(SCENE_ID, a.SCENES)
    path = _write_tmp(tmp_path, f"{SCENE_ID}.usda", a.to_usda(m))
    res = a.check_usd(path)
    assert res["verdict"] == "PASS", res["failures"]
    assert res["metrics"]["rigid_bodies"] == 3
    assert res["metrics"]["joints"] == 2


@pytest.mark.parametrize(
    "breaker_name,expected_rule",
    [
        ("zero_mass", "usd.inertial.mass_positive"),
        ("inverted_limits", "usd.joints.limits"),
        ("missing_collision", "usd.collision.present"),
        ("millimetre_scale", "usd.inertial.inertia_valid"),
        ("tiny_inertia", "usd.inertial.inertia_valid"),
        ("parts_overlap", "usd.geometry.rest_overlap"),
    ],
)
def test_usd_gate_catches_every_breaker_with_typed_rule(tmp_path, breaker_name, expected_rule):
    pytest.importorskip("pxr.Usd")
    m = a.build("drawer-cabinet-02")
    broken = a.BREAKERS[breaker_name](m)
    assert broken is not None
    path = _write_tmp(tmp_path, f"{breaker_name}.usda", a.to_usda(broken))
    res = a.check_usd(path)
    assert res["verdict"] == "FAIL", f"{breaker_name} unexpectedly passed usd.sim-ready@1"
    rules = {f["rule"] for f in res["failures"]}
    assert expected_rule in rules, f"{breaker_name}: expected {expected_rule}, got {rules}"


def test_usd_gate_catches_missing_units_and_nested_rigid_bodies(tmp_path):
    pytest.importorskip("pxr.Usd")
    bad_usda = """#usda 1.0
(
    defaultPrim = "BadBot"
)
def Xform "BadBot"
{
    def Xform "parent" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 2
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        def Cube "c0" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.5
        }
        def Xform "child" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            float physics:mass = 1
            float3 physics:diagonalInertia = (0.05, 0.05, 0.05)
            def Cube "c1" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {
                double size = 0.2
            }
        }
    }
}
"""
    path = _write_tmp(tmp_path, "nested.usda", bad_usda)
    res = a.check_usd(path)
    assert res["verdict"] == "FAIL"
    rules = {f["rule"] for f in res["failures"]}
    assert "usd.units" in rules
    assert "usd.articulation_root" in rules


def test_usd_gate_catches_malformed_usda_with_parse_rule(tmp_path):
    pytest.importorskip("pxr.Usd")
    path = _write_tmp(tmp_path, "broken_syntax.usda", "#usda 1.0\ndef Xform \"Oops\" { bad syntax\n")
    res = a.check_usd(path)
    assert res["verdict"] == "FAIL"
    assert any(f["rule"] == "usd.parse" for f in res["failures"])


def test_cli_check_usd_exit_codes(tmp_path, capsys):
    pytest.importorskip("pxr.Usd")
    m = a.build("chest-01")
    good_p = _write_tmp(tmp_path, "good.usda", a.to_usda(m))
    bad_p = _write_tmp(tmp_path, "bad.usda", a.to_usda(a.BREAKERS["zero_mass"](m)))

    assert cli_main(["check-usd", good_p]) == 0
    assert "PASS" in capsys.readouterr().out

    assert cli_main(["check-usd", bad_p, "--json"]) == 1
    out = capsys.readouterr().out
    assert '"verdict": "FAIL"' in out and "usd.inertial.mass_positive" in out
