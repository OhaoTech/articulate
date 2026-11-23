"""Tests for unified check (OpenUSD + URDF/MuJoCo + sweep) and wrap-mesh (3D prop -> OpenUSD)."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import articulate as a  # noqa: E402
from articulate.cli import main as cli_main  # noqa: E402
from articulate.sim_check import check_any  # noqa: E402
from articulate.wrap_mesh import wrap_mesh_to_usda  # noqa: E402


def test_wrap_mesh_glb_to_sim_ready_openusd_passes_gate(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pxr.Usd")

    box = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    glb_path = tmp_path / "crate-prop.glb"
    box.export(str(glb_path))

    usda_path = tmp_path / "crate-prop.usda"
    rc = cli_main(["wrap-mesh", str(glb_path), "--out", str(usda_path), "--mass", "3.5", "--verify"])
    assert rc == 0
    assert usda_path.is_file()
    res = a.check_usd(str(usda_path))
    assert res["verdict"] == "PASS", res["failures"]
    assert res["metrics"]["rigid_bodies"] == 1


def test_unified_check_urdf_with_sweep_passes_good_scene_and_catches_console_collision(tmp_path):
    pytest.importorskip("simcheck")
    pytest.importorskip("mujoco")

    m = a.build("room-door-drawer-01", a.SCENES)
    good_urdf = tmp_path / "good.urdf"
    good_urdf.write_text(a.to_urdf(m), encoding="utf-8", newline="\n")

    bad_m = a.SCENE_BREAKERS["door_into_console"](m)
    bad_urdf = tmp_path / "door_into_console.urdf"
    bad_urdf.write_text(a.to_urdf(bad_m), encoding="utf-8", newline="\n")

    res_good = check_any(str(good_urdf), sweep=True)
    assert res_good["verdict"] == "PASS", res_good["failures"]
    assert all(not j["collides"] for j in res_good["sweep"]["joints"])

    res_bad = check_any(str(bad_urdf), sweep=True)
    assert res_bad["verdict"] == "FAIL"
    assert any(f["rule"] == "sweep.collision_free" and f["subject"] == "door_hinge" for f in res_bad["failures"])
