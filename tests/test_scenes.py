"""python -m pytest open/articulate/tests/test_scenes.py   (pure Python, no MuJoCo: safe anywhere).

These tests check what a laptop can: the files are well formed, the numbers are
consistent, the output is byte-stable, the broken variants are broken the way
they say, and the door and the drawer clear the room through their whole range
by plain geometry. They do not run the gate or the joint sweep: that is CI's
job in CI (see README). A pass here is not a gate verdict.
"""
import base64
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import articulate as a  # noqa: E402
import build  # noqa: E402
import gate as gt  # noqa: E402
from articulate import scenes  # noqa: E402
from articulate.model import DENSITY, Box, Cylinder  # noqa: E402

SID = "room-door-drawer-01"


def good():
    return a.build(SID, a.SCENES)


# ── geometry helpers (axis-aligned room shapes, a door that turns about Z) ──

def aabbs(m, link):
    """Every shape of a static link as (lo, hi), in the room frame (the root sits at the origin)."""
    out = []
    for s in m.get_link(link).shapes:
        ext = s.size if isinstance(s, Box) else s.extents()
        out.append(([s.at[k] - ext[k] / 2 for k in range(3)], [s.at[k] + ext[k] / 2 for k in range(3)]))
    return out


def overlap(a_lo, a_hi, b_lo, b_hi, tol=1e-9):
    return all(a_lo[k] < b_hi[k] - tol and b_lo[k] < a_hi[k] - tol for k in range(3))


def _rect_hits_aabb(centre, half, theta, lo, hi, margin):
    """Does a rectangle (half extents `half`, turned by theta about the origin, then moved to `centre`)
    come within `margin` of the XY box lo..hi? Separating axis test."""
    c, s = math.cos(theta), math.sin(theta)
    corners = [(centre[0] + sx * half[0] * c - sy * half[1] * s, centre[1] + sx * half[0] * s + sy * half[1] * c)
               for sx in (-1, 1) for sy in (-1, 1)]
    box = [(lo[0] - margin, lo[1] - margin), (hi[0] + margin, hi[1] + margin)]
    for axis in ((1, 0), (0, 1), (c, s), (-s, c)):
        p = [x * axis[0] + y * axis[1] for x, y in corners]
        q = [x * axis[0] + y * axis[1] for x in (box[0][0], box[1][0]) for y in (box[0][1], box[1][1])]
        if max(p) <= min(q) or max(q) <= min(p):
            return False
    return True


def _circle_hits_aabb(centre, r, lo, hi, margin):
    dx = max(lo[0] - centre[0], 0.0, centre[0] - hi[0])
    dy = max(lo[1] - centre[1], 0.0, centre[1] - hi[1])
    return math.hypot(dx, dy) < r + margin


def door_hits(m, theta, margin=0.0):
    """The names of the room shapes (by index) the door touches, or comes within `margin` of, at hinge angle theta."""
    j = next(j for j in m.joints if j.name == "door_hinge")
    hits = []
    for i, (lo, hi) in enumerate(aabbs(m, "room")):
        for s in m.get_link("door").shapes:
            z_lo = j.origin[2] + s.at[2] - (s.size[2] if isinstance(s, Box) else s.length) / 2
            z_hi = j.origin[2] + s.at[2] + (s.size[2] if isinstance(s, Box) else s.length) / 2
            if not (z_lo < hi[2] + margin and lo[2] < z_hi + margin):
                continue
            c, sn = math.cos(theta), math.sin(theta)
            centre = (j.origin[0] + s.at[0] * c - s.at[1] * sn, j.origin[1] + s.at[0] * sn + s.at[1] * c)
            if isinstance(s, Box):
                hit = _rect_hits_aabb(centre, (s.size[0] / 2, s.size[1] / 2), theta, lo, hi, margin)
            else:
                hit = _circle_hits_aabb(centre, s.radius, lo, hi, margin)
            if hit:
                hits.append(i)
    return sorted(set(hits))


def sweep_door(m, steps=400, margin=0.0):
    j = next(j for j in m.joints if j.name == "door_hinge")
    return {k: door_hits(m, j.lower + (j.upper - j.lower) * k / steps, margin) for k in range(steps + 1)}


def drawer_hits(m, steps=100, margin=0.0):
    j = next(j for j in m.joints if j.name == "drawer_slide")
    room = aabbs(m, "room")
    hits = []
    for k in range(steps + 1):
        q = j.lower + (j.upper - j.lower) * k / max(steps, 1)
        for s in m.get_link("drawer").shapes:
            ext = s.size if isinstance(s, Box) else s.extents()
            lo = [j.origin[i] + s.at[i] - ext[i] / 2 + (j.axis[i] * q) - margin for i in range(3)]
            hi = [j.origin[i] + s.at[i] + ext[i] / 2 + (j.axis[i] * q) + margin for i in range(3)]
            hits += [(k, i) for i, (rlo, rhi) in enumerate(room) if overlap(lo, hi, rlo, rhi)]
    return hits


# ── what the scene is ──

def test_the_scene_is_new_and_the_pinned_sets_are_untouched():
    assert list(a.SCENES) == [SID]
    assert SID not in a.PILOT and len(a.PILOT) == 20
    assert len({a.PILOT[i][0] for i in a.PILOT}) == 5 and len(a.FAMILIES) == 5
    assert not any(SID in k for k in a.catalog(5, seed=0))


def test_the_scene_is_a_room_a_hinged_door_and_a_sliding_drawer():
    m = good()
    assert [lk.name for lk in m.links] == ["room", "door", "drawer"]
    assert m.root().name == "room"
    door, drawer = m.joints
    assert (door.name, door.kind, door.parent, door.child, door.axis) == ("door_hinge", "revolute", "room", "door", (0.0, 0.0, 1.0))
    assert (drawer.name, drawer.kind, drawer.parent, drawer.child) == ("drawer_slide", "prismatic", "room", "drawer")
    assert m.name == SID


def test_joint_limits_and_dynamics_are_positive_and_consistent():
    for j in good().joints:
        assert j.lower == 0.0 < j.upper
        assert j.effort > 0 and j.velocity > 0 and j.damping >= 0
    door, drawer = good().joints
    assert 1.0 < door.upper < math.pi          # a door that opens most of the way, not a full turn
    assert 0.1 < drawer.upper < 0.5            # metres: shorter than the drawer is deep


def test_masses_and_inertia_follow_the_geometry():
    m = good()
    for lk in m.links:
        it = lk.inertial()
        x, y, z = it.diag
        assert it.mass > 0 and min(it.diag) > 0
        assert x + y >= z and y + z >= x and x + z >= y, (lk.name, it.diag)
        # mass is density times volume, every shape counted once
        assert it.mass == pytest.approx(sum(DENSITY[s.material] * s.volume() for s in lk.shapes))
    # a 2.4 x 2.0 x 2.4 m wooden room is a tonne or so, a door tens of kilograms, a drawer a few
    assert 500 < m.get_link("room").inertial().mass < 3000
    assert 5 < m.get_link("door").inertial().mass < 100
    assert 0.5 < m.get_link("drawer").inertial().mass < 20


def test_no_shape_is_counted_twice_inside_a_link():
    """Shapes of one link may touch but not overlap, or mass and inertia would double count volume."""
    m = good()
    for lk in m.links:
        boxes = []
        for s in lk.shapes:
            ext = s.size if isinstance(s, Box) else s.extents()
            boxes.append(([s.at[k] - ext[k] / 2 for k in range(3)], [s.at[k] + ext[k] / 2 for k in range(3)]))
        for i, (alo, ahi) in enumerate(boxes):
            for blo, bhi in boxes[i + 1:]:
                assert not overlap(alo, ahi, blo, bhi, tol=1e-9), lk.name


def test_door_and_drawer_clear_the_room_at_rest():
    m = good()
    assert door_hits(m, 0.0) == []
    assert drawer_hits(m, steps=0) == []


def test_door_clears_every_wall_and_the_console_through_its_whole_range():
    """Swept in 400 steps with a 2 mm margin: the 3 mm gap to the jamb holds all the way round."""
    hits = {k: v for k, v in sweep_door(good(), margin=0.002).items() if v}
    assert hits == {}, sorted(hits)[:5]


def test_drawer_clears_the_console_and_room_through_its_whole_range():
    assert drawer_hits(good(), margin=0.002) == []


def test_the_drawer_comes_out_toward_the_room_and_stays_inside_it():
    m = good()
    j = m.joints[1]
    assert j.axis == (0.0, -1.0, 0.0)
    front = j.origin[1] - j.upper      # the drawer front at full travel
    wall_inner = -m.params["depth"] / 2 + scenes.WALL
    assert wall_inner < front < j.origin[1]


# ── the three formats ──

def test_urdf_is_well_formed_and_its_numbers_match_the_model():
    m = good()
    urdf = ET.fromstring(a.to_urdf(m))
    assert urdf.get("name") == SID
    links = {l.get("name"): l for l in urdf.findall("link")}
    assert list(links) == ["room", "door", "drawer"]
    for lk in m.links:
        node = links[lk.name]
        assert float(node.find("inertial/mass").get("value")) == pytest.approx(lk.inertial().mass, rel=1e-5)
        inertia = node.find("inertial/inertia")
        for k, v in zip(("ixx", "iyy", "izz"), lk.inertial().diag):
            assert float(inertia.get(k)) == pytest.approx(v, rel=1e-5)
        assert len(node.findall("collision")) == len(node.findall("visual")) == len(lk.shapes)
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    assert joints["door_hinge"].get("type") == "revolute" and joints["drawer_slide"].get("type") == "prismatic"
    for j in joints.values():
        lim = j.find("limit")
        assert float(lim.get("lower")) < float(lim.get("upper"))
        assert float(lim.get("effort")) > 0 and float(lim.get("velocity")) > 0
    assert all(math.isfinite(float(v)) for v in re.findall(r'="(-?[0-9.]+(?:e[-+]?[0-9]+)?)"', a.to_urdf(m)))


def test_mjcf_has_the_same_bodies_and_joints():
    mj = ET.fromstring(a.to_mjcf(good()))
    assert [b.get("name") for b in mj.findall(".//body")] == ["room", "door", "drawer"]
    assert {j.get("name"): j.get("type") for j in mj.findall(".//joint")} == {"door_hinge": "hinge", "drawer_slide": "slide"}
    assert [j.get("name") for j in mj.findall(".//freejoint")] == ["room_free"]


def test_gltf_is_valid_json_with_the_expected_node_tree():
    g = json.loads(a.to_gltf(good()))
    nodes = g["nodes"]
    by_name = {n["name"]: n for n in nodes}
    root = nodes[g["scenes"][0]["nodes"][0]]
    assert root["name"] == SID
    assert [nodes[i]["name"] for i in root["children"]] == ["room"]
    assert [nodes[i]["name"] for i in by_name["room"]["children"]] == ["door", "drawer"]
    assert by_name["door"]["extras"]["joint"]["name"] == "door_hinge"
    assert by_name["drawer"]["extras"]["joint"]["name"] == "drawer_slide"
    assert by_name["door"]["extras"]["joint"]["upper"] == scenes.DOOR_OPEN
    blob = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    assert len(blob) == g["buffers"][0]["byteLength"]
    assert all(v["byteOffset"] + v["byteLength"] <= len(blob) for v in g["bufferViews"])
    assert len(g["meshes"]) == 3 and all(n["mesh"] < 3 for n in by_name.values() if "mesh" in n)
    for acc in g["accessors"]:
        assert acc["bufferView"] < len(g["bufferViews"]) and acc["count"] > 0


# ── the writer ──

def test_the_writer_is_byte_deterministic():
    with tempfile.TemporaryDirectory() as x, tempfile.TemporaryDirectory() as y:
        build.build_scenes(x)
        build.build_scenes(y)
        seen = 0
        for root, _, files in os.walk(x):
            for f in files:
                p = os.path.join(root, f)
                assert open(p, "rb").read() == open(os.path.join(y, os.path.relpath(p, x)), "rb").read(), p
                seen += 1
        assert seen == 1 + 1 + 4 + 8   # index.json, object.json, four formats, four broken variants (urdf + usda)
    assert a.to_urdf(good()) == a.to_urdf(good()) and a.to_gltf(good()) == a.to_gltf(good()) and a.to_usda(good()) == a.to_usda(good())


def test_the_writer_lays_out_the_files_and_hashes_them():
    import hashlib
    with tempfile.TemporaryDirectory() as d:
        objs = build.build_scenes(d)
        assert len(objs) == 1
        idx = json.load(open(os.path.join(d, "index.json")))
        assert idx["objects"] == [SID]
        meta = json.load(open(os.path.join(d, SID, "object.json")))
        assert meta == objs[0] and meta["kind"] == "scene" and meta["generated"] is False
        assert set(meta["files"]) == {"model.urdf", "model.xml", "model.gltf", "model.usda"}
        assert set(meta["broken"]) == {"zero_mass", "inverted_limits", "missing_collision", "door_into_console"}
        assert set(meta["broken_usd"]) == {"zero_mass", "inverted_limits", "missing_collision", "door_into_console"}
        assert meta["sweep"] == {"good": "collision_free", "broken": {"door_into_console": "door_hinge"}}
        for name, digest in meta["files"].items():
            assert hashlib.sha256(open(os.path.join(d, SID, name), "rb").read()).hexdigest() == digest
        for name, digest in meta["broken"].items():
            assert hashlib.sha256(open(os.path.join(d, SID, "broken", f"{name}.urdf"), "rb").read()).hexdigest() == digest
            assert digest != meta["files"]["model.urdf"]
        for name, digest in meta["broken_usd"].items():
            assert hashlib.sha256(open(os.path.join(d, SID, "broken", f"{name}.usda"), "rb").read()).hexdigest() == digest
            assert digest != meta["files"]["model.usda"]
        assert [j["name"] for j in meta["joints"]] == ["door_hinge", "drawer_slide"]


def test_the_object_writer_is_unchanged_for_objects():
    """Scenes added keys of their own; the pilot's object.json must not gain any."""
    with tempfile.TemporaryDirectory() as d:
        meta = build.build_all(d)[0]
        assert "kind" not in meta and "sweep" not in meta


def test_the_cli_writes_scenes():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "build.py"), d, "--scenes"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert "1 scenes, 4 broken variants" in p.stdout
        assert os.path.exists(os.path.join(d, SID, "model.urdf"))


# ── the broken variants are broken the way they say ──

def test_zero_mass_zeroes_the_door():
    b = scenes.SCENE_BREAKERS["zero_mass"](good())
    assert b.get_link("door").inertial().mass == 0 and good().get_link("door").inertial().mass > 0
    assert 'mass value="0"' in a.to_urdf(b)


def test_inverted_limits_swaps_the_door_limits():
    j = scenes.SCENE_BREAKERS["inverted_limits"](good()).joints[0]
    assert j.name == "door_hinge" and j.lower > j.upper


def test_missing_collision_takes_the_collision_off_the_door():
    urdf = ET.fromstring(a.to_urdf(scenes.SCENE_BREAKERS["missing_collision"](good())))
    door = next(l for l in urdf.findall("link") if l.get("name") == "door")
    assert door.findall("visual") and not door.findall("collision")
    room = next(l for l in urdf.findall("link") if l.get("name") == "room")
    assert room.findall("collision")


def test_door_into_console_touches_nothing_at_rest_and_only_the_sweep_finds_it():
    m = good()
    b = scenes.SCENE_BREAKERS["door_into_console"](m)
    assert len(b.get_link("room").shapes) == len(m.get_link("room").shapes) + 1
    assert door_hits(b, 0.0) == [] and drawer_hits(b, steps=0) == []          # a rest check sees nothing
    hits = {k: v for k, v in sweep_door(b).items() if v}
    assert hits, "the sweep must find the console"
    new = len(m.get_link("room").shapes)
    assert all(v == [new] for v in hits.values())                              # and it is the console it finds
    # the drawer is not what is wrong
    assert drawer_hits(b, margin=0.002) == []
    # the good scene is untouched by the breaker
    assert len(m.get_link("room").shapes) == new


def test_door_into_console_goes_deep_enough_to_count():
    """The sweep's tolerance is 1 mm: the console must sit well inside the arc, not graze it."""
    b = scenes.SCENE_BREAKERS["door_into_console"](good())
    j = b.joints[0]
    assert any(door_hits(b, j.upper * k / 400, margin=-0.05) for k in range(401)), "no 5 cm overlap anywhere in the swing"


def test_door_into_console_does_not_apply_without_a_door():
    assert scenes.door_into_console(a.build("drawer-cabinet-02")) is None


# ── gate.py, scenes mode, with a fake checker and a fake sweep ──

def fake_run_gate(monkeypatch):
    """Contracts: good files pass; broken ones fail, except door_into_console, which only the sweep can see."""
    def run(gate, contract, paths, fields=gt.FIELDS, log=None):
        out = {}
        for p in paths:
            broken = os.sep + "broken" + os.sep in p
            name = os.path.basename(p)[:-5]
            fail = broken and name != "door_into_console"
            out[p] = {"verdict": "FAIL" if fail else "PASS",
                      "failures": [{"rule": "x.y", "severity": "error"}] if fail else []}
        return out
    monkeypatch.setattr(gt, "run_gate", run)


def fake_sweep(monkeypatch, found=True):
    """A urdf_sweep report built from the geometric sweep above (and a lie when found is False)."""
    def run(script, urdf_path, python=None):
        broken = os.sep + "broken" + os.sep in urdf_path
        m = good()
        if broken:
            m = scenes.SCENE_BREAKERS["door_into_console"](m)
        door = bool(broken and found) or (not broken and not found)
        joints = [{"name": "door_hinge", "type": "revolute", "unit": "rad", "lower": 0.0, "upper": scenes.DOOR_OPEN, "samples": 48,
                   "max_penetration_m": 0.05 if door else 0.0, "worst": {"bodies": ["door", "room"], "at": 0.7} if door else None, "collides": door},
                  {"name": "drawer_slide", "type": "prismatic", "unit": "m", "lower": 0.0, "upper": 0.231, "samples": 48,
                   "max_penetration_m": 0.0, "worst": None, "collides": False}]
        return {"ok": True, "mujoco": "x", "mode": "kinematic sweep", "tolerance_m": 0.001,
                "neutral": {"max_penetration_m": 0.0, "bodies": None}, "joints": joints}
    monkeypatch.setattr(gt, "run_sweep", run)


def test_gate_scenes_mode_passes_when_contracts_and_sweep_agree(monkeypatch, capsys):
    fake_run_gate(monkeypatch)
    fake_sweep(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        build.build_scenes(d)
        assert gt.main(d, "gate_check", sweep="urdf_sweep.py") == 0
        out = capsys.readouterr().out
        assert "NOTE room-door-drawer-01/broken/door_into_console: passes every contract; the sweep is what catches it" in out
        assert "door_hinge" in out and "COLLIDES" in out and "collision_free" in out
        doc = json.load(open(os.path.join(d, "gate.json")))
        assert doc["objects"][SID]["good"]["model.urdf"]["urdf.sim-ready@1"]["verdict"] == "PASS"
        assert set(doc["sweeps"][SID]) == {"good", "door_into_console"}
        assert os.path.exists(os.path.join(d, SID, "sweep.json"))


def test_gate_scenes_mode_fails_when_the_sweep_misses_the_broken_scene(monkeypatch, capsys):
    fake_run_gate(monkeypatch)
    fake_sweep(monkeypatch, found=False)
    with tempfile.TemporaryDirectory() as d:
        build.build_scenes(d)
        assert gt.main(d, "gate_check", sweep="urdf_sweep.py") == 1
        out = capsys.readouterr().out
        assert "MISMATCH" in out and "expected door_hinge to collide" in out and "good scene, joint door_hinge collides" in out


def test_gate_without_sweep_treats_a_contract_miss_on_a_sweep_variant_as_fine(monkeypatch):
    fake_run_gate(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        build.build_scenes(d)
        assert gt.main(d, "gate_check") == 0
        assert "sweeps" not in json.load(open(os.path.join(d, "gate.json")))


def test_gate_still_fails_a_contract_miss_on_an_ordinary_broken_variant(monkeypatch, capsys):
    def run(gate, contract, paths, fields=gt.FIELDS, log=None):
        return {p: {"verdict": "PASS", "failures": []} for p in paths}
    monkeypatch.setattr(gt, "run_gate", run)
    with tempfile.TemporaryDirectory() as d:
        build.build_scenes(d)
        assert gt.main(d, "gate_check") == 1
        out = capsys.readouterr().out
        assert "broken/zero_mass: expected FAIL" in out and "broken/door_into_console: expected FAIL" not in out


# ── the CI wiring (CI is the authority; this only keeps the wiring from rotting) ──

def test_ci_builds_gates_and_sweeps_the_scene():
    # Monorepo only: the published open repo has no workflow file to check.
    wf_path = os.path.join(ROOT, "..", "..", ".github", "workflows", "deploy.yml")
    if not os.path.exists(wf_path):
        pytest.skip("no .github/workflows/deploy.yml (not the monorepo)")
    wf = open(wf_path, encoding="utf-8").read()
    assert 'build.py "$RUNNER_TEMP/articulate-scene" --scenes' in wf
    assert 'gate.py "$RUNNER_TEMP/articulate-scene" niua-core/target/debug/examples/gate_check --sweep niua-core/sim/urdf_sweep.py' in wf
    assert "- 'niua-core/sim/urdf_sweep.py'" in wf
