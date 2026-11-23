"""python -m pytest open/articulate   (pure Python, no MuJoCo: safe anywhere)."""
import base64
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import articulate as a  # noqa: E402
import build  # noqa: E402
from articulate.model import Box, Cylinder  # noqa: E402

IDS = list(a.PILOT)
GEN = a.catalog(40, seed=0)   # 200 generated objects: the same checks as the pilot
GIDS = list(GEN)
CAT = {**a.PILOT, **GEN}


def world_boxes(m):
    """Every shape at rest as a world-space AABB, tagged with its link."""
    offset = {m.root().name: (0.0, 0.0, 0.0)}
    todo = [m.root().name]
    while todo:
        p = todo.pop()
        for j in m.children_of(p):
            offset[j.child] = tuple(offset[p][k] + j.origin[k] for k in range(3))
            todo.append(j.child)
    out = []
    for lk in m.links:
        o = offset[lk.name]
        for s in lk.shapes:
            ext = s.size if isinstance(s, Box) else s.extents()
            c = [o[k] + s.at[k] for k in range(3)]
            out.append((lk.name, [c[k] - ext[k] / 2 for k in range(3)], [c[k] + ext[k] / 2 for k in range(3)]))
    return out


def overlap(a_lo, a_hi, b_lo, b_hi, tol=1e-9):
    return all(a_lo[k] < b_hi[k] - tol and b_lo[k] < a_hi[k] - tol for k in range(3))


def test_the_pilot_is_20_objects_in_5_families():
    assert len(IDS) == 20
    assert len({a.PILOT[i][0] for i in IDS}) == 5


@pytest.mark.parametrize("oid", IDS + GIDS[::10])
def test_every_format_parses_and_names_match(oid):
    m = a.build(oid, CAT)
    urdf = ET.fromstring(a.to_urdf(m))
    assert [l.get("name") for l in urdf.findall("link")] == [lk.name for lk in m.links]
    assert len(urdf.findall("joint")) == len(m.joints) >= 1
    mjcf = ET.fromstring(a.to_mjcf(m))
    assert len(mjcf.findall(".//body")) == len(m.links)
    assert {j.get("name") for j in mjcf.findall(".//joint")} == {j.name for j in m.joints}
    g = json.loads(a.to_gltf(m))
    blob = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])
    assert len(blob) == g["buffers"][0]["byteLength"]
    assert all(v["byteOffset"] + v["byteLength"] <= len(blob) for v in g["bufferViews"])
    assert {n["name"] for n in g["nodes"]} >= {lk.name for lk in m.links}


@pytest.mark.parametrize("oid", IDS + GIDS)
def test_one_root_and_unique_names(oid):
    m = a.build(oid, CAT)
    m.root()
    assert len({lk.name for lk in m.links}) == len(m.links)
    assert len({j.name for j in m.joints}) == len(m.joints)


@pytest.mark.parametrize("oid", IDS + GIDS)
def test_nothing_touches_at_rest(oid):
    """Parts of different links never overlap in the closed pose (all shapes are axis-aligned, so AABBs are exact)."""
    boxes = world_boxes(a.build(oid, CAT))
    for i, (la, lo_a, hi_a) in enumerate(boxes):
        for lb, lo_b, hi_b in boxes[i + 1:]:
            if la != lb:
                assert not overlap(lo_a, hi_a, lo_b, hi_b), f"{oid}: {la} overlaps {lb}"


@pytest.mark.parametrize("oid", IDS + GIDS)
def test_physics_follows_geometry(oid):
    for lk in a.build(oid, CAT).links:
        it = lk.inertial()
        assert 0.05 < it.mass < 200, (lk.name, it.mass)
        x, y, z = it.diag
        assert min(it.diag) > 0
        assert x + y >= z and y + z >= x and x + z >= y, (lk.name, it.diag)


def test_box_and_cylinder_inertia_match_the_textbook():
    from articulate.model import Link
    b = Link("b", [Box((0.2, 0.4, 0.6))]).inertial()
    m = 600 * 0.2 * 0.4 * 0.6
    assert b.mass == pytest.approx(m)
    assert b.diag == pytest.approx((m * (0.16 + 0.36) / 12, m * (0.04 + 0.36) / 12, m * (0.04 + 0.16) / 12))
    c = Link("c", [Cylinder(0.1, 0.5, axis="x", material="wood")]).inertial()
    mc = c.mass
    assert c.diag == pytest.approx((mc * 0.01 / 2, mc * (0.03 + 0.25) / 12, mc * (0.03 + 0.25) / 12))


def test_parallel_axis_two_boxes():
    from articulate.model import Link
    one = Box((0.1, 0.1, 0.1), (-0.5, 0, 0))
    two = Box((0.1, 0.1, 0.1), (0.5, 0, 0))
    it = Link("l", [one, two]).inertial()
    m = it.mass / 2
    assert it.com == pytest.approx((0, 0, 0))
    own = m * 0.02 / 12
    assert it.diag == pytest.approx((2 * own, 2 * (own + m * 0.25), 2 * (own + m * 0.25)))


def test_build_is_deterministic():
    with tempfile.TemporaryDirectory() as x, tempfile.TemporaryDirectory() as y:
        build.build_all(x)
        build.build_all(y)
        for root, _, files in os.walk(x):
            for f in files:
                p = os.path.join(root, f)
                q = os.path.join(y, os.path.relpath(p, x))
                assert open(p, "rb").read() == open(q, "rb").read(), p


def test_every_object_has_its_broken_variants():
    with tempfile.TemporaryDirectory() as d:
        objs = build.build_all(d)
        for o in objs:
            assert set(o["broken"]) >= {"zero_mass", "inverted_limits", "missing_collision", "millimetre_scale", "tiny_inertia"}
            for name, digest in o["broken"].items():
                assert digest != o["files"]["model.urdf"], (o["id"], name)
        multi = {o["id"] for o in objs if "parts_overlap" in o["broken"]}
        assert "drawer-cabinet-02" in multi and "chest-01" not in multi


def test_breakers_change_what_they_say():
    m = a.build("drawer-cabinet-02")
    assert a.BREAKERS["zero_mass"](m).get_link("drawer_1").inertial().mass == 0
    j = a.BREAKERS["inverted_limits"](m).joints[0]
    assert j.lower > j.upper
    assert not a.BREAKERS["missing_collision"](m).get_link("drawer_1").collision
    big = a.BREAKERS["millimetre_scale"](m)
    assert big.get_link("carcass").inertial().mass == pytest.approx(m.get_link("carcass").inertial().mass)
    assert max(b.size[2] for b in big.get_link("carcass").shapes) == pytest.approx(900)
    assert a.BREAKERS["tiny_inertia"](m).get_link("drawer_1").inertial().diag == (1e-6, 1e-6, 1e-6)
    over = a.BREAKERS["parts_overlap"](m)
    assert over.joints[0].origin == over.joints[1].origin
    # the good model is untouched
    assert m.get_link("drawer_1").inertial().mass > 0 and m.joints[0].lower < m.joints[0].upper


@pytest.mark.parametrize("oid", [i for i in IDS + GIDS if a.BREAKERS["parts_overlap"](a.build(i, CAT)) is not None])
def test_parts_overlap_really_overlaps(oid):
    """The breaker must make a real overlap, or a pass on it says nothing about the gate."""
    boxes = world_boxes(a.BREAKERS["parts_overlap"](a.build(oid, CAT)))
    hits = [(la, lb) for i, (la, lo_a, hi_a) in enumerate(boxes) for lb, lo_b, hi_b in boxes[i + 1:]
            if la != lb and overlap(lo_a, hi_a, lo_b, hi_b)]
    assert hits, oid


def test_catalog_is_deterministic_distinct_and_new():
    assert a.catalog(40, seed=0) == GEN
    assert a.catalog(40, seed=1) != GEN
    assert len(GEN) == 40 * len(a.FAMILIES)
    keys = [(fn, tuple(sorted(p.items()))) for fn, p in GEN.values()]
    pilot = {(fn, tuple(sorted(p.items()))) for fn, p in a.PILOT.values()}
    assert len(set(keys)) == len(keys) and not set(keys) & pilot


def test_generated_sizes_are_whole_centimetres_and_in_range():
    for oid, (fn, p) in GEN.items():
        for k in ("width", "height", "depth"):
            assert abs(p[k] * 100 - round(p[k] * 100)) < 1e-9, (oid, k)
            assert 0.2 <= p[k] <= 2.0, (oid, k, p[k])


@pytest.mark.parametrize("oid", GIDS[::7])
def test_breakers_apply_to_generated_objects(oid):
    m = a.build(oid, CAT)
    for name, br in a.BREAKERS.items():
        b = br(m)
        if b is not None:
            assert a.to_urdf(b) != a.to_urdf(m), (oid, name)


def test_build_with_generated_variants():
    with tempfile.TemporaryDirectory() as d:
        objs = build.build_all(d, per_family=2, seed=0)
        assert len(objs) == 20 + 10
        assert sum(o["generated"] for o in objs) == 10
        idx = json.load(open(os.path.join(d, "index.json")))
        assert idx["per_family"] == 2 and len(idx["objects"]) == 30
