"""The bench's spec scorer: our own answers match, wrong objects don't."""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import articulate as a  # noqa: E402
import bench as cli  # noqa: E402
from articulate import bench as b  # noqa: E402
from articulate import families as f  # noqa: E402

TASKS = b.tasks(10)
FNS = {fn.__name__: fn for fn, _ in a.FAMILIES.values()}


def ref(t):
    r = t["reference"]
    return FNS[r["family"]](**r["params"])


def rules(t, m):
    return [x["rule"] for x in b.check_spec(t, a.to_urdf(m))]


def test_50_tasks_from_a_seed_the_dataset_does_not_use():
    assert len(TASKS) == 50 and len({t["id"] for t in TASKS}) == 50
    data = {(fn, tuple(sorted(p.items()))) for fn, p in {**a.PILOT, **a.catalog(200, seed=0)}.values()}
    for t in TASKS:
        assert (FNS[t["reference"]["family"]], tuple(sorted(t["reference"]["params"].items()))) not in data


def test_prompts_are_plain_and_complete():
    for t in TASKS:
        p = t["prompt"]
        assert "cm wide" in p and "tall" in p and "deep" in p and "metres" in p and "—" not in p
        assert "{" not in p


@pytest.mark.parametrize("t", TASKS, ids=[t["id"] for t in TASKS])
def test_reference_answers_match_their_spec(t):
    assert b.check_spec(t, a.to_urdf(ref(t))) == []


def test_wrong_size_fails():
    t = next(x for x in TASKS if x["reference"]["family"] == "chest")
    p = dict(t["reference"]["params"], width=t["reference"]["params"]["width"] + 0.3)
    assert "spec.size" in rules(t, f.chest(**p))


def test_millimetres_fail_the_size():
    t = TASKS[0]
    assert rules(t, ref(t).scaled(1000.0)).count("spec.size") == 3


def test_wrong_number_of_drawers_fails():
    t = next(x for x in TASKS if x["reference"]["family"] == "drawer_cabinet")
    p = dict(t["reference"]["params"])
    p["drawers"] = p["drawers"] - 1
    assert "spec.joints" in rules(t, f.drawer_cabinet(**p))


def test_wrong_kind_of_object_fails():
    t = next(x for x in TASKS if x["reference"]["family"] == "sliding_cabinet")
    p = t["reference"]["params"]
    assert "spec.joints" in rules(t, f.door_cabinet(p["width"], p["height"], p["depth"], 2))


def test_joint_along_the_wrong_axis_fails():
    t = next(x for x in TASKS if x["reference"]["family"] == "desk")
    m = ref(t)
    m.joints[0].axis = (1, 0, 0)
    assert "spec.joints" in rules(t, m)


def test_swapped_limits_fail_the_range():
    t = next(x for x in TASKS if x["reference"]["family"] == "door_cabinet")
    assert "spec.range" in rules(t, a.BREAKERS["inverted_limits"](ref(t)))


def test_floating_object_fails():
    t = TASKS[0]
    xml = a.to_urdf(ref(t))
    # lift everything 30 cm by adding a fixed joint under a new root
    xml = xml.replace("<link ", '<link name="floor_offset"/><joint name="lift" type="fixed"><parent link="floor_offset"/>'
                      f'<child link="{ref(t).root().name}"/><origin xyz="0 0 0.3"/></joint><link ', 1)
    assert "spec.on_floor" in [x["rule"] for x in b.check_spec(t, xml)]


def test_meshes_and_garbage_are_reported_not_crashed():
    t = TASKS[0]
    mesh = '<robot name="r"><link name="a"><visual><geometry><mesh filename="a.obj"/></geometry></visual></link></robot>'
    assert [x["rule"] for x in b.check_spec(t, mesh)][0] == "spec.primitives"
    assert b.check_spec(t, "not xml")[0]["rule"] == "spec.parse"
    assert b.check_spec(t, '<robot name="r"><link name="a"/><link name="b"/></robot>')[0]["rule"] == "spec.parse"


@pytest.mark.parametrize("old, new", [('<axis xyz="0 -1 0"/>', '<axis xyz="0 1"/>'),     # 2 numbers, not 3
                                      ('<axis xyz="0 -1 0"/>', '<axis xyz="0 -1 0 0"/>'),
                                      ('upper="0.4192"', 'upper="half"')])               # not a number
def test_an_unreadable_joint_axis_or_limit_is_a_parse_failure_not_a_crash(old, new):
    t = TASKS[0]
    xml = a.to_urdf(ref(t))
    assert old in xml and b.check_spec(t, xml) == []
    f = b.check_spec(t, xml.replace(old, new, 1))
    assert f and f[-1]["rule"] == "spec.parse"


def test_score_without_the_gate_counts_missing_answers():
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, "tasks.jsonl")
        cli.cmd_tasks(tp, 2)
        cli.cmd_reference(tp, os.path.join(d, "ans"))
        ts = cli.load(tp)
        os.remove(os.path.join(d, "ans", f"{ts[0]['id']}.urdf"))
        res = cli.score(ts, os.path.join(d, "ans"), gate=None)
        assert res["tasks"] == 10
        assert res["summary"]["answered"] == 9 and res["summary"]["spec_ok"] == 9
        assert res["summary"]["pass"] == 0   # no gate run, so nothing passes both


def test_the_published_task_file_matches_the_generator():
    """tasks-v1.jsonl is the benchmark as released; a code change must not move it."""
    assert cli.load(os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl")) == TASKS


def test_an_undefined_world_parent_is_named_in_the_failure():
    """A common LLM habit: a fixed joint from 'world' without defining the link."""
    t = TASKS[0]
    xml = a.to_urdf(ref(t)).replace("<link ", '<joint name="to_world" type="fixed"><parent link="world"/>'
                                    f'<child link="{ref(t).root().name}"/></joint><link ', 1)
    f = b.check_spec(t, xml)
    assert f[0]["rule"] == "spec.parse" and "'world'" in f[0]["got"] and "not defined" in f[0]["got"]
