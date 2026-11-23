"""The Feedback Lift loop with a fake model and a fake checker: no network, no key, no MuJoCo."""
import json
import os
import sys
import tempfile
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import articulate as a  # noqa: E402
import bench as cli  # noqa: E402
import bench_groq as g  # noqa: E402
import bench_loop as lp  # noqa: E402
import gate as gt  # noqa: E402

TP = os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl")
TASKS = cli.load(TP)
KEY = "test-key-not-real"
ALL = list(lp.MODES)

GOOD = "<robot name='good'><link name='base'/></robot>"
BAD = "<robot name='bad'><link name='base'/></robot>"
RAW = "MUJOCO-SAYS drawer_1 overlaps the carcass by 45 mm at rest"
FIX = "FIX-HINT move drawer_1 forward until it clears the carcass"


def ok(content, usage=(100, 900), finish="stop"):
    return 200, {}, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def xml(urdf):
    return ok(f"```xml\n{urdf}\n```\n")


class Fake:
    """Replies in order (a reply may be a function of the request body); records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        r = self.replies.pop(0)
        return r(body) if callable(r) else r


class Clock:
    def __init__(self):
        self.t = 0.0

    def sleep(self, s):
        self.t += s

    def now(self):
        return self.t


def learns_from_fixes(body):
    """A model that gets it right only once it is told the fix."""
    return xml(GOOD if FIX in body["messages"][-1]["content"] else BAD)


FAILED = {"answered": True,
          "spec": [{"rule": "spec.size", "subject": "z", "expected": 1.2, "got": 0.8}],
          "gate": {"verdict": "FAIL", "failures": [
              {"rule": "simcheck.no_errors", "severity": "error", "path": "/mujoco/engine.rest_contacts",
               "subject": "drawer_1", "expected": "no simcheck error for engine.rest_contacts", "got": RAW, "fix": FIX}]},
          "pass": False}
MISSING = {"answered": False, "spec": [{"rule": "answer.missing"}], "gate": None, "pass": False}


class FakeCheck:
    """Passes a URDF named 'good'; anything else fails one spec rule and one gate rule."""

    def __init__(self):
        self.calls = []

    def __call__(self, task, urdf, path):
        self.calls.append((task["id"], urdf, path))
        if urdf is None:
            return dict(MISSING)
        if "good" in urdf:
            assert path and os.path.isabs(path) and open(path, encoding="utf-8").read() == urdf
            return {"answered": True, "spec": [], "gate": {"verdict": "PASS", "failures": []}, "pass": True}
        return json.loads(json.dumps(FAILED))


def loop(d, tasks, post, modes=ALL, turns=3, check=None, **kw):
    c = Clock()
    kw.setdefault("log", lambda s: None)
    return lp.run(tasks, d, g.DEFAULT_MODEL, KEY, modes=modes, turns=turns, check=check or FakeCheck(), post=post,
                  pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), sleep=c.sleep, date="2026-11-10", **kw)


def record(d, mode, task):
    return json.load(open(lp.record_path(os.path.join(d, "loop", g.slug(g.DEFAULT_MODEL)), mode, task["id"])))


def board(d):
    return [json.loads(x) for x in open(os.path.join(d, "loop-leaderboard.jsonl"))]


# ---- feedback -----------------------------------------------------------------

def test_none_says_only_that_it_failed():
    text = lp.feedback("none", FAILED)
    assert "failed" in text and "```xml" in text
    for leak in (RAW, FIX, "spec.size", "simcheck.no_errors", "0.8"):
        assert leak not in text
    assert lp.feedback("none", MISSING) == text   # the same words whatever went wrong


def test_mujoco_gives_the_raw_text_and_nothing_structured():
    text = lp.feedback("mujoco", FAILED)
    assert RAW in text
    for leak in (FIX, "spec.size", "simcheck.no_errors", "/mujoco/engine.rest_contacts", "expected"):
        assert leak not in text
    parse = {"answered": True, "spec": [{"rule": "spec.parse", "got": "mismatched tag: line 3, column 2"}],
             "gate": {"verdict": "FAIL", "failures": [{"rule": "artifact.parse", "severity": "error", "subject": "artifact",
                                                       "expected": "a bare .urdf", "got": "model.urdf is not well-formed XML",
                                                       "fix": "Zip the model file"}]}, "pass": False}
    text = lp.feedback("mujoco", parse)
    assert "mismatched tag" in text and "not well-formed XML" in text
    assert "artifact.parse" not in text and "spec.parse" not in text and "Zip the model file" not in text
    spec_only = {"answered": True, "spec": [{"rule": "spec.size", "subject": "x", "expected": 1.0, "got": 2.0}],
                 "gate": {"verdict": "PASS", "failures": []}, "pass": False}
    assert "printed no error" in lp.feedback("mujoco", spec_only) and "spec.size" not in lp.feedback("mujoco", spec_only)
    assert lp.NO_URDF in lp.feedback("mujoco", MISSING)


def test_niua_gives_rule_ids_where_expected_got_and_fix():
    text = lp.feedback("niua", FAILED)
    for want in ("simcheck.no_errors /mujoco/engine.rest_contacts drawer_1", "no simcheck error for engine.rest_contacts",
                 RAW, f"fix: {FIX}"):
        assert want in text, want
    assert "answer.missing" in lp.feedback("niua", MISSING)


def test_no_mode_hands_over_the_spec_answer_key():
    # The spec's expected values (one of them, the minimum joint range, is not in the task text)
    # are the grader's answer key: no mode passes them on, so mujoco and niua cover the same failures.
    spec_only = {"answered": True, "gate": {"verdict": "PASS", "failures": []}, "pass": False,
                 "spec": [{"rule": "spec.size", "subject": "z", "expected": 1.234, "got": 0.8},
                          {"rule": "spec.range", "subject": "drawer_1", "expected": ">= 0.2", "got": 0.05}]}
    for mode in lp.MODES:
        for check in (spec_only, FAILED):
            text = lp.feedback(mode, check)
            for leak in ("spec.size", "spec.range", "1.234", ">= 0.2", "expected 1.2"):
                assert leak not in text, (mode, leak)
    assert "reported no failure" in lp.feedback("niua", spec_only)
    parse = {"answered": True, "spec": [{"rule": "spec.parse", "got": "mismatched tag: line 3, column 2"}],
             "gate": {"verdict": "FAIL", "failures": [{"rule": "artifact.parse", "severity": "error", "subject": "artifact",
                                                       "expected": "a bare .urdf", "got": "not well-formed XML",
                                                       "fix": "Zip the model file"}]}, "pass": False}
    text = lp.feedback("niua", parse)
    assert "spec.parse: got mismatched tag" in text and "artifact.parse artifact: expected a bare .urdf" in text
    with pytest.raises(ValueError):
        lp.feedback("louder", FAILED)


def test_feedback_is_bounded():
    many = {"answered": True, "spec": [], "pass": False,
            "gate": {"verdict": "FAIL", "failures": [{"rule": "simcheck.no_errors", "got": f"{i} " + "x" * 2000, "fix": "f"}
                                                     for i in range(40)]}}
    for mode in ("mujoco", "niua"):
        lines = [x for x in lp.feedback(mode, many).splitlines() if x.startswith("- ")]
        assert len(lines) == lp.MAX_NOTES + 1 and "(and 28 more)" in lines[-1]
        assert all(len(x) <= lp.MAX_NOTE_CHARS + 2 for x in lines)


# ---- the retry request ----------------------------------------------------------

def test_a_retry_carries_the_task_the_last_urdf_and_fits_the_minute_budget():
    msgs, cap = lp.retry_messages(TASKS[0], BAD, "note", 7500)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == TASKS[0]["prompt"] and BAD in msgs[1]["content"] and msgs[2]["content"] == "note"
    assert cap < 7500 and lp.estimate(msgs) + cap <= lp.FREE_TPM
    msgs, _ = lp.retry_messages(TASKS[0], None, "note", 7500)
    assert [m["role"] for m in msgs] == ["user", "user"]       # no URDF last time: nothing to hand back


def test_a_huge_last_urdf_is_cut_so_the_reply_keeps_its_floor():
    huge = "<robot name='r'>" + "<link name='l'/>" * 3000 + "</robot>"
    msgs, cap = lp.retry_messages(TASKS[0], huge, "note", 7500)
    assert lp.MIN_REPLY <= cap < lp.MIN_REPLY + 50 and "cut here" in msgs[1]["content"]
    assert lp.estimate(msgs) + cap <= lp.FREE_TPM


def test_request_body_without_messages_is_the_single_try_request():
    body = g.request_body(g.DEFAULT_MODEL, "p", 7500, "low")
    assert body["messages"] == [{"role": "system", "content": g.SYSTEM}, {"role": "user", "content": "p"}]
    turns = [{"role": "user", "content": "p"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "f"}]
    body = g.request_body(g.DEFAULT_MODEL, "ignored", 10, "low", turns)
    assert body["messages"] == [{"role": "system", "content": g.SYSTEM}, *turns]


# ---- a whole loop ---------------------------------------------------------------

def test_three_modes_share_turn_one_and_measure_the_lift():
    tasks = TASKS[:4]
    with tempfile.TemporaryDirectory() as d:
        fake, check = Fake([learns_from_fixes] * 24), FakeCheck()
        logs = []
        r = loop(d, tasks, fake, check=check, log=logs.append)
        assert r["complete"] and not r["stopped"] and r["requests"] == 24 and len(fake.calls) == 24
        # 4 first tries (shared), then none 2 + mujoco 2 + niua 1 retries per task
        firsts = [b for _, _, b in fake.calls if b["messages"][-1]["content"] in {t["prompt"] for t in tasks}]
        assert len(firsts) == 4 and len(check.calls) == 24
        for t in tasks:
            assert record(d, "niua", t)["pass_turn"] == 2
            assert record(d, "mujoco", t)["pass_turn"] is None and record(d, "none", t)["pass_turn"] is None
        rec = record(d, "none", tasks[0])
        assert rec["feedback"] == "none" and rec["model"] == g.DEFAULT_MODEL and rec["requests"] == 3
        assert rec["tokens"] == {"prompt_tokens": 300, "completion_tokens": 2700, "total_tokens": 3000}
        assert [a["turn"] for a in rec["attempts"]] == [1, 2, 3]
        assert rec["attempts"][0]["rules"] == ["spec.size", "simcheck.no_errors"]
        lines = board(d)
        assert [x["feedback"] for x in lines] == ["none", "mujoco", "niua"]
        niua = lines[2]
        assert niua["pass_at_1"] == 0 and niua["pass_at_turns"] == 4 and niua["lift"] == 4
        assert niua["pass_by_turn"] == [0, 4, 0] and niua["score_at_1"] == 0.0
        assert niua["score_at_turns"] == 1.0 and niua["lift_points"] == 1.0
        assert niua["tasks"] == 4 and niua["done"] == 4 and niua["complete"] and niua["tokens"] == 8000
        assert niua["turns"] == 3 and niua["date"] == "2026-11-10" and niua["contract"] == cli.CONTRACT
        assert lines[0]["lift"] == 0 and lines[1]["lift"] == 0
        per_task = [json.loads(x) for x in open(os.path.join(d, "loop", g.slug(g.DEFAULT_MODEL), "niua", "results.jsonl"))]
        assert [x["id"] for x in per_task] == [t["id"] for t in tasks]
        assert any("lift +4" in x for x in logs)
        for root, _, files in os.walk(d):
            for f in files:
                assert KEY not in open(os.path.join(root, f), encoding="utf-8").read()
        assert KEY not in "\n".join(logs)


def test_turn_one_is_the_single_try_request_and_retries_are_the_mode_feedback():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([learns_from_fixes] * 3)
        loop(d, TASKS[:1], fake, modes=["mujoco"])
        first, retry = fake.calls[0][2], fake.calls[1][2]
        assert first == g.request_body(g.DEFAULT_MODEL, TASKS[0]["prompt"], 7500, "low")
        assert [m["role"] for m in retry["messages"]] == ["system", "user", "assistant", "user"]
        assert retry["messages"][1]["content"] == TASKS[0]["prompt"] and BAD in retry["messages"][2]["content"]
        assert retry["messages"][3]["content"] == lp.feedback("mujoco", FAILED)
        assert lp.estimate(retry["messages"][1:]) + retry["max_tokens"] <= lp.FREE_TPM
        raw = json.load(open(os.path.join(d, "loop", g.slug(g.DEFAULT_MODEL), "mujoco", "turn2", "raw",
                                          f"{TASKS[0]['id']}.json")))
        assert raw["feedback"] == lp.feedback("mujoco", FAILED) and raw["max_tokens"] == retry["max_tokens"]


def test_a_pass_on_turn_one_sends_no_retry():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([xml(GOOD)] * 2)
        r = loop(d, TASKS[:2], fake)
        assert r["complete"] and len(fake.calls) == 2
        for line in board(d):
            assert line["pass_at_1"] == 2 and line["pass_at_turns"] == 2 and line["lift"] == 0


def test_no_urdf_is_a_failed_try_and_is_retried():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([ok("Sorry, no."), xml(GOOD)])
        loop(d, TASKS[:1], fake, modes=["niua"])
        retry = fake.calls[1][2]["messages"]
        assert [m["role"] for m in retry] == ["system", "user", "user"] and "answer.missing" in retry[-1]["content"]
        rec = record(d, "niua", TASKS[0])
        assert rec["pass_turn"] == 2 and rec["attempts"][0]["urdf_found"] is False


def test_a_cut_off_retry_asks_for_the_code_block_with_its_context():
    def continuation(body):
        assert body["messages"][-1]["content"] == g.CONTINUE and body["messages"][1]["content"] == TASKS[0]["prompt"]
        return xml(GOOD)

    with tempfile.TemporaryDirectory() as d:
        fake = Fake([xml(BAD), ok("thinking...", finish="length"), continuation])
        loop(d, TASKS[:1], fake, modes=["niua"])
        rec = record(d, "niua", TASKS[0])
        assert rec["pass_turn"] == 2 and rec["requests"] == 3 and len(fake.calls) == 3


def test_a_daily_limit_stops_the_loop_and_a_rerun_resumes_without_resending():
    tasks = TASKS[:2]
    with tempfile.TemporaryDirectory() as d:
        # per task: turn 1, none 2, none 3, niua 2. Task 1 stops on its none retry.
        day1 = Fake([learns_from_fixes] * 5 + [(429, {"retry-after": "40000"}, {"error": {"message": "TPD"}})])
        r1 = loop(d, tasks, day1, modes=["none", "niua"])
        assert not r1["complete"] and not r1["fatal"] and "40000" in r1["stopped"]
        assert [x["done"] for x in board(d)] == [1, 1]
        day2 = Fake([learns_from_fixes] * 3)
        r2 = loop(d, tasks, day2, modes=["none", "niua"])
        assert r2["complete"] and len(day2.calls) == 3
        assert all(b["messages"][-1]["content"] != tasks[1]["prompt"] for _, _, b in day2.calls)   # turn 1 not resent
        assert [x["pass_at_turns"] for x in board(d)] == [0, 2]
        again = Fake([])
        assert loop(d, tasks, again, modes=["none", "niua"])["complete"] and again.calls == []


def test_limit_caps_requests_and_a_rejected_key_is_fatal():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([learns_from_fixes] * 5)
        r = loop(d, TASKS[:1], fake, modes=["none"], limit=2)
        assert len(fake.calls) == 2 and r["stopped"] == "--limit 2 reached" and not r["fatal"]
    with tempfile.TemporaryDirectory() as d:
        logs = []
        r = loop(d, TASKS[:1], Fake([(401, {}, {"error": {"message": "Invalid API Key"}})]), log=logs.append)
        assert r["fatal"] and not r["complete"] and "401" in r["stopped"] and KEY not in "\n".join(logs)


def test_a_resume_must_match_the_settings():
    with tempfile.TemporaryDirectory() as d:
        loop(d, TASKS[:1], Fake([xml(GOOD)]))
        run = json.load(open(os.path.join(d, "loop", g.slug(g.DEFAULT_MODEL), "run.json")))
        assert run["turns"] == 3 and run["feedback_version"] == lp.FEEDBACK_VERSION and KEY not in json.dumps(run)
        with pytest.raises(SystemExit, match="other settings"):
            loop(d, TASKS[:1], Fake([]), turns=2)


def test_the_loop_refuses_one_turn_and_unknown_modes():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            loop(d, TASKS[:1], Fake([]), turns=1)
        with pytest.raises(ValueError):
            loop(d, TASKS[:1], Fake([]), modes=["louder"])
    assert lp.parse_modes("niua, none,niua") == ["none", "niua"]
    with pytest.raises(ValueError):
        lp.parse_modes("niua,louder")


def test_other_models_and_modes_keep_their_leaderboard_lines():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "loop-leaderboard.jsonl"), "w") as f:
            f.write(json.dumps({"model": "openai/gpt-oss-20b", "feedback": "niua", "lift": 9}) + "\n")
        loop(d, TASKS[:1], Fake([xml(GOOD)]), modes=["niua"])
        loop(d, TASKS[:1], Fake([]), modes=["niua"])        # a rerun replaces its own line, once
        lines = board(d)
        assert [(x["model"], x["feedback"]) for x in lines] == [(g.DEFAULT_MODEL, "niua"), ("openai/gpt-oss-20b", "niua")]


# ---- the real checker, with gate_check faked ----------------------------------

def test_gate_checker_applies_the_bench_pass_rule(monkeypatch):
    seen = {}

    def fake_gate(gate, contract, paths, fields=gt.FIELDS, log=print):
        seen.update(gate=gate, contract=contract, fields=fields)
        return {paths[0]: {"verdict": "PASS", "failures": []}}

    monkeypatch.setattr(gt, "run_gate", fake_gate)
    check = lp.gate_checker("/opt/gate_check")
    t = TASKS[0]
    fns = {fn.__name__: fn for fn, _ in a.FAMILIES.values()}
    ref = a.to_urdf(fns[t["reference"]["family"]](**t["reference"]["params"]))
    assert check(t, None, None) == MISSING
    c = check(t, ref, "/tmp/a.urdf")
    assert seen == {"gate": "/opt/gate_check", "contract": cli.CONTRACT, "fields": lp.GATE_FIELDS}
    assert c["answered"] and c["spec"] == [] and c["gate"]["verdict"] == "PASS" and c["pass"]
    monkeypatch.setattr(gt, "run_gate", lambda *a, **k: {})
    assert check(t, ref, "/tmp/a.urdf")["gate"]["verdict"] == "ERROR" and not check(t, ref, "/tmp/a.urdf")["pass"]


def test_gate_json_fields_are_unchanged_and_the_loop_also_gets_the_check_path(monkeypatch):
    line = {"file": "a.urdf", "outcome": {"verdict": "fail", "failures": [
        {"rule": "simcheck.no_errors", "severity": "error", "path": "/mujoco/engine.sweep", "subject": "hinge",
         "expected": "e", "got": "g", "fix": "f"}]}}
    out = (json.dumps(line) + "\n" + json.dumps(line) + "\n").encode()
    monkeypatch.setattr(gt.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=out, stderr=b""))
    plain = gt.run_gate("gc", cli.CONTRACT, ["a.urdf"], log=lambda s: None)["a.urdf"]["failures"][0]
    assert "path" not in plain and set(plain) == set(gt.FIELDS)
    loop_ = gt.run_gate("gc", cli.CONTRACT, ["a.urdf"], fields=lp.GATE_FIELDS, log=lambda s: None)["a.urdf"]["failures"][0]
    assert loop_["path"] == "/mujoco/engine.sweep"


# ---- the command line --------------------------------------------------------

def test_answer_with_feedback_runs_the_loop_with_three_turns_by_default(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    seen = {}

    def capture(tasks, out, model, key, **kw):
        seen.update(kw, n=len(tasks), key_ok=key == KEY)
        return {"complete": False, "fatal": False}

    monkeypatch.setattr(lp, "run", capture)
    monkeypatch.setattr(g, "answer", lambda *a, **k: pytest.fail("the single-try path must not run"))
    with tempfile.TemporaryDirectory() as d:
        rc = g.main(["answer", TP, d, "--feedback", "niua,none", "--gate", "gc", "--limit", "9", "--max-tokens", "6000"])
    assert rc == g.EXIT_RESUME
    assert seen["turns"] == lp.LIFT_TURNS == 3 and seen["modes"] == ["none", "niua"] and seen["n"] == 50
    assert seen["key_ok"] and seen["limit"] == 9 and seen["max_tokens"] == 6000 and callable(seen["check"])
    out = capsys.readouterr()
    assert KEY not in out.out and KEY not in out.err


def test_answer_without_feedback_is_the_single_try_bench(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setattr(lp, "run", lambda *a, **k: pytest.fail("the loop must not run"))
    monkeypatch.setattr(g, "answer", lambda *a, **k: {"complete": True, "fatal": False})
    with tempfile.TemporaryDirectory() as d:
        assert g.main(["answer", TP, d]) == g.EXIT_DONE
        assert g.main(["answer", TP, d, "--turns", "1"]) == g.EXIT_DONE


@pytest.mark.parametrize("argv", [["--feedback", "niua"],                          # no gate
                                  ["--turns", "3"],                                # no feedback
                                  ["--turns", "1", "--feedback", "niua", "--gate", "gc"],
                                  ["--turns", "0"],
                                  ["--feedback", "louder", "--gate", "gc"]])
def test_answer_refuses_a_loop_it_cannot_run(monkeypatch, argv):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setattr(lp, "run", lambda *a, **k: pytest.fail("nothing may run"))
    monkeypatch.setattr(g, "answer", lambda *a, **k: pytest.fail("nothing may run"))
    with pytest.raises(SystemExit):
        g.main(["answer", TP, "out", *argv])


# ---- the provider -------------------------------------------------------------

POLL_URL = "https://gen.pollinations.ai/v1/chat/completions"
POLL_MODEL = "deepseek/deepseek-v4.1-flash"
POLL_KEY = "pollinations-test-key-not-real"


def poll_loop(d, tasks, post, modes=ALL, turns=3, **kw):
    c = Clock()
    kw.setdefault("log", lambda s: None)
    return lp.run(tasks, d, POLL_MODEL, POLL_KEY, url=POLL_URL, provider="pollinations", modes=modes, turns=turns,
                  check=FakeCheck(), post=post, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), sleep=c.sleep,
                  max_tokens=32000, date="2026-11-10", **kw)


def test_pollinations_sends_every_try_to_its_url_key_and_model_with_the_full_reply_cap():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([learns_from_fixes] * 8)   # 2 shared first tries, 2 x 2 none retries, 2 x 1 niua retries
        logs = []
        r = poll_loop(d, TASKS[:2], fake, modes=["none", "niua"], log=logs.append)
        assert r["complete"] and len(fake.calls) == 2 + 2 * 2 + 2 * 1
        for url, headers, body in fake.calls:          # turn 1 and every retry: one provider
            assert url == POLL_URL and headers["Authorization"] == f"Bearer {POLL_KEY}"
            assert body["model"] == POLL_MODEL and body["max_tokens"] == 32000   # no Groq minute cap
            assert "reasoning_effort" not in body
        retry = fake.calls[1][2]["messages"]
        assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
        d_ = os.path.join(d, "loop", g.slug(POLL_MODEL))
        run = json.load(open(os.path.join(d_, "run.json")))
        assert run["provider"] == "pollinations" and run["tpm"] is None and run["max_tokens"] == 32000
        assert run["model"] == POLL_MODEL and run["reasoning_effort"] is None
        lines = board(d)
        assert [(x["provider"], x["model"], x["feedback"]) for x in lines] == \
            [("pollinations", POLL_MODEL, "none"), ("pollinations", POLL_MODEL, "niua")]
        assert lines[1]["lift"] == 2 and lines[1]["max_tokens"] == 32000
        for root, _, files in os.walk(d):
            for f in files:
                assert POLL_KEY not in open(os.path.join(root, f), encoding="utf-8").read()
        assert POLL_KEY not in "\n".join(logs)


def test_pollinations_hands_back_a_huge_last_urdf_whole():
    huge = "<robot name='r'>" + "<link name='l'/>" * 3000 + "</robot>"
    msgs, cap = lp.retry_messages(TASKS[0], huge, "note", 32000, tpm=None)
    assert cap == 32000 and huge in msgs[1]["content"] and "cut here" not in msgs[1]["content"]
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([xml(huge.replace("name='r'", "name='bad'")), xml(GOOD)])
        poll_loop(d, TASKS[:1], fake, modes=["none"])
        retry = fake.calls[1][2]
        assert retry["max_tokens"] == 32000 and "cut here" not in retry["messages"][2]["content"]


def test_a_pollinations_retry_that_fails_goes_back_to_pollinations_and_errors_hide_the_key():
    with tempfile.TemporaryDirectory() as d:
        logs = []
        fake = Fake([xml(BAD), (503, {}, {"error": {"message": "busy"}}), xml(GOOD)])
        r = poll_loop(d, TASKS[:1], fake, modes=["niua"], log=logs.append)
        assert r["complete"] and [u for u, _, _ in fake.calls] == [POLL_URL] * 3
        rec = json.load(open(lp.record_path(os.path.join(d, "loop", g.slug(POLL_MODEL)), "niua", TASKS[0]["id"])))
        assert rec["pass_turn"] == 2
    with tempfile.TemporaryDirectory() as d:
        logs = []
        r = poll_loop(d, TASKS[:1], Fake([(401, {}, {"error": {"message": f"bad key {POLL_KEY}"}})]),
                      log=logs.append)
        assert r["fatal"] and POLL_KEY not in r["stopped"] and POLL_KEY not in "\n".join(logs)


def test_groq_keeps_its_url_and_its_minute_cap():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([learns_from_fixes] * 2)
        loop(d, TASKS[:1], fake, modes=["niua"])
        assert [u for u, _, _ in fake.calls] == [g.GROQ_URL, g.GROQ_URL]
        assert fake.calls[1][2]["max_tokens"] < 7500
        run = json.load(open(os.path.join(d, "loop", g.slug(g.DEFAULT_MODEL), "run.json")))
        assert run["provider"] == "groq" and run["tpm"] == lp.FREE_TPM == lp.PROVIDER_TPM["groq"]
        assert board(d)[0]["provider"] == "groq"


def test_answer_with_feedback_on_pollinations_runs_the_loop_on_pollinations(monkeypatch, capsys):
    monkeypatch.setenv("POLLINATIONS_API_KEY", POLL_KEY)
    monkeypatch.delenv("POLLINATIONS_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = {}

    def capture(tasks, out, model, key, **kw):
        seen.update(kw, model=model, key_ok=key == POLL_KEY)
        return {"complete": True, "fatal": False}

    monkeypatch.setattr(lp, "run", capture)
    monkeypatch.setattr(g, "answer", lambda *a, **k: pytest.fail("the single-try path must not run"))
    with tempfile.TemporaryDirectory() as d:
        rc = g.main(["answer", TP, d, "--provider", "pollinations", "--model", POLL_MODEL, "--max-tokens", "32000",
                     "--feedback", "none,mujoco,niua", "--gate", "gc"])
    assert rc == g.EXIT_DONE
    assert seen["url"] == POLL_URL and seen["provider"] == "pollinations" and seen["model"] == POLL_MODEL
    assert seen["key_ok"] and seen["max_tokens"] == 32000 and seen["modes"] == ALL
    out = capsys.readouterr()
    assert POLL_KEY not in out.out and POLL_KEY not in out.err


def test_answer_with_feedback_on_groq_still_runs_the_loop_on_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    seen = {}
    monkeypatch.setattr(lp, "run", lambda tasks, out, model, key, **kw: seen.update(kw, model=model) or
                        {"complete": True, "fatal": False})
    with tempfile.TemporaryDirectory() as d:
        assert g.main(["answer", TP, d, "--feedback", "niua", "--gate", "gc"]) == g.EXIT_DONE
    assert seen["url"] == g.GROQ_URL and seen["provider"] == "groq" and seen["model"] == g.DEFAULT_MODEL


def test_a_pollinations_loop_without_its_key_sends_nothing(monkeypatch, capsys):
    monkeypatch.delenv("POLLINATIONS_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", KEY)            # the other provider's key is not used
    monkeypatch.setattr(lp, "run", lambda *a, **k: pytest.fail("nothing may run"))
    rc = g.main(["answer", TP, "out", "--provider", "pollinations", "--model", POLL_MODEL,
                 "--feedback", "niua", "--gate", "gc"])
    assert rc == g.EXIT_NO_KEY and "POLLINATIONS_API_KEY is not set" in capsys.readouterr().err


def _rec(tid, mode, pass_turn, attempts):
    return {"id": tid, "model": POLL_MODEL, "feedback": mode, "turns": 3, "pass_turn": pass_turn,
            "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "requests": len(attempts),
            "attempts": [{"turn": k, "pass": p, "urdf_found": found, "rules": rules}
                         for k, (p, found, rules) in enumerate(attempts, 1)]}


def test_the_failure_table_shows_each_try_of_every_task_that_failed_turn_one(capsys):
    t1 = (False, True, ["spec.joint_range", "articulated.collision", "articulated.collision"])
    recs = {
        "none": [_rec("bench-b", "none", None, [t1, (False, False, []), (False, True, [])]),
                 _rec("bench-a", "none", 1, [(True, True, [])])],
        "mujoco": [_rec("bench-b", "mujoco", 3, [t1, (False, True, ["articulated.collision"]), (True, True, [])]),
                   _rec("bench-a", "mujoco", 1, [(True, True, [])])],
        "niua": [_rec("bench-b", "niua", 2, [t1, (True, True, [])]), _rec("bench-a", "niua", 1, [(True, True, [])])],
    }
    with tempfile.TemporaryDirectory() as d:
        for mode, rows in recs.items():
            os.makedirs(os.path.join(d, "loop", g.slug(POLL_MODEL), mode))
            with open(os.path.join(d, "loop", g.slug(POLL_MODEL), mode, "results.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
        lines = lp.failure_table(d)
        assert lp.main(["failures", d]) == 0
    t1_text = "t1 spec.joint_range,articulated.collision"
    assert lines == [
        f"{POLL_MODEL}: 1 tasks did not pass on turn 1",
        f"bench-b none: {t1_text} | t2 no URDF | t3 fail (no rule) => no pass",
        f"bench-b mujoco: {t1_text} | t2 articulated.collision | t3 pass => pass on turn 3",
        f"bench-b niua: {t1_text} | t2 pass => pass on turn 2",
    ]
    assert capsys.readouterr().out.splitlines() == lines


def test_the_failure_table_caps_rule_ids_and_handles_an_empty_folder():
    many = [f"r{i}" for i in range(lp.MAX_TABLE_RULES + 2)]
    assert lp.try_text({"pass": False, "urdf_found": True, "rules": many}).endswith(",+2 more")
    with tempfile.TemporaryDirectory() as d:
        assert lp.failure_table(d) == []
