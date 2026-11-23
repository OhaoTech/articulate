"""The Groq runner with a fake client: no network, no key, no real sleeping."""
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import articulate as a  # noqa: E402
import bench as cli  # noqa: E402
import bench_groq as g  # noqa: E402

TASKS = cli.load(os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl"))
FNS = {fn.__name__: fn for fn, _ in a.FAMILIES.values()}
KEY = "test-key-not-real"


def reference_urdf(task):
    r = task["reference"]
    return a.to_urdf(FNS[r["family"]](**r["params"]))


def ok(content, usage=(100, 900), finish="stop"):
    return 200, {}, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


class Fake:
    """Replies in order; records every call."""

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
        self.slept = []

    def sleep(self, s):
        self.slept.append(s)
        self.t += s

    def now(self):
        return self.t


def reply_with_reference(body):
    prompt = body["messages"][1]["content"]
    task = next(t for t in TASKS if t["prompt"] == prompt)
    return ok(f"Here you go.\n```xml\n{reference_urdf(task)}```\n")


# ---- pieces ------------------------------------------------------------------

@pytest.mark.parametrize("v,want", [("7", 7.0), ("1.5", 1.5), ("7.66s", 7.66), ("1m2.5s", 62.5), ("200ms", 0.2),
                                    ("1h0m0s", 3600.0), ("2m59.56s", 179.56), (None, None), ("", None), ("soon", None)])
def test_seconds_reads_every_header_form(v, want):
    got = g.seconds(v)
    assert got == pytest.approx(want) if want is not None else got is None


def test_pacer_spaces_request_starts():
    c = Clock()
    p = g.Pacer(20.0, sleep=c.sleep, clock=c.now)
    p.wait()
    c.t += 5
    p.wait()
    p.wait()
    assert c.slept == [pytest.approx(15.0), pytest.approx(20.0)]


@pytest.mark.parametrize("text", [
    "```xml\n<robot name='r'><link name='a'/></robot>\n```",
    "Sure!\n```\n<?xml version='1.0'?>\n<robot name='r'><link name='a'/></robot>\n```\nDone.",
    "```urdf\r\n<robot name='r'><link name='a'/></robot>\r\n```",
    "<?xml version='1.0'?><robot name='r'><link name='a'/></robot> trailing words",
])
def test_extract_urdf_finds_the_robot(text):
    u = g.extract_urdf(text)
    assert u is not None and u.lstrip().startswith(("<robot", "<?xml")) and u.rstrip().endswith("</robot>")


def test_extract_urdf_skips_non_robot_blocks_and_nothing():
    assert g.extract_urdf("```python\nprint(1)\n```\n```xml\n<robot name='x'/>\n```") == "<robot name='x'/>\n"
    assert g.extract_urdf("I can't do that.") is None
    assert g.extract_urdf(None) is None


def test_request_shape_is_the_openai_chat_api_with_the_key_in_the_header_only():
    fake = Fake([ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, "qwen/qwen3.8-27b", "make a chest", g.Pacer(0, sleep=c.sleep, clock=c.now),
          max_tokens=123, max_wait=60, sleep=c.sleep, reasoning_effort="low")
    url, headers, body = fake.calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["User-Agent"] == g.USER_AGENT and "urllib" not in headers["User-Agent"].lower()
    assert body["model"] == "qwen/qwen3.8-27b" and body["max_tokens"] == 123 and body["temperature"] == 0
    assert "reasoning_effort" not in body   # only gpt-oss models take it
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and body["messages"][1]["content"] == "make a chest"
    assert KEY not in json.dumps(body)


def test_reasoning_effort_goes_to_gpt_oss_only():
    assert g.request_body("openai/gpt-oss-120b", "p", 10, "low")["reasoning_effort"] == "low"
    assert "reasoning_effort" not in g.request_body("openai/gpt-oss-120b", "p", 10, None)
    assert "reasoning_effort" not in g.request_body("qwen/qwen3.8-27b", "p", 10, "low")


def test_success_holds_the_next_request_until_the_minute_budget_refills():
    """remaining 3000 tokens < next request's 8000 -> the NEXT start waits for the reset Groq names.
    Held, not slept inside ask(), so the reply is on disk before any waiting."""
    h = {"x-ratelimit-remaining-tokens": "3000", "x-ratelimit-reset-tokens": "41.5s"}
    fake = Fake([(200, h, ok("```xml\n<robot name='r'/>\n```")[2])])
    c = Clock()
    p = g.Pacer(0, sleep=c.sleep, clock=c.now)
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", p, max_tokens=7000, max_wait=120, sleep=c.sleep)
    assert c.slept == []            # nothing slept before returning the reply
    p.wait()                        # the next request
    assert c.slept == [pytest.approx(41.5)]
    assert g.budget_wait({"x-ratelimit-remaining-tokens": "7999"}, 8000) is None   # no reset header: no guess
    assert g.budget_wait({"x-ratelimit-remaining-tokens": "9000", "x-ratelimit-reset-tokens": "5s"}, 8000) is None
    assert g.budget_wait({}, 8000) is None


def test_429_with_retry_after_waits_then_succeeds():
    fake = Fake([(429, {"retry-after": "7"}, {"error": {"message": "rate limit"}}), ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60, sleep=c.sleep)
    assert len(fake.calls) == 2 and 7.0 in c.slept


def test_429_uses_groq_reset_header_when_retry_after_is_missing():
    fake = Fake([(429, {"x-ratelimit-reset-tokens": "7.5s"}, None), ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60, sleep=c.sleep)
    assert 7.5 in c.slept


def test_a_long_wait_stops_the_run_instead_of_spinning():
    fake = Fake([(429, {"retry-after": "5400"}, {"error": {"message": "tokens per day"}})])
    c = Clock()
    with pytest.raises(g.StopRun, match="5400"):
        g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=120, sleep=c.sleep)
    assert len(fake.calls) == 1 and c.slept == []


def test_server_errors_back_off_then_give_up():
    fake = Fake([(503, {}, None)] * 5)
    c = Clock()
    with pytest.raises(g.StopRun, match="gave up"):
        g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=120,
              retries=4, sleep=c.sleep)
    assert len(fake.calls) == 5 and c.slept == [2.0, 4.0, 8.0, 16.0]


def test_a_rejected_request_is_an_error_not_a_retry():
    fake = Fake([(401, {}, {"error": {"message": "Invalid API Key"}})])
    c = Clock()
    with pytest.raises(RuntimeError, match="401"):
        g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=120, sleep=c.sleep)
    assert len(fake.calls) == 1


# ---- a whole run -------------------------------------------------------------

def test_full_mocked_run_answers_paces_and_scores_like_the_reference():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([reply_with_reference] * len(TASKS))
        c = Clock()
        logs = []
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(20.0, sleep=c.sleep, clock=c.now),
                     log=logs.append, sleep=c.sleep)
        assert r["complete"] and r["sent"] == 50 and r["no_urdf"] == 0
        assert r["usage"] == {"prompt_tokens": 5000, "completion_tokens": 45000}
        assert sum(c.slept) == pytest.approx(20.0 * 49)       # one request every 20 s, none in parallel
        out = os.path.join(d, g.slug(g.DEFAULT_MODEL))
        assert len([f for f in os.listdir(out) if f.endswith(".urdf")]) == 50
        # spec half without the gate (the gate half runs in CI)
        line = g.score(os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl"), d, g.DEFAULT_MODEL, None,
                       date="2026-11-05")
        assert line["answered"] == 50 and line["spec_ok"] == 50 and line["pass"] == 0
        assert line["model"] == g.DEFAULT_MODEL and line["bench"] == "articulate/tasks-v1" and line["date"] == "2026-11-05"
        res = json.load(open(os.path.join(out, "results.json")))
        assert res["model"] == g.DEFAULT_MODEL and res["tasks"] == 50
        board = [json.loads(x) for x in open(os.path.join(d, "leaderboard.jsonl"))]
        assert board == [line]
        # nothing on disk holds the key
        for root, _, files in os.walk(d):
            for f in files:
                assert KEY not in open(os.path.join(root, f), encoding="utf-8").read()


def test_run_stops_on_the_daily_limit_and_resumes_without_resending():
    with tempfile.TemporaryDirectory() as d:
        c = Clock()
        day1 = Fake([reply_with_reference] * 3 + [(429, {"retry-after": "40000"}, {"error": {"message": "TPD"}})])
        r1 = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=day1, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                      log=lambda s: None, sleep=c.sleep)
        assert not r1["complete"] and r1["done"] == 3 and "40000" in r1["stopped"]
        day2 = Fake([reply_with_reference] * 47)
        r2 = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=day2, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                      log=lambda s: None, sleep=c.sleep)
        assert r2["complete"] and r2["sent"] == 47
        sent2 = {b["messages"][1]["content"] for _, _, b in day2.calls}
        assert not sent2 & {b["messages"][1]["content"] for _, _, b in day1.calls[:3]}


def test_a_reply_without_urdf_is_recorded_and_counts_as_missing():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([ok("Sorry, I can't help with that.")] + [reply_with_reference] * 49)
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     log=lambda s: None, sleep=c.sleep)
        assert r["complete"] and r["no_urdf"] == 1
        raw = json.load(open(os.path.join(d, g.slug(g.DEFAULT_MODEL), "raw", f"{TASKS[0]['id']}.json")))
        assert raw["urdf_found"] is False and "Sorry" in raw["content"]
        res = cli.score(TASKS, os.path.join(d, g.slug(g.DEFAULT_MODEL)), None)
        assert res["summary"]["answered"] == 49


def test_limit_caps_requests_per_run():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([reply_with_reference] * 5)
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     limit=5, log=lambda s: None, sleep=c.sleep)
        assert r["sent"] == 5 and not r["complete"] and len(fake.calls) == 5


# ---- the command line --------------------------------------------------------

@pytest.mark.parametrize("key,msg", [(None, "is not set"), ("", "is not set"), ("   ", "is not set"),
                                     ("abc\ndef", "line breaks"), ("abc def", "spaces"), ("clé", "non-ASCII")])
def test_answer_without_a_usable_key_sends_nothing(monkeypatch, capsys, key, msg):
    if key is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", key)
    # patch what the code really calls, so a broken key check would fail here instead of reaching the network
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the network"))
    monkeypatch.setattr(g, "answer", lambda *a, **k: pytest.fail("answer must not run without a usable key"))
    with tempfile.TemporaryDirectory() as d:
        assert g.main(["answer", TP, d]) == g.EXIT_NO_KEY
    out = capsys.readouterr()
    assert msg in out.err
    if key and key.strip():
        assert key not in out.err and key not in out.out


def test_only_free_tier_models_are_accepted(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    rc = g.main(["answer", "t.jsonl", "out", "--model", "some-paid-model"])
    assert rc != 0
    out = capsys.readouterr()
    assert "must" in out.err
    assert g.DEFAULT_MODEL in g.FREE_MODELS


def test_a_rejected_key_stops_the_run_cleanly():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([(401, {}, {"error": {"message": "Invalid API Key"}})])
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     log=lambda s: None, sleep=c.sleep)
        assert not r["complete"] and r["sent"] == 0 and "401" in r["stopped"] and len(fake.calls) == 1


def test_settings_are_recorded_and_a_resume_must_match_them():
    with tempfile.TemporaryDirectory() as d:
        c = Clock()
        kw = dict(pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=lambda s: None, sleep=c.sleep, limit=1)
        g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake([reply_with_reference]), **kw)
        run = json.load(open(os.path.join(d, g.slug(g.DEFAULT_MODEL), "run.json")))
        assert run["max_tokens"] == 7500 and run["reasoning_effort"] == "low" and KEY not in json.dumps(run)
        with pytest.raises(SystemExit, match="other settings"):
            g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake([]), max_tokens=500, **kw)
        line = g.score(TP, d, g.DEFAULT_MODEL, None, date="2026-11-05", partial=True)
        assert line["max_tokens"] == 7500 and line["reasoning_effort"] == "low" and line["answered"] == 1


TP = os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl")


def full_run(d, replies=None):
    c = Clock()
    return g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake(replies or [reply_with_reference] * len(TASKS)),
                    pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=lambda s: None, sleep=c.sleep)


def test_score_refuses_an_unfinished_run_unless_asked_for_partial():
    with tempfile.TemporaryDirectory() as d:
        c = Clock()
        g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake([reply_with_reference] * 3),
                 pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=lambda s: None, sleep=c.sleep, limit=3)
        with pytest.raises(SystemExit, match="3/50 tasks asked"):
            g.score(TP, d, g.DEFAULT_MODEL, None)
        assert not os.path.exists(os.path.join(d, "leaderboard.jsonl"))
        assert g.score(TP, d, g.DEFAULT_MODEL, None, partial=True)["answered"] == 3


def test_score_keeps_exactly_one_line_per_model():
    with tempfile.TemporaryDirectory() as d:
        full_run(d)
        g.score(TP, d, g.DEFAULT_MODEL, None, date="2026-11-05")
        g.score(TP, d, g.DEFAULT_MODEL, None, date="2026-11-06")
        rows = [json.loads(x) for x in open(os.path.join(d, "leaderboard.jsonl"))]
        assert len(rows) == 1 and rows[0]["date"] == "2026-11-06"


def test_check_accepts_an_honest_leaderboard_and_catches_every_drift():
    quiet = lambda s: None
    with tempfile.TemporaryDirectory() as d:
        full_run(d)
        board = os.path.join(d, "leaderboard.jsonl")
        # no line yet: check says which line to add
        missing = g.check(TP, d, None, log=quiet)
        assert len(missing) == 1 and "no line" in missing[0]
        honest = g.score(TP, d, g.DEFAULT_MODEL, None, date="2026-11-05")
        assert g.check(TP, d, None, log=quiet) == []
        fake_row = {**honest, "pass": 50, "score": 1.0, "date": "2026-11-09"}
        # a hand-typed score alone is caught
        g.write_board(board, [fake_row])
        drift = g.check(TP, d, None, log=quiet)
        assert len(drift) == 1 and "pass=50" in drift[0]
        # a hand-typed score BEFORE an honest one is caught too (not only the last line counts)
        with open(board, "w") as f:
            f.write(json.dumps(fake_row) + "\n" + json.dumps(honest) + "\n")
        probs = g.check(TP, d, None, log=quiet)
        assert any("2 lines" in p for p in probs) and any("pass=50" in p for p in probs)
        # a line with no committed answers is caught
        g.write_board(board, [honest, {**honest, "model": "qwen/qwen3.8-27b"}])
        assert any("not committed" in p for p in g.check(TP, d, None, log=quiet))


def test_check_rejects_an_unfinished_committed_run():
    with tempfile.TemporaryDirectory() as d:
        c = Clock()
        g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake([reply_with_reference] * 3),
                 pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=lambda s: None, sleep=c.sleep, limit=3)
        probs = g.check(TP, d, None, log=lambda s: None)
        assert len(probs) == 1 and "only 3/50" in probs[0]


def test_check_with_no_committed_runs_is_clean():
    with tempfile.TemporaryDirectory() as d:
        assert g.check(os.path.join(os.path.dirname(HERE), "bench", "tasks-v1.jsonl"), os.path.join(d, "none"),
                       None, log=lambda s: None) == []


# ---- review findings ---------------------------------------------------------

import http.client  # noqa: E402
import io  # noqa: E402
import urllib.error  # noqa: E402


def wire_headers(raw: bytes):
    return http.client.parse_headers(io.BytesIO(raw))


class FakeResp:
    def __init__(self, status, headers, body):
        self.status, self.headers, self._body = status, headers, body

    def read(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_post_lowercases_wire_headers_on_success_and_on_429(monkeypatch):
    ok_hdrs = wire_headers(b"X-RateLimit-Remaining-Tokens: 300\r\nX-RateLimit-Reset-Tokens: 41.5s\r\n\r\n")
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp(200, ok_hdrs, b'{"choices": [{"message": {"content": "x"}}]}'))
    status, h, data = g.http_post(g.GROQ_URL, {"User-Agent": g.USER_AGENT}, {"a": 1})
    assert status == 200 and h["x-ratelimit-reset-tokens"] == "41.5s" and data["choices"]

    def raise_429(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                     wire_headers(b"Retry-After: 40000\r\nX-RateLimit-Reset-Tokens: 7.66s\r\n\r\n"),
                                     io.BytesIO(b'{"error": {"message": "tokens per day"}}'))
    monkeypatch.setattr(g.urllib.request, "urlopen", raise_429)
    status, h, data = g.http_post(g.GROQ_URL, {}, {})
    assert status == 429 and h["retry-after"] == "40000" and data["error"]["message"] == "tokens per day"


@pytest.mark.parametrize("err", [http.client.IncompleteRead(b"par"), http.client.BadStatusLine("x"),
                                 ConnectionResetError("reset"), TimeoutError("slow"),
                                 urllib.error.URLError("dns"), ValueError("Invalid header value b'Bearer x\\ny'")])
def test_http_post_turns_any_network_failure_into_599_without_echoing_headers(monkeypatch, err):
    def boom(req, timeout):
        if isinstance(err, http.client.IncompleteRead):
            return FakeResp(200, wire_headers(b"\r\n"), err)   # fails mid-body
        raise err
    monkeypatch.setattr(g.urllib.request, "urlopen", boom)
    status, h, data = g.http_post(g.GROQ_URL, {"Authorization": f"Bearer {KEY}"}, {})
    assert status == 599 and h == {}
    assert KEY not in json.dumps(data) and "Bearer" not in json.dumps(data)


def test_a_599_is_retried_not_fatal():
    fake = Fake([(599, {}, {"error": {"message": "network: ConnectionResetError"}}), ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60, sleep=c.sleep)
    assert len(fake.calls) == 2 and c.slept == [2.0]


def test_retry_after_wins_over_the_per_minute_reset_header():
    """A daily-limit 429 names a long Retry-After next to a short per-minute reset: the long one must win."""
    fake = Fake([(429, {"retry-after": "40000", "x-ratelimit-reset-tokens": "7.66s",
                        "x-ratelimit-reset-requests": "2m59.56s"}, {"error": {"message": "tokens per day"}})])
    c = Clock()
    with pytest.raises(g.StopRun, match="40000"):
        g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=120,
              sleep=c.sleep)
    assert len(fake.calls) == 1 and c.slept == []


def test_a_retry_still_waits_out_the_min_interval():
    """503 -> 2 s backoff, then the pacer adds 18 s so request starts stay 20 s apart."""
    fake = Fake([(503, {}, None), ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(20.0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60,
          sleep=c.sleep)
    assert c.slept == [pytest.approx(2.0), pytest.approx(18.0)]


@pytest.mark.parametrize("bad", [None, {}, {"choices": []}])
def test_a_200_without_choices_is_retried_not_fatal(bad):
    fake = Fake([(200, {}, bad), ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    data = g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60,
                 sleep=c.sleep)
    assert data["choices"] and len(fake.calls) == 2


def test_the_budget_hold_is_capped_by_max_wait():
    h = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2h0m0s"}
    fake = Fake([(200, h, ok("```xml\n<robot name='r'/>\n```")[2])])
    c = Clock()
    p = g.Pacer(0, sleep=c.sleep, clock=c.now)
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "p", p, max_tokens=7000, max_wait=120, sleep=c.sleep)
    p.wait()
    assert c.slept == [pytest.approx(120.0)]


def test_the_reply_is_saved_before_the_budget_wait():
    """A run stopped during the polite wait must not lose the reply it just got."""
    spent = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "50s"}

    def reply(body):
        return (200, spent, reply_with_reference(body)[2])

    with tempfile.TemporaryDirectory() as d:
        c = Clock()
        saved_when_sleeping = []

        def sleep(s):
            saved_when_sleeping.append(len(os.listdir(os.path.join(d, g.slug(g.DEFAULT_MODEL), "raw"))))
            c.sleep(s)

        g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake([reply, reply]), pacer=g.Pacer(0, sleep=sleep, clock=c.now),
                 log=lambda s: None, sleep=sleep, limit=2)
        assert saved_when_sleeping == [1]   # the one budget wait came after reply 1 was on disk


@pytest.mark.parametrize("text,want", [
    ("<think>draft:\n```xml\n<robot name='draft'/>\n```\nnow the lid</think>\n```xml\n<robot name='final'/>\n```",
     "<robot name='final'/>\n"),
    ("<think>draft <robot name='draft'></robot></think>\n```xml\n<robot name='final'><link name='a'/>", None),
    ("<think>still thinking about <robot name='draft'></robot> and", None),
    ("<THINK>x</THINK><robot name='bare'></robot>", "<robot name='bare'></robot>\n"),
])
def test_inline_reasoning_is_never_scored_as_the_answer(text, want):
    assert g.extract_urdf(text) == want


def test_limit_counts_retries_too():
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([(503, {}, None), (503, {}, None), reply_with_reference, reply_with_reference])
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     limit=3, log=lambda s: None, sleep=c.sleep)
        assert len(fake.calls) == 3 and r["requests"] == 3 and r["sent"] == 1 and "--limit 3" in r["stopped"]


def test_a_reply_without_urdf_is_never_resent():
    with tempfile.TemporaryDirectory() as d:
        full_run(d, [ok("Sorry, I can't help with that.")] + [reply_with_reference] * 49)
        again = Fake([])
        c = Clock()
        r2 = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=again, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                      log=lambda s: None, sleep=c.sleep)
        assert r2["sent"] == 0 and r2["complete"] and again.calls == []


def test_logs_and_stop_messages_never_hold_the_key():
    with tempfile.TemporaryDirectory() as d:
        logs = []
        c = Clock()
        replies = [reply_with_reference] * 2 + [(429, {"retry-after": "40000"}, {"error": {"message": "TPD"}})]
        g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=Fake(replies), pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                 log=logs.append, sleep=c.sleep)
        g.answer(TASKS, os.path.join(d, "b"), g.DEFAULT_MODEL, KEY,
                 post=Fake([(401, {}, {"error": {"message": "Invalid API Key"}})]),
                 pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=logs.append, sleep=c.sleep)
        text = "\n".join(logs)
        assert "stopped:" in text and "rejected:" in text and KEY not in text


@pytest.mark.parametrize("result,code", [({"complete": True, "fatal": False}, 0),
                                         ({"complete": False, "fatal": False}, 3),
                                         ({"complete": False, "fatal": True}, 4)])
def test_answer_cli_wires_flags_and_exit_codes(monkeypatch, capsys, result, code):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    seen = {}

    def capture(tasks, out, model, key, **kw):
        seen.update(kw, n=len(tasks), model=model, key_ok=key == KEY)
        return result

    monkeypatch.setattr(g, "answer", capture)
    with tempfile.TemporaryDirectory() as d:
        rc = g.main(["answer", TP, d, "--min-interval", "33", "--max-wait", "77", "--limit", "5",
                     "--max-tokens", "6000", "--reasoning-effort", "medium"])
    assert rc == code
    assert seen["n"] == 50 and seen["model"] == g.DEFAULT_MODEL and seen["key_ok"]
    assert seen["pacer"].min_interval == 33 and seen["max_wait"] == 77 and seen["limit"] == 5
    assert seen["max_tokens"] == 6000 and seen["reasoning_effort"] == "medium"
    out = capsys.readouterr()
    assert KEY not in out.out and KEY not in out.err


def test_score_cli_only_takes_free_models():
    with pytest.raises(SystemExit):
        g.main(["score", TP, "out", "gate", "--model", "some-paid-model"])


# ---- length-retry ------------------------------------------------------------

def test_length_retry_extracts_urdf_from_continuation():
    """finish_reason=length with no URDF triggers one retry with CONTINUE prompt."""
    truncated = ok("<think>Let me start with the base link, then the drawer...</think>\nHere is the ", finish="length")

    def continuation(body):
        # The retry should get the CONTINUE prompt
        assert body["messages"][1]["content"] == g.CONTINUE
        # Return a valid URDF for the first task (continuation gets CONTINUE, not original prompt)
        return ok(f"```xml\n{reference_urdf(TASKS[0])}```\n")

    with tempfile.TemporaryDirectory() as d:
        fake = Fake([truncated, continuation] + [reply_with_reference] * 49)
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     log=lambda s: None, sleep=c.sleep)
        assert r["complete"] and r["no_urdf"] == 0
        assert len(fake.calls) == 51  # 50 tasks + 1 retry
        raw = json.load(open(os.path.join(d, g.slug(g.DEFAULT_MODEL), "raw", f"{TASKS[0]['id']}.json")))
        assert raw["urdf_found"] is True
        assert "---length-retry---" in raw["content"]


def test_length_retry_does_not_happen_when_urdf_extracted_despite_truncation():
    """If the URDF is complete before truncation, no retry needed."""
    truncated_but_complete = ok("```xml\n<robot name='r'><link name='a'/></robot>\n```\nI would also like to add",
                                finish="length")
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([truncated_but_complete] + [reply_with_reference] * 49)
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     log=lambda s: None, sleep=c.sleep)
        assert r["complete"] and r["no_urdf"] == 0
        assert len(fake.calls) == 50  # no retry


def test_length_retry_counts_against_limit():
    """A length retry consumes one HTTP request from the limit."""
    truncated = ok("Starting to think...", finish="length")
    # Continuation gets CONTINUE prompt, so provide a static URDF response
    continuation_ok = ok(f"```xml\n{reference_urdf(TASKS[0])}```\n")
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([truncated, continuation_ok, reply_with_reference])
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     limit=2, log=lambda s: None, sleep=c.sleep)
        assert len(fake.calls) == 2 and r["requests"] == 2
        assert r["done"] == 1  # only task 0 completed (truncated + retry)


def test_length_retry_still_no_urdf_is_recorded_as_missing():
    """If retry also fails to produce URDF, the task is marked as no_urdf."""
    truncated1 = ok("Let me think about this...", finish="length")
    truncated2 = ok("Still thinking about the best way to", finish="length")  # retry also truncated
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([truncated1, truncated2] + [reply_with_reference] * 49)
        c = Clock()
        r = g.answer(TASKS, d, g.DEFAULT_MODEL, KEY, post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now),
                     log=lambda s: None, sleep=c.sleep)
        assert r["complete"] and r["no_urdf"] == 1
        raw = json.load(open(os.path.join(d, g.slug(g.DEFAULT_MODEL), "raw", f"{TASKS[0]['id']}.json")))
        assert raw["urdf_found"] is False
        assert "---length-retry---" in raw["content"]


# ---- pollinations provider ---------------------------------------------------

POLLINATIONS_KEY = "pollinations-test-key"
POLLINATIONS_MODEL = "gpt-4o-mini"
POLLINATIONS_BASE = "https://gen.pollinations.ai/v1"


def test_pollinations_posts_to_base_url_chat_completions():
    """Pollinations uses {base}/chat/completions with Bearer key."""
    fake = Fake([ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    url = POLLINATIONS_BASE + "/chat/completions"
    g.ask(fake, url, POLLINATIONS_KEY, POLLINATIONS_MODEL, "make a chest", g.Pacer(0, sleep=c.sleep, clock=c.now),
          max_tokens=100, max_wait=60, sleep=c.sleep)
    posted_url, headers, body = fake.calls[0]
    assert posted_url == "https://gen.pollinations.ai/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {POLLINATIONS_KEY}"
    assert body["model"] == POLLINATIONS_MODEL


def test_groq_path_still_hits_api_groq_com():
    """Groq provider uses api.groq.com and FREE_MODELS unchanged."""
    fake = Fake([ok("```xml\n<robot name='r'/>\n```")])
    c = Clock()
    g.ask(fake, g.GROQ_URL, KEY, g.DEFAULT_MODEL, "make a chest", g.Pacer(0, sleep=c.sleep, clock=c.now),
          max_tokens=100, max_wait=60, sleep=c.sleep)
    url, headers, body = fake.calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert g.DEFAULT_MODEL in g.FREE_MODELS


def test_answer_with_pollinations_writes_provider_to_run_json():
    """answer() with provider=pollinations writes 'pollinations' to run.json."""
    with tempfile.TemporaryDirectory() as d:
        fake = Fake([reply_with_reference] * 2)
        c = Clock()
        url = POLLINATIONS_BASE + "/chat/completions"
        r = g.answer(TASKS[:2], d, POLLINATIONS_MODEL, POLLINATIONS_KEY, url=url, provider="pollinations",
                     post=fake, pacer=g.Pacer(0, sleep=c.sleep, clock=c.now), log=lambda s: None, sleep=c.sleep)
        assert r["done"] == 2
        run = json.load(open(os.path.join(d, g.slug(POLLINATIONS_MODEL), "run.json")))
        assert run["provider"] == "pollinations"
        assert run["model"] == POLLINATIONS_MODEL


def test_leaderboard_line_uses_provider_from_settings():
    """leaderboard_line reads provider from settings, not hardcoded."""
    res = {"summary": {"answered": 1, "spec_ok": 1, "gate_ok": 1, "pass": 1}, "tasks": 1, "score": 1.0, "contract": "test"}
    line_groq = g.leaderboard_line(res, "model1", "2026-11-06", {"provider": "groq", "max_tokens": 100})
    assert line_groq["provider"] == "groq"
    line_poll = g.leaderboard_line(res, "model2", "2026-11-06", {"provider": "pollinations", "max_tokens": 200})
    assert line_poll["provider"] == "pollinations"
    line_default = g.leaderboard_line(res, "model3", "2026-11-06", {})
    assert line_default["provider"] == "groq"  # default when not specified


def test_cli_groq_still_restricts_to_free_models(monkeypatch, capsys):
    """Groq provider CLI refuses models outside FREE_MODELS."""
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    rc = g.main(["answer", "tasks.jsonl", "out", "--provider", "groq", "--model", "some-paid-model"])
    assert rc == g.EXIT_NO_KEY
    out = capsys.readouterr()
    assert "must" in out.err


def test_cli_pollinations_requires_model(monkeypatch, capsys):
    """Pollinations provider CLI requires --model or POLLINATIONS_MODEL."""
    monkeypatch.setenv("POLLINATIONS_API_KEY", POLLINATIONS_KEY)
    monkeypatch.delenv("POLLINATIONS_MODEL", raising=False)
    rc = g.main(["answer", "tasks.jsonl", "out", "--provider", "pollinations"])
    assert rc == g.EXIT_NO_KEY
    out = capsys.readouterr()
    assert "model" in out.err.lower() and "required" in out.err.lower()


def test_cli_pollinations_uses_env_base_url(monkeypatch):
    """Pollinations CLI reads base URL from POLLINATIONS_BASE_URL env."""
    calls = []

    def capture(tasks, out, model, key, **kw):
        calls.append(kw)
        return {"complete": True, "fatal": False}

    monkeypatch.setattr(g, "answer", capture)
    monkeypatch.setattr(g.cli, "load", lambda p: [])
    monkeypatch.setenv("POLLINATIONS_API_KEY", POLLINATIONS_KEY)
    monkeypatch.setenv("POLLINATIONS_BASE_URL", "https://custom.api/v1")
    monkeypatch.setenv("POLLINATIONS_MODEL", POLLINATIONS_MODEL)
    g.main(["answer", "tasks.jsonl", "out", "--provider", "pollinations"])
    assert calls[0]["url"] == "https://custom.api/v1/chat/completions"
    assert calls[0]["provider"] == "pollinations"


def test_a_provider_error_that_echoes_the_key_never_repeats_it():
    fake = Fake([(401, {}, {"error": {"message": f"Invalid API key: {POLLINATIONS_KEY}"}})])
    c = Clock()
    with pytest.raises(RuntimeError) as e:
        g.ask(fake, POLLINATIONS_BASE + "/chat/completions", POLLINATIONS_KEY, POLLINATIONS_MODEL, "p",
              g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60, sleep=c.sleep)
    assert POLLINATIONS_KEY not in str(e.value) and "[key]" in str(e.value)


def test_a_key_cut_by_the_300_character_limit_leaves_no_part_behind():
    msg = "x" * 285 + f" key {POLLINATIONS_KEY} was refused"   # the key starts at character 290
    fake = Fake([(401, {}, {"error": {"message": msg}})])
    c = Clock()
    with pytest.raises(RuntimeError) as e:
        g.ask(fake, POLLINATIONS_BASE + "/chat/completions", POLLINATIONS_KEY, POLLINATIONS_MODEL, "p",
              g.Pacer(0, sleep=c.sleep, clock=c.now), max_tokens=10, max_wait=60, sleep=c.sleep)
    assert POLLINATIONS_KEY[:6] not in str(e.value)
