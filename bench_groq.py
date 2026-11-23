"""Run the Articulate bench against an OpenAI-compatible API provider.

    GROQ_API_KEY=... python bench_groq.py answer bench/tasks-v1.jsonl runs/
    POLLINATIONS_API_KEY=... python bench_groq.py answer bench/tasks-v1.jsonl runs/ --provider pollinations --model <model>
    python bench_groq.py score bench/tasks-v1.jsonl runs/ GATE_CHECK
    python bench_groq.py check bench/tasks-v1.jsonl bench/results GATE_CHECK     (CI)
    GROQ_API_KEY=... python bench_groq.py answer bench/tasks-v1.jsonl runs/ --feedback none,mujoco,niua \\
        --gate GATE_CHECK [--turns 3]                                  (Feedback Lift: bench_loop.py)

`answer` sends each task to the model (OpenAI-compatible chat completions),
pulls the URDF out of the reply and writes runs/<model>/<task id>.urdf, with
the raw reply in runs/<model>/raw/. It is resumable: a task that already has
a reply is skipped, so a run cut short by a daily token limit picks up where
it stopped on the next run. Exit 0: every task answered. Exit 3: stopped
early, rerun to resume. Exit 4: the provider rejected the request (bad key,
unknown model); rerunning won't help.

Supported providers:
  groq (default): key from GROQ_API_KEY, model restricted to FREE_MODELS
  pollinations: key from POLLINATIONS_API_KEY, base URL from POLLINATIONS_BASE_URL
                or --base-url (default https://gen.pollinations.ai/v1), model required

`score` runs bench.py's scorer (spec + gate) over a finished run, writes
runs/<model>/results.json and keeps exactly one line per model in
runs/leaderboard.jsonl. `check` re-scores every committed run and fails if
leaderboard.jsonl disagrees with what the gate measures.

Keys are never printed or written. Pacing is polite by default: one request
at a time, at least --min-interval seconds apart, a wait for the minute's
token budget when the headers say it is spent, Retry-After honoured, and
the run stops rather than spins when the provider asks for a long wait.
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench as cli  # noqa: E402

# Provider endpoints.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
POLLINATIONS_BASE_DEFAULT = "https://gen.pollinations.ai/v1"

DEFAULT_MODEL = "openai/gpt-oss-120b"
# Text models seen on Groq's free plan (GET /openai/v1/models, 2026-11-05: each
# 1,000 requests a day and 8,000 tokens a minute). Anything else is refused, so
# a typo can't reach a model we did not choose. Groq's catalog changes; check
# the list before adding one.
FREE_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
# gpt-oss models think before answering, and the thinking counts as reply tokens.
REASONING_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
# Cloudflare in front of the API refuses Python's default User-Agent (error 1010).
USER_AGENT = "niua-articulate-bench/1.0"

PROVIDERS = ("groq", "pollinations")

SYSTEM = ("Output a complete URDF file inside a single ```xml code block. "
          "Do not explain or comment. Start the code block immediately.")

# On finish_reason=length with no extractable URDF, the answer loop retries once with this message.
CONTINUE = "Output the complete URDF now, inside a single ```xml code block."

# Exit codes of `answer`.
EXIT_DONE, EXIT_NO_KEY, EXIT_RESUME, EXIT_REJECTED = 0, 2, 3, 4

# What a client returns: (HTTP status, lower-cased headers, parsed JSON body or None).
Response = tuple[int, dict, Optional[dict]]
Post = Callable[[str, dict, dict], Response]


class StopRun(Exception):
    """The provider asked for a longer wait than we accept (the daily limit), or kept failing."""


def http_post(url: str, headers: dict, body: dict, timeout: float = 180.0) -> Response:
    """POST JSON. Never raises for the network: a dropped or broken connection is status 599,
    and its message names only the error type (never a header, so never the key)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, _json(r.read())
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except (http.client.HTTPException, OSError):
            raw = b""
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, _json(raw)
    except (http.client.HTTPException, OSError, ValueError) as e:
        # OSError covers URLError, timeouts, resets and SSL errors; HTTPException covers
        # IncompleteRead and BadStatusLine; ValueError is an illegal header value.
        return 599, {}, {"error": {"message": f"network: {type(e).__name__}"}}


def _json(raw: bytes) -> Optional[dict]:
    try:
        v = json.loads(raw or b"null")
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def seconds(v: Optional[str]) -> Optional[float]:
    """Retry-After is plain seconds. Groq's x-ratelimit-reset-* look like '1m2.5s', '7.66s', '200ms', '1h0m0s'."""
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", v)
    if not parts:
        return None
    return sum(float(n) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[u] for n, u in parts)


class Pacer:
    """At least `min_interval` seconds between the starts of two requests, and no start
    before a time named by hold() (a spent token budget)."""

    def __init__(self, min_interval: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.min_interval, self.sleep, self.clock = min_interval, sleep, clock
        self.last: Optional[float] = None
        self.not_before: Optional[float] = None

    def hold(self, s: float) -> None:
        """The next request starts no sooner than `s` seconds from now."""
        t = self.clock() + s
        self.not_before = t if self.not_before is None else max(self.not_before, t)

    def wait(self) -> None:
        now = self.clock()
        target = now
        if self.last is not None:
            target = max(target, self.last + self.min_interval)
        if self.not_before is not None:
            target = max(target, self.not_before)
        if target > now:
            self.sleep(target - now)
        self.last = self.clock()
        self.not_before = None


def request_body(model: str, prompt: str, max_tokens: int, reasoning_effort: Optional[str],
                 messages: Optional[list[dict]] = None) -> dict:
    """The chat request. `messages` (the Feedback Lift retries) replaces the single user turn;
    the system message is always ours."""
    turns = messages if messages is not None else [{"role": "user", "content": prompt}]
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, *turns]}
    if reasoning_effort and model in REASONING_MODELS:
        body["reasoning_effort"] = reasoning_effort
    return body


def budget_wait(h: dict, need: int) -> Optional[float]:
    """After a success: if this minute's token budget can't cover the next request, how long until it resets."""
    try:
        left = int(float(h.get("x-ratelimit-remaining-tokens", "")))
    except ValueError:
        return None
    return seconds(h.get("x-ratelimit-reset-tokens")) if left < need else None


def ask(post: Post, url: str, key: str, model: str, prompt: str, pacer: Pacer, *, max_tokens: int,
        max_wait: float, retries: int = 4, sleep: Callable[[float], None] = time.sleep,
        reasoning_effort: Optional[str] = None, messages: Optional[list[dict]] = None) -> dict:
    """Send one task. Returns the response body. Raises StopRun on a long wait or repeated failure,
    RuntimeError on a request the provider rejects outright (bad key, unknown model)."""
    body = request_body(model, prompt, max_tokens, reasoning_effort, messages)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": USER_AGENT}
    last = ""
    for attempt in range(retries + 1):
        pacer.wait()
        status, h, data = post(url, headers, body)
        if status == 200 and data and data.get("choices"):
            # Polite: hold the next request until this minute's token budget refills,
            # rather than draw a 429. Held, not slept, so the reply is saved first.
            wait = budget_wait(h, max_tokens + 1000)
            if wait:
                pacer.hold(min(wait, max_wait))
            return data
        # Some providers echo the key they were sent in the error text; never pass it on.
        last = f"HTTP {status}: {_message(data, key)}"
        if status == 429 or status >= 500 or (status == 200 and not (data and data.get("choices"))):
            wait = (seconds(h.get("retry-after")) if status == 429 else None)
            if wait is None and status == 429:
                wait = seconds(h.get("x-ratelimit-reset-tokens")) or seconds(h.get("x-ratelimit-reset-requests"))
            if wait is None:
                wait = 2.0 * 2 ** attempt
            if wait > max_wait:
                raise StopRun(f"provider asks to wait {wait:.0f}s (over --max-wait {max_wait:.0f}s); rerun later to resume")
            if attempt < retries:
                sleep(wait)
                continue
            raise StopRun(f"gave up after {retries} retries ({last})")
        raise RuntimeError(last)
    raise StopRun(f"gave up after {retries} retries ({last})")


def _message(data: Optional[dict], key: str = "") -> str:
    """The provider's error text, cut to 300 characters. The key is replaced before the cut,
    so a key that straddles the cut can't leave its first characters behind."""
    err = (data or {}).get("error")
    msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
    if key:
        msg = msg.replace(key, "[key]")
    return msg[:300]


FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)```", re.S)
BARE = re.compile(r"(?:<\?xml[^>]*\?>\s*)?<robot\b.*?</robot>", re.S)
# Reasoning some models write inline (Qwen3 on Groq: <think>...</think> in the content).
# An unterminated block runs to the end: a reply cut off mid-thought has no answer.
THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S | re.I)


def extract_urdf(text: Optional[str]) -> Optional[str]:
    """The URDF in a reply, ignoring any inline reasoning: the first fenced block holding
    <robot ...>, else a bare <robot>...</robot>."""
    text = THINK.sub("", text or "")
    for block in FENCE.findall(text):
        if "<robot" in block:
            return block.strip() + "\n"
    m = BARE.search(text)
    return m.group(0).strip() + "\n" if m else None


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def answer(tasks: list[dict], out: str, model: str, key: str, *, url: str = GROQ_URL,
           provider: str = "groq", post: Post = http_post, pacer: Optional[Pacer] = None,
           max_tokens: int = 7500, max_wait: float = 120.0, limit: Optional[int] = None,
           log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
           reasoning_effort: Optional[str] = "low") -> dict:
    """Answer every task that has no reply yet. Never raises on a provider limit or rejection:
    it stops and reports. `limit` caps HTTP requests this run, retries included.
    On finish_reason=length with no extractable URDF, retries once with a continuation prompt."""
    d = os.path.join(out, slug(model))
    raw = os.path.join(d, "raw")
    os.makedirs(raw, exist_ok=True)
    pacer = pacer or Pacer(20.0, sleep=sleep)
    settings = {"model": model, "provider": provider, "temperature": 0, "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort if model in REASONING_MODELS else None, "system": SYSTEM}
    run_path = os.path.join(d, "run.json")
    if os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as f:
            before = json.load(f)
        if before != settings:
            raise SystemExit(f"{run_path} was made with other settings; use a new output folder")
    else:
        with open(run_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
            f.write("\n")

    calls = 0

    def counted(url: str, headers: dict, body: dict) -> Response:
        nonlocal calls
        if limit is not None and calls >= limit:
            raise StopRun(f"--limit {limit} reached")
        calls += 1
        return post(url, headers, body)

    done = sent = no_urdf = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    stopped, fatal = None, False
    for t in tasks:
        raw_path = os.path.join(raw, f"{t['id']}.json")
        if os.path.exists(raw_path):
            done += 1
            continue
        if limit is not None and calls >= limit:
            stopped = f"--limit {limit} reached"
            break
        try:
            data = ask(counted, url, key, model, t["prompt"], pacer, max_tokens=max_tokens, max_wait=max_wait,
                       sleep=sleep, reasoning_effort=reasoning_effort)
        except StopRun as e:
            stopped = str(e)
            log(f"stopped: {e}")
            break
        except RuntimeError as e:
            stopped, fatal = str(e), True
            log(f"rejected: {e}")
            break
        sent += 1
        choice = data["choices"][0] if data.get("choices") else {}
        text = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason")
        u = data.get("usage") or {}
        task_usage = {k: int(u.get(k) or 0) for k in ("prompt_tokens", "completion_tokens")}
        for k in usage:
            usage[k] += task_usage.get(k, 0)
        urdf = extract_urdf(text)
        # Length-retry: if truncated without a complete URDF, try once more with a continuation prompt.
        if urdf is None and finish == "length":
            if limit is not None and calls >= limit:
                stopped = f"--limit {limit} reached"
                break
            try:
                data2 = ask(counted, url, key, model, CONTINUE, pacer, max_tokens=max_tokens, max_wait=max_wait,
                            sleep=sleep, reasoning_effort=reasoning_effort)
            except StopRun as e:
                stopped = str(e)
                log(f"stopped: {e}")
                break
            except RuntimeError as e:
                stopped, fatal = str(e), True
                log(f"rejected: {e}")
                break
            choice2 = data2["choices"][0] if data2.get("choices") else {}
            text2 = (choice2.get("message") or {}).get("content") or ""
            u2 = data2.get("usage") or {}
            for k in usage:
                usage[k] += int(u2.get(k) or 0)
            for k in task_usage:
                task_usage[k] += int(u2.get(k) or 0)
            urdf = extract_urdf(text2)
            text = text + "\n---length-retry---\n" + text2
            finish = choice2.get("finish_reason")
        if urdf is None:
            no_urdf += 1
        else:
            with open(os.path.join(d, f"{t['id']}.urdf"), "w", encoding="utf-8", newline="\n") as f:
                f.write(urdf)
        # The raw reply is written last: it marks the task done for a resumed run.
        with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"id": t["id"], "model": model, "content": text, "usage": task_usage,
                       "finish_reason": finish, "urdf_found": urdf is not None},
                      f, indent=2, sort_keys=True)
            f.write("\n")
        done += 1
        log(f"  {done}/{len(tasks)} {t['id']}{'' if urdf else '  (no URDF in reply)'}")
    return {"model": model, "tasks": len(tasks), "done": done, "sent": sent, "requests": calls, "no_urdf": no_urdf,
            "usage": usage, "stopped": stopped, "fatal": fatal, "complete": done == len(tasks)}


def asked(tasks: list[dict], d: str) -> int:
    """How many tasks have a reply on disk (a reply without a URDF counts: the model was asked)."""
    return sum(os.path.exists(os.path.join(d, "raw", f"{t['id']}.json")) for t in tasks)


def leaderboard_line(res: dict, model: str, date: str, settings: Optional[dict] = None) -> dict:
    s = res["summary"]
    st = settings or {}
    return {"bench": "articulate/tasks-v1", "contract": res["contract"], "date": date, "model": model,
            "provider": st.get("provider", "groq"), "tasks": res["tasks"], "answered": s["answered"],
            "spec_ok": s["spec_ok"], "gate_ok": s["gate_ok"], "pass": s["pass"], "score": res["score"],
            "max_tokens": st.get("max_tokens"), "reasoning_effort": st.get("reasoning_effort")}


def read_board(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_board(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in sorted(rows, key=lambda r: r["model"]):
            f.write(json.dumps(row, sort_keys=True) + "\n")


def score(tasks_path: str, out: str, model: str, gate: Optional[str], date: Optional[str] = None,
          partial: bool = False) -> dict:
    """Score runs/<model>/ with bench.py, write results.json, and set this model's one line in
    leaderboard.jsonl. Refuses an unfinished run unless partial=True: unasked tasks are not
    model failures."""
    tasks = cli.load(tasks_path)
    d = os.path.join(out, slug(model))
    n = asked(tasks, d)
    if n < len(tasks) and not partial:
        raise SystemExit(f"{d}: {n}/{len(tasks)} tasks asked; run 'answer' again to finish before scoring")
    res = cli.score(tasks, d, gate)
    res["model"] = model
    with open(os.path.join(d, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2, sort_keys=True)
        f.write("\n")
    run_path = os.path.join(d, "run.json")
    settings = None
    if os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as f:
            settings = json.load(f)
    line = leaderboard_line(res, model, date or dt.datetime.now(dt.timezone.utc).date().isoformat(), settings)
    board_path = os.path.join(out, "leaderboard.jsonl")
    write_board(board_path, [r for r in read_board(board_path) if r.get("model") != model] + [line])
    return line


# Fields a committed leaderboard line must share with a fresh scoring (the date may differ).
CHECKED = ("bench", "contract", "model", "provider", "tasks", "answered", "spec_ok", "gate_ok", "pass", "score",
           "max_tokens", "reasoning_effort")


def check(tasks_path: str, out: str, gate: Optional[str], log: Callable[[str], None] = print) -> list[str]:
    """Re-score every committed run under `out` and compare with out/leaderboard.jsonl.
    Returns the problems; empty means every line is what the gate measures today."""
    if not os.path.isdir(out):
        log(f"no committed runs under {out}")
        return []
    rows: dict[str, list[dict]] = {}
    for row in read_board(os.path.join(out, "leaderboard.jsonl")):
        rows.setdefault(row.get("model"), []).append(row)
    tasks = cli.load(tasks_path)
    problems, seen = [], set()
    for model, rs in sorted(rows.items(), key=lambda kv: str(kv[0])):
        if len(rs) > 1:
            problems.append(f"{model}: {len(rs)} lines in leaderboard.jsonl; keep one")
    for name in sorted(os.listdir(out)):
        run_path = os.path.join(out, name, "run.json")
        if not os.path.exists(run_path):
            continue
        with open(run_path, encoding="utf-8") as f:
            settings = json.load(f)
        model = settings["model"]
        seen.add(model)
        d = os.path.join(out, name)
        n = asked(tasks, d)
        if n < len(tasks):
            problems.append(f"{model}: only {n}/{len(tasks)} tasks asked; finish the run before committing it")
            continue
        res = cli.score(tasks, d, gate)
        fresh = leaderboard_line(res, model, "", settings)
        log(f"{model}: {fresh['pass']}/{fresh['tasks']} pass (spec {fresh['spec_ok']}, gate {fresh['gate_ok']}, "
            f"answered {fresh['answered']})  score {fresh['score']:.2%}")
        if model not in rows:
            problems.append(f"{model}: no line in leaderboard.jsonl; add {json.dumps({k: fresh[k] for k in CHECKED}, sort_keys=True)}")
            continue
        for row in rows[model]:
            diff = {k: (row.get(k), fresh[k]) for k in CHECKED if row.get(k) != fresh[k]}
            if diff:
                problems.append(f"{model}: leaderboard.jsonl says {', '.join(f'{k}={a}' for k, (a, _) in diff.items())} "
                                f"but the gate measures {', '.join(f'{k}={b}' for k, (_, b) in diff.items())}")
    for model in sorted(set(rows) - seen, key=str):
        problems.append(f"{model}: in leaderboard.jsonl but its answers are not committed")
    return problems


KEY_OK = re.compile(r"[\x21-\x7e]+")   # printable ASCII, no spaces or line breaks


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer", help="send the tasks to the model")
    a.add_argument("tasks")
    a.add_argument("out")
    a.add_argument("--provider", default="groq", choices=PROVIDERS,
                   help="API provider (default groq)")
    a.add_argument("--model", help="model name (default for groq: openai/gpt-oss-120b; required for pollinations)")
    a.add_argument("--base-url", help="base URL for pollinations (default https://gen.pollinations.ai/v1)")
    a.add_argument("--min-interval", type=float, default=20.0, help="seconds between requests (default 20)")
    a.add_argument("--max-tokens", type=int, default=7500,
                   help="reply cap per task (default 7500; free tier allows 8000 tokens a minute)")
    a.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high"),
                   help="gpt-oss models only (default low, so thinking leaves room for the URDF)")
    a.add_argument("--max-wait", type=float, default=120.0, help="stop instead of waiting longer than this (default 120 s)")
    a.add_argument("--limit", type=int, help="send at most this many HTTP requests this run, retries included")
    a.add_argument("--turns", type=int,
                   help="tries per task (default 1: the single-try bench; with --feedback, 3)")
    a.add_argument("--feedback",
                   help="Feedback Lift (bench_loop.py): retry a failed task with none, mujoco or niua feedback; "
                        "comma separated runs several, e.g. none,mujoco,niua. Needs --gate")
    a.add_argument("--gate", help="path to the gate's gate_check binary (with --feedback: every try is checked)")
    s = sub.add_parser("score", help="score a finished run with bench.py (spec + gate)")
    s.add_argument("tasks")
    s.add_argument("out")
    s.add_argument("gate", help="path to the gate's gate_check binary")
    s.add_argument("--model", help="model name (default: openai/gpt-oss-120b)")
    s.add_argument("--date", help="leaderboard date (default today, UTC)")
    s.add_argument("--partial", action="store_true", help="score an unfinished run (unasked tasks count as missing)")
    k = sub.add_parser("check", help="CI: re-score every committed run and compare with leaderboard.jsonl")
    k.add_argument("tasks")
    k.add_argument("out")
    k.add_argument("gate", help="path to the gate's gate_check binary")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        problems = check(args.tasks, args.out, os.path.abspath(args.gate))
        for p in problems:
            print("MISMATCH", p)
        return 1 if problems else 0
    if args.cmd == "score":
        model = args.model or DEFAULT_MODEL
        line = score(args.tasks, args.out, model, os.path.abspath(args.gate), args.date, partial=args.partial)
        print(f"{line['model']}: {line['pass']}/{line['tasks']} pass (spec {line['spec_ok']}, gate {line['gate_ok']}, "
              f"answered {line['answered']})  score {line['score']:.2%}")
        return 0
    # answer command
    modes, turns = [], 1
    if args.feedback is not None or (args.turns is not None and args.turns != 1):
        import bench_loop as lp
        turns = args.turns if args.turns is not None else lp.LIFT_TURNS
        if args.feedback is None or turns < 2:
            ap.error("the Feedback Lift loop takes --feedback (none, mujoco, niua) with --turns 2 or more; "
                     "--turns 1 without --feedback is the single-try bench")
        try:
            modes = lp.parse_modes(args.feedback)
        except ValueError as e:
            ap.error(str(e))
        if not args.gate:
            ap.error("--feedback needs --gate GATE_CHECK: every try is checked before the next one")
    provider = args.provider
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        key_name = "GROQ_API_KEY"
        url = GROQ_URL
        model = args.model or DEFAULT_MODEL
        if model not in FREE_MODELS:
            print(f"groq provider: --model must be one of {FREE_MODELS}", file=sys.stderr)
            return EXIT_NO_KEY
    elif provider == "pollinations":
        key = os.environ.get("POLLINATIONS_API_KEY", "").strip()
        key_name = "POLLINATIONS_API_KEY"
        base = args.base_url or os.environ.get("POLLINATIONS_BASE_URL", "").strip() or POLLINATIONS_BASE_DEFAULT
        url = base.rstrip("/") + "/chat/completions"
        model = args.model or os.environ.get("POLLINATIONS_MODEL", "").strip()
        if not model:
            print("pollinations provider: --model or POLLINATIONS_MODEL is required", file=sys.stderr)
            return EXIT_NO_KEY
    else:
        print(f"unknown provider: {provider}", file=sys.stderr)
        return EXIT_NO_KEY
    if not key:
        print(f"{key_name} is not set; nothing sent", file=sys.stderr)
        return EXIT_NO_KEY
    if not KEY_OK.fullmatch(key):
        print(f"{key_name} contains spaces, line breaks or non-ASCII characters; nothing sent", file=sys.stderr)
        return EXIT_NO_KEY
    if modes:
        r = lp.run(cli.load(args.tasks), args.out, model, key, url=url, provider=provider, modes=modes, turns=turns,
                   check=lp.gate_checker(os.path.abspath(args.gate)), pacer=Pacer(args.min_interval),
                   max_tokens=args.max_tokens, max_wait=args.max_wait, limit=args.limit,
                   reasoning_effort=args.reasoning_effort)
    else:
        r = answer(cli.load(args.tasks), args.out, model, key, url=url, provider=provider,
                   pacer=Pacer(args.min_interval), max_tokens=args.max_tokens, max_wait=args.max_wait,
                   limit=args.limit, reasoning_effort=args.reasoning_effort)
    print(json.dumps(r, sort_keys=True))
    if r["complete"]:
        return EXIT_DONE
    return EXIT_REJECTED if r["fatal"] else EXIT_RESUME


if __name__ == "__main__":
    sys.exit(main())
