"""Feedback Lift: how many extra passes each kind of feedback buys on the Articulate bench.

    GROQ_API_KEY=... python bench_groq.py answer bench/tasks-v1.jsonl runs/ \\
        --feedback none,mujoco,niua --gate GATE_CHECK [--turns 3]
    POLLINATIONS_API_KEY=... python bench_groq.py answer bench/tasks-v1.jsonl runs/ \\
        --provider pollinations --model deepseek/deepseek-v4.1-flash --max-tokens 32000 \\
        --feedback none,mujoco,niua --gate GATE_CHECK

The single-try bench asks once per task. Here a task that fails is asked again,
up to --turns tries in all. Every retry carries the task, the model's own last
URDF and one kind of feedback about it:

  none     only "that failed, try again"
  mujoco   only the raw text the sim check printed: simcheck's MuJoCo messages
           and the parser's error, with no rule ids and no fixes
  niua     NIUA's structured notes: every gate failure as rule id, simcheck
           check, subject, expected, got and fix (what articulated.sim-ready@1
           already reports), and the parser's error

No mode sees the bench's own spec failures (size, joint count, joint range):
they are the grader's answer key, not anything the simulator or the gate
reports, and one of them (the minimum joint range) is not in the task text. A
try that fails only the spec hears only that it failed, in every mode, so the
mujoco and niua modes differ in the form of the same report, not in what it
covers.

Turn 1 is the single-try bench's own request (same system prompt, same task,
same reply cap, same one retry on a reply cut off at the length limit) and is
asked once and shared by every mode, so the modes differ only in what they say
after a failure: a paired comparison.

A pass is what bench.py counts: the spec holds and the gate passes. Each task
gets one record per mode: the turn it passed on (or none), the mode, the model,
the tokens and requests it used (turn 1 included). Each (model, mode) gets one
summary line in OUT/loop-leaderboard.jsonl: pass@1, pass@turns, and the lift
between them.

Any provider bench_groq.py answer takes (groq or pollinations), with its
pacing: one request at a time, the minute's token budget respected when the
response headers name one, the run stopped (exit 3) on a daily limit and
resumed on the next run from what is on disk. Every request of a run, turn 1
and every retry, goes to the same provider, URL, key and model. On Groq's
free tier a retry is longer than a first try (it carries the last URDF), so
its reply cap shrinks to keep the request inside 8,000 tokens a minute. Other
providers have no such cap here, so a retry keeps the full --max-tokens and
the last URDF whole.

On disk, under OUT/loop/<model>/:
  run.json                        settings; a resume must match them
  turn1/<id>.urdf, raw/, check/   the shared first try
  <mode>/turn<k>/...              the retries of that mode
  <mode>/tasks/<id>.json          the record, written last (the task is done)
  <mode>/results.jsonl            every record, rewritten at the end of a run

Simulation only, no hardware tested.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench as cli  # noqa: E402
import bench_groq as gq  # noqa: E402
from articulate import bench as b  # noqa: E402

MODES = ("none", "mujoco", "niua")
LIFT_TURNS = 3            # tries per task when --feedback is given without --turns
FREE_TPM = 8000           # Groq free plan: tokens a minute per model (see bench_groq.FREE_MODELS)
# The tokens-a-minute cap a retry must fit, per provider. A provider not named here has none:
# its retries keep the full reply cap and the last URDF whole.
PROVIDER_TPM = {"groq": FREE_TPM}
MIN_REPLY = 2000          # never cap a retry's reply below this; the last URDF is cut instead
CHARS_PER_TOKEN = 2       # cautious: URDF runs 2.6 to 3.0 characters a token (o200k, the committed
                          # gpt-oss answers), so 3 would undercount a long URDF and draw a 413
MAX_NOTES = 12            # feedback lines per retry
MAX_NOTE_CHARS = 400      # characters per feedback line
FEEDBACK_VERSION = 1      # bump when the feedback wording changes; a resume must match it
GATE_FIELDS = ("rule", "severity", "path", "subject", "expected", "got", "fix")

TRY_AGAIN = "Output the complete corrected URDF inside a single ```xml code block."
NO_URDF = "no URDF found in the reply"

Check = Callable[[dict, Optional[str], Optional[str]], dict]


# ---- feedback -----------------------------------------------------------------

def _clip(s: object) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= MAX_NOTE_CHARS else s[:MAX_NOTE_CHARS - 3] + "..."


def _unique(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        line = _clip(line)
        if line and line not in out:
            out.append(line)
    if len(out) > MAX_NOTES:
        out = out[:MAX_NOTES] + [f"(and {len(out) - MAX_NOTES} more)"]
    return out


def raw_lines(check: dict) -> list[str]:
    """What the sim check printed, as text: simcheck's messages (MuJoCo), the parser's error."""
    if not check["answered"]:
        return [NO_URDF]
    lines = [s.get("got", "") for s in check["spec"] if s["rule"] == "spec.parse"]
    lines += [f.get("got", "") for f in (check["gate"] or {}).get("failures", [])]
    return _unique(lines)


def niua_lines(check: dict) -> list[str]:
    """NIUA's notes as the gate writes them: rule id, where, expected, got, fix; and the parser's error.
    The bench's spec failures are left out, as in every mode: they are the grader's answer key."""
    if not check["answered"]:
        return [f"answer.missing: {NO_URDF}"]
    lines = [f"spec.parse: got {s.get('got', '')}" for s in check["spec"] if s["rule"] == "spec.parse"]
    for f in (check["gate"] or {}).get("failures", []):
        where = " ".join(x for x in (f.get("path", ""), f.get("subject", "")) if x)
        lines.append(f"{f['rule']}{' ' + where if where else ''}: expected {f.get('expected', '')}; "
                     f"got {f.get('got', '')}; fix: {f.get('fix', '')}")
    return _unique(lines)


def feedback(mode: str, check: dict) -> str:
    """The user turn of a retry, for a failed check."""
    if mode == "none":
        return f"That URDF failed. Try again. {TRY_AGAIN}"
    if mode == "mujoco":
        lines = raw_lines(check)
        head = "That URDF failed. The simulator and parser said:" if lines else \
            "That URDF failed. The simulator and parser printed no error."
    elif mode == "niua":
        lines = niua_lines(check)
        head = ("That URDF failed. NIUA's check (articulated.sim-ready@1 in MuJoCo) reported, one line per "
                "failure: rule, where, expected, got, fix.") if lines else             "That URDF failed. NIUA's check (articulated.sim-ready@1 in MuJoCo) reported no failure."
    else:
        raise ValueError(f"unknown feedback mode {mode!r}; known: {', '.join(MODES)}")
    return "\n".join([head, *(f"- {x}" for x in lines), TRY_AGAIN])


# ---- the retry request ----------------------------------------------------------

def estimate(messages: list[dict]) -> int:
    """Tokens a request's messages take, guessed high (system prompt included)."""
    chars = len(gq.SYSTEM) + sum(len(m["content"]) for m in messages)
    return chars // CHARS_PER_TOKEN + 20 * (len(messages) + 1)


def retry_messages(task: dict, last_urdf: Optional[str], note: str, max_tokens: int,
                   tpm: Optional[int] = FREE_TPM) -> tuple[list[dict], int]:
    """The task, the model's last URDF, the feedback; and a reply cap that keeps prompt plus reply
    inside the minute's tokens. When the URDF is too long for a reply of MIN_REPLY, it is cut.
    tpm None (a provider with no minute cap): the URDF goes back whole and the cap is max_tokens."""
    def build(urdf: Optional[str]) -> list[dict]:
        msgs = [{"role": "user", "content": task["prompt"]}]
        if urdf:
            msgs.append({"role": "assistant", "content": f"```xml\n{urdf}```"})
        msgs.append({"role": "user", "content": note})
        return msgs

    msgs = build(last_urdf)
    if tpm is None:
        return msgs, max_tokens
    room = tpm - estimate(msgs) - 500   # 500: headroom for the provider's own count
    if room < MIN_REPLY and last_urdf:
        mark = "\n<!-- cut here: too long to send back in full -->\n"
        keep = max(0, len(last_urdf) - (MIN_REPLY - room) * CHARS_PER_TOKEN - len(mark) - CHARS_PER_TOKEN)
        msgs = build(last_urdf[:keep] + mark)
        room = tpm - estimate(msgs) - 500
    return msgs, max(MIN_REPLY, min(max_tokens, room))


# ---- the checker ----------------------------------------------------------------

def gate_checker(gate_bin: str) -> Check:
    """bench.py's pass rule for one answer: the spec, then articulated.sim-ready@1 via gate_check."""
    import gate as gt

    def check(task: dict, urdf: Optional[str], path: Optional[str]) -> dict:
        if urdf is None or path is None:
            return {"answered": False, "spec": [{"rule": "answer.missing"}], "gate": None, "pass": False}
        spec = b.check_spec(task, urdf)
        verdicts = gt.run_gate(gate_bin, cli.CONTRACT, [path], fields=GATE_FIELDS, log=lambda s: None)
        verdict = verdicts.get(path, {"verdict": "ERROR", "failures": []})
        return {"answered": True, "spec": spec, "gate": verdict, "pass": not spec and verdict["verdict"] == "PASS"}

    return check


def rule_ids(check: dict) -> list[str]:
    return [s["rule"] for s in check["spec"]] + [f["rule"] for f in (check["gate"] or {}).get("failures", [])]


# ---- the run --------------------------------------------------------------------

def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def turn_dir(d: str, mode: str, k: int) -> str:
    return os.path.join(d, "turn1") if k == 1 else os.path.join(d, mode, f"turn{k}")


def record_path(d: str, mode: str, task_id: str) -> str:
    return os.path.join(d, mode, "tasks", f"{task_id}.json")


def run(tasks: list[dict], out: str, model: str, key: str, *, modes: list[str], turns: int, check: Check,
        url: str = gq.GROQ_URL, provider: str = "groq", post: gq.Post = gq.http_post,
        pacer: Optional[gq.Pacer] = None, max_tokens: int = 7500, max_wait: float = 120.0,
        limit: Optional[int] = None, log: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep, reasoning_effort: Optional[str] = "low",
        tpm: Optional[int] = None, date: Optional[str] = None) -> dict:
    """Run every (task, mode) that has no record yet, then write each mode's results and summary.
    Never raises on a provider limit or rejection: it stops and reports, like bench_groq.answer.
    Every request goes to `url` with `key` and `model`, as bench_groq.answer sends them. `tpm` None
    means the provider's own cap (PROVIDER_TPM: 8,000 on groq, none elsewhere)."""
    if tpm is None:
        tpm = PROVIDER_TPM.get(provider)
    if turns < 2:
        raise ValueError("the Feedback Lift loop needs turns >= 2; turns 1 is bench_groq.py answer")
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise ValueError(f"feedback modes must be some of {', '.join(MODES)}; got {', '.join(modes) or 'none'}")
    d = os.path.join(out, "loop", gq.slug(model))
    os.makedirs(d, exist_ok=True)
    pacer = pacer or gq.Pacer(20.0, sleep=sleep)
    settings = {"model": model, "provider": provider, "temperature": 0, "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort if model in gq.REASONING_MODELS else None, "system": gq.SYSTEM,
                "turns": turns, "tpm": tpm, "feedback_version": FEEDBACK_VERSION}
    run_path = os.path.join(d, "run.json")
    if os.path.exists(run_path):
        if _read(run_path) != settings:
            raise SystemExit(f"{run_path} was made with other settings; use a new output folder")
    else:
        _write(run_path, settings)

    calls = 0

    def counted(url: str, headers: dict, body: dict) -> gq.Response:
        nonlocal calls
        if limit is not None and calls >= limit:
            raise gq.StopRun(f"--limit {limit} reached")
        calls += 1
        return post(url, headers, body)

    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def ask(prompt: str, messages: Optional[list[dict]], cap: int) -> dict:
        data = gq.ask(counted, url, key, model, prompt, pacer, max_tokens=cap, max_wait=max_wait, sleep=sleep,
                      reasoning_effort=reasoning_effort, messages=messages)
        choice = data["choices"][0] if data.get("choices") else {}
        u = data.get("usage") or {}
        got = {k: int(u.get(k) or 0) for k in usage}
        for k in usage:
            usage[k] += got[k]
        return {"text": (choice.get("message") or {}).get("content") or "", "finish": choice.get("finish_reason"),
                "usage": got}

    def reply(task: dict, tdir: str, messages: Optional[list[dict]], cap: int) -> dict:
        """One try, from disk if it was asked before. Like bench_groq.answer: a reply cut off at the
        length limit with no URDF gets one more request asking for the code block only."""
        raw_path = os.path.join(tdir, "raw", f"{task['id']}.json")
        urdf_path = os.path.join(tdir, f"{task['id']}.urdf")
        if os.path.exists(raw_path):
            r = _read(raw_path)
            urdf = None
            if r["urdf_found"]:
                with open(urdf_path, encoding="utf-8", errors="replace") as f:
                    urdf = f.read()
            return {**r, "urdf": urdf, "urdf_path": urdf_path if urdf is not None else None}
        first = ask(task["prompt"], messages, cap)
        text, finish, got, requests = first["text"], first["finish"], dict(first["usage"]), 1
        urdf = gq.extract_urdf(text)
        if urdf is None and finish == "length":
            # Turn 1 sends CONTINUE alone, exactly as the single-try bench does; a retry keeps its context.
            more = ask(gq.CONTINUE, None if messages is None else messages + [{"role": "user", "content": gq.CONTINUE}],
                       cap)
            for k in got:
                got[k] += more["usage"][k]
            urdf = gq.extract_urdf(more["text"])
            text = text + "\n---length-retry---\n" + more["text"]
            finish = more["finish"]
            requests += 1
        if urdf is not None:
            os.makedirs(tdir, exist_ok=True)
            with open(urdf_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(urdf)
        r = {"id": task["id"], "model": model, "content": text, "usage": got, "finish_reason": finish,
             "urdf_found": urdf is not None, "requests": requests, "max_tokens": cap,
             "feedback": messages[-1]["content"] if messages else None}
        _write(raw_path, r)   # written last: the try is done
        return {**r, "urdf": urdf, "urdf_path": urdf_path if urdf is not None else None}

    def checked(task: dict, tdir: str, r: dict) -> dict:
        path = os.path.join(tdir, "check", f"{task['id']}.json")
        if os.path.exists(path):
            return _read(path)
        c = check(task, r["urdf"], os.path.abspath(r["urdf_path"]) if r["urdf_path"] else None)
        _write(path, c)
        return c

    def solve(task: dict, mode: str) -> dict:
        attempts, tokens, requests, pass_turn = [], {"prompt_tokens": 0, "completion_tokens": 0}, 0, None
        last: Optional[tuple[dict, dict]] = None
        for k in range(1, turns + 1):
            tdir = turn_dir(d, mode, k)
            if k == 1:
                r = reply(task, tdir, None, max_tokens)
            else:
                msgs, cap = retry_messages(task, last[0]["urdf"], feedback(mode, last[1]), max_tokens, tpm)
                r = reply(task, tdir, msgs, cap)
            c = checked(task, tdir, r)
            for key_ in tokens:
                tokens[key_] += int(r["usage"].get(key_, 0))
            requests += int(r.get("requests", 1))
            attempts.append({"turn": k, "pass": c["pass"], "urdf_found": r["urdf_found"], "rules": rule_ids(c)})
            if c["pass"]:
                pass_turn = k
                break
            last = (r, c)
        tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
        return {"id": task["id"], "model": model, "feedback": mode, "turns": turns, "pass_turn": pass_turn,
                "tokens": tokens, "requests": requests, "attempts": attempts}

    stopped, fatal = None, False
    for t in tasks:
        try:
            for mode in modes:
                rp = record_path(d, mode, t["id"])
                if os.path.exists(rp):
                    continue
                rec = solve(t, mode)
                _write(rp, rec)
                log(f"  {t['id']} {mode}: " + (f"pass on turn {rec['pass_turn']}" if rec["pass_turn"] else
                                                 f"no pass in {turns} turns"))
        except gq.StopRun as e:
            stopped = str(e)
            log(f"stopped: {e}")
            break
        except RuntimeError as e:
            stopped, fatal = str(e), True
            log(f"rejected: {e}")
            break

    lines = {m: summarize(tasks, d, m, settings, date) for m in modes}
    board = os.path.join(out, "loop-leaderboard.jsonl")
    keep = [r for r in gq.read_board(board) if (r.get("model"), r.get("feedback")) not in {(model, m) for m in modes}]
    write_board(board, keep + list(lines.values()))
    for line in lines.values():
        log(summary_text(line))
    return {"model": model, "tasks": len(tasks), "modes": lines, "requests": calls, "usage": usage,
            "stopped": stopped, "fatal": fatal, "complete": all(x["complete"] for x in lines.values())}


def summarize(tasks: list[dict], d: str, mode: str, settings: dict, date: Optional[str] = None) -> dict:
    """Write <mode>/results.jsonl (one record per task done) and return the mode's summary line."""
    recs = [_read(record_path(d, mode, t["id"])) for t in tasks if os.path.exists(record_path(d, mode, t["id"]))]
    os.makedirs(os.path.join(d, mode), exist_ok=True)
    with open(os.path.join(d, mode, "results.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    n, turns = len(tasks), settings["turns"]
    by_turn = [sum(r["pass_turn"] == k for r in recs) for k in range(1, turns + 1)]
    p1, pk = by_turn[0], sum(by_turn)
    return {"bench": "articulate/tasks-v1", "contract": cli.CONTRACT, "model": settings["model"], "provider": settings["provider"],
            "feedback": mode, "turns": turns, "tasks": n, "done": len(recs), "complete": len(recs) == n,
            "pass_at_1": p1, "pass_at_turns": pk, "lift": pk - p1, "pass_by_turn": by_turn,
            "score_at_1": round(p1 / n, 4) if n else 0.0, "score_at_turns": round(pk / n, 4) if n else 0.0,
            "lift_points": round((pk - p1) / n, 4) if n else 0.0,
            "tokens": sum(r["tokens"]["total_tokens"] for r in recs), "requests": sum(r["requests"] for r in recs),
            "max_tokens": settings["max_tokens"], "reasoning_effort": settings["reasoning_effort"],
            "date": date or dt.datetime.now(dt.timezone.utc).date().isoformat()}


def summary_text(line: dict) -> str:
    return (f"{line['model']} feedback={line['feedback']}: pass@1 {line['pass_at_1']}/{line['tasks']}, "
            f"pass@{line['turns']} {line['pass_at_turns']}/{line['tasks']}, lift +{line['lift']} "
            f"({line['done']}/{line['tasks']} tasks done, {line['tokens']} tokens)")


def write_board(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in sorted(rows, key=lambda r: (str(r.get("model")), MODES.index(r["feedback"])
                                               if r.get("feedback") in MODES else len(MODES))):
            f.write(json.dumps(row, sort_keys=True) + "\n")


def parse_modes(text: str) -> list[str]:
    """'none,mujoco,niua' -> ['none', 'mujoco', 'niua'], in that order, no repeats."""
    got = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in got if m not in MODES]
    if bad or not got:
        raise ValueError(f"--feedback takes {', '.join(MODES)} (comma separated); got {text!r}")
    return [m for m in MODES if m in got]


# ---- why a retry did not fix it ------------------------------------------------------

MAX_TABLE_RULES = 6       # rule ids shown per try in the failure table


def try_text(attempt: dict) -> str:
    """One try of a record, short: pass, no URDF, or its failing rule ids (each once)."""
    if attempt["pass"]:
        return "pass"
    if not attempt["urdf_found"]:
        return "no URDF"
    rules = list(dict.fromkeys(attempt["rules"]))
    if not rules:
        return "fail (no rule)"
    more = len(rules) - MAX_TABLE_RULES
    return ",".join(rules[:MAX_TABLE_RULES]) + (f",+{more} more" if more > 0 else "")


def failure_line(rec: dict) -> str:
    """'<task> <mode>: t1 <rules> | t2 <rules> | t3 pass => pass on turn 3'."""
    tries = " | ".join(f"t{a['turn']} {try_text(a)}" for a in rec["attempts"])
    end = f"pass on turn {rec['pass_turn']}" if rec["pass_turn"] else "no pass"
    return f"{rec['id']} {rec['feedback']}: {tries} => {end}"


def failure_table(out: str) -> list[str]:
    """One line per (task, mode) that did not pass on turn 1, from every OUT/loop/<model>/<mode>/results.jsonl,
    tasks sorted by id and modes in MODES order, with one header line per model."""
    root = os.path.join(out, "loop")
    lines: list[str] = []
    if not os.path.isdir(root):
        return lines
    for name in sorted(os.listdir(root)):
        recs: dict[str, dict[str, dict]] = {}
        model = name
        for mode in MODES:
            path = os.path.join(root, name, mode, "results.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for row in f:
                    if row.strip():
                        r = json.loads(row)
                        model = r.get("model", model)
                        recs.setdefault(r["id"], {})[mode] = r
        failed = [tid for tid in sorted(recs) if any(r["pass_turn"] != 1 for r in recs[tid].values())]
        lines.append(f"{model}: {len(failed)} tasks did not pass on turn 1")
        lines += [failure_line(recs[tid][m]) for tid in failed for m in MODES if m in recs[tid]]
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Feedback Lift tools. The loop itself runs via bench_groq.py answer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("failures", help="one line per task and mode that failed turn 1: each try's rule ids")
    f.add_argument("out", help="the OUT folder of a loop run (holds loop/<model>/<mode>/results.jsonl)")
    args = ap.parse_args(argv)
    lines = failure_table(args.out)
    print("\n".join(lines) if lines else f"no results.jsonl under {os.path.join(args.out, 'loop')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
