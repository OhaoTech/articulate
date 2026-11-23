"""NIUA Articulate bench: plain requests an agent answers with a URDF.

    python bench.py tasks TASKS.jsonl [--per-family 10]
    python bench.py reference TASKS.jsonl ANSWERS_DIR      our own answers (the ceiling)
    python bench.py score TASKS.jsonl ANSWERS_DIR GATE_CHECK [--out RESULTS.json]

An answer is ANSWERS_DIR/<task id>.urdf. A task passes when the URDF is the
object asked for (spec) and it passes articulated.sim-ready@1 (gate, MuJoCo).
A missing answer fails. Score is passed / tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import articulate as a  # noqa: E402
from articulate import bench as b  # noqa: E402

CONTRACT = "articulated.sim-ready@1"


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_tasks(out: str, per_family: int) -> None:
    ts = b.tasks(per_family)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for t in ts:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    print(f"{len(ts)} tasks -> {out}")


def cmd_reference(tasks_path: str, answers: str) -> None:
    fns = {fn.__name__: fn for fn, _ in a.FAMILIES.values()}
    os.makedirs(answers, exist_ok=True)
    for t in load(tasks_path):
        r = t["reference"]
        with open(os.path.join(answers, f"{t['id']}.urdf"), "w", encoding="utf-8", newline="\n") as f:
            f.write(a.to_urdf(fns[r["family"]](**r["params"])))
    print(f"reference answers -> {answers}")


def score(tasks: list[dict], answers: str, gate: str | None) -> dict:
    rows, paths = {}, {}
    for t in tasks:
        p = os.path.join(answers, f"{t['id']}.urdf")
        if not os.path.exists(p):
            rows[t["id"]] = {"answered": False, "spec": [{"rule": "answer.missing"}], "gate": None}
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            rows[t["id"]] = {"answered": True, "spec": b.check_spec(t, f.read()), "gate": None}
        paths[t["id"]] = p
    if gate and paths:
        import gate as g
        verdicts = g.run_gate(gate, CONTRACT, list(paths.values()))
        for tid, p in paths.items():
            rows[tid]["gate"] = verdicts.get(p, {"verdict": "ERROR", "failures": []})
    for r in rows.values():
        r["spec_ok"] = not r["spec"]
        r["gate_ok"] = bool(r["gate"]) and r["gate"]["verdict"] == "PASS"
        r["pass"] = r["spec_ok"] and r["gate_ok"]
    n = len(tasks)
    summary = {k: sum(r[k] for r in rows.values()) for k in ("answered", "spec_ok", "gate_ok", "pass")}
    return {"schema": "niua.articulate.bench/v1", "contract": CONTRACT, "tasks": n,
            "summary": summary, "score": round(summary["pass"] / n, 4) if n else 0.0, "results": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("out")
    t.add_argument("--per-family", type=int, default=10)
    r = sub.add_parser("reference")
    r.add_argument("tasks")
    r.add_argument("answers")
    s = sub.add_parser("score")
    s.add_argument("tasks")
    s.add_argument("answers")
    s.add_argument("gate")
    s.add_argument("--out")
    s.add_argument("--min-score", type=float, default=0.0, help="exit 1 below this (CI: 1.0 for the reference)")
    args = ap.parse_args()
    if args.cmd == "tasks":
        cmd_tasks(args.out, args.per_family)
    elif args.cmd == "reference":
        cmd_reference(args.tasks, args.answers)
    else:
        res = score(load(args.tasks), args.answers, os.path.abspath(args.gate))
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(res, f, indent=2, sort_keys=True)
                f.write("\n")
        s = res["summary"]
        print(f"{res['tasks']} tasks: answered {s['answered']}, spec {s['spec_ok']}, gate {s['gate_ok']}, "
              f"pass {s['pass']}  score {res['score']:.2%}")
        for tid, row in res["results"].items():
            if not row["pass"]:
                why = [x["rule"] for x in row["spec"]] + ([x["rule"] for x in row["gate"]["failures"]] if row["gate"] else [])
                print(f"  FAIL {tid}: {', '.join(why)}")
        if res["score"] < args.min_score:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
