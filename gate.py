"""Run every built object, good and broken, through the NIUA gate and write the results.

    python gate.py OUT_DIR GATE_CHECK [--sweep URDF_SWEEP]   (CI only: the physics contract runs MuJoCo)

GATE_CHECK is the gate's own offline checker (niua-core example `gate_check`),
the same Rust code as POST /api/v1/check minus upload and signing. Two
contracts, both built in:

  articulated.sim-ready@1   simcheck in MuJoCo on CPU (URDF and MJCF)
  urdf.sim-ready@1          static robot-file rules (URDF)

Good files must pass every contract that applies to them. A broken file must
fail at least one. OUT_DIR/gate.json keeps every verdict and every failure
(rule, what was expected, what was found, the fix): the repair pairs.

Scenes (build.py --scenes) carry a `sweep` block in object.json. With --sweep
URDF_SWEEP (niua-core/sim/urdf_sweep.py) each such scene's good URDF is also
swept joint by joint through its range in MuJoCo and must be collision free,
and each broken variant the block names must collide on the joint it names.
Those variants are caught by the sweep, not by a contract, so a contract that
passes them is reported, not counted as a mismatch. The sweep output is
printed per joint and kept in OUT_DIR/<scene>/sweep.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

CONTRACTS = {
    "articulated.sim-ready@1": ("model.urdf", "model.xml"),
    "urdf.sim-ready@1": ("model.urdf",),
}


CHUNK = 400   # files per gate_check call: keeps each call well under its timeout
# The failure fields gate.json keeps. The Feedback Lift loop (bench_loop.py) also asks for
# "path", where the gate names the simcheck check that failed.
FIELDS = ("rule", "severity", "subject", "expected", "got", "fix")


def _progress(line: str) -> None:
    print(line, flush=True)


def run_gate(gate: str, contract: str, paths: list[str], fields: tuple = FIELDS, log=_progress) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(paths), CHUNK):
        out.update(_run_chunk(gate, contract, paths[i:i + CHUNK], fields))
        log(f"  {contract}: {min(i + CHUNK, len(paths))}/{len(paths)}")
    return out


def _run_chunk(gate: str, contract: str, paths: list[str], fields: tuple = FIELDS) -> dict[str, dict]:
    """One gate_check call; with several paths it prints one JSON line per file."""
    if len(paths) == 1:
        paths = paths + paths   # force the one-line-per-file form
    p = subprocess.run([gate, contract, *paths], capture_output=True, timeout=3600)
    if p.returncode != 0:
        sys.exit(f"gate_check {contract} failed: {p.stderr.decode(errors='replace')[-2000:]}")
    out = {}
    for line in p.stdout.decode().splitlines():
        if line.strip():
            r = json.loads(line)
            o = r["outcome"]
            out[r["file"]] = {
                "verdict": o["verdict"].upper(),
                "failures": [{k: f[k] for k in fields if k in f}
                             for f in o["failures"]],
            }
    return out


def run_sweep(script: str, urdf_path: str, python: str = sys.executable) -> dict:
    """One urdf_sweep.py run on a bare URDF: the report it printed."""
    with open(urdf_path, "rb") as f:
        p = subprocess.run([python, script, "sweep"], stdin=f, capture_output=True, timeout=600)
    if p.returncode != 0:
        sys.exit(f"urdf_sweep failed on {urdf_path}: {p.stderr.decode(errors='replace')[-2000:]}")
    return json.loads(p.stdout.decode())


def _sweep_lines(label: str, rep: dict) -> list[str]:
    if not rep.get("ok"):
        return [f"  {label}: sweep failed: {rep.get('error')}"]
    n = rep["neutral"]
    out = [f"  {label}: MuJoCo {rep['mujoco']}, {rep['mode']}",
           f"    at rest: max penetration {n['max_penetration_m']} m (tolerance {rep['tolerance_m']} m)"]
    for j in rep["joints"]:
        w = j["worst"]
        where = f", deepest {w['bodies'][0]} against {w['bodies'][1]} at {w['at']} {j['unit']}" if w else ""
        out.append(f"    {j['name']:14} {j['type']:9} {j['lower']}..{j['upper']} {j['unit']:3} "
                   f"max penetration {j['max_penetration_m']} m  {'COLLIDES' if j['collides'] else 'collision_free'}{where}")
    return out


def sweep_scenes(out: str, metas: dict, script: str, log=_progress) -> tuple[dict, list[str]]:
    """Sweep every object that carries a `sweep` block. Returns (reports, mismatches)."""
    reports: dict[str, dict] = {}
    bad: list[str] = []
    for oid, meta in metas.items():
        spec = meta.get("sweep")
        if not spec:
            continue
        d = os.path.join(out, oid)
        mine: dict[str, dict] = {"good": run_sweep(script, os.path.join(d, "model.urdf"))}
        for name in spec["broken"]:
            mine[name] = run_sweep(script, os.path.join(d, "broken", f"{name}.urdf"))
        reports[oid] = mine
        with open(os.path.join(d, "sweep.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(mine, f, indent=2, sort_keys=True)
            f.write("\n")
        log(f"joint sweep, {oid}:")
        for label, rep in mine.items():
            for line in _sweep_lines("good scene" if label == "good" else f"broken/{label}", rep):
                log(line)
        good = mine["good"]
        if not good.get("ok"):
            bad.append(f"{oid}: sweep of the good scene failed: {good.get('error')}")
        else:
            if good["neutral"]["max_penetration_m"] > good["tolerance_m"]:
                bad.append(f"{oid}: good scene touches itself at rest ({good['neutral']['max_penetration_m']} m)")
            for j in good["joints"]:
                if j["collides"]:
                    bad.append(f"{oid}: good scene, joint {j['name']} collides in its range (max penetration {j['max_penetration_m']} m)")
        for name, joint in spec["broken"].items():
            rep = mine[name]
            if not rep.get("ok"):
                bad.append(f"{oid}/broken/{name}: sweep failed: {rep.get('error')}")
                continue
            hit = {j["name"] for j in rep["joints"] if j["collides"]}
            if joint not in hit:
                bad.append(f"{oid}/broken/{name}: expected {joint} to collide in the sweep, colliding joints: {sorted(hit) or 'none'}")
    return reports, bad


def check_usd_all(out: str, metas: dict, log=_progress) -> tuple[dict, list[str]]:
    """Run usd.sim-ready@1 over every built model.usda and broken/<name>.usda."""
    from articulate.usd_check import CONTRACT as USD_CONTRACT, check_usd

    reports: dict[str, dict] = {}
    bad: list[str] = []
    for oid, meta in metas.items():
        d = os.path.join(out, oid)
        good_res = check_usd(os.path.join(d, "model.usda"))
        per_broken: dict[str, dict] = {}
        if good_res["verdict"] != "PASS":
            bad.append(f"{oid}/model.usda {USD_CONTRACT}: expected PASS, got {[x['rule'] for x in good_res['failures']]}")
        sweep_only = set((meta.get("sweep") or {}).get("broken", {}))
        for name in meta.get("broken_usd", {}):
            res = check_usd(os.path.join(d, "broken", f"{name}.usda"))
            per_broken[name] = res
            if res["verdict"] == "PASS" and name not in sweep_only:
                bad.append(f"{oid}/broken/{name}.usda {USD_CONTRACT}: expected FAIL, got PASS")
        reports[oid] = {"good": good_res, "broken": per_broken}
    log(f"  {USD_CONTRACT}: checked {len(metas)} objects/scenes")
    return reports, bad


def main(out: str, gate: str, sweep: str | None = None, check_usd_flag: bool = False) -> int:
    index = json.load(open(os.path.join(out, "index.json")))
    metas = {oid: json.load(open(os.path.join(out, oid, "object.json"))) for oid in index["objects"]}

    def good_path(oid, fname):
        return os.path.join(out, oid, fname)

    def broken_path(oid, name):
        return os.path.join(out, oid, "broken", f"{name}.urdf")

    verdicts: dict[str, dict] = {}
    for contract, fnames in CONTRACTS.items():
        paths = [good_path(oid, f) for oid in metas for f in fnames]
        paths += [broken_path(oid, n) for oid, m in metas.items() for n in m["broken"]]
        print(f"running {contract} on {len(paths)} files", flush=True)
        verdicts[contract] = run_gate(gate, contract, paths)

    results, bad = {}, []
    caught: dict[str, dict[str, list[int]]] = {}
    for oid, meta in metas.items():
        res = {"good": {}, "broken": {}}
        for contract, fnames in CONTRACTS.items():
            for f in fnames:
                r = verdicts[contract][good_path(oid, f)]
                res["good"].setdefault(f, {})[contract] = r
                if r["verdict"] != "PASS":
                    bad.append(f"{oid}/{f} {contract}: expected PASS, got {[x['rule'] for x in r['failures']]}")
        for name in meta["broken"]:
            per = {c: verdicts[c][broken_path(oid, name)] for c in CONTRACTS}
            res["broken"][name] = per
            for c, r in per.items():
                tally = caught.setdefault(name, {}).setdefault(c, [0, 0])
                tally[0] += r["verdict"] == "FAIL"
                tally[1] += 1
            if all(r["verdict"] == "PASS" for r in per.values()):
                if name in (meta.get("sweep") or {}).get("broken", {}):
                    print(f"NOTE {oid}/broken/{name}: passes every contract; the sweep is what catches it")
                else:
                    bad.append(f"{oid}/broken/{name}: expected FAIL, passed every contract")
        results[oid] = res

    sweeps: dict[str, dict] = {}
    if sweep:
        sweeps, sweep_bad = sweep_scenes(out, metas, sweep)
        bad += sweep_bad
    usd_reports: dict[str, dict] = {}
    if check_usd_flag:
        usd_reports, usd_bad = check_usd_all(out, metas)
        bad += usd_bad
    contracts = list(CONTRACTS) + (["usd.sim-ready@1"] if check_usd_flag else [])
    doc = {"schema": "niua.articulate.gate/v1", "contracts": contracts, "objects": results}
    if sweeps:
        doc["sweeps"] = sweeps
    if usd_reports:
        doc["usd"] = usd_reports
    with open(os.path.join(out, "gate.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")

    n_good = sum(len(fn) for fn in CONTRACTS.values()) * len(metas)
    n_pass = sum(1 for res in results.values() for per in res["good"].values() for r in per.values() if r["verdict"] == "PASS")
    print(f"good files: {n_pass}/{n_good} checks pass")
    print("broken files, caught by contract:")
    for name in sorted(caught):
        row = "  ".join(f"{c} {k}/{n}" for c, (k, n) in caught[name].items())
        rules = sorted({x["rule"] for res in results.values() if name in res["broken"]
                        for r in res["broken"][name].values() for x in r["failures"] if x["severity"] == "error"})
        print(f"  {name:18} {row}   rules: {', '.join(rules)}")
    for line in bad:
        print("MISMATCH", line)
    return 1 if bad else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run every built object through the NIUA gate.")
    ap.add_argument("out")
    ap.add_argument("gate_check")
    ap.add_argument("--sweep", metavar="URDF_SWEEP", help="also sweep the scenes' joints with niua-core/sim/urdf_sweep.py")
    ap.add_argument("--usd", action="store_true", help="also run usd.sim-ready@1 on model.usda and broken/*.usda (requires usd-core)")
    args = ap.parse_args()
    sys.exit(main(args.out, os.path.abspath(args.gate_check), os.path.abspath(args.sweep) if args.sweep else None, check_usd_flag=args.usd))
