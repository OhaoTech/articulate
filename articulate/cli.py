"""Standalone CLI entry point (`niua-articulate`) for building, wrapping, and
validating articulated objects, scenes, and 3D props (URDF, MJCF, glTF, OpenUSD).

Simulation only, no hardware tested.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .sim_check import check_any
from .usd_check import check_usd
from .wrap_mesh import wrap_mesh_to_usda


def _print_results(results: list[dict], as_json: bool) -> int:
    failed = sum(1 for r in results if r["verdict"] != "PASS")
    if as_json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, sort_keys=True))
    else:
        for r in results:
            print(f"{r['verdict']:4}  {r['file']}  ({r['contract']})")
            for f in r["failures"]:
                print(f"      [{f['rule']}] {f['subject']}: expected {f['expected']}; got {f['got']}")
                print(f"        fix: {f['fix']}")
            if "sweep" in r:
                for j in r["sweep"].get("joints", []):
                    tag = "COLLIDES" if j["collides"] else "collision_free"
                    print(f"      sweep {j['name']}: max_penetration={j['max_penetration_m']} m ({tag})")
    return 1 if failed else 0


def cmd_check(args: argparse.Namespace) -> int:
    results = []
    for p in args.paths:
        res = check_any(p, sweep=args.sweep)
        res["file"] = p
        results.append(res)
    return _print_results(results, args.json)


def cmd_check_usd(args: argparse.Namespace) -> int:
    results = []
    for p in args.paths:
        res = check_usd(p)
        res["file"] = p
        results.append(res)
    return _print_results(results, args.json)


def cmd_wrap_mesh(args: argparse.Namespace) -> int:
    usda = wrap_mesh_to_usda(
        args.mesh,
        name=args.name,
        mass_kg=args.mass,
        density=args.density,
        scale=args.scale,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(usda, encoding="utf-8", newline="\n")
    print(f"wrote sim-ready OpenUSD -> {out}")
    if args.verify:
        res = check_usd(str(out))
        res["file"] = str(out)
        return _print_results([res], as_json=False)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import build as build_mod

    objs = build_mod.build_scenes(args.out) if args.scenes else build_mod.build_all(args.out, args.per_family, args.seed)
    print(f"{len(objs)} {'scenes' if args.scenes else 'objects'}, {sum(len(o['broken']) for o in objs)} broken variants -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="niua-articulate", description="NIUA Articulate: objects-as-code + OpenUSD/MuJoCo physics gate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_any = sub.add_parser("check", help="Validate .usda/.usd (usd.sim-ready@1) or .urdf/.xml (articulated.sim-ready@1 + optional --sweep).")
    p_any.add_argument("paths", nargs="+", help="One or more asset paths.")
    p_any.add_argument("--sweep", action="store_true", help="Also run MuJoCo kinematic joint sweep on URDF files.")
    p_any.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    p_any.set_defaults(func=cmd_check)

    p_check = sub.add_parser("check-usd", help="Validate OpenUSD (.usda/.usd) files against usd.sim-ready@1 (CPU, usd-core).")
    p_check.add_argument("paths", nargs="+", help="One or more .usda/.usd/.usdc files.")
    p_check.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    p_check.set_defaults(func=cmd_check_usd)

    p_wrap = sub.add_parser("wrap-mesh", help="Wrap a 3D mesh (GLB/OBJ/STL/PLY) into a sim-ready OpenUSD (.usda) rigid body.")
    p_wrap.add_argument("mesh", help="Input 3D mesh file.")
    p_wrap.add_argument("--out", required=True, help="Output .usda file path.")
    p_wrap.add_argument("--name", default=None, help="Root USD prim name (defaults to mesh filename).")
    p_wrap.add_argument("--mass", type=float, default=None, help="Explicit mass in kg (overrides density).")
    p_wrap.add_argument("--density", type=float, default=600.0, help="Density in kg/m^3 when --mass is omitted (default: 600).")
    p_wrap.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor (e.g. 0.001 for mm -> m).")
    p_wrap.add_argument("--verify", action="store_true", help="Immediately verify the output with usd.sim-ready@1.")
    p_wrap.set_defaults(func=cmd_wrap_mesh)

    p_build = sub.add_parser("build", help="Generate articulated objects or scenes (URDF, MJCF, glTF, OpenUSD .usda).")
    p_build.add_argument("out", help="Output directory.")
    p_build.add_argument("--per-family", type=int, default=0, help="Generated variants per family (default 0: pilot only).")
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--scenes", action="store_true", help="Write scenes instead of furniture objects.")
    p_build.set_defaults(func=cmd_build)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
