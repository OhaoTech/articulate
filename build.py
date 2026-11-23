"""Write the dataset: every object as URDF, MJCF, glTF and OpenUSD (.usda), plus its broken variants.

    python build.py OUT_DIR                     the 20 pilot objects
    python build.py OUT_DIR --per-family N [--seed S]
                                                pilot + N generated variants per family
    python build.py OUT_DIR --scenes            the scenes (a room with a door and a drawer) instead

OUT_DIR/<object id>/
    object.json            family, parameters, joints, sha256 of every file
    model.urdf  model.xml  model.gltf  model.usda
    broken/<breaker>.urdf  broken/<breaker>.usda  the same object with one known mistake

Deterministic: building twice gives identical bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import articulate as a  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha(text)


def write_object(out: str, oid: str, m: "a.Model", family: str, generated: bool, breakers: dict, extra: dict | None = None) -> dict:
    """Write one object's folder and return its object.json content."""
    d = os.path.join(out, oid)
    files = {
        "model.urdf": write(os.path.join(d, "model.urdf"), a.to_urdf(m)),
        "model.xml": write(os.path.join(d, "model.xml"), a.to_mjcf(m)),
        "model.gltf": write(os.path.join(d, "model.gltf"), a.to_gltf(m)),
        "model.usda": write(os.path.join(d, "model.usda"), a.to_usda(m)),
    }
    broken = {}
    broken_usd = {}
    for name, breaker in breakers.items():
        b = breaker(m)
        if b is not None:
            broken[name] = write(os.path.join(d, "broken", f"{name}.urdf"), a.to_urdf(b))
            broken_usd[name] = write(os.path.join(d, "broken", f"{name}.usda"), a.to_usda(b))
    meta = {
        "schema": "niua.articulate.object/v1",
        "id": oid,
        "family": family,
        "generated": generated,
        "description": m.description,
        "params": m.params,
        "links": [{"name": lk.name, "mass_kg": round(lk.inertial().mass, 6)} for lk in m.links],
        "joints": [{"name": j.name, "type": j.kind, "parent": j.parent, "child": j.child,
                    "axis": list(j.axis), "lower": j.lower, "upper": j.upper} for j in m.joints],
        "files": files,
        "broken": broken,
        "broken_usd": broken_usd,
        **(extra or {}),
    }
    write(os.path.join(d, "object.json"), json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def write_index(out: str, index: list[dict], seed: int = 0, per_family: int = 0) -> None:
    write(os.path.join(out, "index.json"), json.dumps(
        {"schema": "niua.articulate.index/v1", "seed": seed, "per_family": per_family,
         "objects": [o["id"] for o in index]}, indent=2) + "\n")


def build_all(out: str, per_family: int = 0, seed: int = 0) -> list[dict]:
    cat = dict(a.PILOT)
    cat.update(a.catalog(per_family, seed) if per_family else {})
    index = [write_object(out, oid, a.build(oid, cat), cat[oid][0].__name__, oid not in a.PILOT, a.BREAKERS) for oid in cat]
    write_index(out, index, seed, per_family)
    return index


def build_scenes(out: str) -> list[dict]:
    """Every scene, written like an object. object.json carries a `sweep` block:
    the good scene's joints must sweep collision free, and each broken variant
    named in it is caught by sweeping the named joint, not by a contract."""
    index = []
    for sid, (fn, _) in a.SCENES.items():
        sweep = {"good": "collision_free", "broken": dict(a.SWEEP_CAUGHT)}
        index.append(write_object(out, sid, a.build(sid, a.SCENES), fn.__name__, False, a.SCENE_BREAKERS,
                                  {"kind": "scene", "sweep": sweep}))
    write_index(out, index)
    return index


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the NIUA Articulate dataset.")
    ap.add_argument("out")
    ap.add_argument("--per-family", type=int, default=0, help="generated variants per family (default 0: pilot only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", action="store_true", help="write the scenes instead of the objects")
    args = ap.parse_args()
    objs = build_scenes(args.out) if args.scenes else build_all(args.out, args.per_family, args.seed)
    print(f"{len(objs)} {'scenes' if args.scenes else 'objects'}, {sum(len(o['broken']) for o in objs)} broken variants -> {args.out}")
