# NIUA Articulate

[![CI](https://github.com/OhaoTech/articulate/actions/workflows/ci.yml/badge.svg)](https://github.com/OhaoTech/articulate/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Data: CC-BY-4.0](https://img.shields.io/badge/Data-CC_BY_4.0-lightgrey.svg)](LICENSE-DATA)
[![PyPI](https://img.shields.io/pypi/v/niua-articulate)](https://pypi.org/project/niua-articulate/)
[![Downloads](https://static.pepy.tech/badge/niua-articulate)](https://pepy.tech/project/niua-articulate)

**Articulated objects and scenes as code — emitted in OpenUSD (`.usda`), URDF, MJCF, and glTF 2.0, with a CPU-only OpenUSD + MuJoCo physics gate and 3D-mesh-to-OpenUSD wrapper.**

> **Scope:** Simulation only, no hardware tested. Every asset is verified on CPU in Pixar OpenUSD (`usd-core`, `UsdPhysics`) and MuJoCo (`simcheck` + kinematic joint sweep).

## Quickstart

```bash
pip install "niua-articulate[usd,sim]"

# 1. Generate articulated furniture or the room-door-drawer scene (URDF, MJCF, glTF, OpenUSD .usda)
niua-articulate build ./out-scenes --scenes

# 2. Validate OpenUSD (usd.sim-ready@1) and URDF + MuJoCo joint sweep
niua-articulate check ./out-scenes/room-door-drawer-01/model.usda
niua-articulate check ./out-scenes/room-door-drawer-01/model.urdf --sweep

# 3. Wrap any 3D prop (GLB / OBJ / STL / PLY) into a sim-ready OpenUSD rigid body
niua-articulate wrap-mesh prop.glb --out prop.usda --mass 2.5 --verify
```

## What is included

- **5 parametric families** (`drawer_cabinet`, `door_cabinet`, `chest`, `sliding_cabinet`, `desk`) — 20 deterministic pilot objects plus unlimited parametric variants (`--per-family N --seed S`).
- **Articulated room scene** (`room-door-drawer-01`) — hinged door + sliding drawer verified collision-free across full joint range, plus `door_into_console` (a variant that passes static checks and is caught only by the MuJoCo kinematic sweep).
- **6 deliberate broken repair pairs** (`zero_mass`, `inverted_limits`, `missing_collision`, `millimetre_scale`, `tiny_inertia`, `parts_overlap`) in both URDF and OpenUSD.
- **Standalone CPU physics gate:**
  - `usd.sim-ready@1` (`usd-core` / `pxr.UsdPhysics`): units, up-axis, articulation root, no nested rigid bodies, mass, inertia plausibility, colliders, acyclic joint targets/limits, rest-pose AABB overlap.
  - `articulated.sim-ready@1` (`simcheck` + MuJoCo CPU) + `--sweep` kinematic collision sweep.
- **50 frozen benchmark tasks** (`bench/tasks-v1.jsonl`) and committed model runs (`bench/results/`).

## Licenses

- **Code:** Apache-2.0 ([LICENSE](LICENSE))
- **Dataset & benchmark outputs:** CC-BY-4.0 ([LICENSE-DATA](LICENSE-DATA)) — *"NIUA Articulate by Ohao (ohao.tech), CC-BY-4.0."*
