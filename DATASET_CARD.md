---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - robotics
  - articulated-objects
  - openusd
  - urdf
  - mjcf
  - mujoco
  - physics-simulation
  - synthetic-data
pretty_name: NIUA Articulate v1
---

# NIUA Articulate v1

**Physically verified articulated objects and scenes as code — shipped in OpenUSD (`.usda`), URDF, MJCF, and glTF 2.0, paired with deliberate failure variants and typed physics repair pairs.**

> **Scope:** Simulation only, no hardware tested. Every asset is verified on CPU in MuJoCo (`3.16.0` + `simcheck 0.2.0`) and Pixar OpenUSD (`usd-core 26.11`, `UsdPhysics`).

## What is inside

- **5 parametric object families** (20 pilot + 1,000 generated variants with `--per-family 200 --seed 0`): `drawer_cabinet`, `door_cabinet`, `chest`, `sliding_cabinet`, `desk`.
- **1 articulated room scene** (`room-door-drawer-01`): 2.4 m x 2.0 m x 2.4 m room with a hinged door (`door_hinge`, revolute `0..1.5 rad`) and sliding drawer (`drawer_slide`, prismatic `0..0.231 m`), swept collision-free in MuJoCo.
- **4 formats per asset:** `model.usda` (OpenUSD with `UsdPhysics`), `model.urdf`, `model.xml` (MJCF), `model.gltf` (glTF 2.0).
- **Deliberate broken repair pairs** (`broken/*.urdf`, `broken/*.usda`): `zero_mass`, `inverted_limits`, `missing_collision`, `millimetre_scale`, `tiny_inertia`, `parts_overlap`, and `door_into_console` (caught only by range-of-motion joint sweep).
- **50 frozen benchmark tasks** (`bench/tasks-v1.jsonl`) for evaluating LLM/agent ability to write physics-passing articulated descriptions.

## Reproduce & Verify Locally

```bash
pip install "niua-articulate[usd,sim]"

# Generate the exact deterministic dataset
niua-articulate build ./out --per-family 200 --seed 0
niua-articulate build ./out-scenes --scenes

# Validate OpenUSD and URDF + kinematic joint sweep
niua-articulate check ./out-scenes/room-door-drawer-01/model.usda
niua-articulate check ./out-scenes/room-door-drawer-01/model.urdf --sweep
```

## License & Attribution

- **Data:** CC-BY-4.0 — *"NIUA Articulate by Ohao (ohao.tech), CC-BY-4.0."*
- **Code:** Apache-2.0 (see repository `LICENSE`).
