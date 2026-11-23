"""Generators: each family gets parameter ranges, sampled with a fixed seed, so
the 20 pilot objects become as many checked variants as you ask for.

Sizes are snapped to whole centimetres. Ranges are the sizes real furniture
comes in, and every sample stays inside the bounds the family functions need
(drawers at least 12 cm tall, door leaves wide enough for a handle, a desk wide
enough for its drawer). Same seed, same catalog, byte for byte.
"""
from __future__ import annotations

import random
from typing import Callable

from . import families as f

Sampler = Callable[[random.Random], dict]


def _cm(rng: random.Random, lo: float, hi: float) -> float:
    return rng.randint(round(lo * 100), round(hi * 100)) / 100


def _drawer_cabinet(rng):
    h = _cm(rng, 0.5, 1.3)
    most = min(6, int((h - 2 * f.T) / 0.12))
    return dict(width=_cm(rng, 0.4, 1.2), height=h, depth=_cm(rng, 0.35, 0.6), drawers=rng.randint(2, most))


def _door_cabinet(rng):
    w = _cm(rng, 0.35, 1.2)
    doors = 1 if w < 0.6 else 2 if w > 0.8 else rng.randint(1, 2)
    return dict(width=w, height=_cm(rng, 0.5, 2.0), depth=_cm(rng, 0.3, 0.6), doors=doors)


def _chest(rng):
    w = _cm(rng, 0.3, 1.2)
    return dict(width=w, height=_cm(rng, 0.2, 0.6), depth=_cm(rng, 0.22, min(0.6, w)))


def _sliding_cabinet(rng):
    return dict(width=_cm(rng, 0.8, 1.8), height=_cm(rng, 0.6, 2.0), depth=_cm(rng, 0.35, 0.6))


def _desk(rng):
    return dict(width=_cm(rng, 0.8, 1.8), height=_cm(rng, 0.70, 0.78), depth=_cm(rng, 0.5, 0.8))


FAMILIES: dict[str, tuple[Callable, Sampler]] = {
    "drawer-cabinet": (f.drawer_cabinet, _drawer_cabinet),
    "door-cabinet": (f.door_cabinet, _door_cabinet),
    "chest": (f.chest, _chest),
    "sliding-cabinet": (f.sliding_cabinet, _sliding_cabinet),
    "desk": (f.desk, _desk),
}


def catalog(per_family: int, seed: int = 0) -> dict[str, tuple[Callable, dict]]:
    """per_family distinct variants of every family: {"chest-g0001": (chest, params), ...}.
    Never repeats a pilot object or an earlier sample."""
    out: dict[str, tuple[Callable, dict]] = {}
    for fam, (fn, sample) in FAMILIES.items():
        rng = random.Random(f"{seed}:{fam}")
        seen = {tuple(sorted(p.items())) for g, p in f.PILOT.values() if g is fn}
        n = 0
        while n < per_family:
            p = sample(rng)
            key = tuple(sorted(p.items()))
            if key in seen:
                continue
            seen.add(key)
            n += 1
            out[f"{fam}-g{n:04d}"] = (fn, p)
    return out
