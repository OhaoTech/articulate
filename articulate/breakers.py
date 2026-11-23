"""Breakers: the mistakes people (and agents) actually make in robot description
files, applied on purpose to a good model. Each returns a broken copy, or None
when it does not apply to that model. The gate's report on the broken copy,
next to the good one, is the repair pair.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

from .model import Inertial, Model


def _first_moving(m: Model) -> str:
    return m.joints[0].child


def zero_mass(m: Model) -> Optional[Model]:
    """A moving part with mass 0: the exporter dropped it, or it was never set."""
    b = m.copy()
    lk = b.get_link(_first_moving(b))
    it = lk.inertial()
    lk.inertial_override = Inertial(0.0, it.com, it.diag)
    return b


def inverted_limits(m: Model) -> Optional[Model]:
    """A joint's lower and upper limits swapped."""
    b = m.copy()
    j = b.joints[0]
    j.lower, j.upper = j.upper, j.lower
    return b


def missing_collision(m: Model) -> Optional[Model]:
    """A moving part you can see but not touch: its collision geometry is gone."""
    b = m.copy()
    b.get_link(_first_moving(b)).collision = False
    return b


def millimetre_scale(m: Model) -> Optional[Model]:
    """Lengths written in millimetres where metres were meant (CAD export default)."""
    return m.scaled(1000.0)


def tiny_inertia(m: Model) -> Optional[Model]:
    """A placeholder inertia (1e-6 on the diagonal) kept from a template: far too
    small for the part's mass and size."""
    b = m.copy()
    lk = b.get_link(_first_moving(b))
    it = lk.inertial()
    lk.inertial_override = Inertial(it.mass, it.com, (1e-6, 1e-6, 1e-6))
    return b


def parts_overlap(m: Model) -> Optional[Model]:
    """Two sibling parts placed into each other at rest (a copy-pasted offset).
    Applies only where two moving parts share a parent."""
    by_parent: dict[str, list] = {}
    for j in m.joints:
        by_parent.setdefault(j.parent, []).append(j)
    pair = next((js for js in by_parent.values() if len(js) >= 2), None)
    if pair is None:
        return None
    b = m.copy()
    a, c = (next(j for j in b.joints if j.name == pair[0].name), next(j for j in b.joints if j.name == pair[1].name))
    # Same place and the same parts as its sibling: a pasted copy that was never moved.
    # (Moving the origin alone is not enough: a mirrored right door at the left
    # hinge swings outside the cabinet and overlaps nothing.)
    c.origin = a.origin
    b.get_link(c.child).shapes = [copy.deepcopy(s) for s in b.get_link(a.child).shapes]
    return b


BREAKERS: dict[str, Callable[[Model], Optional[Model]]] = {
    "zero_mass": zero_mass,
    "inverted_limits": inverted_limits,
    "missing_collision": missing_collision,
    "millimetre_scale": millimetre_scale,
    "tiny_inertia": tiny_inertia,
    "parts_overlap": parts_overlap,
}
