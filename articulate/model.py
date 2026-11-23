"""Articulated objects as code: links made of axis-aligned boxes and cylinders,
joined by revolute and prismatic joints. One model writes URDF, MJCF and glTF.

Inertia is never typed by hand: each link's mass, centre of mass and inertia
tensor come from its shapes and a density, so the physics follows the geometry.
Units are metres, kilograms and radians. Z is up.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

Vec = tuple[float, float, float]

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class Box:
    size: Vec                 # full extents (x, y, z)
    at: Vec = (0.0, 0.0, 0.0) # centre, in the link frame
    material: str = "wood"

    def volume(self) -> float:
        x, y, z = self.size
        return x * y * z

    def inertia_diag(self, m: float) -> Vec:
        x, y, z = self.size
        return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)


@dataclass
class Cylinder:
    radius: float
    length: float
    at: Vec = (0.0, 0.0, 0.0)
    axis: str = "z"           # the cylinder's long axis: "x", "y" or "z"
    material: str = "metal"

    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.length

    def inertia_diag(self, m: float) -> Vec:
        r, h = self.radius, self.length
        along = m * r * r / 2
        across = m * (3 * r * r + h * h) / 12
        return tuple(along if a == self.axis else across for a in "xyz")  # type: ignore[return-value]

    def extents(self) -> Vec:
        d = 2 * self.radius
        return tuple(self.length if a == self.axis else d for a in "xyz")  # type: ignore[return-value]


Shape = Box | Cylinder

DENSITY = {"wood": 600.0, "metal": 7800.0, "plastic": 1100.0, "glass": 2500.0}
RGBA = {"wood": (0.66, 0.48, 0.32, 1.0), "metal": (0.72, 0.73, 0.76, 1.0),
        "plastic": (0.15, 0.16, 0.18, 1.0), "glass": (0.8, 0.9, 0.95, 0.35)}


@dataclass
class Inertial:
    mass: float
    com: Vec
    diag: Vec                 # ixx, iyy, izz about the COM (shapes are axis-aligned, so off-diagonals are 0)


@dataclass
class Link:
    name: str
    shapes: list[Shape]
    collision: bool = True
    inertial_override: Optional[Inertial] = None   # only breakers set this

    def inertial(self) -> Inertial:
        if self.inertial_override is not None:
            return self.inertial_override
        masses = [DENSITY[s.material] * s.volume() for s in self.shapes]
        m = sum(masses)
        com = tuple(sum(mi * s.at[k] for mi, s in zip(masses, self.shapes)) / m for k in range(3))
        diag = [0.0, 0.0, 0.0]
        for mi, s in zip(masses, self.shapes):
            own = s.inertia_diag(mi)
            d = [s.at[k] - com[k] for k in range(3)]
            for k in range(3):
                # parallel axis: add m * (distance squared to the axis through the COM)
                others = [d[j] for j in range(3) if j != k]
                diag[k] += own[k] + mi * (others[0] ** 2 + others[1] ** 2)
        return Inertial(m, com, tuple(diag))  # type: ignore[arg-type]


@dataclass
class Joint:
    name: str
    kind: str                 # "revolute" or "prismatic"
    parent: str
    child: str
    origin: Vec               # child frame origin, in the parent frame
    axis: Vec
    lower: float
    upper: float
    effort: float = 50.0
    velocity: float = 1.0
    damping: float = 0.5


@dataclass
class Model:
    name: str
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    description: str = ""
    params: dict = field(default_factory=dict)

    def link(self, name: str, *shapes: Shape) -> Link:
        lk = Link(name, list(shapes))
        self.links.append(lk)
        return lk

    def joint(self, name: str, kind: str, parent: str, child: str, origin: Vec, axis: str | Vec,
              lower: float, upper: float, **kw) -> Joint:
        j = Joint(name, kind, parent, child, origin, AXES[axis] if isinstance(axis, str) else axis, lower, upper, **kw)
        self.joints.append(j)
        return j

    def get_link(self, name: str) -> Link:
        return next(lk for lk in self.links if lk.name == name)

    def root(self) -> Link:
        children = {j.child for j in self.joints}
        roots = [lk for lk in self.links if lk.name not in children]
        if len(roots) != 1:
            raise ValueError(f"{self.name}: expected one root link, found {[r.name for r in roots]}")
        return roots[0]

    def children_of(self, link: str) -> list[Joint]:
        return [j for j in self.joints if j.parent == link]

    def copy(self) -> "Model":
        return copy.deepcopy(self)

    def scaled(self, k: float) -> "Model":
        """Every length times k, masses left as they were (a units mistake, not a bigger object)."""
        m = self.copy()
        for lk in m.links:
            fixed = lk.inertial()
            for s in lk.shapes:
                s.at = tuple(v * k for v in s.at)  # type: ignore[assignment]
                if isinstance(s, Box):
                    s.size = tuple(v * k for v in s.size)  # type: ignore[assignment]
                else:
                    s.radius *= k
                    s.length *= k
            lk.inertial_override = Inertial(fixed.mass, tuple(v * k for v in fixed.com), fixed.diag)  # type: ignore[arg-type]
        for j in m.joints:
            j.origin = tuple(v * k for v in j.origin)  # type: ignore[assignment]
            if j.kind == "prismatic":
                j.lower, j.upper = j.lower * k, j.upper * k
        return m
