"""Signed distance fields, and the constructive-solid-geometry algebra over them.

An SDF is a function f(p) that returns the signed distance from a point p to a
surface: negative inside, zero on it, positive outside. Modelling with SDFs is
appealing because the boolean operations that are hard on meshes are trivial on
distances — the union of two solids is the pointwise minimum of their fields, the
intersection is the maximum, and cutting B out of A is max(A, −B). Whole shapes
compose as one arithmetic expression, and the result is still a clean field you
can mesh, raymarch, or query.

Every field here is vectorised: call it on an (N, 3) array of points and get back
N distances. The gradient (the surface normal direction) is available by central
differences on any field, which is what lets the mesher orient its output
outward without knowing which primitive produced a given face.
"""

from __future__ import annotations

import numpy as np


class SDF:
    """Base class: subclasses implement `distance(points)`; operators build a tree."""

    def distance(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        return self.distance(p)

    # -- CSG operators -------------------------------------------------------

    def __or__(self, other):   # union
        return Union(self, other)

    def __and__(self, other):  # intersection
        return Intersection(self, other)

    def __sub__(self, other):  # difference (A minus B)
        return Difference(self, other)

    def translate(self, offset):
        return Translate(self, offset)

    def scale(self, factor):
        return Scale(self, factor)

    def shell(self, thickness):
        return Shell(self, thickness)

    def round(self, radius):
        return Round(self, radius)

    def smooth_union(self, other, k=1.0):
        return SmoothUnion(self, other, k)

    # -- differential / bounds ----------------------------------------------

    def gradient(self, p: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Central-difference gradient ∇f (points outward for a distance field)."""
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        grad = np.empty_like(p)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            grad[:, axis] = (self(p + step) - self(p - step)) / (2 * h)
        return grad

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """A conservative axis-aligned box that contains the surface. Overridden by
        primitives; combinators combine their children's boxes."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


class Sphere(SDF):
    def __init__(self, radius=1.0, center=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=np.float64)

    def distance(self, p):
        return np.linalg.norm(p - self.center, axis=1) - self.radius

    def bounds(self):
        r = self.radius
        return self.center - r, self.center + r


class Box(SDF):
    def __init__(self, half_extents=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
        self.half = np.asarray(half_extents, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)

    def distance(self, p):
        q = np.abs(p - self.center) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def bounds(self):
        return self.center - self.half, self.center + self.half


class Torus(SDF):
    """Ring in the z=0 plane: `major` is the ring radius, `minor` the tube radius."""

    def __init__(self, major=1.0, minor=0.3, center=(0.0, 0.0, 0.0)):
        self.major = float(major)
        self.minor = float(minor)
        self.center = np.asarray(center, dtype=np.float64)

    def distance(self, p):
        d = p - self.center
        q = np.hypot(np.hypot(d[:, 0], d[:, 1]) - self.major, d[:, 2])
        return q - self.minor

    def bounds(self):
        r = self.major + self.minor
        return self.center - [r, r, self.minor], self.center + [r, r, self.minor]


class Cylinder(SDF):
    """Capped cylinder along z, radius `radius`, total height `height`."""

    def __init__(self, radius=1.0, height=2.0, center=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        self.height = float(height)
        self.center = np.asarray(center, dtype=np.float64)

    def distance(self, p):
        d = p - self.center
        radial = np.hypot(d[:, 0], d[:, 1]) - self.radius
        axial = np.abs(d[:, 2]) - self.height / 2
        outside = np.linalg.norm(np.maximum(np.stack([radial, axial], axis=1), 0.0), axis=1)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside

    def bounds(self):
        r, h = self.radius, self.height / 2
        return self.center - [r, r, h], self.center + [r, r, h]


# ---------------------------------------------------------------------------
# combinators
# ---------------------------------------------------------------------------


class Union(SDF):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def distance(self, p):
        return np.minimum(self.a(p), self.b(p))

    def bounds(self):
        la, ha = self.a.bounds()
        lb, hb = self.b.bounds()
        return np.minimum(la, lb), np.maximum(ha, hb)


class Intersection(SDF):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def distance(self, p):
        return np.maximum(self.a(p), self.b(p))

    def bounds(self):
        la, ha = self.a.bounds()
        lb, hb = self.b.bounds()
        return np.maximum(la, lb), np.minimum(ha, hb)


class Difference(SDF):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def distance(self, p):
        return np.maximum(self.a(p), -self.b(p))

    def bounds(self):
        return self.a.bounds()  # A minus B fits inside A


class SmoothUnion(SDF):
    """A blended union with a rounded fillet of width ~k where the shapes meet —
    the "metaball" look, and the reason SDFs are loved for organic modelling."""

    def __init__(self, a, b, k=1.0):
        self.a, self.b, self.k = a, b, float(k)

    def distance(self, p):
        da, db = self.a(p), self.b(p)
        h = np.clip(0.5 + 0.5 * (db - da) / self.k, 0.0, 1.0)
        return db * (1 - h) + da * h - self.k * h * (1 - h)

    def bounds(self):
        la, ha = self.a.bounds()
        lb, hb = self.b.bounds()
        return np.minimum(la, lb) - self.k, np.maximum(ha, hb) + self.k


class Shell(SDF):
    """A hollow shell of the surface: |f| − thickness/2 (turns a solid into a wall)."""

    def __init__(self, a, thickness):
        self.a, self.thickness = a, float(thickness)

    def distance(self, p):
        return np.abs(self.a(p)) - self.thickness / 2

    def bounds(self):
        lo, hi = self.a.bounds()
        return lo - self.thickness, hi + self.thickness


class Round(SDF):
    """Inflate a field outward by `radius`, which rounds every convex edge to that
    radius — the difference between a sharp box and a rounded die. Build a rounded
    box as `Box(half - r).round(r)` so the outer size stays `half`."""

    def __init__(self, a, radius):
        self.a, self.radius = a, float(radius)

    def distance(self, p):
        return self.a(p) - self.radius

    def bounds(self):
        lo, hi = self.a.bounds()
        return lo - self.radius, hi + self.radius


class Translate(SDF):
    def __init__(self, a, offset):
        self.a = a
        self.offset = np.asarray(offset, dtype=np.float64)

    def distance(self, p):
        return self.a(p - self.offset)

    def bounds(self):
        lo, hi = self.a.bounds()
        return lo + self.offset, hi + self.offset


class Scale(SDF):
    def __init__(self, a, factor):
        self.a = a
        self.factor = float(factor)

    def distance(self, p):
        return self.a(p / self.factor) * self.factor

    def bounds(self):
        lo, hi = self.a.bounds()
        return lo * self.factor, hi * self.factor
