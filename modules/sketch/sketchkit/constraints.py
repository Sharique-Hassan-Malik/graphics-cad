"""The constraint vocabulary: each constraint is one or more scalar equations
r(q) = 0 over the sketch's parameter vector, plus the analytic partial
derivatives of those equations (its rows of the global Jacobian).

Every residual is written in a smooth, root-findable form — distances as squared
lengths, angles via the sin/cos identity below — so Newton's method sees a well
behaved system. The Jacobians are analytic and exact; `tests/` cross-checks each
one against a finite-difference approximation, so a wrong derivative cannot slip
through and merely slow the solver down.

A Jacobian "row" is a dict {global-parameter-index: ∂residual/∂param}. Only the
handful of parameters a constraint actually touches appear, which is what keeps
the assembled Jacobian sparse and the whole thing fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Constraint:
    """Interface: residuals(q) → array, and jacobian(q) → list of index→partial dicts,
    one dict per residual (same order)."""

    def residuals(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, q: np.ndarray) -> list[dict[int, float]]:
        raise NotImplementedError

    @property
    def n_equations(self) -> int:
        return 1


# --- helpers to pull entity coordinates out of the flat vector ---------------

def _p(q, point):
    return q[point.ix], q[point.iy]


def _line(q, line):
    ax, ay = _p(q, line.a)
    bx, by = _p(q, line.b)
    return ax, ay, bx, by, (bx - ax), (by - ay)


# ---------------------------------------------------------------------------
# incidence
# ---------------------------------------------------------------------------


@dataclass
class Coincident(Constraint):
    p: object
    q_: object

    @property
    def n_equations(self):
        return 2

    def residuals(self, q):
        px, py = _p(q, self.p)
        qx, qy = _p(q, self.q_)
        return np.array([px - qx, py - qy])

    def jacobian(self, q):
        return [
            {self.p.ix: 1.0, self.q_.ix: -1.0},
            {self.p.iy: 1.0, self.q_.iy: -1.0},
        ]


@dataclass
class PointOnLine(Constraint):
    p: object
    line: object

    def residuals(self, q):
        px, py = _p(q, self.p)
        ax, ay, bx, by, ex, ey = _line(q, self.line)
        wx, wy = px - ax, py - ay
        return np.array([ex * wy - ey * wx])

    def jacobian(self, q):
        px, py = _p(q, self.p)
        ax, ay, bx, by, ex, ey = _line(q, self.line)
        wx, wy = px - ax, py - ay
        return [{
            self.p.ix: -ey, self.p.iy: ex,
            self.line.a.ix: ey - wy, self.line.a.iy: wx - ex,
            self.line.b.ix: wy, self.line.b.iy: -wx,
        }]


@dataclass
class PointOnCircle(Constraint):
    p: object
    circle: object

    def residuals(self, q):
        px, py = _p(q, self.p)
        cx, cy = _p(q, self.circle.center)
        r = q[self.circle.ir]
        return np.array([(px - cx) ** 2 + (py - cy) ** 2 - r ** 2])

    def jacobian(self, q):
        px, py = _p(q, self.p)
        cx, cy = _p(q, self.circle.center)
        r = q[self.circle.ir]
        dx, dy = px - cx, py - cy
        return [{
            self.p.ix: 2 * dx, self.p.iy: 2 * dy,
            self.circle.center.ix: -2 * dx, self.circle.center.iy: -2 * dy,
            self.circle.ir: -2 * r,
        }]


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


@dataclass
class Distance(Constraint):
    p: object
    q_: object
    d: float

    def residuals(self, q):
        px, py = _p(q, self.p)
        qx, qy = _p(q, self.q_)
        return np.array([(px - qx) ** 2 + (py - qy) ** 2 - self.d ** 2])

    def jacobian(self, q):
        px, py = _p(q, self.p)
        qx, qy = _p(q, self.q_)
        dx, dy = px - qx, py - qy
        return [{
            self.p.ix: 2 * dx, self.q_.ix: -2 * dx,
            self.p.iy: 2 * dy, self.q_.iy: -2 * dy,
        }]


@dataclass
class Radius(Constraint):
    circle: object
    r: float

    def residuals(self, q):
        return np.array([q[self.circle.ir] - self.r])

    def jacobian(self, q):
        return [{self.circle.ir: 1.0}]


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------


@dataclass
class Horizontal(Constraint):
    p: object
    q_: object

    def residuals(self, q):
        return np.array([q[self.p.iy] - q[self.q_.iy]])

    def jacobian(self, q):
        return [{self.p.iy: 1.0, self.q_.iy: -1.0}]


@dataclass
class Vertical(Constraint):
    p: object
    q_: object

    def residuals(self, q):
        return np.array([q[self.p.ix] - q[self.q_.ix]])

    def jacobian(self, q):
        return [{self.p.ix: 1.0, self.q_.ix: -1.0}]


@dataclass
class Parallel(Constraint):
    l1: object
    l2: object

    def residuals(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return np.array([ux * vy - uy * vx])

    def jacobian(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return [{
            self.l1.a.ix: -vy, self.l1.b.ix: vy,
            self.l1.a.iy: vx, self.l1.b.iy: -vx,
            self.l2.a.ix: uy, self.l2.b.ix: -uy,
            self.l2.a.iy: -ux, self.l2.b.iy: ux,
        }]


@dataclass
class Perpendicular(Constraint):
    l1: object
    l2: object

    def residuals(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return np.array([ux * vx + uy * vy])

    def jacobian(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return [{
            self.l1.a.ix: -vx, self.l1.b.ix: vx,
            self.l1.a.iy: -vy, self.l1.b.iy: vy,
            self.l2.a.ix: -ux, self.l2.b.ix: ux,
            self.l2.a.iy: -uy, self.l2.b.iy: uy,
        }]


@dataclass
class EqualLength(Constraint):
    l1: object
    l2: object

    def residuals(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return np.array([(ux ** 2 + uy ** 2) - (vx ** 2 + vy ** 2)])

    def jacobian(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        return [{
            self.l1.a.ix: -2 * ux, self.l1.b.ix: 2 * ux,
            self.l1.a.iy: -2 * uy, self.l1.b.iy: 2 * uy,
            self.l2.a.ix: 2 * vx, self.l2.b.ix: -2 * vx,
            self.l2.a.iy: 2 * vy, self.l2.b.iy: -2 * vy,
        }]


@dataclass
class Angle(Constraint):
    """Fix the directed angle from line l1 to line l2 to `theta` radians.

    The residual is r = cos θ · (u×v) − sin θ · (u·v) = |u||v|·sin(∠(u,v) − θ),
    which is zero exactly when the lines are at angle θ, and is a smooth function
    of the coordinates (no arctangent, no division). Its Jacobian is a linear
    combination of the cross-product rows (as in Parallel) and dot-product rows
    (as in Perpendicular).
    """
    l1: object
    l2: object
    theta: float

    def residuals(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        cross = ux * vy - uy * vx
        dot = ux * vx + uy * vy
        return np.array([np.cos(self.theta) * cross - np.sin(self.theta) * dot])

    def jacobian(self, q):
        *_, ux, uy = _line(q, self.l1)
        *_, vx, vy = _line(q, self.l2)
        c, s = np.cos(self.theta), np.sin(self.theta)
        cross_row = {
            self.l1.a.ix: -vy, self.l1.b.ix: vy,
            self.l1.a.iy: vx, self.l1.b.iy: -vx,
            self.l2.a.ix: uy, self.l2.b.ix: -uy,
            self.l2.a.iy: -ux, self.l2.b.iy: ux,
        }
        dot_row = {
            self.l1.a.ix: -vx, self.l1.b.ix: vx,
            self.l1.a.iy: -vy, self.l1.b.iy: vy,
            self.l2.a.ix: -ux, self.l2.b.ix: ux,
            self.l2.a.iy: -uy, self.l2.b.iy: uy,
        }
        return [{k: c * cross_row[k] - s * dot_row[k] for k in cross_row}]


@dataclass
class Tangent(Constraint):
    """Two circles tangent to each other. External (default): centre distance =
    sum of radii. Internal: centre distance = |difference of radii|."""
    c1: object
    c2: object
    internal: bool = False

    def residuals(self, q):
        cx1, cy1 = _p(q, self.c1.center)
        cx2, cy2 = _p(q, self.c2.center)
        r1, r2 = q[self.c1.ir], q[self.c2.ir]
        s = (r1 - r2) if self.internal else (r1 + r2)
        return np.array([(cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 - s ** 2])

    def jacobian(self, q):
        cx1, cy1 = _p(q, self.c1.center)
        cx2, cy2 = _p(q, self.c2.center)
        r1, r2 = q[self.c1.ir], q[self.c2.ir]
        dx, dy = cx1 - cx2, cy1 - cy2
        s = (r1 - r2) if self.internal else (r1 + r2)
        dr2 = 2 * s if self.internal else -2 * s
        return [{
            self.c1.center.ix: 2 * dx, self.c1.center.iy: 2 * dy,
            self.c2.center.ix: -2 * dx, self.c2.center.iy: -2 * dy,
            self.c1.ir: -2 * s, self.c2.ir: dr2,
        }]
