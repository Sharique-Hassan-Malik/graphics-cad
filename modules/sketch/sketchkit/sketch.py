"""The user-facing sketch: add points, lines and circles, pin what should not
move, declare constraints, and solve.

Entities are thin. A `Point` owns two slots in the sketch's flat parameter vector
(its x and y); a `Circle` is a centre `Point` plus one radius slot; a `Line` owns
no parameters at all — it is just a reference to two points, so a line and the
points it is drawn through stay the same object, and a constraint on the line
moves those points. That shared-parameter model is the whole reason a constraint
solver feels "connected": dragging one entity ripples through everything wired to
it.

`Sketch.solve()` assembles the residual vector and the sparse Jacobian from the
constraints and hands them to the numerical core. `Sketch.profile()` reads an
ordered set of points back out as a closed loop, ready to extrude in Blender.
"""

from __future__ import annotations

import numpy as np

from .constraints import Constraint
from .solver import SolveResult, solve


class Point:
    def __init__(self, ix: int, iy: int, name: str = ""):
        self.ix, self.iy = ix, iy
        self.name = name


class Line:
    def __init__(self, a: Point, b: Point):
        self.a, self.b = a, b


class Circle:
    def __init__(self, center: Point, ir: int):
        self.center, self.ir = center, ir


class Sketch:
    def __init__(self):
        self._params: list[float] = []
        self._free: list[bool] = []
        self.constraints: list[Constraint] = []
        self.points: list[Point] = []
        self.circles: list[Circle] = []
        self._result: SolveResult | None = None

    # -- building ------------------------------------------------------------

    def _alloc(self, value: float) -> int:
        self._params.append(float(value))
        self._free.append(True)
        return len(self._params) - 1

    def point(self, x: float, y: float, name: str = "") -> Point:
        p = Point(self._alloc(x), self._alloc(y), name)
        self.points.append(p)
        return p

    def line(self, a: Point, b: Point) -> Line:
        return Line(a, b)

    def circle(self, cx: float, cy: float, r: float) -> Circle:
        center = self.point(cx, cy)
        c = Circle(center, self._alloc(r))
        self.circles.append(c)
        return c

    def fix(self, *points: Point) -> "Sketch":
        """Pin points in place: their coordinates become constants the solver may
        not change. Anchoring at least one point (and usually one direction) is
        what removes the sketch's free rigid-body motion."""
        for p in points:
            self._free[p.ix] = False
            self._free[p.iy] = False
        return self

    def fix_x(self, p: Point) -> "Sketch":
        self._free[p.ix] = False
        return self

    def fix_y(self, p: Point) -> "Sketch":
        self._free[p.iy] = False
        return self

    def add(self, *constraints: Constraint) -> "Sketch":
        self.constraints.extend(constraints)
        return self

    # -- solving -------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return len(self._params)

    @property
    def n_equations(self) -> int:
        return sum(c.n_equations for c in self.constraints)

    def _residual_fn(self):
        cons = self.constraints
        if not cons:
            return lambda q: np.zeros(0)
        return lambda q: np.concatenate([c.residuals(q) for c in cons])

    def _jacobian_fn(self):
        n = self.n_params
        cons = self.constraints
        rows_per = [c.n_equations for c in cons]
        total = sum(rows_per)

        def jac(q):
            J = np.zeros((total, n))
            r = 0
            for c in cons:
                for row in c.jacobian(q):
                    for idx, val in row.items():
                        J[r, idx] = val
                    r += 1
            return J

        return jac

    def solve(self, tol: float = 1e-12, max_iter: int = 200) -> SolveResult:
        q0 = np.array(self._params, dtype=np.float64)
        free = np.array(self._free, dtype=bool)
        result = solve(self._residual_fn(), self._jacobian_fn(), q0, free, tol, max_iter)
        # Commit the solved values so entity queries reflect the solution.
        self._params = list(result.q)
        self._result = result
        return result

    # -- reading the solution back ------------------------------------------

    def xy(self, p: Point) -> tuple[float, float]:
        return self._params[p.ix], self._params[p.iy]

    def radius(self, c: Circle) -> float:
        return self._params[c.ir]

    def coords(self, points: list[Point]) -> np.ndarray:
        return np.array([self.xy(p) for p in points])

    def profile(self, points: list[Point]) -> np.ndarray:
        """The solved coordinates of `points`, as an ordered closed loop (Nx2),
        ready to hand to the Blender extruder."""
        return self.coords(points)
