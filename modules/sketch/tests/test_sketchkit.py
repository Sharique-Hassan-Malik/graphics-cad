"""Tests for the constraint solver.

The through-line: a solved sketch is right only if it is right as *numbers* — the
constraints are satisfied to machine precision, the reported degrees of freedom
match what the constraint set actually implies, and (the deepest check) the
analytic Jacobian of every constraint agrees with a finite-difference of its own
residual. A wrong derivative can't hide behind a solve that still limps to an
answer, and a wrong DOF verdict can't hide behind a picture.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sketchkit import constraints as C  # noqa: E402
from sketchkit import demos  # noqa: E402
from sketchkit.sketch import Sketch  # noqa: E402


# ---------------------------------------------------------------------------
# every constraint's analytic Jacobian must match finite differences
# ---------------------------------------------------------------------------


def _numerical_jacobian(constraint, q, k, h=1e-6):
    n = len(q)
    grad = np.zeros(n)
    for i in range(n):
        qp, qm = q.copy(), q.copy()
        qp[i] += h
        qm[i] -= h
        grad[i] = (constraint.residuals(qp)[k] - constraint.residuals(qm)[k]) / (2 * h)
    return grad


def _all_constraints_and_state():
    rng = np.random.default_rng(0)
    s = Sketch()
    pts = [s.point(*(rng.normal(size=2) * 5)) for _ in range(6)]
    c1 = s.circle(*(rng.normal(size=2) * 5), 3.0)
    c2 = s.circle(*(rng.normal(size=2) * 5), 2.0)
    line = lambda i, j: s.line(pts[i], pts[j])
    cons = [
        C.Coincident(pts[0], pts[1]),
        C.PointOnLine(pts[2], line(3, 4)),
        C.PointOnCircle(pts[5], c1),
        C.Distance(pts[0], pts[2], 7.0),
        C.Radius(c2, 2.5),
        C.Horizontal(pts[0], pts[3]),
        C.Vertical(pts[1], pts[4]),
        C.Parallel(line(0, 1), line(2, 3)),
        C.Perpendicular(line(1, 2), line(3, 4)),
        C.EqualLength(line(0, 1), line(4, 5)),
        C.Angle(line(0, 2), line(3, 5), math.radians(37)),
        C.Tangent(c1, c2),
        C.Tangent(c1, c2, internal=True),
    ]
    return cons, np.array(s._params)


@pytest.mark.parametrize("index", range(13))
def test_constraint_jacobian_matches_finite_difference(index):
    cons, q = _all_constraints_and_state()
    con = cons[index]
    rows = con.jacobian(q)
    assert len(rows) == con.n_equations
    for k, row in enumerate(rows):
        analytic = np.zeros(len(q))
        for idx, val in row.items():
            analytic[idx] = val
        numeric = _numerical_jacobian(con, q, k)
        assert np.max(np.abs(analytic - numeric)) < 1e-5, type(con).__name__


# ---------------------------------------------------------------------------
# solving well-posed sketches
# ---------------------------------------------------------------------------


def test_rectangle_solves_to_an_exact_rectangle():
    sketch, pts, _ = demos.rectangle(40.0, 25.0)
    result = sketch.solve()
    assert result.status == "fully-constrained"
    assert result.dof == 0 and result.redundant == 0
    assert result.residual_norm < 1e-9
    c = sketch.coords(pts)
    # opposite sides equal, adjacent sides perpendicular, right dimensions
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(40.0, abs=1e-9)
    assert np.linalg.norm(c[2] - c[1]) == pytest.approx(25.0, abs=1e-9)
    assert np.dot(c[1] - c[0], c[2] - c[1]) == pytest.approx(0.0, abs=1e-7)


def test_regular_polygon_is_actually_regular():
    sketch, pts, _ = demos.regular_polygon(6, 20.0)
    result = sketch.solve()
    assert result.status == "fully-constrained"
    c = sketch.coords(pts)
    edges = [np.linalg.norm(c[(k + 1) % 6] - c[k]) for k in range(6)]
    radii = [np.linalg.norm(c[k]) for k in range(6)]
    assert max(edges) - min(edges) < 1e-9      # all sides equal
    assert max(radii) - min(radii) < 1e-9      # all on the circumcircle
    assert radii[0] == pytest.approx(20.0, abs=1e-9)


def test_l_bracket_solves_to_the_expected_corners():
    sketch, pts, _ = demos.l_bracket(40.0, 30.0, 8.0)
    result = sketch.solve()
    assert result.status == "fully-constrained"
    expected = [(0, 0), (40, 0), (40, 8), (8, 8), (8, 30), (0, 30)]
    got = sketch.coords(pts)
    assert np.allclose(got, expected, atol=1e-9)


def test_solver_converges_in_a_handful_of_iterations():
    for factory in (lambda: demos.rectangle(30, 20),
                    lambda: demos.regular_polygon(8, 15),
                    lambda: demos.l_bracket(50, 40, 10)):
        sketch, _, _ = factory()
        result = sketch.solve()
        assert result.converged
        assert result.iterations <= 12
        assert result.residual_norm < 1e-9


def test_fixed_points_do_not_move():
    sketch, pts, _ = demos.rectangle(40, 25)
    before = sketch.xy(pts[0])
    sketch.solve()
    after = sketch.xy(pts[0])
    assert before == after == (0.0, 0.0)


# ---------------------------------------------------------------------------
# the degrees-of-freedom verdict
# ---------------------------------------------------------------------------


def test_under_constrained_is_detected():
    # A rectangle missing one dimension can still change size in one direction.
    s = Sketch()
    p = [s.point(0, 0), s.point(30, 4), s.point(35, 20), s.point(-2, 18)]
    s.fix(p[0])
    s.add(C.Horizontal(p[0], p[1]), C.Vertical(p[1], p[2]),
          C.Horizontal(p[2], p[3]), C.Vertical(p[3], p[0]),
          C.Distance(p[0], p[1], 40))          # height left free
    result = s.solve()
    assert result.status == "under-constrained"
    assert result.dof == 1


def test_over_constrained_is_detected():
    # A perpendicularity implied by the horizontal+vertical edges is redundant.
    s = Sketch()
    p = [s.point(0, 0), s.point(30, 4), s.point(35, 20), s.point(-2, 18)]
    s.fix(p[0])
    s.add(C.Horizontal(p[0], p[1]), C.Vertical(p[1], p[2]),
          C.Horizontal(p[2], p[3]), C.Vertical(p[3], p[0]),
          C.Distance(p[0], p[1], 40), C.Distance(p[1], p[2], 25),
          C.Perpendicular(s.line(p[0], p[1]), s.line(p[1], p[2])))
    result = s.solve()
    assert result.status == "over-constrained"
    assert result.redundant == 1
    assert result.dof == 0


# ---------------------------------------------------------------------------
# the nonlinear headline: Descartes' Circle Theorem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radii", [(3.0, 4.0, 5.0), (2.0, 2.0, 2.0), (1.0, 2.5, 6.0)])
def test_tangent_circles_match_descartes(radii):
    sketch, circles = demos.tangent_circles(*radii)
    result = sketch.solve()
    assert result.status == "fully-constrained"
    assert result.residual_norm < 1e-9
    solved = sketch.radius(circles[3])
    predicted = demos.descartes_radius(*radii)
    # The solver found the inner Soddy circle from tangency alone; it must match
    # the closed-form curvature identity to machine precision.
    assert abs(solved - predicted) < 1e-9


def test_angle_constraint_realises_the_angle():
    s = Sketch()
    a = s.point(0, 0)
    b = s.point(10, 0)
    c = s.point(3, 9)
    s.fix(a, b)  # first line pinned along the x-axis
    s.add(C.Distance(a, c, 10.0),
          C.Angle(s.line(a, b), s.line(a, c), math.radians(60)))
    s.solve()
    cx, cy = s.xy(c)
    measured = math.degrees(math.atan2(cy, cx))
    assert measured == pytest.approx(60.0, abs=1e-6)
    assert math.hypot(cx, cy) == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Blender bridge
# ---------------------------------------------------------------------------


def test_bpy_script_is_emitted_without_blender():
    from sketchkit.blender_export import to_bpy_script

    sketch, pts, _ = demos.regular_polygon(6, 20)
    sketch.solve()
    script = to_bpy_script(sketch.coords(pts), thickness=5.0, name="hex")
    assert "extrude_face_region" in script   # Blender does the modelling
    assert "bmesh" in script
    assert script.count("(") > 6             # the solved coordinates are inlined


def _blender():
    from sketchkit.blender_export import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_extrudes_and_renders(tmp_path):
    from sketchkit.blender_export import render

    sketch, pts, _ = demos.regular_polygon(6, 20)
    sketch.solve()
    result = render(sketch.coords(pts), out_dir=str(tmp_path), name="hex", thickness=6.0, samples=8)
    assert result["ran"], result
    assert result["blend_written"] and os.path.getsize(result["blend"]) > 1000
    assert result["png_written"] and os.path.getsize(result["png"]) > 1000
