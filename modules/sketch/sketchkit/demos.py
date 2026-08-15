"""A few parametric sketches, each built only from constraints, used by both the
CLI and the showcase. Each returns the sketch, the ordered points that form its
profile (for extrusion), and a label. None of them places its geometry directly —
the coordinates come out of the solver.
"""

from __future__ import annotations

import numpy as np

from .constraints import (
    Angle, Distance, Horizontal, Radius, Tangent, Vertical,
)
from .sketch import Sketch


def rectangle(width: float = 40.0, height: float = 25.0):
    """Four corners, from horizontal/vertical edges and two dimensions. The
    initial guess is a crooked quadrilateral; the solver squares it up."""
    s = Sketch()
    p0 = s.point(0.0, 0.0)
    p1 = s.point(width * 0.8, height * 0.1)
    p2 = s.point(width * 0.9, height * 1.1)
    p3 = s.point(width * 0.2, height * 0.9)
    s.fix(p0)
    s.add(
        Horizontal(p0, p1), Vertical(p1, p2), Horizontal(p2, p3), Vertical(p3, p0),
        Distance(p0, p1, width), Distance(p1, p2, height),
    )
    return s, [p0, p1, p2, p3], f"rectangle {width:g}x{height:g}"


def regular_polygon(sides: int = 6, circumradius: float = 20.0):
    """A regular n-gon defined with no coordinates: pin a centre, put every vertex
    at the same distance from it, and set the angle between consecutive radial
    lines to 2π/n. That is exactly how a CAD sketcher builds a polygon — a centre,
    a radius, and equal angular divisions — and it is well-conditioned, so the
    solver squares a perturbed ring of points into an exact polygon in a handful
    of Newton steps.
    """
    s = Sketch()
    center = s.point(0.0, 0.0)
    pts = []
    for k in range(sides):
        a = 2 * np.pi * k / sides + 0.15 * np.sin(2 * k)  # a perturbed initial guess
        pts.append(s.point(circumradius * np.cos(a) * 0.9, circumradius * np.sin(a) * 1.1))
    s.fix(center)           # centre the polygon
    s.fix_y(pts[0])         # drop the first vertex on the x-axis → pins rotation
    radial = [s.line(center, p) for p in pts]
    cons = [Distance(center, p, circumradius) for p in pts]
    cons += [Angle(radial[k], radial[k + 1], 2 * np.pi / sides) for k in range(sides - 1)]
    s.add(*cons)
    return s, pts, f"regular {sides}-gon r{circumradius:g}"


def l_bracket(length: float = 40.0, height: float = 30.0, wall: float = 8.0):
    """An L-shaped profile (one reflex corner) from horizontal/vertical edges and
    dimensions — the kind of outline a real bracket sketch has."""
    s = Sketch()
    p = [
        s.point(0, 0), s.point(length, 2), s.point(length * 0.9, wall * 1.2),
        s.point(wall * 1.1, wall * 0.9), s.point(wall * 0.8, height),
        s.point(-2, height * 0.95),
    ]
    s.fix(p[0])
    s.add(
        # every edge is horizontal or vertical...
        Horizontal(p[0], p[1]), Vertical(p[1], p[2]), Horizontal(p[2], p[3]),
        Vertical(p[3], p[4]), Horizontal(p[4], p[5]), Vertical(p[5], p[0]),
        # ...and exactly four independent dimensions define the rest (the two
        # remaining edge lengths follow from the profile closing on itself).
        Distance(p[0], p[1], length), Distance(p[5], p[0], height),
        Distance(p[1], p[2], wall), Distance(p[4], p[5], wall),
    )
    return s, p, f"L-bracket {length:g}x{height:g}"


def tangent_circles(r1: float = 3.0, r2: float = 4.0, r3: float = 5.0):
    """Three mutually tangent circles of fixed radius, plus a fourth nestled in
    the gap tangent to all three. Verification-only (not a single extrudable
    loop): the solved fourth radius is checked against Descartes' Circle Theorem.
    Returns (sketch, [c1, c2, c3, c4])."""
    s = Sketch()
    c1 = s.circle(0.0, 0.0, r1)
    c2 = s.circle(r1 + r2, 0.0, r2)
    c3 = s.circle(r1 * 0.6, r1 + r3, r3)
    c4 = s.circle(r1 * 0.8, r1 * 0.7, 0.5)
    s.fix(c1.center)        # pin position
    s.fix_y(c2.center)      # pin orientation
    s.add(
        Radius(c1, r1), Radius(c2, r2), Radius(c3, r3),
        Tangent(c1, c2), Tangent(c2, c3), Tangent(c1, c3),
        Tangent(c1, c4), Tangent(c2, c4), Tangent(c3, c4),
    )
    return s, [c1, c2, c3, c4]


def descartes_radius(r1: float, r2: float, r3: float) -> float:
    """The inner Soddy circle radius predicted by Descartes' Circle Theorem."""
    k1, k2, k3 = 1 / r1, 1 / r2, 1 / r3
    k4 = k1 + k2 + k3 + 2 * np.sqrt(k1 * k2 + k2 * k3 + k3 * k1)
    return 1 / k4


DEMOS = {
    "rectangle": rectangle,
    "polygon": regular_polygon,
    "bracket": l_bracket,
}
