"""Turning 2D profiles into watertight 3D solids by extrusion.

Three cap strategies, each matched to a shape it triangulates robustly:

  * `extrude` — a simple polygon (bracket, plate). Cap by **ear clipping**, the
    standard algorithm for an arbitrary simple polygon.

  * `extrude_star` — a star-shaped loop (circle, regular polygon, a solid gear
    about its axis). Cap by a **fan from the centre**: one triangle per edge.
    This sidesteps ear clipping, whose point-in-triangle tests are numerically
    fragile on a gear's hundreds of near-collinear flank segments.

  * `extrude_ring` — the region between two **radially-aligned** loops (a bored
    gear, a washer, a tube). With one inner vertex per outer vertex at the same
    angle, the annular cap is one quad per index — no sweep, no sorting.
    `bore_matching` builds such an aligned bore for any star-shaped outline.

Each produces a closed 2-manifold: caps plus side walls, welded so a shared
boundary is shared topology. `mesh.is_watertight()` is the acceptance test.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _ensure_ccw(loop: np.ndarray) -> np.ndarray:
    return loop if _signed_area(loop) > 0 else loop[::-1].copy()


# ---------------------------------------------------------------------------
# ear-clipping triangulation of a simple polygon
# ---------------------------------------------------------------------------


def triangulate_simple(loop: np.ndarray) -> np.ndarray:
    """Triangulate a simple CCW polygon by ear clipping.

    An "ear" is a triangle of three consecutive vertices that lies inside the
    polygon and contains no other vertex. Clipping ears one at a time reduces the
    polygon by a vertex each step and always terminates, because every simple
    polygon with more than three vertices has at least two ears (the two-ears
    theorem). The output is a set of triangles whose total area equals the
    polygon's — which is exactly how the tests check it.
    """
    loop = _ensure_ccw(loop)
    n = len(loop)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    indices = list(range(n))
    triangles = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > 2 * n * n:
            raise RuntimeError("ear clipping failed to terminate; polygon may be self-intersecting")
        clipped = False
        m = len(indices)
        for i in range(m):
            prev = indices[(i - 1) % m]
            cur = indices[i]
            nxt = indices[(i + 1) % m]
            if _is_ear(loop, indices, prev, cur, nxt):
                triangles.append([prev, cur, nxt])
                indices.pop(i)
                clipped = True
                break
        if not clipped:
            raise RuntimeError("no ear found; polygon is not simple")
    triangles.append(indices[:])
    return np.array(triangles, dtype=np.int64)


def _is_ear(loop, indices, prev, cur, nxt) -> bool:
    a, b, c = loop[prev], loop[cur], loop[nxt]
    if _cross(a, b, c) <= 0:
        return False  # reflex corner, not a convex ear
    for j in indices:
        if j in (prev, cur, nxt):
            continue
        if _point_in_triangle(loop[j], a, b, c):
            return False
    return True


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_triangle(p, a, b, c) -> bool:
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# ---------------------------------------------------------------------------
# extrusion
# ---------------------------------------------------------------------------


def extrude(loop: np.ndarray, height: float) -> Mesh:
    """Extrude a simple polygon along +Z into a watertight prism."""
    loop = _ensure_ccw(np.asarray(loop, dtype=np.float64))
    n = len(loop)
    cap = triangulate_simple(loop)

    bottom = np.column_stack([loop, np.zeros(n)])
    top = np.column_stack([loop, np.full(n, height)])
    vertices = np.vstack([bottom, top])

    faces = []
    # bottom cap faces downward (reverse winding), top cap upward.
    for a, b, c in cap:
        faces.append([a, c, b])            # bottom (−Z)
        faces.append([a + n, b + n, c + n])  # top (+Z)
    # side walls: one quad per edge, split into two triangles, outward-facing.
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    return Mesh(vertices, np.array(faces))


def extrude_star(loop: np.ndarray, height: float, center=(0.0, 0.0)) -> Mesh:
    """Extrude a star-shaped loop, capping each end with a fan from `center`.

    A loop is star-shaped about a point if the whole boundary is visible from it —
    true of a circle, a regular polygon, and a gear about its axis. For such a
    loop the cap triangulates trivially and robustly: one triangle per edge,
    fanning out from the centre. This sidesteps ear clipping entirely, which
    matters for a gear whose hundreds of near-collinear flank segments make the
    general algorithm numerically fragile.
    """
    loop = _ensure_ccw(np.asarray(loop, dtype=np.float64))
    n = len(loop)
    cx, cy = center
    bottom = np.column_stack([loop, np.zeros(n)])
    top = np.column_stack([loop, np.full(n, height)])
    center_bottom = np.array([[cx, cy, 0.0]])
    center_top = np.array([[cx, cy, height]])
    vertices = np.vstack([bottom, top, center_bottom, center_top])
    cb, ct = 2 * n, 2 * n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([cb, j, i])              # bottom fan (−Z)
        faces.append([ct, i + n, j + n])      # top fan (+Z)
        faces.append([i, j, j + n])           # side wall
        faces.append([i, j + n, i + n])
    return Mesh(vertices, np.array(faces))


def extrude_ring(outer: np.ndarray, inner: np.ndarray, height: float) -> Mesh:
    """Extrude the region between two **radially-aligned** loops (a bored part).

    The two loops must have the same number of vertices, with vertex *i* of the
    inner loop at the same polar angle as vertex *i* of the outer loop. Under that
    alignment the annular cap triangulates trivially and robustly — one quad per
    index, split into two triangles — with no sweep, no sorting, and no
    numerical fragility. `bore_matching` builds a bore loop that satisfies the
    requirement for any star-shaped outer boundary; two circles of equal segment
    count are aligned by construction.
    """
    outer = _ensure_ccw(np.asarray(outer, dtype=np.float64))
    inner = _ensure_ccw(np.asarray(inner, dtype=np.float64))
    n = len(outer)
    if len(inner) != n:
        raise ValueError("extrude_ring needs equal-length, radially-aligned loops "
                         f"(got {n} outer, {len(inner)} inner); see bore_matching")

    ring2d = np.vstack([outer, inner])
    bottom = np.column_stack([ring2d, np.zeros(2 * n)])
    top = np.column_stack([ring2d, np.full(2 * n, height)])
    vertices = np.vstack([bottom, top])
    base_top = 2 * n
    # index map: outer bottom i in 0..n-1, inner bottom n+i, tops add base_top.

    faces = []
    for i in range(n):
        j = (i + 1) % n
        oi, oj = i, j                 # outer bottom
        ii, ij = n + i, n + j         # inner bottom
        # bottom cap (−Z): annulus quad (oi, oj, ij, ii), wound to face down
        faces.append([oi, ij, oj])
        faces.append([oi, ii, ij])
        # top cap (+Z): same quad on the top ring, wound to face up
        faces.append([oi + base_top, oj + base_top, ij + base_top])
        faces.append([oi + base_top, ij + base_top, ii + base_top])
        # outer wall — outward
        faces.append([oi, oj, oj + base_top])
        faces.append([oi, oj + base_top, oi + base_top])
        # inner (bore) wall — inward
        faces.append([ii, ij + base_top, ij])
        faces.append([ii, ii + base_top, ij + base_top])
    return Mesh(vertices, np.array(faces))


def bore_matching(outer: np.ndarray, radius: float) -> np.ndarray:
    """A bore loop aligned to `outer`: one point per outer vertex, at its angle.

    Because a star-shaped boundary has one vertex per angle, placing a bore point
    at each outer vertex's angle produces two radially-aligned loops — the
    precondition `extrude_ring` needs to triangulate the cap without any sweep.
    """
    outer = np.asarray(outer, dtype=np.float64)
    angles = np.arctan2(outer[:, 1], outer[:, 0])
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


# ---------------------------------------------------------------------------
# convenience primitives
# ---------------------------------------------------------------------------


def box(width: float, depth: float, height: float) -> Mesh:
    from .profiles import rectangle

    return extrude(rectangle(width, depth), height).translated(dz=-height / 2)


def cylinder(radius: float, height: float, segments: int = 64) -> Mesh:
    from .profiles import circle

    return extrude(circle(radius, segments), height)


def tube(outer_radius: float, inner_radius: float, height: float, segments: int = 64) -> Mesh:
    from .profiles import circle

    outer = circle(outer_radius, segments)
    return extrude_ring(outer, bore_matching(outer, inner_radius), height)
