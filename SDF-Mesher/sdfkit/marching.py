"""Marching tetrahedra: extract the zero-isosurface of an SDF as a triangle mesh.

Marching *tetrahedra*, not the more famous marching cubes, on purpose. A cube has
256 corner sign patterns with genuine topological ambiguities — the "which way
does the saddle connect" cases that make naïve marching cubes leak holes. A
tetrahedron has four corners and no ambiguity at all: the isosurface either
misses it, cuts off one corner (a triangle), or separates two corners from two (a
quad, split into two triangles). Split every grid cube into six tetrahedra the
same way everywhere (the Freudenthal / Kuhn decomposition, all six sharing the
cube's main diagonal) and neighbouring cells triangulate their shared faces
identically — so the extracted surface is watertight by construction, before any
cleanup.

The grid is sampled vectorised, and all cells are processed together per case, so
a 64³ field meshes in a moment rather than a Python-loop age.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .sdf import SDF

# The eight cube corners, indexed by (dx, dy, dz) bits: corner c = dx + 2·dy + 4·dz.
_CORNER = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])

# Freudenthal decomposition: six tetrahedra, each a path (0,0,0) → (1,1,1) adding
# one unit axis step at a time (the six axis orderings), all sharing corners 0↔7.
_TETS = np.array([
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
])


def triangulate(sdf: SDF, bounds=None, resolution: int = 48, isolevel: float = 0.0) -> Mesh:
    """Mesh the surface `sdf(p) = isolevel` inside `bounds` at `resolution` cells
    along the longest axis. `bounds` defaults to the SDF's own conservative box,
    padded slightly so the surface never touches the grid edge."""
    if bounds is None:
        lo, hi = sdf.bounds()
        pad = 0.05 * (hi - lo) + 1e-3
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = (np.asarray(b, dtype=np.float64) for b in bounds)

    extent = hi - lo
    steps = np.maximum((resolution * extent / extent.max()).astype(int), 1)
    nx, ny, nz = steps
    gx = np.linspace(lo[0], hi[0], nx + 1)
    gy = np.linspace(lo[1], hi[1], ny + 1)
    gz = np.linspace(lo[2], hi[2], nz + 1)

    # Sample the field on the full grid in one vectorised call.
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1)  # (nx+1,ny+1,nz+1,3)
    values = (sdf(grid.reshape(-1, 3)) - isolevel).reshape(grid.shape[:3])
    # Nudge exact zeros so no edge has a zero-length crossing.
    values = np.where(values == 0.0, -1e-12, values)

    # Per-corner value and position blocks over all cells at once.
    corner_val = []
    corner_pos = []
    for dx, dy, dz in _CORNER:
        corner_val.append(values[dx:dx + nx, dy:dy + ny, dz:dz + nz].reshape(-1))
        px = grid[dx:dx + nx, dy:dy + ny, dz:dz + nz, :].reshape(-1, 3)
        corner_pos.append(px)
    corner_val = np.stack(corner_val)   # (8, cells)
    corner_pos = np.stack(corner_pos)   # (8, cells, 3)

    tris = []
    for tet in _TETS:
        v = corner_val[tet]             # (4, cells)
        p = corner_pos[tet]             # (4, cells, 3)
        _emit_tetra(v, p, tris)

    if not tris:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))

    tri_pts = np.concatenate(tris, axis=0)            # (K, 3, 3)
    verts = tri_pts.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    return Mesh(verts, faces).welded().oriented(outward=sdf.gradient)


def _interp(va, vb, pa, pb):
    """Linear zero-crossing between corner a and corner b (vectorised)."""
    t = (va / (va - vb))[:, None]
    return pa + t * (pb - pa)


def _emit_tetra(v, p, out):
    """Append this tetrahedron-type's triangles (over all cells) to `out`.

    `v` is (4, cells), `p` is (4, cells, 3). No winding care is taken — the mesh
    is oriented afterwards from the SDF gradient — only that the emitted triangles
    tile the crossing correctly (one triangle for a 1|3 split, two for a 2|2).
    """
    inside = v < 0.0                    # (4, cells)
    ninside = inside.sum(axis=0)        # (cells,)

    # 1|3 splits: one corner is the minority; three edges from it cross.
    for lone in range(4):
        others = [k for k in range(4) if k != lone]
        mask = ((ninside == 1) & inside[lone]) | ((ninside == 3) & ~inside[lone])
        if not mask.any():
            continue
        vl, pl = v[lone, mask], p[lone, mask]
        pts = [_interp(vl, v[o, mask], pl, p[o, mask]) for o in others]
        out.append(np.stack(pts, axis=1))          # (m, 3, 3)

    # 2|2 splits: two corners inside, two outside; four crossing edges form a quad.
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        c, d = [k for k in range(4) if k not in (a, b)]
        mask = (ninside == 2) & inside[a] & inside[b]
        if not mask.any():
            continue
        q_ac = _interp(v[a, mask], v[c, mask], p[a, mask], p[c, mask])
        q_ad = _interp(v[a, mask], v[d, mask], p[a, mask], p[d, mask])
        q_bd = _interp(v[b, mask], v[d, mask], p[b, mask], p[d, mask])
        q_bc = _interp(v[b, mask], v[c, mask], p[b, mask], p[c, mask])
        out.append(np.stack([q_ac, q_ad, q_bd], axis=1))   # quad triangle 1
        out.append(np.stack([q_ac, q_bd, q_bc], axis=1))   # quad triangle 2
