"""The triangle mesh for this generator — the shared implementation.

There were three near-identical `Mesh` classes in this repository. They agreed
on the representation (vertices N×3, triangles M×3) and disagreed about which
checks existed, so a mesh from one generator could not be validated with
another's tools. `geokit.mesh` is the union of them, and every generator now
produces it.

Re-exported here so `from terrainkit.mesh import Mesh` keeps working.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from geokit.mesh import Mesh  # noqa: E402,F401

__all__ = ["Mesh", "terrain_mesh"]


def terrain_mesh(heightmap: np.ndarray, width: float = 10.0, height_scale: float = 3.0,
                 base: float = 0.0) -> Mesh:
    """A watertight terrain slab from a heightmap. `width` is the world size of
    the longest side; `height_scale` maps the [0,1] heightmap to world height."""
    rows, cols = heightmap.shape
    cell = width / max(rows, cols)
    z = base + heightmap * height_scale
    floor = base - 0.5 * height_scale

    verts = []
    top = np.zeros((rows, cols), int)
    bot = np.zeros((rows, cols), int)
    for r in range(rows):
        for c in range(cols):
            top[r, c] = len(verts)
            verts.append((c * cell, (rows - 1 - r) * cell, float(z[r, c])))
    for r in range(rows):
        for c in range(cols):
            bot[r, c] = len(verts)
            verts.append((c * cell, (rows - 1 - r) * cell, floor))

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b, d, e = top[r, c], top[r, c + 1], top[r + 1, c + 1], top[r + 1, c]
            faces += [(a, b, d), (a, d, e)]
            a2, b2, d2, e2 = bot[r, c], bot[r, c + 1], bot[r + 1, c + 1], bot[r + 1, c]
            faces += [(a2, d2, b2), (a2, e2, d2)]

    def wall(t0, t1, b0, b1):
        faces.append((t0, t1, b1))
        faces.append((t0, b1, b0))

    for c in range(cols - 1):
        wall(top[0, c], top[0, c + 1], bot[0, c], bot[0, c + 1])
        wall(top[rows - 1, c + 1], top[rows - 1, c], bot[rows - 1, c + 1], bot[rows - 1, c])
    for r in range(rows - 1):
        wall(top[r + 1, 0], top[r, 0], bot[r + 1, 0], bot[r, 0])
        wall(top[r, cols - 1], top[r + 1, cols - 1], bot[r, cols - 1], bot[r + 1, cols - 1])

    return Mesh(verts, faces)
