"""Lift a solved *terrain* tiling into a 3D island.

The Wang-corner terrain tileset keys each tile on its four corner materials, and
WFC guarantees neighbouring tiles agree on the corners they share. So the grid of
tile *corners* is a globally consistent (H+1)×(W+1) lattice of land/water values —
a clean binary heightmap that falls out of the constraint solution for free. This
turns it into a watertight island slab: a stepped top surface (land raised, water
low), a flat underside, and side walls, ready to extrude/render in Blender.
"""

from __future__ import annotations

import numpy as np


def terrain_lattice(result) -> np.ndarray:
    """The (H+1)×(W+1) corner land/water lattice implied by a terrain result."""
    ts, grid = result.tileset, result.grid
    h, w = grid.shape
    lattice = np.full((h + 1, w + 1), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            nw, ne, se, sw = ts.tiles[grid[r, c]].meta["corners"]
            for rr, cc, val in ((r, c, nw), (r, c + 1, ne), (r + 1, c + 1, se), (r + 1, c, sw)):
                if lattice[rr, cc] == -1:
                    lattice[rr, cc] = val
                elif lattice[rr, cc] != val:                      # WFC forbids this
                    raise ValueError("inconsistent corner lattice — adjacency was violated")
    return lattice


def island_mesh(result, cell=1.0, land_z=1.0, water_z=0.28, base_z=0.0):
    """A watertight island slab: raised land, low water, closed underneath."""
    lattice = terrain_lattice(result)
    hp, wp = lattice.shape
    z = np.where(lattice == 1, land_z, water_z)

    verts, top, bot = [], np.zeros((hp, wp), int), np.zeros((hp, wp), int)
    for r in range(hp):
        for c in range(wp):
            top[r, c] = len(verts)
            verts.append((c * cell, (hp - 1 - r) * cell, float(z[r, c])))
    for r in range(hp):
        for c in range(wp):
            bot[r, c] = len(verts)
            verts.append((c * cell, (hp - 1 - r) * cell, base_z))

    faces = []
    for r in range(hp - 1):
        for c in range(wp - 1):
            a, b, d, e = top[r, c], top[r, c + 1], top[r + 1, c + 1], top[r + 1, c]
            faces += [(a, b, d), (a, d, e)]                       # top surface
            a2, b2, d2, e2 = bot[r, c], bot[r, c + 1], bot[r + 1, c + 1], bot[r + 1, c]
            faces += [(a2, d2, b2), (a2, e2, d2)]                 # underside (reversed)

    def wall(t0, t1, b0, b1):
        faces.append((t0, t1, b1))
        faces.append((t0, b1, b0))

    for c in range(wp - 1):
        wall(top[0, c], top[0, c + 1], bot[0, c], bot[0, c + 1])           # north
        wall(top[hp - 1, c + 1], top[hp - 1, c], bot[hp - 1, c + 1], bot[hp - 1, c])  # south
    for r in range(hp - 1):
        wall(top[r + 1, 0], top[r, 0], bot[r + 1, 0], bot[r, 0])           # west
        wall(top[r, wp - 1], top[r + 1, wp - 1], bot[r, wp - 1], bot[r + 1, wp - 1])  # east

    return np.array(verts, dtype=float), np.array(faces, dtype=int)
