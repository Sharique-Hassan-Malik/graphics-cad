"""Independent verification that a finished tiling is legal.

The solver is *supposed* to guarantee legality, so this checks it the other way
round — it never looks at the solver's internal state, only at the final grid of
tile indices, and re-tests every shared edge against the tileset's adjacency
rules. Zero violations across thousands of edges is the headline correctness
claim: the output isn't merely plausible, it provably satisfies every constraint.
"""

from __future__ import annotations

import numpy as np

from .tiles import DIRS, TileSet


def adjacency_violations(grid: np.ndarray, tileset: TileSet) -> list:
    """Return every illegal neighbour pair as (r, c, direction, a, b)."""
    height, width = grid.shape
    bad = []
    for r in range(height):
        for c in range(width):
            a = grid[r, c]
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    b = grid[nr, nc]
                    if not tileset.compat[d][a, b]:
                        bad.append((r, c, d, int(a), int(b)))
    return bad


def is_valid(grid: np.ndarray, tileset: TileSet) -> bool:
    return len(adjacency_violations(grid, tileset)) == 0


def edge_count(height: int, width: int) -> int:
    """Number of shared interior edges (each checked twice → distinct pairs)."""
    return 2 * (height * (width - 1) + width * (height - 1))


def tile_histogram(grid: np.ndarray, tileset: TileSet) -> np.ndarray:
    return np.bincount(grid.reshape(-1), minlength=tileset.n)
