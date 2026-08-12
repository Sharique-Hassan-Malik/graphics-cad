"""Wave Function Collapse: fill a grid with tiles so that every adjacency is legal.

The name is borrowed from physics but the algorithm is pure constraint solving.
Every cell starts in *superposition* — all tiles are possible. Then repeat:

  1. **Observe.** Pick the most-constrained undecided cell (lowest entropy — the
     fewest remaining options) and *collapse* it to a single tile, chosen at
     random weighted by the tiles' frequencies.
  2. **Propagate.** That choice forbids some options in the neighbours (anything
     no longer compatible across the shared edge), which forbids more in *their*
     neighbours, and so on. The wave of eliminations is pushed out until nothing
     changes — an arc-consistency (AC-3-style) fixpoint.

Occasionally propagation paints a cell into a corner: zero options left, a
*contradiction*. This tiled formulation simply restarts (deterministically), and
in practice succeeds within a few attempts. The output is guaranteed legal — the
verification module checks that independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiles import DIRS, TileSet


@dataclass
class Result:
    grid: np.ndarray          # (H, W) int tile indices, or None on total failure
    success: bool
    attempts: int
    collapses: int            # observation steps in the successful run
    tileset: TileSet
    seed: int


class Contradiction(Exception):
    pass


def collapse(tileset: TileSet, width: int, height: int, seed: int = 0,
             max_attempts: int = 40, border=None) -> Result:
    """Generate a `height × width` tiling. `border`, if given, is a socket label
    every outward-facing edge of the grid must present (e.g. "shut" to fence a
    pipe network, or a water corner for an island) — a hard boundary constraint."""
    rng = np.random.default_rng(seed)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            grid, collapses = _run(tileset, width, height, rng, border)
            return Result(grid, True, attempts, collapses, tileset, seed)
        except Contradiction:
            continue
    return Result(None, False, attempts, 0, tileset, seed)


def _run(tileset: TileSet, width, height, rng, border):
    n = tileset.n
    possible = np.ones((height, width, n), dtype=bool)
    weights = tileset.weights

    if border is not None:
        _apply_border(tileset, possible, border)
        _propagate_all(tileset, possible)

    collapses = 0
    while True:
        counts = possible.sum(axis=2)
        if (counts == 0).any():
            raise Contradiction()
        undecided = counts > 1
        if not undecided.any():
            break  # every cell settled on exactly one tile

        # observe: the undecided cell of lowest weighted entropy (+ tiny noise)
        entropy = _entropy(possible, weights)
        entropy[~undecided] = np.inf
        entropy += rng.random(entropy.shape) * 1e-6
        r, c = np.unravel_index(np.argmin(entropy), entropy.shape)

        options = np.flatnonzero(possible[r, c])
        w = weights[options]
        choice = options[rng.choice(len(options), p=w / w.sum())]
        possible[r, c] = False
        possible[r, c, choice] = True
        collapses += 1

        _propagate(tileset, possible, [(r, c)])

    return np.argmax(possible, axis=2), collapses


def _entropy(possible, weights):
    """Weighted Shannon entropy of each cell's remaining options."""
    w = np.where(possible, weights[None, None, :], 0.0)
    total = w.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.log(total) - (np.where(w > 0, w * np.log(np.where(w > 0, w, 1)), 0).sum(axis=2)) / total
    return ent


def _propagate(tileset, possible, seeds):
    """Push eliminations out from `seeds` until the grid is arc-consistent."""
    height, width, _ = possible.shape
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        here = possible[r, c]
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            # tiles allowed in the neighbour = those compatible with some option here
            allowed = tileset.compat[d][here].any(axis=0)
            before = possible[nr, nc]
            after = before & allowed
            if not after.any():
                raise Contradiction()
            if (after != before).any():
                possible[nr, nc] = after
                stack.append((nr, nc))


def _propagate_all(tileset, possible):
    height, width, _ = possible.shape
    _propagate(tileset, possible, [(r, c) for r in range(height) for c in range(width)])


def _apply_border(tileset, possible, label):
    """Forbid, on the grid's outer edges, any tile whose outward socket ≠ label."""
    height, width, _ = possible.shape
    socks = [t.sockets for t in tileset.tiles]
    north = np.array([s[0] == label for s in socks])
    east = np.array([s[1] == label for s in socks])
    south = np.array([s[2] == label for s in socks])
    west = np.array([s[3] == label for s in socks])
    possible[0, :, :] &= north
    possible[-1, :, :] &= south
    possible[:, -1, :] &= east
    possible[:, 0, :] &= west
