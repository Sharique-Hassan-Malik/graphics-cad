"""Tiles, their edge sockets, and the adjacency they imply.

A tile carries four edge *sockets* — one per side (N, E, S, W) — and a little
bitmap. Two tiles may sit next to each other exactly when the sockets on their
touching edges match: tile A's east socket must equal the west socket of the tile
to its east. Because a socket is defined to encode what the tile's border looks
like, "sockets match" and "the border pixels line up seamlessly" are the same
statement — so a valid WFC output is automatically a seamless image.

`TileSet` precomputes, for each tile and each of the four directions, the boolean
mask of tiles allowed to sit there. That mask is the only thing the solver needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# directions: 0=N, 1=E, 2=S, 3=W ; opposite is (d+2) % 4
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # (drow, dcol) for N, E, S, W


@dataclass
class Tile:
    name: str
    sockets: tuple            # (north, east, south, west), each an equality-matched label
    pixels: np.ndarray        # (h, w, 3) uint8
    weight: float = 1.0
    meta: dict = field(default_factory=dict)   # e.g. corner values for terrain tiles


def _rot90(tile: Tile, name: str) -> Tile:
    """Rotate a tile 90° clockwise: sockets shift N←W E←N S←E W←S, pixels rot."""
    n, e, s, w = tile.sockets
    return Tile(name, (w, n, e, s), np.rot90(tile.pixels, k=-1).copy(), tile.weight)


def _with_rotations(tile: Tile, count: int) -> list[Tile]:
    out = [tile]
    for i in range(1, count):
        out.append(_rot90(out[-1], f"{tile.name}_r{i}"))
    return out


class TileSet:
    def __init__(self, tiles: list[Tile]):
        self.tiles = tiles
        self.n = len(tiles)
        self.weights = np.array([t.weight for t in tiles], dtype=np.float64)
        self.compat = self._build_compat()

    def _build_compat(self) -> list[np.ndarray]:
        """compat[d][a, b] is True iff tile b may sit in direction d from tile a."""
        compat = []
        for d in range(4):
            opp = (d + 2) % 4
            m = np.zeros((self.n, self.n), dtype=bool)
            for a in range(self.n):
                for b in range(self.n):
                    m[a, b] = self.tiles[a].sockets[d] == self.tiles[b].sockets[opp]
            compat.append(m)
        return compat

    def tile_size(self) -> int:
        return self.tiles[0].pixels.shape[0]


# ---------------------------------------------------------------------------
# a small drawing helper
# ---------------------------------------------------------------------------

def _canvas(size, color):
    img = np.empty((size, size, 3), np.uint8)
    img[:] = color
    return img


def _bar(img, r0, r1, c0, c1, color):
    img[r0:r1, c0:c1] = color


# ---------------------------------------------------------------------------
# tileset 1 — "pipes" (the classic connected-network / Knots set)
# ---------------------------------------------------------------------------

def make_pipes(size: int = 15) -> TileSet:
    """Pipe segments that always connect: every edge socket is 'open' or 'shut',
    and open only ever meets open. The blank tile ties the empty space together.
    Renders as bright conduits on a dark board."""
    bg = (24, 26, 32)
    wire = (90, 200, 255)
    mid = size // 2
    t = max(1, size // 5)

    def piece(name, opens, weight):
        img = _canvas(size, bg)
        _bar(img, mid - t, mid + t, mid - t, mid + t, wire)   # centre pad
        if "N" in opens: _bar(img, 0, mid + t, mid - t, mid + t, wire)
        if "S" in opens: _bar(img, mid - t, size, mid - t, mid + t, wire)
        if "W" in opens: _bar(img, mid - t, mid + t, 0, mid + t, wire)
        if "E" in opens: _bar(img, mid - t, mid + t, mid - t, size, wire)
        sock = tuple(("open" if d in opens else "shut") for d in "NESW")
        return Tile(name, sock, img, weight)

    tiles = []
    tiles.append(piece("blank", "", 1.6))
    tiles += _with_rotations(piece("line", "NS", 1.0), 2)          # straight (2 rots)
    tiles += _with_rotations(piece("elbow", "NE", 1.0), 4)         # corner (4 rots)
    tiles += _with_rotations(piece("tee", "NES", 0.5), 4)          # T-junction (4 rots)
    tiles.append(piece("cross", "NESW", 0.3))
    return TileSet(tiles)


# ---------------------------------------------------------------------------
# tileset 2 — "terrain" (Wang tiles keyed on the four corners)
# ---------------------------------------------------------------------------

def make_terrain(size: int = 16) -> TileSet:
    """Every tile is defined by its four corner materials (land or water); the
    edge socket is the pair of corner values along that edge, so shared edges must
    agree on both corners. That is the classic Wang/blob scheme, and it produces
    coherent coastlines: islands and lakes with matching shores."""
    deep = np.array([40, 82, 170], np.float64)
    water = np.array([56, 110, 200], np.float64)
    sand = np.array([214, 203, 140], np.float64)
    grass = np.array([76, 156, 74], np.float64)
    hill = np.array([104, 170, 92], np.float64)

    def shade(h):  # map a land-fraction h in [0,1] to a terrain colour with shores
        if h < 0.30:
            return deep
        if h < 0.46:
            return water
        if h < 0.56:
            return sand
        if h < 0.78:
            return grass
        return hill

    def corner_tile(nw, ne, se, sw):
        # a per-pixel land fraction: bilinear interpolation of the four corner
        # values (0=water, 1=land), then binned into clean terrain bands so the
        # coastline is a crisp shore rather than a speckle.
        img = np.empty((size, size, 3), np.float64)
        for r in range(size):
            v = r / (size - 1)
            for cc in range(size):
                u = cc / (size - 1)
                top = nw * (1 - u) + ne * u
                bot = sw * (1 - u) + se * u
                h = top * (1 - v) + bot * v
                img[r, cc] = shade(h)
        n = (nw, ne)
        e = (ne, se)
        s = (sw, se)
        w = (nw, sw)
        name = f"{nw}{ne}{se}{sw}"
        weight = 6.0 if len({nw, ne, se, sw}) == 1 else 1.0   # favour solid land/water
        return Tile(name, (n, e, s, w), img.astype(np.uint8), weight,
                    meta={"corners": (nw, ne, se, sw)})

    tiles = [corner_tile(nw, ne, se, sw)
             for nw in (0, 1) for ne in (0, 1) for se in (0, 1) for sw in (0, 1)]
    return TileSet(tiles)


TILESETS = {
    "pipes": make_pipes,
    "terrain": make_terrain,
}
