"""Turn a solved grid of tile indices into an image by stamping each tile's
bitmap into place. Because adjacent tiles have matching sockets — and the sockets
encode the border pixels — the stamped result is seamless."""

from __future__ import annotations

import numpy as np

from .image import write_png
from .tiles import TileSet


def to_image(grid: np.ndarray, tileset: TileSet, scale: int = 1) -> np.ndarray:
    ts = tileset.tile_size()
    height, width = grid.shape
    out = np.empty((height * ts, width * ts, 3), np.uint8)
    for r in range(height):
        for c in range(width):
            out[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = tileset.tiles[grid[r, c]].pixels
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)
    return out


def save(grid: np.ndarray, tileset: TileSet, path: str, scale: int = 1) -> str:
    return write_png(path, to_image(grid, tileset, scale))
