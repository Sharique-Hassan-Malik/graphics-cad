"""wfckit — Wave Function Collapse: constraint-based procedural tilemaps."""

from .solver import Result, collapse
from .tiles import TILESETS, Tile, TileSet
from .verify import adjacency_violations, is_valid

__all__ = ["collapse", "Result", "TileSet", "Tile", "TILESETS", "is_valid", "adjacency_violations"]
