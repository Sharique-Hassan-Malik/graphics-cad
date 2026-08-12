"""terrainkit — fractal-noise heightmaps, hydraulic erosion, meshed 3D terrain."""

from .erosion import ErosionStats, erode
from .mesh import Mesh, terrain_mesh
from .noise import fbm, heightmap, value_noise

__all__ = ["heightmap", "fbm", "value_noise", "erode", "ErosionStats", "terrain_mesh", "Mesh"]
