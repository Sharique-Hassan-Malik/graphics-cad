"""Render a heightmap as a top-down hypsometric map: elevation → biome colour,
lit by hillshading so ridges and valleys read at a glance. Pure NumPy.

The colour ramp is the familiar cartographic one — deep water, shallows, sand,
grass, forest, rock, snow. Hillshading multiplies each pixel by the Lambert
shading of the surface normal against a low sun, which is what makes an otherwise
flat colour map look three-dimensional and reveals the drainage that erosion
carved.
"""

from __future__ import annotations

import numpy as np

from .image import write_png

_STOPS = [
    (0.00, (36, 64, 128)),    # deep water
    (0.30, (54, 100, 180)),   # water
    (0.37, (60, 120, 200)),   # shallows
    (0.40, (214, 203, 140)),  # sand
    (0.46, (86, 158, 80)),    # grass
    (0.66, (54, 118, 58)),    # forest
    (0.80, (120, 110, 96)),   # rock
    (0.90, (150, 140, 128)),  # scree
    (0.97, (245, 245, 250)),  # snow
]


def _ramp(h: np.ndarray) -> np.ndarray:
    stops = _STOPS
    out = np.zeros((*h.shape, 3), dtype=np.float64)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        mask = (h >= t0) & (h < t1)
        f = ((h[mask] - t0) / (t1 - t0))[:, None]
        out[mask] = np.array(c0) * (1 - f) + np.array(c1) * f
    out[h >= stops[-1][0]] = stops[-1][1]
    out[h < stops[0][0]] = stops[0][1]
    return out


def hillshade(h: np.ndarray, scale: float = 3.0, light=(-1.0, -1.0, 1.6)) -> np.ndarray:
    """Lambert shading of the surface normal against a light direction, in [0,1]."""
    gy, gx = np.gradient(h * scale)
    nz = np.ones_like(h)
    norm = np.stack([-gx, -gy, nz], axis=-1)
    norm /= np.linalg.norm(norm, axis=-1, keepdims=True)
    light = np.array(light, dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = np.clip(norm @ light, 0, 1)
    return 0.55 + 0.45 * shade      # keep some ambient so shadows aren't black


def shade(heightmap: np.ndarray, relief_scale: float = 3.0) -> np.ndarray:
    """Colour + hillshade a heightmap into an (H, W, 3) uint8 image."""
    rgb = _ramp(heightmap)
    lit = hillshade(heightmap, scale=relief_scale)[:, :, None]
    # do not shade the flat water surface (it should read as water, not lit rock)
    water = (heightmap < 0.38)[:, :, None]
    rgb = np.where(water, rgb, rgb * lit)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_map(heightmap: np.ndarray, path: str, scale: int = 1, relief_scale: float = 3.0) -> str:
    img = shade(heightmap, relief_scale)
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return write_png(path, img)
