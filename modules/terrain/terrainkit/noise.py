"""Coherent value noise and fractal Brownian motion — the raw shape of terrain.

A single octave of value noise is a lattice of random values, smoothly
interpolated: gentle, blobby hills with one characteristic size. Real landscapes
have detail at every scale, so **fractal Brownian motion** sums several octaves,
each at twice the frequency and about half the amplitude of the last. The result
has broad mountains, medium ridges, and fine roughness at once — and it is a
deterministic function of the seed, so the same seed always yields the same land.
"""

from __future__ import annotations

import numpy as np


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise(shape: tuple[int, int], cells: int, seed: int) -> np.ndarray:
    """One octave: a `cells × cells` lattice of random values, smoothly upsampled
    to `shape`, returned in [0, 1]."""
    rng = np.random.default_rng(seed)
    lattice = rng.random((cells + 1, cells + 1))

    ys = np.linspace(0, cells, shape[0], endpoint=False)
    xs = np.linspace(0, cells, shape[1], endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = _smoothstep(ys - y0)[:, None]
    fx = _smoothstep(xs - x0)[None, :]

    v00 = lattice[np.ix_(y0, x0)]
    v01 = lattice[np.ix_(y0, x0 + 1)]
    v10 = lattice[np.ix_(y0 + 1, x0)]
    v11 = lattice[np.ix_(y0 + 1, x0 + 1)]
    top = v00 * (1 - fx) + v01 * fx
    bot = v10 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


def fbm(shape: tuple[int, int], seed: int = 0, octaves: int = 6,
        base_cells: int = 4, lacunarity: float = 2.0, persistence: float = 0.5) -> np.ndarray:
    """Fractal Brownian motion: `octaves` of value noise summed at rising
    frequency and falling amplitude, normalised to [0, 1]."""
    total = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    cells = base_cells
    norm = 0.0
    for o in range(octaves):
        total += amplitude * value_noise(shape, int(round(cells)), seed + 1013 * o)
        norm += amplitude
        amplitude *= persistence
        cells *= lacunarity
    total /= norm
    return total


def heightmap(size: int = 256, seed: int = 0, octaves: int = 6, relief: float = 1.6) -> np.ndarray:
    """A terrain heightmap in [0, 1]. `relief` > 1 sharpens peaks and flattens
    valleys (a mild exponent), which reads as more mountainous."""
    h = fbm((size, size), seed=seed, octaves=octaves)
    h = (h - h.min()) / (h.max() - h.min() + 1e-12)
    return h ** relief
