"""Droplet-based hydraulic erosion — the step that turns noise into landscape.

Fractal noise gives plausible elevation but no *history*: no valleys carved by
water, no sediment fans where rivers slow. This simulates many rain droplets.
Each droplet follows the downhill gradient, picks up soil where the flow is fast
and steep (its carrying capacity exceeds its load), and drops soil where the flow
slows or pools. Run enough droplets and drainage networks emerge — ridgelines
sharpen, valleys deepen, plains silt up.

The invariant that makes it *simulation* rather than decoration is **conservation
of mass**: a droplet never creates or destroys soil, only moves it. Everything a
droplet erodes it carries, and everything it carries it eventually deposits —
during travel, on evaporating, or (crucially) before it leaves the map. So the
total elevation summed over the grid is unchanged by erosion, which the tests and
showcase check to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErosionStats:
    droplets: int
    eroded: float           # total soil lifted from the terrain
    deposited: float        # total soil laid back down
    height_sum_before: float
    height_sum_after: float

    @property
    def mass_error(self) -> float:
        """Relative change in total elevation — zero if mass was conserved."""
        return abs(self.height_sum_after - self.height_sum_before) / (self.height_sum_before + 1e-12)


def _height_and_gradient(h, x, y):
    """Bilinear height and its gradient at a floating point (x, y)."""
    ix, iy = int(x), int(y)
    fx, fy = x - ix, y - iy
    h00 = h[iy, ix]
    h10 = h[iy, ix + 1]
    h01 = h[iy + 1, ix]
    h11 = h[iy + 1, ix + 1]
    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
    gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    height = (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy
    return height, gx, gy


def erode(heightmap: np.ndarray, droplets: int = 20000, seed: int = 0,
          inertia: float = 0.05, capacity: float = 4.0, deposition: float = 0.3,
          erosion: float = 0.3, evaporation: float = 0.02, gravity: float = 4.0,
          lifetime: int = 30, min_slope: float = 0.01) -> tuple[np.ndarray, ErosionStats]:
    """Return an eroded copy of `heightmap` and the mass-conservation statistics.

    Deposition and erosion are spread bilinearly over the four cells under the
    droplet, so the terrain stays smooth. A droplet deposits any remaining load
    when it stops, evaporates, or reaches the border — which is what makes the
    process exactly mass-conserving.
    """
    h = heightmap.astype(np.float64).copy()
    size = h.shape[0]
    rng = np.random.default_rng(seed)
    total_eroded = 0.0
    total_deposited = 0.0
    before = float(h.sum())

    def deposit(x, y, amount):
        nonlocal total_deposited
        ix, iy = int(x), int(y)
        fx, fy = x - ix, y - iy
        h[iy, ix] += amount * (1 - fx) * (1 - fy)
        h[iy, ix + 1] += amount * fx * (1 - fy)
        h[iy + 1, ix] += amount * (1 - fx) * fy
        h[iy + 1, ix + 1] += amount * fx * fy
        total_deposited += amount

    def take(x, y, amount):
        nonlocal total_eroded
        ix, iy = int(x), int(y)
        fx, fy = x - ix, y - iy
        h[iy, ix] -= amount * (1 - fx) * (1 - fy)
        h[iy, ix + 1] -= amount * fx * (1 - fy)
        h[iy + 1, ix] -= amount * (1 - fx) * fy
        h[iy + 1, ix + 1] -= amount * fx * fy
        total_eroded += amount

    for _ in range(droplets):
        x = rng.uniform(0, size - 1)
        y = rng.uniform(0, size - 1)
        dx = dy = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for _step in range(lifetime):
            height, gx, gy = _height_and_gradient(h, x, y)
            # steer downhill, blended with momentum
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            length = np.hypot(dx, dy)
            if length < 1e-8:
                break
            dx, dy = dx / length, dy / length
            nx, ny = x + dx, y + dy

            if not (0 <= nx < size - 1 and 0 <= ny < size - 1):
                deposit(x, y, sediment)     # drop the load before leaving the map
                sediment = 0.0
                break

            new_height, _, _ = _height_and_gradient(h, nx, ny)
            dh = new_height - height

            if dh >= 0:
                # flowing uphill / into a pit: fill it, up to what we carry
                drop = min(sediment, dh + 1e-12)
                deposit(x, y, drop)
                sediment -= drop
            else:
                cap = max(-dh, min_slope) * speed * water * capacity
                if sediment > cap:
                    drop = (sediment - cap) * deposition
                    deposit(x, y, drop)
                    sediment -= drop
                else:
                    grab = min((cap - sediment) * erosion, -dh)
                    take(x, y, grab)
                    sediment += grab

            speed = np.sqrt(max(speed * speed + dh * gravity, 0.0))
            water *= (1 - evaporation)
            x, y = nx, ny
        else:
            deposit(x, y, sediment)         # lifetime ended: drop the load
            sediment = 0.0

        if sediment > 0:                    # any leftover (e.g. stalled droplet)
            deposit(x, y, sediment)

    stats = ErosionStats(droplets, total_eroded, total_deposited, before, float(h.sum()))
    return h, stats
