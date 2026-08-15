"""A few higher-level parametric parts, composed from the primitives.

These are thin: each is a profile plus an extrude. They exist to show the kit
covers ordinary mechanical shapes, not only gears, and to exercise the general
ear-clipping path on a genuinely non-convex outline (the L-bracket has a reflex
corner, which is where a naive triangulator fails).
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .profiles import rounded_rectangle
from .solids import bore_matching, extrude, extrude_ring


def plate(width: float, depth: float, thickness: float, corner_radius: float = 0.0) -> Mesh:
    """A rectangular plate, optionally with rounded corners."""
    profile = rounded_rectangle(width, depth, corner_radius)
    return extrude(profile, thickness).translated(dz=-thickness / 2)


def washer(outer_diameter: float, inner_diameter: float, thickness: float, segments: int = 64) -> Mesh:
    """A flat annulus — the bored-ring case, watertight with a genuine hole."""
    from .profiles import circle

    if inner_diameter >= outer_diameter:
        raise ValueError("washer bore must be smaller than its outer diameter")
    outer = circle(outer_diameter / 2, segments)
    return extrude_ring(outer, bore_matching(outer, inner_diameter / 2), thickness).translated(dz=-thickness / 2)


def l_bracket(length: float, height: float, thickness: float, wall: float) -> Mesh:
    """An L-shaped bracket, extruded from a reflex (non-convex) profile.

    The L profile has exactly one reflex vertex — the inside corner — which is
    the case ear clipping exists to handle and a fan-from-centre triangulation
    would get wrong. So this doubles as the test that the general extruder is
    correct, not just the star-shaped shortcut.
    """
    w = wall
    profile = np.array([
        [0, 0],
        [length, 0],
        [length, w],
        [w, w],          # the reflex corner
        [w, height],
        [0, height],
    ], dtype=np.float64)
    return extrude(profile, thickness)
