"""A few models built from the SDF algebra, shared by the CLI and the showcase.
Each is one expression — a tree of primitives and boolean/blend operators — and
returns an SDF plus a suggested meshing resolution and the topology its shape
demands (Euler characteristic), so the showcase can check the mesh against it.
"""

from __future__ import annotations

from .sdf import Box, Cylinder, Sphere, Torus


def rounded_box(half=(1.0, 1.0, 1.0), radius=0.15):
    hx, hy, hz = half
    return Box((hx - radius, hy - radius, hz - radius)).round(radius)


def die(size=1.0, pip=0.14, inset=0.86):
    """A six-sided die: a rounded cube with the standard 1–6 pip pattern carved
    out as little spheres. Twenty-one indentations, one solid — a recognisable
    game object, and a good watertight-CSG stress test (χ stays 2)."""
    s = size
    body = rounded_box((s, s, s), radius=0.18 * s)
    a = 0.42 * s  # pip offset from centre on a face
    d = inset * s  # how far out the pip centres sit (so they cut the surface)

    faces = {
        1: [(0, 0)],
        2: [(-a, -a), (a, a)],
        3: [(-a, -a), (0, 0), (a, a)],
        4: [(-a, -a), (-a, a), (a, -a), (a, a)],
        5: [(-a, -a), (-a, a), (0, 0), (a, -a), (a, a)],
        6: [(-a, -a), (-a, 0), (-a, a), (a, -a), (a, 0), (a, a)],
    }
    # place each face's pips on one of the six cube faces (opposite faces sum to 7)
    placements = {
        1: ("z", +1), 6: ("z", -1),
        2: ("y", -1), 5: ("y", +1),
        3: ("x", +1), 4: ("x", -1),
    }
    solid = body
    for value, pips in faces.items():
        axis, sign = placements[value]
        for (u, v) in pips:
            if axis == "z":
                c = (u, v, sign * d)
            elif axis == "y":
                c = (u, sign * d, v)
            else:
                c = (sign * d, u, v)
            solid = solid - Sphere(pip * s, center=c)
    return solid, 96, 2


def blob():
    """Four spheres melted together with smooth-union — the classic metaball
    look that SDFs make trivial and meshes make hard."""
    b = Sphere(0.75, center=(-0.7, 0, 0))
    for c, r in [((0.7, 0.1, 0.1), 0.75), ((0.1, 0.7, -0.1), 0.6), ((0.15, -0.4, 0.5), 0.55)]:
        b = b.smooth_union(Sphere(r, center=c), k=0.45)
    return b, 72, 2


def bracket():
    """A mounting plate with four bolt holes: a flat box with four cylinders cut
    through it. Four through-holes make it a genus-4 surface (χ = −6)."""
    plate = Box((1.4, 1.0, 0.16))
    for sx in (-1, 1):
        for sy in (-1, 1):
            plate = plate - Cylinder(0.22, 1.0, center=(sx * 1.05, sy * 0.62, 0))
    return plate, 110, -6


def cross():
    """Three axis-aligned bars unioned into a plus/jack shape (χ = 2)."""
    bar = Box((1.2, 0.35, 0.35))
    return (bar | Box((0.35, 1.2, 0.35)) | Box((0.35, 0.35, 1.2))), 72, 2


def ring():
    """A torus — the canonical through-hole (χ = 0, genus 1)."""
    return Torus(1.0, 0.38), 72, 0


SCENES = {
    "die": die,
    "blob": blob,
    "bracket": bracket,
    "cross": cross,
    "ring": ring,
}
