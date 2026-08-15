"""2D profiles: closed loops of points in the XY plane, ready to be extruded.

A loop is an ordered `(n, 2)` array of points with an implied closing edge from
the last back to the first. Counter-clockwise winding means the enclosed region
is "material"; the extruder relies on that convention to orient the solid.

The interesting one is the involute gear. Everything else is a helper for tests
and for the fillet/chamfer demos.
"""

from __future__ import annotations

import numpy as np


def circle(radius: float, segments: int = 64) -> np.ndarray:
    """A CCW circle. Used as a bore, and as the round cross-section of a cylinder."""
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def rectangle(width: float, height: float) -> np.ndarray:
    w, h = width / 2, height / 2
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]])


def rounded_rectangle(width: float, height: float, radius: float, segments: int = 8) -> np.ndarray:
    """A rectangle with rounded corners — the outline of most brackets and plates.

    Each corner is a quarter-arc, which is where fillet radius enters a real part:
    a sharp internal corner is a stress concentrator and a printer artifact, so
    CAD rounds them, and so does this.
    """
    if radius <= 0:
        return rectangle(width, height)
    radius = min(radius, width / 2, height / 2)
    w, h = width / 2 - radius, height / 2 - radius
    centers = [(w, h), (-w, h), (-w, -h), (w, -h)]
    starts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    points = []
    for (cx, cy), start in zip(centers, starts):
        arc = np.linspace(start, start + np.pi / 2, segments)
        points.append(np.column_stack([cx + radius * np.cos(arc), cy + radius * np.sin(arc)]))
    return np.vstack(points)


def regular_polygon(radius: float, sides: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / sides
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


# ---------------------------------------------------------------------------
# involute gear
# ---------------------------------------------------------------------------


def involute_point(base_radius: float, roll: float) -> tuple[float, float]:
    """One point on the involute of a circle, at parameter (roll angle) `roll`.

    The involute is the curve traced by the end of a taut string unwound from the
    base circle. Its parametric form is

        x = rb (cos t + t sin t)
        y = rb (sin t - t cos t)

    and it is exactly the shape a gear tooth flank must have for the fundamental
    law of gearing to hold: two involute teeth in contact transmit a *constant*
    angular-velocity ratio regardless of centre distance. That property — not
    aesthetics — is why real gears are involute, and why this is computed from
    the equation rather than approximated by an arc.
    """
    x = base_radius * (np.cos(roll) + roll * np.sin(roll))
    y = base_radius * (np.sin(roll) - roll * np.cos(roll))
    return x, y


def _inv(alpha: float) -> float:
    """The involute function inv(α) = tan(α) − α, the angular position along an
    involute of the point whose pressure angle is α."""
    return np.tan(alpha) - alpha


def involute_gear_profile(
    module: float,
    teeth: int,
    pressure_angle_deg: float = 20.0,
    flank_samples: int = 8,
) -> np.ndarray:
    """A closed CCW outline of an involute spur gear.

    Standard proportions (ISO): the pitch, base, addendum (outer) and dedendum
    (root) circles are

        pitch  r  = module * teeth / 2
        base   rb = r * cos(pressure_angle)
        outer  ra = r + module          (one module of addendum)
        root   rf = r - 1.25 * module   (1.25 modules of dedendum)

    Each tooth is two involute flanks (mirror images) joined by a tip arc, and
    neighbouring teeth are joined by a root arc. The whole outline is one loop of
    `teeth` such tooth-and-gap units.
    """
    if teeth < 4:
        raise ValueError("a gear needs at least 4 teeth")
    alpha = np.radians(pressure_angle_deg)
    r = module * teeth / 2.0
    rb = r * np.cos(alpha)
    ra = r + module
    rf = r - 1.25 * module

    # The flank is sampled by radius, from where it leaves the base circle
    # (or the root, if the root is outside the base circle) up to the tip.
    start_radius = max(rb, rf)
    radii = np.linspace(start_radius, ra, flank_samples)

    # Angular half-thickness of a tooth at the pitch circle is a quarter of the
    # angular pitch; `theta0` places the involute so the flank crosses the pitch
    # circle exactly there.
    half_tooth_angle = np.pi / (2 * teeth)
    theta0 = half_tooth_angle + _inv(alpha)

    def flank_angle(radius: float) -> float:
        # Pressure angle at this radius, then its involute angular position.
        ratio = np.clip(rb / radius, -1.0, 1.0)
        return theta0 - _inv(np.arccos(ratio))

    # Build one tooth traversed counter-clockwise — angle strictly increasing —
    # so the whole outline is a proper CCW star polygon (radius is a single-valued
    # function of angle). Order: up the LEFT flank (base→tip, angle rising from
    # −theta0), across the tip arc, down the RIGHT flank (tip→base, angle rising
    # to +theta0). Getting this order wrong traverses the tooth clockwise, which
    # subtracts it instead of adding it.
    tip_angle = flank_angle(ra)
    tooth_polar = []  # list of (radius, angle) with angle increasing
    if rf < rb:
        tooth_polar.append([rf, -theta0])                       # radial up from root, left side
    tooth_polar += [[radius, -flank_angle(radius)] for radius in radii]          # left flank base→tip
    tip_arc = np.linspace(-tip_angle, tip_angle, 4)[1:-1]
    tooth_polar += [[ra, a] for a in tip_arc]                    # tip arc left→right
    tooth_polar += [[radius, flank_angle(radius)] for radius in radii[::-1]]     # right flank tip→base
    if rf < rb:
        tooth_polar.append([rf, theta0])                        # radial down to root, right side

    tooth_polar = np.array(tooth_polar)

    # Replicate around the gear, joining consecutive teeth with a root arc that
    # also runs in increasing angle.
    loop = []
    step = 2 * np.pi / teeth
    for k in range(teeth):
        offset = k * step
        for radius, angle in tooth_polar:
            a = angle + offset
            loop.append([radius * np.cos(a), radius * np.sin(a)])
        # root arc from this tooth's right root to the next tooth's left root
        this_end = theta0 + offset
        next_start = -theta0 + offset + step
        for a in np.linspace(this_end, next_start, 3)[1:-1]:
            loop.append([rf * np.cos(a), rf * np.sin(a)])

    return np.array(loop)


def gear_parameters(module: float, teeth: int, pressure_angle_deg: float = 20.0) -> dict:
    """The analytic dimensions, so a generated gear can be measured against them."""
    alpha = np.radians(pressure_angle_deg)
    r = module * teeth / 2.0
    return {
        "pitch_radius": r,
        "base_radius": r * np.cos(alpha),
        "outer_radius": r + module,
        "root_radius": r - 1.25 * module,
        "circular_pitch": np.pi * module,
        "pressure_angle_deg": pressure_angle_deg,
    }
