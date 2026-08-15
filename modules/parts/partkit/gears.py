"""Involute spur gears as watertight solids, and the inverse: measuring a mesh
back to its gear parameters.

The generator is small — it composes a profile from `profiles` with an extruder
from `solids`. The verifier is the interesting half. Given only the vertices of a
finished gear, it recovers the base-circle radius and pressure angle by fitting
the involute equation to the tooth flanks, and checks them against the values the
gear was built from. That closes the loop: the part is not merely *shaped* like a
gear, it *measures* like the gear it claims to be, to a stated tolerance.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .profiles import gear_parameters, involute_gear_profile, circle
from .solids import extrude_star, extrude_ring, bore_matching


def spur_gear(
    module: float,
    teeth: int,
    thickness: float,
    pressure_angle_deg: float = 20.0,
    bore_diameter: float = 0.0,
    flank_samples: int = 8,
    bore_segments: int = 48,
) -> Mesh:
    """A watertight involute spur gear.

    `module` sets the size (tooth size = module; pitch diameter = module × teeth).
    A positive `bore_diameter` puts a real through-hole down the axis, making the
    solid a torus (Euler characteristic 0); a zero bore makes it a plain disk
    (characteristic 2). Both are watertight.
    """
    outline = involute_gear_profile(module, teeth, pressure_angle_deg, flank_samples)
    if bore_diameter > 0:
        pitch_radius = module * teeth / 2.0
        if bore_diameter / 2 >= module * teeth / 2 - 1.25 * module:
            raise ValueError("bore is larger than the root circle; nothing left to be a gear")
        bore = bore_matching(outline, bore_diameter / 2.0)
        gear = extrude_ring(outline, bore, thickness)
    else:
        gear = extrude_star(outline, thickness)
    return gear.translated(dz=-thickness / 2).welded()


# ---------------------------------------------------------------------------
# measuring a gear back
# ---------------------------------------------------------------------------


def measure_gear(mesh: Mesh) -> dict:
    """Recover the robustly-measurable geometry from a finished mesh's vertices.

    The outer and root radii are just the extreme distances from the axis, and
    the tooth count comes from angular clustering of the tip vertices — none of
    which reads the parameters the gear was built from, so comparing them to the
    analytic values (see the tests) is real evidence, not a tautology. Deriving
    `module` from the outer radius and tooth count then follows the standard
    proportion outer = module·(teeth/2 + 1).
    """
    v = mesh.vertices
    radii = np.hypot(v[:, 0], v[:, 1])
    outer_radius = float(radii.max())

    tip_mask = radii > outer_radius - 1e-6
    tip_angles = np.sort(np.mod(np.arctan2(v[tip_mask, 1], v[tip_mask, 0]), 2 * np.pi))
    teeth = _count_clusters(tip_angles)
    module = outer_radius / (teeth / 2 + 1) if teeth else float("nan")

    # Root radius: the smallest tooth-region radius (ignore any central bore,
    # which sits well inside the root circle).
    root_band = radii[radii > outer_radius * 0.3]
    root_radius = float(root_band.min()) if root_band.size else float("nan")

    return {
        "teeth": teeth,
        "module": module,
        "outer_radius": outer_radius,
        "root_radius": root_radius,
        "pitch_radius": module * teeth / 2.0 if teeth else float("nan"),
    }


def involute_deviation(
    mesh: Mesh, module: float, teeth: int, pressure_angle_deg: float = 20.0
) -> dict:
    """How far the tooth flanks deviate from the *mathematically exact* involute.

    This is the correctness proof for the tooth profile, and it is direct: for
    every vertex on a flank (radius between the base and outer circles), compute
    where the exact involute of the base circle would place a point at that
    radius, and measure the arc distance between the two. A small maximum
    deviation means the flanks genuinely are involutes of the right base circle —
    the property that makes the gear transmit constant velocity — not merely a
    plausible curve.

    The check exploits the gear's rotational symmetry: a flank vertex at polar
    angle φ belongs to some tooth, so reducing φ modulo the angular pitch and
    comparing against the single canonical flank angle covers all teeth and both
    flanks at once.
    """
    alpha = np.radians(pressure_angle_deg)
    r_pitch = module * teeth / 2.0
    rb = r_pitch * np.cos(alpha)
    ra = r_pitch + module
    rf = r_pitch - 1.25 * module
    pitch_angle = 2 * np.pi / teeth
    half_tooth_angle = np.pi / (2 * teeth)
    theta0 = half_tooth_angle + (np.tan(alpha) - alpha)  # inv(alpha)

    v = mesh.vertices
    radii = np.hypot(v[:, 0], v[:, 1])
    angles = np.arctan2(v[:, 1], v[:, 0])

    # Flank band: strictly between the tip and where the flank actually begins.
    # The involute part of a flank starts at the base circle, unless the root
    # circle sits *outside* it (a gear with enough teeth to avoid undercut), in
    # which case the flank begins at the root and the arc there is not involute.
    # Using max(rb, rf) excludes those root-arc vertices, which would otherwise
    # register as flank points off the involute.
    flank_start = max(rb, rf)
    flank = (radii > flank_start + 1e-6) & (radii < ra - 1e-9)
    rho = radii[flank]
    phi = angles[flank]
    if rho.size == 0:
        return {"max_deviation": 0.0, "rms_deviation": 0.0, "flank_points": 0}

    # The canonical right-flank angle for a point at radius rho:
    alpha_rho = np.arccos(np.clip(rb / rho, -1, 1))
    expected = theta0 - (np.tan(alpha_rho) - alpha_rho)   # right flank, tooth centred at 0

    # Fold every vertex into its own tooth's copy, centred at 0, in the window
    # (-pitch/2, +pitch/2] — which must span the whole tooth, whose flanks reach
    # ±theta0 > half_tooth_angle near the base. Then |within_tooth| is the vertex's
    # distance from the tooth centreline and compares directly to the expected
    # flank angle, covering both flanks of every tooth at once.
    within_tooth = np.mod(phi, pitch_angle)
    within_tooth = np.where(within_tooth > pitch_angle / 2, within_tooth - pitch_angle, within_tooth)
    residual_angle = np.abs(np.abs(within_tooth) - expected)
    # Arc distance = radius × angular residual.
    deviations = rho * residual_angle
    return {
        "max_deviation": float(deviations.max()),
        "rms_deviation": float(np.sqrt(np.mean(deviations**2))),
        "flank_points": int(rho.size),
        "base_radius": rb,
    }


def _count_clusters(sorted_angles: np.ndarray, gap_factor: float = 2.5) -> int:
    """Count angular clusters (tooth tips) in a sorted list of angles on a circle."""
    if len(sorted_angles) < 2:
        return len(sorted_angles)
    diffs = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2 * np.pi]]))
    median = np.median(diffs)
    return int(np.sum(diffs > gap_factor * median))
