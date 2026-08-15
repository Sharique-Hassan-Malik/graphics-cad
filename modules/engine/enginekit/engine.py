"""A multi-cylinder four-stroke engine: the geometry and the timing that a
working-engine animation is driven by.

Each cylinder is a slider-crank (see `slider_crank`) whose crank throw is offset
around the shaft. Over the four-stroke cycle — two crank revolutions, 720° —
every cylinder does intake, compression, power, exhaust in turn, and the *firing
order* schedules the power strokes so the engine runs smoothly: on an N-cylinder
engine they land exactly 720/N degrees apart. The class exposes, for any crank
angle, every piston's position, every rod's swing, the crank-pin locations, the
intake/exhaust valve lift from a cam lobe, and which cylinder is firing — the
full state the Blender rig keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import slider_crank as sc


@dataclass
class Engine:
    n_cylinders: int
    crank_radius: float = 0.5       # stroke = 2·r
    rod_length: float = 1.6
    bore: float = 0.9
    firing_order: list = field(default_factory=list)   # 1-based cylinder numbers
    throw_offsets: list = field(default_factory=list)  # crank-throw angle per cyl, degrees
    layout: str = "inline"          # "inline" or "vee"
    bank_angle: float = 90.0        # degrees, for vee layouts

    def __post_init__(self):
        if not self.firing_order:
            self.firing_order = list(range(1, self.n_cylinders + 1))
        if not self.throw_offsets:
            # Derive the crank throws from the firing order: a cylinder must be at
            # top dead centre when it fires, so its throw offset is the negative of
            # its firing angle. This reproduces the classic cranks exactly (an
            # inline-4 comes out 0-180-180-0) and is correct by construction.
            fire = self._firing_angles_from_order()
            self.throw_offsets = [(-fire[c]) % 360.0 for c in range(1, self.n_cylinders + 1)]
        self.throw_offsets = [float(a) for a in self.throw_offsets]

    def _firing_angles_from_order(self) -> dict[int, float]:
        interval = 720.0 / self.n_cylinders
        return {cyl: (k * interval) % 720.0 for k, cyl in enumerate(self.firing_order)}

    # -- presets -------------------------------------------------------------

    @classmethod
    def inline4(cls):
        return cls(4, firing_order=[1, 3, 4, 2])            # throws derived: 0-180-180-0

    @classmethod
    def inline6(cls):
        return cls(6, firing_order=[1, 5, 3, 6, 2, 4])      # throws derived: 0-240-120-120-240-0

    @classmethod
    def v8(cls):
        return cls(8, firing_order=[1, 8, 4, 3, 6, 5, 7, 2],
                   layout="vee", bank_angle=90.0)           # cross-plane throws 0/90/180/270

    # -- derived timing ------------------------------------------------------

    @property
    def stroke(self) -> float:
        return sc.stroke(self.crank_radius)

    @property
    def firing_interval(self) -> float:
        """Degrees of crank rotation between successive power strokes."""
        return 720.0 / self.n_cylinders

    def firing_angles(self) -> dict[int, float]:
        """Crank angle (deg, in [0,720)) at which each cylinder's power stroke
        begins, assigned by the firing order at even 720/N spacing."""
        return self._firing_angles_from_order()

    def bank_of(self, cyl_index: int) -> int:
        """Which bank a cylinder is on (0/1); inline engines are all bank 0."""
        if self.layout != "vee":
            return 0
        return cyl_index % 2

    # -- kinematic state at a crank angle -----------------------------------

    def piston_displacements(self, crank_deg) -> np.ndarray:
        """Each cylinder's piston displacement from TDC (0..stroke)."""
        theta = np.radians(crank_deg)
        offs = np.radians(self.throw_offsets)
        return sc.piston_displacement(theta + offs, self.crank_radius, self.rod_length)

    def rod_angles(self, crank_deg) -> np.ndarray:
        theta = np.radians(crank_deg)
        offs = np.radians(self.throw_offsets)
        return sc.rod_angle(theta + offs, self.crank_radius, self.rod_length)

    def crank_pins(self, crank_deg):
        theta = np.radians(crank_deg)
        offs = np.radians(self.throw_offsets)
        return sc.crank_pin(theta + offs, self.crank_radius)

    def valve_lift(self, crank_deg) -> np.ndarray:
        """(N, 2) array of (intake, exhaust) valve lift in [0,1] per cylinder.

        A cam lobe opens the exhaust valve through the exhaust up-stroke and the
        intake valve through the intake down-stroke, timed off each cylinder's own
        power-stroke start."""
        fire = self.firing_angles()
        out = np.zeros((self.n_cylinders, 2))
        for i in range(self.n_cylinders):
            rel = (crank_deg - fire[i + 1]) % 720.0
            out[i, 1] = _lobe(rel, center=270.0, duration=250.0)   # exhaust
            out[i, 0] = _lobe(rel, center=450.0, duration=250.0)   # intake
        return out

    def firing_cylinder(self, crank_deg, window: float = 25.0):
        """The cylinder whose power stroke just began (for a combustion flash), or
        None if no cylinder is within `window` degrees after its firing angle."""
        for cyl, angle in self.firing_angles().items():
            if 0.0 <= (crank_deg - angle) % 720.0 < window:
                return cyl
        return None


def _lobe(phase, center, duration):
    """A smooth cam lobe: raised cosine of unit height over `duration` degrees."""
    d = abs(((phase - center + 360.0) % 720.0) - 360.0)
    if d > duration / 2:
        return 0.0
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * d / duration))
