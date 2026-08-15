"""A truck ↔ robot transformer, built as rigid parts with two keyposes each.

This is an original, stylised homage — boxes, not a screen-accurate model. Each
part carries the box size it keeps forever (rigidity), a vehicle pose and a robot
pose, and a time window controlling *when* in the morph it swings across. The
windows are staggered so the legs deploy first, then the arms unfold, then the
head rises last — the beats that make a transformation read.

Coordinates: +x is the robot's right, +y is forward (vehicle length), +z is up.
"""

from __future__ import annotations

import numpy as np

from . import quat
from .rig import Part, Pose, Rig

# colours — a red/blue/silver homage
RED = (0.78, 0.12, 0.12)
BLUE = (0.13, 0.24, 0.70)
SILVER = (0.62, 0.64, 0.68)
STEEL = (0.30, 0.34, 0.52)
DARK = (0.12, 0.12, 0.14)


def _pose(pos, deg=(0, 0, 0)) -> Pose:
    return Pose(np.array(pos, float), quat.from_euler(*np.radians(deg)))


def _part(name, size, color, veh, rob, window):
    (vp, vd), (rp, rd) = veh, rob
    return Part(name, np.array(size, float), color, _pose(vp, vd), _pose(rp, rd), *window)


def optimus() -> Rig:
    parts = [
        # name        size              colour  vehicle (pos, deg)          robot (pos, deg)            window
        _part("head", (1.0, 1.1, 1.0), RED,
              ((0, 2.4, 0.95), (0, 0, 0)), ((0, 0.1, 6.0), (0, 0, 0)), (0.55, 0.85)),
        _part("torso", (2.6, 1.5, 2.2), RED,
              ((0, 0.6, 1.35), (0, 0, 0)), ((0, 0, 4.5), (0, 0, 0)), (0.0, 0.5)),
        _part("pelvis", (2.1, 1.4, 1.0), SILVER,
              ((0, -1.9, 1.2), (0, 0, 0)), ((0, 0, 3.0), (0, 0, 0)), (0.05, 0.55)),

        _part("uarm_L", (0.75, 0.75, 1.7), BLUE,
              ((1.45, 0.7, 1.5), (90, 0, 0)), ((1.75, 0, 4.6), (0, 0, 0)), (0.30, 0.70)),
        _part("uarm_R", (0.75, 0.75, 1.7), BLUE,
              ((-1.45, 0.7, 1.5), (90, 0, 0)), ((-1.75, 0, 4.6), (0, 0, 0)), (0.30, 0.70)),
        _part("farm_L", (0.62, 0.62, 1.6), SILVER,
              ((1.45, -0.7, 1.5), (90, 0, 0)), ((1.75, 0.05, 3.0), (0, 0, 0)), (0.45, 0.85)),
        _part("farm_R", (0.62, 0.62, 1.6), SILVER,
              ((-1.45, -0.7, 1.5), (90, 0, 0)), ((-1.75, 0.05, 3.0), (0, 0, 0)), (0.45, 0.85)),

        _part("thigh_L", (0.85, 0.85, 1.7), BLUE,
              ((1.15, 1.6, 0.55), (0, 90, 0)), ((0.72, 0, 2.0), (0, 0, 0)), (0.10, 0.60)),
        _part("thigh_R", (0.85, 0.85, 1.7), BLUE,
              ((-1.15, 1.6, 0.55), (0, 90, 0)), ((-0.72, 0, 2.0), (0, 0, 0)), (0.10, 0.60)),
        _part("shin_L", (0.72, 0.72, 1.6), STEEL,
              ((1.15, -1.6, 0.55), (0, 90, 0)), ((0.72, 0.05, 0.5), (0, 0, 0)), (0.20, 0.72)),
        _part("shin_R", (0.72, 0.72, 1.6), STEEL,
              ((-1.15, -1.6, 0.55), (0, 90, 0)), ((-0.72, 0.05, 0.5), (0, 0, 0)), (0.20, 0.72)),
    ]
    return Rig(parts)
