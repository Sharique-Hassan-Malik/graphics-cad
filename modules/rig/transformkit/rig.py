"""Rigid parts, their two keyposes, and the staggered transformation between them.

Each part is a box with a fixed size (rigid — the size never changes) and two
poses: one that assembles the vehicle, one that assembles the robot. A pose is a
position and a unit-quaternion orientation. The transformation is a per-part
SLERP+lerp from vehicle to robot, but the parts do not move in lockstep — each has
its own time *window* within the overall morph, so the vehicle unfolds limb by
limb rather than inflating all at once. That staggering is what reads as
"transforming" instead of "scaling."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray            # unit quaternion (w, x, y, z)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = quat.normalize(np.asarray(self.orientation, dtype=np.float64))


@dataclass
class Part:
    name: str
    size: np.ndarray                   # box dimensions (fixed — the part is rigid)
    color: tuple
    vehicle: Pose
    robot: Pose
    t_start: float = 0.0               # when this part begins transforming (in [0,1])
    t_end: float = 1.0

    def progress(self, t: float) -> float:
        if self.t_end <= self.t_start:
            return 1.0 if t >= self.t_end else 0.0
        return _smoothstep((t - self.t_start) / (self.t_end - self.t_start))

    def pose_at(self, t: float) -> Pose:
        """The part's pose at global morph time t in [0,1] (0 = vehicle, 1 = robot)."""
        p = self.progress(t)
        pos = self.vehicle.position * (1 - p) + self.robot.position * p
        ori = quat.slerp(self.vehicle.orientation, self.robot.orientation, p)
        return Pose(pos, ori)

    def corners(self, pose: Pose) -> np.ndarray:
        """World positions of the box's 8 corners under a pose (for rigidity checks)."""
        h = self.size / 2.0
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        local = signs * h
        r = quat.to_matrix(pose.orientation)
        return (local @ r.T) + pose.position


@dataclass
class Rig:
    parts: list = field(default_factory=list)

    def pose_all(self, t: float) -> dict:
        return {p.name: p.pose_at(t) for p in self.parts}

    def add(self, *parts: Part) -> "Rig":
        self.parts.extend(parts)
        return self
