"""Quaternions and the spherical interpolation a transformation rig turns on.

A transforming robot is a set of rigid parts, each swinging from a vehicle pose
to a robot pose. "Rigid" is the whole point: a part may rotate and translate but
never stretch, shear, or scale — the panel that was 40 cm wide is 40 cm wide in
every intermediate frame. Rotations are represented as unit quaternions and
blended with **SLERP** (spherical linear interpolation), which walks the shortest
great-circle arc between two orientations at constant angular speed. Crucially, a
SLERP of two *unit* quaternions is again a unit quaternion, so the rotation it
produces is always a genuine, proper rotation — the algebraic reason the parts
stay rigid, which the tests verify by checking RᵀR = I and det R = +1 throughout.

Quaternions are stored as (w, x, y, z) with w the scalar part.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic X→Y→Z Euler angles (radians) to a unit quaternion."""
    qx = from_axis_angle((1, 0, 0), rx)
    qy = from_axis_angle((0, 1, 0), ry)
    qz = from_axis_angle((0, 0, 1), rz)
    return normalize(multiply(multiply(qx, qy), qz))


def to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """The rotation angle (radians) taking orientation a to b, in [0, π]."""
    d = abs(float(np.dot(normalize(a), normalize(b))))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical interpolation between unit quaternions a and b."""
    a = normalize(a)
    b = normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:            # take the shorter of the two equivalent arcs
        b = -b
        dot = -dot
    if dot > 0.9995:        # nearly parallel: fall back to normalized lerp
        return normalize(a + t * (b - a))
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q_perp = normalize(b - a * dot)
    return a * np.cos(theta) + q_perp * np.sin(theta)


def to_euler_xyz(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion to intrinsic X→Y→Z Euler angles, for handing Blender a rotation."""
    m = to_matrix(q)
    sy = -m[2, 0]
    if abs(sy) < 0.99999:
        rx = np.arctan2(m[2, 1], m[2, 2])
        ry = np.arcsin(sy)
        rz = np.arctan2(m[1, 0], m[0, 0])
    else:                   # gimbal lock
        rx = np.arctan2(-m[1, 2], m[1, 1])
        ry = np.arcsin(sy)
        rz = 0.0
    return float(rx), float(ry), float(rz)
