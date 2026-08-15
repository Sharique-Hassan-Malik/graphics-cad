"""The slider-crank mechanism — the exact kinematics that turn a spinning
crankshaft into a reciprocating piston.

A crank of radius r spins at angle θ; a connecting rod of length l joins the
crank pin to the piston, which is constrained to a straight line (the cylinder
bore). The piston's distance from the crank axis is

    x(θ) = r·cos θ + √(l² − r²·sin²θ)

The first term is the crank throw projected onto the bore; the second is the rod
foreshortened by its swing. Top dead centre (piston highest) is θ = 0 with
x = r + l; bottom dead centre is θ = π with x = l − r; so the stroke is exactly
2r. Everything an engine animation needs — piston position, the rod's swing
angle, and the piston's velocity and acceleration — comes from this one relation
and its derivatives, all in closed form and all tested against the geometry they
claim to describe.
"""

from __future__ import annotations

import numpy as np


def piston_position(theta, crank_radius: float, rod_length: float):
    """Distance from the crank axis to the piston pin, along the bore."""
    theta = np.asarray(theta, dtype=np.float64)
    r, l = crank_radius, rod_length
    return r * np.cos(theta) + np.sqrt(l * l - (r * np.sin(theta)) ** 2)


def piston_displacement(theta, crank_radius: float, rod_length: float):
    """Piston position measured *down* from top dead centre, in [0, stroke]."""
    r, l = crank_radius, rod_length
    return (r + l) - piston_position(theta, crank_radius, rod_length)


def rod_angle(theta, crank_radius: float, rod_length: float):
    """Angle of the connecting rod from the bore axis. Bounded by ±asin(r/l)."""
    theta = np.asarray(theta, dtype=np.float64)
    r, l = crank_radius, rod_length
    return np.arcsin(r * np.sin(theta) / l)


def piston_velocity(theta, crank_radius: float, rod_length: float, omega: float = 1.0):
    """dx/dt for a crank turning at angular velocity `omega` (rad/s)."""
    theta = np.asarray(theta, dtype=np.float64)
    r, l = crank_radius, rod_length
    s = np.sin(theta)
    root = np.sqrt(l * l - (r * s) ** 2)
    dxdtheta = -r * s - (r * r * s * np.cos(theta)) / root
    return dxdtheta * omega


def piston_acceleration(theta, crank_radius: float, rod_length: float, omega: float = 1.0):
    """d²x/dt² at constant `omega`, by central differences of the analytic velocity."""
    theta = np.asarray(theta, dtype=np.float64)
    h = 1e-6
    v1 = piston_velocity(theta + h, crank_radius, rod_length, omega)
    v0 = piston_velocity(theta - h, crank_radius, rod_length, omega)
    return (v1 - v0) / (2 * h) * omega


def crank_pin(theta, crank_radius: float):
    """The crank pin's (bore-axis, perpendicular) coordinates — where the rod's
    big end sits. Bore axis is x; the crank pin swings in the x–y plane."""
    theta = np.asarray(theta, dtype=np.float64)
    return crank_radius * np.cos(theta), crank_radius * np.sin(theta)


def stroke(crank_radius: float) -> float:
    return 2.0 * crank_radius
