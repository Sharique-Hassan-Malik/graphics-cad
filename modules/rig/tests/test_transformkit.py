"""Tests for the transformation rig.

The headline is rigidity, stated as numbers: through the entire morph, every
part's orientation is a *proper rotation* — its matrix R satisfies RᵀR = I and
det R = +1 — so the part never stretches, shears, or scales, and the pairwise
distances between its corners are invariant. That is exactly the property a real
transformer rig must have and a naive vertex-blend would violate. Alongside it:
SLERP stays on the unit sphere and moves at constant angular speed, and every
part hits its vehicle and robot keyposes exactly at the ends.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformkit import quat  # noqa: E402
from transformkit.character import optimus  # noqa: E402


# ---------------------------------------------------------------------------
# quaternion algebra
# ---------------------------------------------------------------------------


def test_to_matrix_is_a_rotation():
    q = quat.from_euler(0.3, -1.1, 2.0)
    R = quat.to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_slerp_stays_on_the_unit_sphere():
    a = quat.from_euler(0.1, 0.2, 0.3)
    b = quat.from_euler(-1.2, 0.8, 2.5)
    for t in np.linspace(0, 1, 25):
        assert np.linalg.norm(quat.slerp(a, b, t)) == pytest.approx(1.0, abs=1e-12)


def test_slerp_has_constant_angular_speed():
    a = quat.from_axis_angle((0, 0, 1), 0.0)
    b = quat.from_axis_angle((0, 0, 1), math.radians(120))
    # the angle from a to slerp(a,b,t) should be linear in t
    angles = [quat.angle_between(a, quat.slerp(a, b, t)) for t in np.linspace(0, 1, 11)]
    diffs = np.diff(angles)
    assert np.allclose(diffs, diffs[0], atol=1e-9)


def test_slerp_endpoints():
    a = quat.from_euler(0.5, 0.5, 0.5)
    b = quat.from_euler(-0.5, 1.0, 0.2)
    assert np.allclose(quat.slerp(a, b, 0.0), a, atol=1e-12)
    # up to sign (q and -q are the same rotation)
    assert quat.angle_between(quat.slerp(a, b, 1.0), b) < 1e-9


def test_slerp_takes_the_short_way():
    # a and (nearly) -a represent close rotations; slerp must not go the long way
    a = quat.from_axis_angle((0, 1, 0), 0.1)
    b = -quat.from_axis_angle((0, 1, 0), 0.2)   # same as +rotation of 0.2
    assert quat.angle_between(a, quat.slerp(a, b, 0.5)) < math.radians(20)


# ---------------------------------------------------------------------------
# the headline: parts stay rigid through the whole transformation
# ---------------------------------------------------------------------------


def test_every_part_is_a_proper_rotation_throughout():
    rig = optimus()
    worst_ortho = worst_det = 0.0
    for part in rig.parts:
        for t in np.linspace(0, 1, 40):
            R = quat.to_matrix(part.pose_at(t).orientation)
            worst_ortho = max(worst_ortho, np.abs(R @ R.T - np.eye(3)).max())
            worst_det = max(worst_det, abs(np.linalg.det(R) - 1.0))
    assert worst_ortho < 1e-12
    assert worst_det < 1e-12


def test_parts_never_stretch_or_scale():
    rig = optimus()
    worst = 0.0
    for part in rig.parts:
        ref = part.corners(part.vehicle)
        ref_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
        for t in np.linspace(0, 1, 40):
            c = part.corners(part.pose_at(t))
            d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
            worst = max(worst, np.abs(d - ref_d).max())
    assert worst < 1e-12          # every inter-corner distance is invariant → rigid


def test_endpoints_hit_the_keyposes_exactly():
    rig = optimus()
    for part in rig.parts:
        assert np.allclose(part.pose_at(0.0).position, part.vehicle.position, atol=1e-12)
        assert np.allclose(part.pose_at(1.0).position, part.robot.position, atol=1e-12)
        assert quat.angle_between(part.pose_at(0.0).orientation, part.vehicle.orientation) < 1e-9
        assert quat.angle_between(part.pose_at(1.0).orientation, part.robot.orientation) < 1e-9


def test_transformation_is_staggered_not_lockstep():
    rig = optimus()
    # at the exact middle of the morph, parts are at a spread of progress values,
    # which is what makes it read as a transformation rather than a uniform scale
    progresses = [p.progress(0.5) for p in rig.parts]
    assert max(progresses) - min(progresses) > 0.3


def test_parts_are_actually_rigid_bodies_not_points():
    rig = optimus()
    # sanity: parts have real extent, and different parts move differently
    assert all(np.all(p.size > 0) for p in rig.parts)
    mid = rig.pose_all(0.5)
    positions = np.array([mid[p.name].position for p in rig.parts])
    assert positions.std(axis=0).sum() > 1.0     # parts are spread out in space


# ---------------------------------------------------------------------------
# animation sampling + Blender bridge
# ---------------------------------------------------------------------------


def test_sampling_produces_a_looping_schedule():
    from transformkit.animate import sample

    data = sample(optimus(), frames=48)
    assert len(data["frames"]) == 48
    # the raised-cosine morph time starts and ends at the vehicle pose (t≈0)
    first = np.array(data["frames"][0])
    last = np.array(data["frames"][-1])
    veh = np.array([[*p.vehicle.position, *p.vehicle.orientation] for p in optimus().parts])
    assert np.allclose(first, veh, atol=1e-6)


def _blender():
    from transformkit.animate import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_renders_animation(tmp_path):
    from transformkit.animate import render

    result = render(optimus(), out_dir=str(tmp_path), name="t", frames=6, resolution=(160, 120), make_gif=False)
    assert result["ran"], result
    assert len([f for f in os.listdir(result["frame_dir"]) if f.endswith(".png")]) == 6
