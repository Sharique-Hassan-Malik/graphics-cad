"""Tests for the SDF mesher.

The through-line: an extracted isosurface is correct only as *numbers*. It must be
watertight and consistently oriented (a real solid, not triangle soup); its Euler
characteristic must equal the one its shape demands (a ball 2, a torus 0, a plate
with n holes 2−2n); and its volume and surface area must converge to the analytic
values as the grid refines. Boolean operations must preserve all of that. Nothing
here trusts a render.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdfkit import scenes  # noqa: E402
from sdfkit.marching import triangulate  # noqa: E402
from sdfkit.mesh import Mesh  # noqa: E402
from sdfkit.sdf import Box, Cylinder, Sphere, Torus  # noqa: E402


# ---------------------------------------------------------------------------
# the extracted surface is a real solid
# ---------------------------------------------------------------------------


def test_sphere_is_a_watertight_oriented_manifold():
    m = triangulate(Sphere(1.0), resolution=32)
    assert m.is_watertight()
    assert m.is_edge_manifold()
    assert m.is_consistently_oriented()
    assert m.euler_characteristic() == 2      # a ball
    assert m.volume() > 0                      # outward-oriented


def test_box_volume_matches_analytic():
    m = triangulate(Box((0.7, 0.5, 0.9)), resolution=48)
    assert m.is_watertight()
    # A box's flat faces are captured essentially exactly by the tetrahedra.
    assert m.volume() == pytest.approx(1.4 * 1.0 * 1.8, rel=0.02)


def test_volume_and_area_converge_to_the_sphere():
    analytic_v = 4 / 3 * math.pi
    analytic_a = 4 * math.pi
    errs_v, errs_a = [], []
    for res in (16, 32, 64):
        m = triangulate(Sphere(1.0), resolution=res)
        errs_v.append(abs(m.volume() - analytic_v) / analytic_v)
        errs_a.append(abs(m.area() - analytic_a) / analytic_a)
    # monotone refinement, and within a fraction of a percent by 64³
    assert errs_v[0] > errs_v[1] > errs_v[2]
    assert errs_a[0] > errs_a[1] > errs_a[2]
    assert errs_v[-1] < 0.005 and errs_a[-1] < 0.005


# ---------------------------------------------------------------------------
# topology is a fingerprint of the shape
# ---------------------------------------------------------------------------


def test_torus_has_euler_zero_and_genus_one():
    m = triangulate(Torus(1.0, 0.35), resolution=48)
    assert m.is_watertight()
    assert m.euler_characteristic() == 0
    assert m.genus() == 1


def test_plate_with_four_holes_is_genus_four():
    plate = Box((1.4, 1.0, 0.16))
    for sx in (-1, 1):
        for sy in (-1, 1):
            plate = plate - Cylinder(0.22, 1.0, center=(sx * 1.05, sy * 0.62, 0))
    m = triangulate(plate, resolution=72)
    assert m.is_watertight()
    assert m.euler_characteristic() == -6      # 2 − 2·4
    assert m.genus() == 4


# ---------------------------------------------------------------------------
# CSG operations preserve solidity
# ---------------------------------------------------------------------------


def test_union_intersection_difference_are_watertight():
    a = Sphere(1.0, center=(-0.6, 0, 0))
    b = Sphere(1.0, center=(0.6, 0, 0))
    for solid in (a | b, a & b, Box((1, 1, 1)) - Sphere(0.7)):
        m = triangulate(solid, resolution=48)
        assert m.is_watertight()
        assert m.is_consistently_oriented()
        assert m.volume() > 0


def test_intersection_is_contained_in_each_operand():
    a = Sphere(1.0, center=(-0.4, 0, 0))
    b = Sphere(1.0, center=(0.4, 0, 0))
    inter = triangulate(a & b, resolution=48).volume()
    just_a = triangulate(a, resolution=48).volume()
    # the lens is smaller than either sphere and positive
    assert 0 < inter < just_a


def test_difference_removes_volume():
    box = triangulate(Box((1, 1, 1)), resolution=48).volume()
    holed = triangulate(Box((1, 1, 1)) - Sphere(0.6), resolution=48).volume()
    assert holed < box


# ---------------------------------------------------------------------------
# the built-in scenes match their declared topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(scenes.SCENES))
def test_scene_has_expected_topology(name):
    sdf, res, expected_chi = scenes.SCENES[name]()
    res = min(res, 64)  # keep the suite quick; topology is resolution-independent here
    m = triangulate(sdf, resolution=res)
    assert m.is_watertight(), name
    assert m.is_consistently_oriented(), name
    assert m.euler_characteristic() == expected_chi, name


# ---------------------------------------------------------------------------
# mesh mechanics
# ---------------------------------------------------------------------------


def test_welding_makes_soup_watertight():
    # Before welding, marching-tetra output is coincident-but-distinct vertices.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],   # tri 1
                      [1, 0, 0], [0, 1, 0], [1, 1, 0]])  # tri 2 shares an edge
    soup = Mesh(verts, [[0, 1, 2], [3, 4, 5]])
    assert soup.vertex_count == 6
    welded = soup.welded()
    assert welded.vertex_count == 4           # the shared edge's vertices merged


def test_stl_round_trip_size():
    m = triangulate(Sphere(1.0), resolution=24)
    data = m.to_stl_binary()
    assert len(data) == 84 + 50 * m.face_count


# ---------------------------------------------------------------------------
# Blender bridge
# ---------------------------------------------------------------------------


def test_bpy_script_emitted_without_blender():
    from sdfkit.blender_export import mesh_to_bpy

    m = triangulate(Sphere(1.0), resolution=16)
    script = mesh_to_bpy(m, name="s")
    assert "from_pydata" in script
    assert script.count("bpy.data") >= 2


def _blender():
    from sdfkit.blender_export import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_builds_blend_and_render(tmp_path):
    from sdfkit.blender_export import render

    m = triangulate(scenes.ring()[0], resolution=40)
    result = render(m, out_dir=str(tmp_path), name="ring", samples=8)
    assert result["ran"], result
    assert result["blend_written"] and os.path.getsize(result["blend"]) > 1000
    assert result["png_written"] and os.path.getsize(result["png"]) > 1000
