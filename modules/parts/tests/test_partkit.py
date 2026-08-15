"""Tests for the CAD kernel.

The through-line is that "manufacturable" is a set of measurable properties, not
an opinion: watertightness, orientation, the right topology, a volume that
matches the analytic answer, and — for gears — flanks that lie on the exact
involute. Every test asserts one of those against a number, so a regression in
the geometry cannot hide behind a plausible-looking render.
"""

from __future__ import annotations

import math
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partkit import parts, profiles, solids  # noqa: E402
from partkit.gears import involute_deviation, measure_gear, spur_gear  # noqa: E402
from partkit.mesh import Mesh  # noqa: E402


# ---------------------------------------------------------------------------
# mesh topology
# ---------------------------------------------------------------------------


def test_box_is_a_watertight_solid():
    box = solids.box(10, 20, 5)
    assert box.is_watertight()
    assert box.is_consistently_oriented()
    assert box.euler_characteristic() == 2  # genus-0 closed surface
    assert box.volume() == pytest.approx(1000.0)


def test_cylinder_volume_matches_analytic():
    cyl = solids.cylinder(5, 10, segments=256)
    assert cyl.is_watertight()
    # More segments → closer to π r² h; 256 sides is within 0.1%.
    assert cyl.volume() == pytest.approx(math.pi * 25 * 10, rel=1e-3)


def test_tube_has_a_through_hole():
    tube = solids.tube(10, 5, 8, segments=128)
    assert tube.is_watertight()
    assert tube.is_consistently_oriented()
    # A solid with one through-hole is a torus: Euler characteristic 0.
    assert tube.euler_characteristic() == 0
    assert tube.volume() == pytest.approx(math.pi * (100 - 25) * 8, rel=1e-3)


def test_watertight_detects_a_hole():
    # Drop one triangle: the mesh now has boundary edges and is not watertight.
    box = solids.box(4, 4, 4)
    holed = Mesh(box.vertices, box.faces[:-1])
    assert not holed.is_watertight()
    assert len(holed.boundary_edges()) > 0


def test_orientation_detects_a_flipped_face():
    box = solids.box(4, 4, 4)
    faces = box.faces.copy()
    faces[0] = faces[0][::-1]  # flip one triangle's winding
    assert not Mesh(box.vertices, faces).is_consistently_oriented()


# ---------------------------------------------------------------------------
# triangulation
# ---------------------------------------------------------------------------


def test_ear_clipping_preserves_area():
    # A triangulation must cover the polygon exactly: triangle areas sum to it.
    loop = profiles.rounded_rectangle(30, 20, 5)
    tris = solids.triangulate_simple(loop)
    total = 0.0
    for a, b, c in tris:
        p, q, r = loop[a], loop[b], loop[c]
        total += 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    assert total == pytest.approx(abs(solids._signed_area(loop)), rel=1e-9)


def test_l_bracket_reflex_corner_extrudes_correctly():
    # The L has one reflex vertex — the case ear clipping exists for.
    bracket = parts.l_bracket(length=40, height=30, thickness=5, wall=6)
    assert bracket.is_watertight()
    assert bracket.is_consistently_oriented()
    # Area of the L = length*wall + (height-wall)*wall; volume = area * thickness.
    area = 40 * 6 + (30 - 6) * 6
    assert bracket.volume() == pytest.approx(area * 5)


# ---------------------------------------------------------------------------
# gears
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("teeth", [12, 20, 40])
def test_solid_gear_is_watertight(teeth):
    gear = spur_gear(module=2.0, teeth=teeth, thickness=6.0)
    assert gear.is_watertight()
    assert gear.is_consistently_oriented()
    assert gear.euler_characteristic() == 2  # solid disk, no hole


def test_bored_gear_is_a_torus():
    gear = spur_gear(module=2.0, teeth=20, thickness=6.0, bore_diameter=8.0)
    assert gear.is_watertight()
    assert gear.is_consistently_oriented()
    assert gear.euler_characteristic() == 0  # through-hole


def test_gear_flanks_are_the_exact_involute():
    # The headline correctness claim: every flank vertex lies on the mathematical
    # involute of the base circle, to machine precision.
    gear = spur_gear(module=3.0, teeth=24, thickness=8.0)
    dev = involute_deviation(gear, module=3.0, teeth=24, pressure_angle_deg=20.0)
    assert dev["flank_points"] > 100
    assert dev["max_deviation"] < 1e-9, f"max flank deviation {dev['max_deviation']} mm"


def test_measured_dimensions_match_the_spec():
    # Measured back from the mesh, without reading the build parameters.
    module, teeth = 2.5, 18
    gear = spur_gear(module=module, teeth=teeth, thickness=5.0)
    analytic = profiles.gear_parameters(module, teeth)
    measured = measure_gear(gear)
    assert measured["teeth"] == teeth
    assert measured["module"] == pytest.approx(module, rel=1e-6)
    assert measured["outer_radius"] == pytest.approx(analytic["outer_radius"], rel=1e-6)
    assert measured["root_radius"] == pytest.approx(analytic["root_radius"], rel=1e-6)


def test_pressure_angle_changes_the_base_circle():
    # A larger pressure angle shrinks the base circle (rb = r cos α).
    dev20 = involute_deviation(spur_gear(2, 20, 5, pressure_angle_deg=20), 2, 20, 20)
    dev25 = involute_deviation(spur_gear(2, 20, 5, pressure_angle_deg=25), 2, 20, 25)
    assert dev25["base_radius"] < dev20["base_radius"]
    assert dev20["max_deviation"] < 1e-9 and dev25["max_deviation"] < 1e-9


def test_too_few_teeth_is_rejected():
    with pytest.raises(ValueError):
        spur_gear(module=2.0, teeth=3, thickness=5.0)


def test_oversized_bore_is_rejected():
    with pytest.raises(ValueError):
        spur_gear(module=2.0, teeth=20, thickness=5.0, bore_diameter=60.0)


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


def test_stl_round_trips():
    gear = spur_gear(module=2.0, teeth=20, thickness=6.0, bore_diameter=8.0)
    data = gear.to_stl_binary()
    # Header (80) + count (4) + 50 bytes per triangle.
    assert len(data) == 84 + 50 * gear.face_count
    back = Mesh.from_stl_binary(data)
    assert back.face_count == gear.face_count
    assert back.volume() == pytest.approx(gear.volume(), rel=1e-6)
    assert back.welded().is_watertight()


def test_obj_export_has_all_vertices_and_faces():
    box = solids.box(2, 2, 2)
    obj = box.to_obj()
    assert obj.count("\nv ") + obj.startswith("v ") == box.vertex_count
    assert obj.count("f ") == box.face_count


# ---------------------------------------------------------------------------
# Blender integration (skipped when Blender is not installed)
# ---------------------------------------------------------------------------


def _blender():
    from partkit.blender_export import find_blender

    return find_blender()


def test_bpy_script_is_emitted_without_blender():
    from partkit.blender_export import mesh_to_bpy

    gear = spur_gear(2, 12, 4)
    script = mesh_to_bpy(gear, name="g")
    assert "from_pydata" in script
    assert script.count("bpy.data") >= 2


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_builds_blend_and_render(tmp_path):
    from partkit.blender_export import render

    gear = spur_gear(module=3.0, teeth=16, thickness=6.0, bore_diameter=8.0)
    result = render(gear, out_dir=str(tmp_path), name="gear16", samples=8)
    assert result["ran"], result
    assert result["blend_written"] and os.path.getsize(result["blend"]) > 1000
    assert result["png_written"] and os.path.getsize(result["png"]) > 1000
