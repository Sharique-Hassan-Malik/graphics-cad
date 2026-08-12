"""Tests for procedural terrain.

The through-line: generation is a *deterministic* function of the seed; hydraulic
erosion *conserves mass* to floating-point precision (it moves soil, never
creates or destroys it); and the meshed terrain is a *watertight* solid, not an
open sheet. Each is a measured number, not an impression of the render.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrainkit.erosion import erode  # noqa: E402
from terrainkit.mesh import terrain_mesh  # noqa: E402
from terrainkit.noise import fbm, heightmap, value_noise  # noqa: E402


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


def test_heightmap_is_deterministic_from_seed():
    a = heightmap(64, seed=7)
    b = heightmap(64, seed=7)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_terrain():
    assert not np.array_equal(heightmap(64, seed=1), heightmap(64, seed=2))


def test_heightmap_is_normalised():
    h = heightmap(96, seed=3)
    assert h.min() == pytest.approx(0.0, abs=1e-9)
    assert h.max() == pytest.approx(1.0, abs=1e-9)


def test_fbm_adds_detail_with_octaves():
    # More octaves add high-frequency detail, raising small-scale variation.
    one = fbm((128, 128), seed=5, octaves=1)
    many = fbm((128, 128), seed=5, octaves=6)
    rough = lambda a: np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
    assert rough(many) > rough(one)


def test_value_noise_is_smooth():
    # A single octave upsampled should vary gently: neighbouring cells differ little.
    n = value_noise((256, 256), cells=8, seed=0)
    step = np.abs(np.diff(n, axis=1))
    assert step.max() < 0.2       # no discontinuities from the smoothstep interp


# ---------------------------------------------------------------------------
# erosion — the headline: it conserves mass
# ---------------------------------------------------------------------------


def test_erosion_conserves_mass():
    h = heightmap(96, seed=11)
    eroded, stats = erode(h, droplets=8000, seed=1)
    # every grain lifted is deposited somewhere
    assert stats.eroded == pytest.approx(stats.deposited, abs=1e-6)
    # so total elevation is unchanged
    assert stats.mass_error < 1e-9
    assert eroded.sum() == pytest.approx(h.sum(), rel=1e-9)


def test_erosion_actually_changes_the_terrain():
    h = heightmap(96, seed=11)
    eroded, stats = erode(h, droplets=8000, seed=1)
    assert stats.eroded > 0
    assert np.abs(eroded - h).max() > 1e-3      # it did real work


def test_erosion_is_deterministic():
    h = heightmap(96, seed=11)
    a, _ = erode(h, droplets=5000, seed=2)
    b, _ = erode(h, droplets=5000, seed=2)
    assert np.array_equal(a, b)


def test_erosion_does_not_move_material_off_map():
    # deposit-on-exit means the border-crossing droplets still conserve mass
    h = heightmap(64, seed=4)
    _, stats = erode(h, droplets=6000, seed=9)
    assert abs(stats.eroded - stats.deposited) < 1e-6


# ---------------------------------------------------------------------------
# meshing
# ---------------------------------------------------------------------------


def test_terrain_mesh_is_a_watertight_solid():
    h = heightmap(48, seed=1)
    mesh = terrain_mesh(h, width=10, height_scale=3)
    assert mesh.is_watertight()
    # a solid slab (disk topology) has Euler characteristic 2
    assert mesh.euler_characteristic() == 2


def test_mesh_face_count_matches_the_grid():
    h = heightmap(20, seed=1)
    mesh = terrain_mesh(h)
    # (n-1)² cells × 2 tris, top and bottom, plus 4 walls × (n-1) × 2 tris
    n = 20
    expected = 2 * 2 * (n - 1) ** 2 + 4 * (n - 1) * 2
    assert mesh.face_count == expected


def test_stl_size_is_correct():
    h = heightmap(16, seed=1)
    mesh = terrain_mesh(h)
    data = mesh.to_stl_binary()
    assert len(data) == 84 + 50 * mesh.face_count


# ---------------------------------------------------------------------------
# 2D map + Blender bridge
# ---------------------------------------------------------------------------


def test_map_png_is_written(tmp_path):
    from terrainkit.colormap import save_map

    h = heightmap(48, seed=2)
    path = str(tmp_path / "m.png")
    save_map(h, path)
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def _blender():
    from terrainkit.blender_export import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_renders_terrain(tmp_path):
    from terrainkit.blender_export import render

    h = heightmap(48, seed=2)
    out = render(h, out_dir=str(tmp_path), name="t", samples=8)
    assert out["ran"], out
    assert out["png_written"] and os.path.getsize(out["png"]) > 1000
