"""Tests for Wave Function Collapse.

The headline is legality as a number: an output is correct only if *every* shared
edge satisfies the tileset's adjacency rule, checked independently of the solver.
Alongside that: the generator is deterministic from its seed, respects hard border
constraints, and (for terrain) yields a corner lattice consistent enough to lift
into a watertight 3D island — which is itself a proof the adjacency held.
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wfckit import verify  # noqa: E402
from wfckit.model3d import island_mesh, terrain_lattice  # noqa: E402
from wfckit.solver import collapse  # noqa: E402
from wfckit.tiles import TILESETS  # noqa: E402


def _border(name):
    return "shut" if name == "pipes" else None


# ---------------------------------------------------------------------------
# the headline: every output is legal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["pipes", "terrain"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_every_output_satisfies_all_adjacencies(name, seed):
    ts = TILESETS[name]()
    result = collapse(ts, 20, 20, seed=seed, border=_border(name))
    assert result.success
    assert verify.is_valid(result.grid, ts), verify.adjacency_violations(result.grid, ts)[:3]


def test_legality_holds_on_a_large_grid():
    ts = TILESETS["terrain"]()
    result = collapse(ts, 48, 48, seed=99)
    assert result.success
    violations = verify.adjacency_violations(result.grid, ts)
    assert len(violations) == 0
    # a real grid, thousands of edges, all satisfied
    assert verify.edge_count(48, 48) > 8000


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_same_seed_is_identical():
    ts = TILESETS["pipes"]()
    a = collapse(ts, 24, 24, seed=42, border="shut")
    b = collapse(ts, 24, 24, seed=42, border="shut")
    assert np.array_equal(a.grid, b.grid)


def test_different_seeds_differ():
    ts = TILESETS["terrain"]()
    a = collapse(ts, 24, 24, seed=1)
    b = collapse(ts, 24, 24, seed=2)
    assert not np.array_equal(a.grid, b.grid)


def test_success_rate_is_high():
    ts = TILESETS["pipes"]()
    ok = sum(collapse(ts, 24, 24, seed=s, border="shut").success for s in range(20))
    assert ok == 20  # this tileset always converges


# ---------------------------------------------------------------------------
# constraints and the tileset itself
# ---------------------------------------------------------------------------


def test_border_constraint_is_respected():
    ts = TILESETS["pipes"]()
    result = collapse(ts, 20, 20, seed=5, border="shut")
    g = result.grid
    # every outward-facing edge socket on the boundary must be 'shut'
    for c in range(20):
        assert ts.tiles[g[0, c]].sockets[0] == "shut"       # north edge
        assert ts.tiles[g[-1, c]].sockets[2] == "shut"      # south edge
    for r in range(20):
        assert ts.tiles[g[r, 0]].sockets[3] == "shut"       # west edge
        assert ts.tiles[g[r, -1]].sockets[1] == "shut"      # east edge


def test_compat_matrix_is_symmetric():
    ts = TILESETS["terrain"]()
    # b may sit east of a  ⟺  a may sit west of b
    assert np.array_equal(ts.compat[1], ts.compat[3].T)
    assert np.array_equal(ts.compat[0], ts.compat[2].T)


def test_pipe_rotations_have_rotated_sockets():
    ts = TILESETS["pipes"]()
    names = {t.name: t for t in ts.tiles}
    # the base elbow opens N+E; a 90° cw rotation opens E+S
    base = names["elbow"].sockets
    rot = names["elbow_r1"].sockets
    assert base == ("open", "open", "shut", "shut")
    assert rot == ("shut", "open", "open", "shut")


# ---------------------------------------------------------------------------
# lifting terrain into 3D (and proving corner consistency)
# ---------------------------------------------------------------------------


def test_terrain_corner_lattice_is_consistent():
    # island_mesh / terrain_lattice raise if any shared corner disagrees, which
    # can only happen if adjacency was violated — so this is a second, structural
    # proof of legality.
    ts = TILESETS["terrain"]()
    result = collapse(ts, 20, 20, seed=8)
    lattice = terrain_lattice(result)
    assert lattice.shape == (21, 21)
    assert set(np.unique(lattice)) <= {0, 1}


def test_island_mesh_is_watertight():
    ts = TILESETS["terrain"]()
    result = collapse(ts, 16, 16, seed=8)
    verts, faces = island_mesh(result)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    assert all(n == 2 for n in counts.values())   # closed surface


# ---------------------------------------------------------------------------
# the from-scratch PNG writer
# ---------------------------------------------------------------------------


def test_png_writer_produces_a_valid_file(tmp_path):
    from wfckit.image import write_png

    img = np.random.randint(0, 256, (12, 20, 3), dtype=np.uint8)
    path = str(tmp_path / "x.png")
    write_png(path, img)
    with open(path, "rb") as handle:
        data = handle.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR width/height are the first chunk's payload
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (20, 12)


# ---------------------------------------------------------------------------
# Blender bridge
# ---------------------------------------------------------------------------


def _blender():
    from wfckit.blender_export import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_renders_island(tmp_path):
    from wfckit.blender_export import render

    ts = TILESETS["terrain"]()
    result = collapse(ts, 16, 16, seed=8)
    out = render(result, out_dir=str(tmp_path), name="island", samples=8)
    assert out["ran"], out
    assert out["png_written"] and os.path.getsize(out["png"]) > 1000
