"""Cross-generator tests — what is only true because these seven share a core.

Each generator's own behaviour is tested in its own folder. What is tested here
is that they all produce the *same* mesh type, that the shared topology checks
therefore work on any of them, and that the one Blender path degrades the same
way everywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geokit import blender, registry  # noqa: E402
from geokit.mesh import Mesh  # noqa: E402


@pytest.fixture(scope="session")
def tetrahedron() -> Mesh:
    """The smallest closed, orientable surface — every check has a known answer."""
    return Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


@pytest.fixture(scope="session")
def open_box() -> Mesh:
    """Two triangles: manifold in the sense of having no bad edges, but open."""
    return Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )


def _generated_meshes() -> dict[str, Mesh]:
    """One mesh from each generator that makes one, through its real API."""
    meshes: dict[str, Mesh] = {}

    registry.add_to_path("parts")
    from partkit.parts import l_bracket

    meshes["parts"] = l_bracket(length=40.0, height=30.0, thickness=4.0, wall=4.0)

    registry.add_to_path("sdf")
    from sdfkit.mesh import Mesh as SdfMesh  # noqa: F401  — the shim under test
    from sdfkit.scenes import SCENES

    registry.add_to_path("terrain")
    from terrainkit.mesh import terrain_mesh

    heights = np.zeros((16, 16))
    meshes["terrain"] = terrain_mesh(heights, width=4.0, height_scale=1.0)
    return meshes


class TestLayout:
    def test_every_generator_exists_with_a_readme(self):
        for gen in registry.generators():
            assert gen.path.is_dir(), f"{gen.name} is missing"
            assert (gen.path / "README.md").is_file(), f"{gen.name} has no README"

    def test_unknown_generator_is_a_clear_error(self):
        with pytest.raises(KeyError, match="unknown generator"):
            registry.generator("nope")

    @staticmethod
    def _sources_containing(needle: str) -> list[Path]:
        """Every source file with *needle* in it, excluding the tests that look
        for it — otherwise this assertion trips over its own text."""
        return sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if "__pycache__" not in str(path)
            and "tests" not in path.parts
            and needle in path.read_text(encoding="utf-8", errors="ignore")
        )

    def test_find_blender_is_defined_exactly_once(self):
        """It was written out seven times, once per generator."""
        found = self._sources_containing("def find_blender")
        assert [p.name for p in found] == ["blender.py"], found

    def test_mesh_is_defined_exactly_once(self):
        found = self._sources_containing("\nclass Mesh")
        assert [p.name for p in found] == ["mesh.py"], found


class TestSharedMesh:
    def test_a_closed_surface_passes_every_check(self, tetrahedron):
        assert tetrahedron.is_watertight()
        assert tetrahedron.is_edge_manifold()
        assert tetrahedron.is_consistently_oriented()
        assert tetrahedron.euler_characteristic() == 2
        assert tetrahedron.genus() == 0
        assert tetrahedron.volume() == pytest.approx(1 / 6)

    def test_an_open_surface_is_reported_as_open(self, open_box):
        assert not open_box.is_watertight()
        assert len(open_box.boundary_edges()) == 4

    def test_the_methods_that_only_one_copy_had_survived_the_merge(self, tetrahedron):
        """`genus`, `face_normals`, `face_centroids` and `oriented` were only in
        the SDF mesher's copy. Losing them would have made the merge a
        regression."""
        assert tetrahedron.genus() == 0
        assert tetrahedron.face_normals().shape == (4, 3)
        assert tetrahedron.face_centroids().shape == (4, 3)
        assert tetrahedron.oriented().is_consistently_oriented()

    def test_stl_round_trip_needs_welding(self, tetrahedron, tmp_path):
        """The reason `geo check` welds on load.

        Binary STL stores three vertices per triangle with no sharing, so a
        closed mesh comes back as loose triangles and every topology check says
        "open" — about the file format, not the geometry.
        """
        path = tmp_path / "t.stl"
        tetrahedron.save_stl(str(path))
        loaded = Mesh.from_stl_binary(path.read_bytes())

        assert loaded.vertex_count == 12          # 4 triangles x 3, unshared
        assert not loaded.is_watertight()

        welded = loaded.welded()
        assert welded.vertex_count == 4
        assert welded.is_watertight()


class TestEveryGeneratorSharesTheType:
    def test_generators_produce_the_shared_mesh(self):
        for name, mesh in _generated_meshes().items():
            assert isinstance(mesh, Mesh), f"{name} produced {type(mesh).__name__}"

    def test_the_shared_checks_run_on_every_generators_output(self):
        """The payoff of one type: before, each generator could only validate
        its own meshes with its own copy of the checks."""
        for name, mesh in _generated_meshes().items():
            assert mesh.vertex_count > 0, name
            assert isinstance(mesh.is_watertight(), (bool, np.bool_)), name
            assert mesh.area() > 0, name

    def test_the_cad_part_is_manufacturable(self):
        mesh = _generated_meshes()["parts"]
        assert mesh.is_watertight()
        assert mesh.is_edge_manifold()
        assert mesh.is_consistently_oriented()
        assert mesh.volume() > 0

    def test_meshes_from_different_generators_can_be_combined(self):
        """Only possible because they are the same type."""
        meshes = _generated_meshes()
        combined = Mesh.concatenate(list(meshes.values()))
        assert combined.face_count == sum(m.face_count for m in meshes.values())

    def test_the_module_shims_resolve_to_the_shared_class(self):
        registry.add_to_path("sdf")
        registry.add_to_path("parts")
        from partkit.mesh import Mesh as PartMesh
        from sdfkit.mesh import Mesh as SdfMesh

        assert SdfMesh is Mesh
        assert PartMesh is Mesh


class TestSharedBlenderPath:
    def test_missing_blender_is_a_state_not_an_error(self, tetrahedron, tmp_path, monkeypatch):
        monkeypatch.setattr(blender, "find_blender", lambda explicit=None: None)
        result = blender.run_script("print('hi')", tmp_path, name="t")
        assert not result.ran
        assert result.note
        # The script is written anyway — the geometry was valid without Blender.
        assert Path(result.script).is_file()

    def test_the_generated_script_rebuilds_the_mesh(self, tetrahedron):
        fragment = blender.mesh_to_pydata(tetrahedron, "tet")
        assert "from_pydata" in fragment
        assert fragment.count("(") > tetrahedron.vertex_count

    def test_camera_framing_scales_with_the_model(self, tetrahedron):
        near = blender.frame_camera(tetrahedron.centroid(), 1.0)
        far = blender.frame_camera(tetrahedron.centroid(), 100.0)
        assert near != far, "framing must depend on size, or one scale breaks"


class TestStandalone:
    """Each generator runs from its own folder — the reason for the layout."""

    @pytest.mark.parametrize("gen", registry.generators(), ids=lambda g: g.name)
    def test_generator_cli_runs_from_its_own_folder(self, gen):
        completed = subprocess.run(
            [sys.executable, "-m", gen.package, "--help"],
            cwd=gen.path, capture_output=True, text=True, timeout=180,
        )
        assert completed.returncode == 0, completed.stderr


class TestCli:
    def test_modules_listing_covers_the_repo(self, capsys):
        from geokit import cli

        assert cli.main(["modules"]) == 0
        out = capsys.readouterr().out
        for gen in registry.generators():
            assert gen.name in out

    def test_check_accepts_a_closed_mesh(self, tetrahedron, tmp_path, capsys):
        from geokit import cli

        path = tmp_path / "t.stl"
        tetrahedron.save_stl(str(path))
        assert cli.main(["check", str(path)]) == 0
        out = capsys.readouterr().out
        assert "watertight" in out and "slicer will accept" in out

    def test_check_rejects_an_open_mesh(self, open_box, tmp_path, capsys):
        from geokit import cli

        path = tmp_path / "open.stl"
        open_box.save_stl(str(path))
        assert cli.main(["check", str(path)]) == 1
        assert "Not manufacturable" in capsys.readouterr().out

    def test_check_without_welding_says_what_the_file_holds(self, tetrahedron, tmp_path, capsys):
        from geokit import cli

        path = tmp_path / "t.stl"
        tetrahedron.save_stl(str(path))
        assert cli.main(["check", str(path), "--no-weld"]) == 1
        assert "vertices_in_file" in capsys.readouterr().out
