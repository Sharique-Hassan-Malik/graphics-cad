"""The seven generators, what they make, and how to run each on its own.

Static data: reading it imports nothing, so `geo modules` works without numpy
and without Blender.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

MODULES_ROOT = Path(__file__).resolve().parents[1] / "modules"


@dataclass(frozen=True)
class Generator:
    name: str
    package: str
    title: str
    summary: str
    produces: str            # "mesh" | "animation" | "layout"
    standalone: str

    @property
    def path(self) -> Path:
        return MODULES_ROOT / self.name


MANIFEST: tuple[Generator, ...] = (
    Generator("terrain", "terrainkit", "Procedural terrain",
              "Layered noise, hydraulic erosion and a biome colour ramp, meshed "
              "into a slab with skirt walls.",
              "mesh", "python -m terrainkit --size 256 --erode 40"),
    Generator("sdf", "sdfkit", "SDF mesher",
              "Signed distance fields combined with boolean and smooth-blend "
              "operators, surfaced by marching cubes.",
              "mesh", "python -m sdfkit --scene die --resolution 96"),
    Generator("wfc", "wfckit", "Wave function collapse",
              "Constraint-propagating tile solver with backtracking, over 2D "
              "layouts and their 3D realisations.",
              "layout", "python -m wfckit --width 24 --height 24"),
    Generator("parts", "partkit", "Parametric parts",
              "CAD primitives, profiles, revolves and involute gears, checked "
              "for manufacturability rather than looks.",
              "mesh", "python -m partkit --part gear --teeth 18"),
    Generator("sketch", "sketchkit", "Sketch solver",
              "2D geometric constraints — coincident, distance, angle, tangent — "
              "solved by Newton iteration, then extruded.",
              "mesh", "python -m sketchkit --demo bracket"),
    Generator("engine", "enginekit", "Mechanical engine",
              "Slider-crank kinematics for an inline-four, sampled over a cycle "
              "into an animation.",
              "animation", "python -m enginekit --cylinders 4 --frames 120"),
    Generator("rig", "transformkit", "Transformer rig",
              "A jointed character rig with quaternion interpolation between "
              "two poses.",
              "animation", "python -m transformkit --frames 90"),
)

_BY_NAME = {g.name: g for g in MANIFEST}


def generators() -> list[Generator]:
    return list(MANIFEST)


def generator(name: str) -> Generator:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown generator {name!r}; choose from {', '.join(sorted(_BY_NAME))}"
        ) from None


def unavailable(gen: Generator) -> str:
    if not gen.path.is_dir():
        return "not present in this repository"
    if importlib.util.find_spec("numpy") is None:
        return "needs numpy"
    return ""


def add_to_path(name: str) -> Path:
    folder = generator(name).path
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return folder
