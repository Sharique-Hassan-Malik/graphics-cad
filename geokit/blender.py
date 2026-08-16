"""Getting geometry into Blender, once, for seven generators.

Blender is a *backend* here and never a dependency of the geometry. Every
generator in this repository produces meshes with plain numpy and validates
them without Blender in the loop — the manifold checks in `geokit.mesh` are
counting, not asking a kernel. Blender is where you go to look at the result.

Seven modules each had their own copy of `find_blender()`, their own
`from_pydata` boilerplate and their own headless-invocation code. That is what
lives here now. What does **not** live here is the scene: a terrain slab lit by
a low sun, a CAD part on a neutral backdrop and an animated crank are different
scenes, and flattening them into one "render()" with fifteen flags would have
been a worse abstraction than three hundred duplicated lines.

So the split is: this module knows how to find Blender, how to turn a mesh into
`from_pydata` calls, and how to run a script headlessly. Each generator still
writes its own scene.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Printed by a generated script when it reaches the end. Checking for it is how
# we know Blender ran the whole program rather than dying halfway with a zero
# exit status, which it will happily do.
DONE_MARKER = "RENDER_DONE"

_COMMON_LOCATIONS = (
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/opt/blender/blender",
    "/snap/bin/blender",
    "~/blender/blender",
    "/Applications/Blender.app/Contents/MacOS/Blender",
)

#: Where an unpacked official tarball ends up. Blender ships as a self-contained
#: archive that needs no install and no root, which is how you get it on a
#: machine you do not administer -- and is therefore the likeliest place for it
#: to be sitting unfound. Globbed because the version is in the directory name.
_PORTABLE_PATTERNS = (
    "~/.local/opt/blender-*/blender",
    "~/opt/blender-*/blender",
    "~/blender-*/blender",
    "/opt/blender-*/blender",
)


def find_blender(explicit: str | None = None) -> str | None:
    """Locate a Blender executable: an explicit path, `$BLENDER_BIN`, `PATH`, then
    the usual places. Returns `None` rather than raising — not having Blender is
    a normal state here, and every generator still works without it."""
    for candidate in (explicit, os.environ.get("BLENDER_BIN")):
        if candidate and os.path.exists(candidate):
            return candidate
    found = shutil.which("blender")
    if found:
        return found
    for path in _COMMON_LOCATIONS:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    for pattern in _PORTABLE_PATTERNS:
        # Newest version last in sort order, so prefer it.
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return matches[-1]
    return None


def available() -> bool:
    return find_blender() is not None


def mesh_to_pydata(mesh: Any, name: str = "object") -> str:
    """The fragment that rebuilds a mesh inside Blender.

    `from_pydata` takes vertices and faces directly, so there is no file format
    in the middle and nothing to import — which means no version skew between
    what was generated and what Blender loads.
    """
    verts = ", ".join(f"({x:.6f},{y:.6f},{z:.6f})" for x, y, z in mesh.vertices)
    faces = ", ".join(f"({a},{b},{c})" for a, b, c in mesh.faces)
    return (
        f"verts = [{verts}]\n"
        f"faces = [{faces}]\n"
        f"mesh = bpy.data.meshes.new({name!r})\n"
        f"mesh.from_pydata(verts, [], faces)\n"
        f"mesh.update()\n"
        f"mesh.validate()\n"
        f"obj = bpy.data.objects.new({name!r}, mesh)\n"
        f"bpy.context.collection.objects.link(obj)\n"
    )


def frame_camera(center: Sequence[float], radius: float, *, elevation: float = 0.9) -> str:
    """Place a camera that frames a sphere of `radius` around `center`.

    Automatic framing is why one script works for a 4 mm bracket and a 200 mm
    gear without a magic distance per generator.
    """
    cx, cy, cz = (float(v) for v in center)
    distance = max(radius, 1e-6) * 3.2
    return (
        f"cam_data = bpy.data.cameras.new('cam')\n"
        f"cam = bpy.data.objects.new('cam', cam_data)\n"
        f"bpy.context.collection.objects.link(cam)\n"
        f"bpy.context.scene.camera = cam\n"
        f"cam.location = ({cx + distance:.6f}, {cy - distance:.6f}, "
        f"{cz + distance * elevation:.6f})\n"
        f"direction = mathutils.Vector(({cx:.6f}, {cy:.6f}, {cz:.6f})) - cam.location\n"
        f"cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()\n"
    )


def cycles_setup(samples: int = 24, resolution: tuple[int, int] = (720, 540)) -> str:
    """Cycles on the CPU, deliberately: no GPU and no display, so it runs where a
    build pipeline actually runs."""
    width, height = resolution
    return (
        f"scene = bpy.context.scene\n"
        f"scene.render.engine = 'CYCLES'\n"
        f"scene.cycles.device = 'CPU'\n"
        f"scene.cycles.samples = {int(samples)}\n"
        f"scene.render.resolution_x = {int(width)}\n"
        f"scene.render.resolution_y = {int(height)}\n"
        f"scene.render.resolution_percentage = 100\n"
    )


def finish(blend_path: str, png_path: str) -> str:
    """Render, save, and print the marker that says it got this far."""
    return (
        f"scene.render.filepath = {png_path!r}\n"
        f"bpy.ops.render.render(write_still=True)\n"
        f"bpy.ops.wm.save_as_mainfile(filepath={blend_path!r})\n"
        f"print({DONE_MARKER!r})\n"
    )


@dataclass
class RenderResult:
    """What a render attempt produced. Never an exception for a missing Blender."""

    script: str
    blend: str = ""
    png: str = ""
    blender: str | None = None
    ran: bool = False
    note: str = ""
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)

    @property
    def blend_written(self) -> bool:
        return bool(self.blend) and os.path.exists(self.blend)

    @property
    def png_written(self) -> bool:
        return bool(self.png) and os.path.exists(self.png)

    def to_dict(self) -> dict[str, Any]:
        return {
            "script": self.script,
            "blend": self.blend,
            "png": self.png,
            "blender": self.blender,
            "ran": self.ran,
            "blend_written": self.blend_written,
            "png_written": self.png_written,
            **({"note": self.note} if self.note else {}),
        }


def run_script(
    script: str,
    out_dir: str | Path,
    name: str = "scene",
    *,
    blend_path: str = "",
    png_path: str = "",
    blender_bin: str | None = None,
    timeout: float = 300.0,
) -> RenderResult:
    """Write a generated script and run it headlessly, if Blender is here.

    With no Blender the script is still written and `ran` is False. The caller
    gets something runnable rather than an exception — the geometry was already
    generated and validated without Blender, and losing that to an
    ImportError-shaped failure would be absurd.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / f"{name}_build.py"
    script_path.write_text(script, encoding="utf-8")

    result = RenderResult(
        script=str(script_path),
        blend=blend_path or str(out / f"{name}.blend"),
        png=png_path or str(out / f"{name}.png"),
        blender=blender_bin or find_blender(),
    )

    if result.blender is None:
        result.note = "no Blender binary found; run the emitted script yourself"
        return result

    try:
        completed = subprocess.run(
            [result.blender, "--background", "--factory-startup", "--python", str(script_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result.note = f"Blender did not finish within {timeout:g}s"
        return result

    result.ran = DONE_MARKER in (completed.stdout or "")
    result.stdout_tail = (completed.stdout or "").strip().splitlines()[-5:]
    if not result.ran:
        result.stderr_tail = (completed.stderr or "").strip().splitlines()[-8:]
    return result
