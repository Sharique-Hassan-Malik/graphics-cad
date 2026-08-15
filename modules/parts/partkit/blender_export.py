"""Taking a `Mesh` into Blender: emit a build script, and optionally run it.

The design keeps Blender at arm's length, for a reason that matters to anyone
who wants to *use* this. The CAD core (mesh, profiles, solids, gears) has no
dependency on Blender at all — it is plain numpy, and every property that
decides manufacturability is checked without Blender in the loop. Blender is a
*backend*: a place to view the part, to render it, and to hand it to the rest of
a Blender pipeline.

Two entry points:

  * `to_bpy_script(mesh)` returns a self-contained Python program that rebuilds
    the mesh inside Blender via `from_pydata` — vertices and faces straight in,
    no import format, no add-on. Paste it into Blender's text editor, or run it
    with `blender --background --python script.py`.

  * `render(mesh, ...)` finds a Blender binary and runs that script headlessly,
    producing a `.blend` file and a Cycles render. If no Blender is found it
    still writes the script and says so, rather than failing — the part is
    already valid without it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from .mesh import Mesh

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from geokit.blender import (  # noqa: E402
    DONE_MARKER,
    RenderResult,
    find_blender,
    mesh_to_pydata,
    run_script,
)

# `find_blender` was written out seven times in this repository, once per
# generator, along with the headless-invocation plumbing. Both now come from
# `geokit.blender`. The *scene* stays here: a terrain slab, a CAD part and an
# animated crank are different scenes, and one render() with fifteen flags
# would have been a worse abstraction than the duplication it replaced.




def mesh_to_bpy(mesh: Mesh, name: str = "part") -> str:
    """The fragment that builds `mesh` as a Blender object via from_pydata."""
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


def render_script(
    mesh: Mesh,
    blend_path: str,
    png_path: str,
    name: str = "part",
    samples: int = 24,
    resolution: tuple[int, int] = (720, 540),
) -> str:
    """A full Blender program: build the part, light it, render it, save the .blend.

    Cycles on the CPU is chosen deliberately: it needs no display or GPU, so it
    runs on a headless machine, which is where a build pipeline lives. The camera
    is framed automatically to the part's bounding sphere so the same script
    works for a 4 mm bracket and a 200 mm gear.
    """
    size = mesh.size()
    radius = float((size**2).sum() ** 0.5) / 2 or 1.0
    center = mesh.vertices.mean(axis=0)
    return f"""import bpy, mathutils, math

# --- clean scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- build the part ---
{mesh_to_bpy(mesh, name)}
# shade smooth-ish is skipped: CAD wants flat faces, which read the geometry honestly.

# --- material ---
mat = bpy.data.materials.new("steel")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.55, 0.57, 0.60, 1.0)
bsdf.inputs["Metallic"].default_value = 0.9
bsdf.inputs["Roughness"].default_value = 0.35
obj.data.materials.append(mat)

center = mathutils.Vector(({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}))
radius = {radius:.6f}

# --- camera framed to the bounding sphere ---
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
direction = mathutils.Vector((1, -1.4, 0.9)).normalized()
cam.location = center + direction * radius * 3.2
look = center - cam.location
cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

# --- lights ---
key = bpy.data.lights.new("key", 'SUN'); key.energy = 4.0
key_obj = bpy.data.objects.new("key", key); bpy.context.collection.objects.link(key_obj)
key_obj.rotation_euler = (math.radians(50), math.radians(20), math.radians(30))
fill = bpy.data.lights.new("fill", 'SUN'); fill.energy = 1.5
fill_obj = bpy.data.objects.new("fill", fill); bpy.context.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (math.radians(-30), math.radians(-40), 0)
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
bpy.context.scene.world = world

# --- render (Cycles CPU: headless-safe) ---
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = {samples}
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.film_transparent = False
scene.render.filepath = {png_path!r}
bpy.ops.wm.save_as_mainfile(filepath={blend_path!r})
bpy.ops.render.render(write_still=True)
print("RENDER_DONE", {png_path!r})
"""


def render(
    mesh: Mesh,
    out_dir: str,
    name: str = "part",
    blender_bin: str | None = None,
    samples: int = 24,
    timeout: float = 180.0,
) -> dict:
    """Build a `.blend` and a rendered `.png` from a mesh, using Blender headlessly.

    Returns a dict describing what happened. If no Blender binary is available the
    build script is still written to `out_dir`, and `ran` is False — the caller
    gets something runnable rather than an exception.
    """
    os.makedirs(out_dir, exist_ok=True)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.png")
    script = render_script(mesh, blend_path, png_path, name=name, samples=samples)
    script_path = os.path.join(out_dir, f"{name}_build.py")
    with open(script_path, "w") as handle:
        handle.write(script)

    binary = blender_bin or find_blender()
    result = {"script": script_path, "blend": blend_path, "png": png_path, "ran": False, "blender": binary}
    if binary is None:
        result["note"] = "no Blender binary found; run the emitted script yourself"
        return result

    proc = subprocess.run(
        [binary, "--background", "--factory-startup", "--python", script_path],
        capture_output=True, text=True, timeout=timeout,
    )
    result["ran"] = "RENDER_DONE" in proc.stdout
    result["blend_written"] = os.path.exists(blend_path)
    result["png_written"] = os.path.exists(png_path)
    result["stdout_tail"] = proc.stdout.strip().splitlines()[-5:] if proc.stdout else []
    if not result["ran"]:
        result["stderr_tail"] = proc.stderr.strip().splitlines()[-8:] if proc.stderr else []
    return result
