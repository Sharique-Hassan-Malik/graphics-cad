"""Take a solved 2D profile into Blender and make it a solid.

The contrast with a mesh-first CAD kernel is deliberate. Here the sketch solver
produces a *constrained, solved outline* — a list of 2D points that satisfy the
constraints exactly — and Blender does the modelling: the emitted script builds
the polygon with `bmesh`, then uses Blender's own `extrude_face_region` operator
to pull it into a prism. That is "parametric sketch → feature" the way a CAD
package does it: solve the sketch, then extrude the result.

As everywhere else, Blender is optional. `to_bpy_script()` returns a self
contained program you can run with `blender --background --python`, and
`render()` runs it headlessly if a Blender binary is found — otherwise it writes
the script and says so. The solver's correctness never depends on Blender.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

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




def to_bpy_script(
    profile: np.ndarray,
    thickness: float = 4.0,
    name: str = "sketch",
    blend_path: str | None = None,
    png_path: str | None = None,
    samples: int = 24,
    resolution: tuple[int, int] = (720, 540),
) -> str:
    """A full Blender program: build the solved profile, extrude it into a solid
    with bmesh, light it, and (if paths are given) render and save."""
    profile = np.asarray(profile, dtype=float)
    coords = ", ".join(f"({x:.9f},{y:.9f})" for x, y in profile)
    center = profile.mean(axis=0)
    extent = np.ptp(profile, axis=0)
    radius = float(np.hypot(*extent)) / 2 + thickness
    cx, cy = float(center[0]), float(center[1])

    render_block = ""
    if blend_path and png_path:
        render_block = f"""
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = {samples}
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.filepath = {png_path!r}
bpy.ops.wm.save_as_mainfile(filepath={blend_path!r})
bpy.ops.render.render(write_still=True)
print("RENDER_DONE", {png_path!r})
"""

    return f"""import bpy, bmesh, mathutils, math

bpy.ops.wm.read_factory_settings(use_empty=True)

# --- build the solved sketch profile and extrude it (Blender does the modelling) ---
coords = [{coords}]
mesh = bpy.data.meshes.new({name!r})
obj = bpy.data.objects.new({name!r}, mesh)
bpy.context.collection.objects.link(obj)

bm = bmesh.new()
verts = [bm.verts.new((x, y, 0.0)) for (x, y) in coords]
face = bm.faces.new(verts)
res = bmesh.ops.extrude_face_region(bm, geom=[face])
top = [g for g in res['geom'] if isinstance(g, bmesh.types.BMVert)]
bmesh.ops.translate(bm, verts=top, vec=(0.0, 0.0, {thickness:.6f}))
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(mesh)
bm.free()
mesh.update()

# --- material ---
mat = bpy.data.materials.new("brass")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.66, 0.53, 0.24, 1.0)
bsdf.inputs["Metallic"].default_value = 0.85
bsdf.inputs["Roughness"].default_value = 0.35
obj.data.materials.append(mat)

center = mathutils.Vector(({cx:.6f}, {cy:.6f}, {thickness/2:.6f}))
radius = {radius:.6f}

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
direction = mathutils.Vector((1, -1.3, 0.9)).normalized()
cam.location = center + direction * radius * 3.4
look = center - cam.location
cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

key = bpy.data.lights.new("key", 'SUN'); key.energy = 4.0
key_obj = bpy.data.objects.new("key", key); bpy.context.collection.objects.link(key_obj)
key_obj.rotation_euler = (math.radians(50), math.radians(20), math.radians(30))
fill = bpy.data.lights.new("fill", 'SUN'); fill.energy = 1.5
fill_obj = bpy.data.objects.new("fill", fill); bpy.context.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (math.radians(-30), math.radians(-40), 0)
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
bpy.context.scene.world = world
{render_block}"""


def render(
    profile: np.ndarray,
    out_dir: str,
    name: str = "sketch",
    thickness: float = 4.0,
    blender_bin: str | None = None,
    samples: int = 24,
    timeout: int = 180,
) -> dict:
    """Emit the build/extrude/render script and run it in Blender if available.

    Returns a dict recording the paths and whether Blender actually ran. With no
    Blender, `ran` is False and the emitted script is still written — the solve
    is already complete and correct without it.
    """
    os.makedirs(out_dir, exist_ok=True)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.png")
    script = to_bpy_script(profile, thickness, name, blend_path, png_path, samples)
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
    if not result["ran"]:
        result["stderr_tail"] = proc.stderr.strip().splitlines()[-8:] if proc.stderr else []
    return result
