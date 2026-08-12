"""Take a meshed SDF into Blender: emit a build script, and optionally render it.

Blender is a backend for viewing and rendering, never a dependency of the
geometry. `to_bpy_script()` rebuilds the mesh with `from_pydata` (vertices and
faces straight in), and `render()` runs it headlessly with Cycles on the CPU —
no GPU, no display — producing a `.blend` and a PNG. With no Blender installed,
the mesh is already extracted and verified; you just get the script.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from .mesh import Mesh


def find_blender() -> str | None:
    candidate = os.environ.get("BLENDER_BIN")
    if candidate and os.path.exists(candidate):
        return candidate
    found = shutil.which("blender")
    if found:
        return found
    for path in ("/usr/bin/blender", "/opt/blender/blender", os.path.expanduser("~/blender/blender")):
        if os.path.exists(path):
            return path
    return None


def mesh_to_bpy(mesh: Mesh, name: str = "surface") -> str:
    verts = ", ".join(f"({x:.6f},{y:.6f},{z:.6f})" for x, y, z in mesh.vertices)
    faces = ", ".join(f"({a},{b},{c})" for a, b, c in mesh.faces)
    return (
        f"verts = [{verts}]\n"
        f"faces = [{faces}]\n"
        f"mesh = bpy.data.meshes.new({name!r})\n"
        f"mesh.from_pydata(verts, [], faces)\n"
        f"mesh.update(); mesh.validate()\n"
        f"obj = bpy.data.objects.new({name!r}, mesh)\n"
        f"bpy.context.collection.objects.link(obj)\n"
    )


def render_script(mesh, blend_path, png_path, name="surface", samples=32,
                  smooth=True, resolution=(720, 540)) -> str:
    size = mesh.size()
    radius = float((size ** 2).sum() ** 0.5) / 2 or 1.0
    center = mesh.vertices.mean(axis=0)
    smooth_line = "\nfor poly in mesh.polygons: poly.use_smooth = True\n" if smooth else "\n"
    return f"""import bpy, mathutils, math

bpy.ops.wm.read_factory_settings(use_empty=True)

{mesh_to_bpy(mesh, name)}{smooth_line}
mat = bpy.data.materials.new("clay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.80, 0.35, 0.28, 1.0)
bsdf.inputs["Roughness"].default_value = 0.55
obj.data.materials.append(mat)

center = mathutils.Vector(({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}))
radius = {radius:.6f}

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
direction = mathutils.Vector((1, -1.3, 0.85)).normalized()
cam.location = center + direction * radius * 3.2
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

key = bpy.data.lights.new("key", 'SUN'); key.energy = 4.0
ko = bpy.data.objects.new("key", key); bpy.context.collection.objects.link(ko)
ko.rotation_euler = (math.radians(52), math.radians(18), math.radians(30))
fill = bpy.data.lights.new("fill", 'SUN'); fill.energy = 1.4
fo = bpy.data.objects.new("fill", fill); bpy.context.collection.objects.link(fo)
fo.rotation_euler = (math.radians(-25), math.radians(-40), 0)
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
bpy.context.scene.world = world

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


def render(mesh, out_dir, name="surface", blender_bin=None, samples=32,
           smooth=True, timeout=300) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.png")
    script = render_script(mesh, blend_path, png_path, name=name, samples=samples, smooth=smooth)
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
