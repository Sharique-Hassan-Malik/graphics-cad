"""Render a WFC island in 3D with Blender (optional, as everywhere in this repo).

`island_bpy_script()` rebuilds the island mesh with `from_pydata` and gives land
and water their own materials by face height. `render()` runs it headless with
Cycles on the CPU. With no Blender present you still get the 2D PNG and the mesh;
this is only the 3D view.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from .model3d import island_mesh

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




def island_bpy_script(verts, faces, blend_path=None, png_path=None, samples=32,
                      land_z=1.0, water_z=0.28, resolution=(760, 560)) -> str:
    vtxt = ", ".join(f"({x:.4f},{y:.4f},{z:.4f})" for x, y, z in verts)
    ftxt = ", ".join(f"({a},{b},{c})" for a, b, c in faces)
    center = verts.mean(axis=0)
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) or 1.0
    render_block = ""
    if blend_path and png_path:
        render_block = f"""
scene = bpy.context.scene
scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = {samples}
scene.render.resolution_x = {resolution[0]}; scene.render.resolution_y = {resolution[1]}
scene.render.filepath = {png_path!r}
bpy.ops.wm.save_as_mainfile(filepath={blend_path!r})
bpy.ops.render.render(write_still=True)
print("RENDER_DONE", {png_path!r})
"""
    return f"""import bpy, bmesh, mathutils, math

bpy.ops.wm.read_factory_settings(use_empty=True)
verts = [{vtxt}]
faces = [{ftxt}]
mesh = bpy.data.meshes.new("island")
mesh.from_pydata(verts, [], faces)
mesh.update(); mesh.validate()
obj = bpy.data.objects.new("island", mesh)
bpy.context.collection.objects.link(obj)

def mat(name, rgb, rough):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0); b.inputs["Roughness"].default_value = rough
    return m
land = mat("land", (0.30, 0.55, 0.26), 0.7)
water = mat("water", (0.20, 0.42, 0.78), 0.15)
obj.data.materials.append(land); obj.data.materials.append(water)
mid = ({land_z} + {water_z}) / 2
for poly in mesh.polygons:
    z = sum(mesh.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
    poly.material_index = 0 if z > mid else 1

center = mathutils.Vector(({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}))
span = {span:.4f}
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = center + mathutils.Vector((0.55, -0.9, 0.8)).normalized() * span * 1.1
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam
key = bpy.data.lights.new("key", 'SUN'); key.energy = 4.0
ko = bpy.data.objects.new("key", key); bpy.context.collection.objects.link(ko)
ko.rotation_euler = (math.radians(55), math.radians(15), math.radians(35))
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.72, 0.85, 1)
bpy.context.scene.world = world
{render_block}"""


def render(result, out_dir, name="island", blender_bin=None, samples=32, timeout=300) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    verts, faces = island_mesh(result)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.png")
    script = island_bpy_script(verts, faces, blend_path, png_path, samples)
    script_path = os.path.join(out_dir, f"{name}_build.py")
    with open(script_path, "w") as handle:
        handle.write(script)

    binary = blender_bin or find_blender()
    out = {"script": script_path, "blend": blend_path, "png": png_path, "ran": False, "blender": binary}
    if binary is None:
        out["note"] = "no Blender binary found; run the emitted script yourself"
        return out
    proc = subprocess.run(
        [binary, "--background", "--factory-startup", "--python", script_path],
        capture_output=True, text=True, timeout=timeout,
    )
    out["ran"] = "RENDER_DONE" in proc.stdout
    out["png_written"] = os.path.exists(png_path)
    out["blend_written"] = os.path.exists(blend_path)
    if not out["ran"]:
        out["stderr_tail"] = proc.stderr.strip().splitlines()[-8:] if proc.stderr else []
    return out
