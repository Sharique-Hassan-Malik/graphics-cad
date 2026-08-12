"""Render a terrain slab in 3D with Blender (optional). Faces are coloured by
elevation with a biome ramp — water, sand, grass, rock, snow — and lit by a low
sun so the eroded valleys catch shadow. Cycles on the CPU; no GPU or display."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from .mesh import terrain_mesh


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


_BIOMES = [
    (0.30, (0.14, 0.30, 0.62), 0.1),   # water
    (0.40, (0.72, 0.68, 0.46), 0.7),   # sand
    (0.66, (0.26, 0.48, 0.24), 0.8),   # grass
    (0.82, (0.42, 0.38, 0.34), 0.9),   # rock
    (1.01, (0.92, 0.92, 0.95), 0.6),   # snow
]


def terrain_bpy_script(mesh, height_scale, base, blend_path=None, png_path=None,
                       samples=40, resolution=(800, 560)) -> str:
    v = mesh.vertices
    vtxt = ", ".join(f"({x:.4f},{y:.4f},{z:.4f})" for x, y, z in v)
    ftxt = ", ".join(f"({a},{b},{c})" for a, b, c in mesh.faces)
    center = v.mean(axis=0)
    span = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) or 1.0
    thresholds = [t for t, _, _ in _BIOMES]
    colors = [c for _, c, _ in _BIOMES]
    roughs = [r for _, _, r in _BIOMES]

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
    return f"""import bpy, mathutils, math

bpy.ops.wm.read_factory_settings(use_empty=True)
verts = [{vtxt}]
faces = [{ftxt}]
mesh = bpy.data.meshes.new("terrain")
mesh.from_pydata(verts, [], faces)
mesh.update(); mesh.validate()
obj = bpy.data.objects.new("terrain", mesh)
bpy.context.collection.objects.link(obj)
for poly in mesh.polygons: poly.use_smooth = True

thresholds = {thresholds}
colors = {colors}
roughs = {roughs}
mats = []
for i, col in enumerate(colors):
    m = bpy.data.materials.new(f"biome{{i}}"); m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*col, 1.0)
    b.inputs["Roughness"].default_value = roughs[i]
    obj.data.materials.append(m); mats.append(m)

base = {base}; hscale = {height_scale}
for poly in mesh.polygons:
    z = sum(mesh.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
    frac = (z - base) / hscale
    idx = 0
    while idx < len(thresholds) - 1 and frac > thresholds[idx]:
        idx += 1
    poly.material_index = idx

center = mathutils.Vector(({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}))
span = {span:.4f}
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = center + mathutils.Vector((0.5, -0.95, 0.62)).normalized() * span * 1.15
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam
sun = bpy.data.lights.new("sun", 'SUN'); sun.energy = 3.2
so = bpy.data.objects.new("sun", sun); bpy.context.collection.objects.link(so)
so.rotation_euler = (math.radians(58), math.radians(12), math.radians(40))
world = bpy.data.worlds.new("w"); world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.68, 0.85, 1)
bpy.context.scene.world = world
{render_block}"""


def render(heightmap, out_dir, name="terrain", width=10.0, height_scale=3.0, base=0.0,
           blender_bin=None, samples=40, timeout=400) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    mesh = terrain_mesh(heightmap, width=width, height_scale=height_scale, base=base)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.png")
    script = terrain_bpy_script(mesh, height_scale, base, blend_path, png_path, samples)
    script_path = os.path.join(out_dir, f"{name}_build.py")
    with open(script_path, "w") as handle:
        handle.write(script)

    binary = blender_bin or find_blender()
    out = {"script": script_path, "blend": blend_path, "png": png_path, "ran": False,
           "blender": binary, "mesh": mesh}
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
