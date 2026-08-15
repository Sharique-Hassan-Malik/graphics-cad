"""Drive a Blender animation of the vehicle↔robot transformation from the rig.

Every frame asks the rig for each part's pose (position + unit quaternion) at the
current morph time and writes it as a Blender keyframe — orientations go in as
quaternions, so there is no Euler-order ambiguity between the maths and Blender.
The morph time follows a raised cosine 0→1→0, so the clip loops.

Rendered with Cycles: automotive-paint and chrome materials, bevelled armour
panels, a softly reflective floor and three-point lighting. Blender is optional;
the motion is defined and tested here.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from .rig import Rig

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




def sample(rig: Rig, frames: int):
    names = [p.name for p in rig.parts]
    sizes = [list(map(float, p.size)) for p in rig.parts]
    colors = [tuple(p.color) for p in rig.parts]
    per_frame = []
    for f in range(frames):
        t = 0.5 - 0.5 * np.cos(2 * np.pi * f / frames)
        row = []
        for p in rig.parts:
            pose = p.pose_at(float(t))
            row.append([*map(float, pose.position), *map(float, pose.orientation)])
        per_frame.append(row)
    return {"names": names, "sizes": sizes, "colors": colors, "frames": per_frame}


def build_script(data, frame_dir, resolution=(640, 520), samples=24) -> str:
    return f"""import bpy, bmesh, math, mathutils

D = {data!r}
names = D['names']; sizes = D['sizes']; colors = D['colors']; frames = D['frames']
nf = len(frames)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene; coll = bpy.context.collection

def material(col):
    r, g, b = col
    m = bpy.data.materials.new("m"); m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    sat = max(r, g, b) - min(r, g, b); light = max(r, g, b)
    if sat < 0.12 and light > 0.5:                 # bare metal → chrome
        bs.inputs["Base Color"].default_value = (r, g, b, 1)
        bs.inputs["Metallic"].default_value = 1.0
        bs.inputs["Roughness"].default_value = 0.12
    else:                                          # automotive paint
        bs.inputs["Base Color"].default_value = (r, g, b, 1)
        bs.inputs["Metallic"].default_value = 0.55
        bs.inputs["Roughness"].default_value = 0.30
        try: bs.inputs["Coat Weight"].default_value = 0.6
        except Exception: pass
    return m

objs = []
for i, name in enumerate(names):
    me = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=bm.verts, vec=[s for s in sizes[i]]); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); coll.objects.link(o)
    o.data.materials.append(material(colors[i]))
    mod = o.modifiers.new("bev", 'BEVEL'); mod.width = 0.03; mod.segments = 2
    o.rotation_mode = 'QUATERNION'
    objs.append(o)

for f in range(nf):
    scene.frame_set(f + 1)
    for i, o in enumerate(objs):
        px, py, pz, qw, qx, qy, qz = frames[f][i]
        o.location = (px, py, pz); o.rotation_quaternion = (qw, qx, qy, qz)
        o.keyframe_insert("location"); o.keyframe_insert("rotation_quaternion")

# reflective ground
gm = bpy.data.meshes.new("ground"); bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=60); bm.to_mesh(gm); bm.free()
g = bpy.data.objects.new("ground", gm); coll.objects.link(g); g.location = (0, 0, -0.02)
gmat = bpy.data.materials.new("floor"); gmat.use_nodes = True
gb = gmat.node_tree.nodes["Principled BSDF"]
gb.inputs["Base Color"].default_value = (0.05, 0.055, 0.07, 1); gb.inputs["Roughness"].default_value = 0.16
gb.inputs["Metallic"].default_value = 0.4
g.data.materials.append(gmat)

# orbiting camera
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); coll.objects.link(cam)
pivot = bpy.data.objects.new("pivot", None); coll.objects.link(pivot); pivot.location = (0, 0, 3.0)
cam.parent = pivot; cam.location = (0, -17, 2.4)
cam.rotation_euler = (mathutils.Vector((0,0,3.0)) - (pivot.location + cam.location)).to_track_quat('-Z','Y').to_euler()
scene.camera = cam
for f in range(nf):
    scene.frame_set(f + 1)
    pivot.rotation_euler = (0, 0, math.radians(38) * math.sin(2*math.pi*f/nf))
    pivot.keyframe_insert("rotation_euler")

def area(loc, energy, size, color=(1,1,1)):
    d = bpy.data.lights.new("a", 'AREA'); d.energy = energy; d.size = size; d.color = color
    o = bpy.data.objects.new("a", d); coll.objects.link(o); o.location = loc
    o.rotation_euler = (mathutils.Vector((0,0,3)) - mathutils.Vector(loc)).to_track_quat('-Z','Y').to_euler()
area((10, -8, 14), 5000, 10, (1.0,0.97,0.92))
area((-11, -5, 9), 2000, 12, (0.7,0.8,1.0))
area((0, 12, 8), 2400, 10, (1.0,0.9,0.8))

world = bpy.data.worlds.new("w"); world.use_nodes = True; scene.world = world
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05,0.06,0.08,1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.5

scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = {samples}
scene.render.use_persistent_data = True
try:
    scene.cycles.use_denoising = True; scene.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
scene.frame_start = 1; scene.frame_end = nf
scene.render.resolution_x = {resolution[0]}; scene.render.resolution_y = {resolution[1]}
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = {os.path.join(frame_dir, "f_")!r}
bpy.ops.render.render(animation=True)
print("RENDER_DONE")
"""


def render(rig: Rig, out_dir: str, name: str = "transformer", frames: int = 72,
           fps: int = 24, resolution=(640, 520), samples: int = 24, blender_bin=None,
           make_gif: bool = True, timeout: int = 3000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"{name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    data = sample(rig, frames)
    script = build_script(data, frame_dir, resolution=resolution, samples=samples)
    script_path = os.path.join(out_dir, f"{name}_build.py")
    with open(script_path, "w") as h:
        h.write(script)

    binary = blender_bin or find_blender()
    result = {"script": script_path, "frame_dir": frame_dir, "ran": False, "blender": binary}
    if binary is None:
        result["note"] = "no Blender binary found; run the emitted script yourself"
        return result

    proc = subprocess.run([binary, "--background", "--factory-startup", "--python", script_path],
                          capture_output=True, text=True, timeout=timeout)
    result["ran"] = "RENDER_DONE" in proc.stdout
    if not result["ran"]:
        result["stderr_tail"] = proc.stderr.strip().splitlines()[-10:]
        return result

    ff = shutil.which("ffmpeg")
    pattern = os.path.join(frame_dir, "f_%04d.png")
    if ff:
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        subprocess.run([ff, "-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "19", mp4], capture_output=True)
        result["mp4"] = mp4
        if make_gif:
            gif = os.path.join(out_dir, f"{name}.gif")
            palette = os.path.join(out_dir, "_pal.png")
            subprocess.run([ff, "-y", "-i", pattern, "-vf",
                            "fps=%d,scale=480:-1:flags=lanczos,palettegen" % fps, palette], capture_output=True)
            subprocess.run([ff, "-y", "-framerate", str(fps), "-i", pattern, "-i", palette, "-lavfi",
                            "fps=%d,scale=480:-1:flags=lanczos [x]; [x][1:v] paletteuse" % fps, gif],
                           capture_output=True)
            result["gif"] = gif
    return result
