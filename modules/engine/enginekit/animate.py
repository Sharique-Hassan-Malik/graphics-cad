"""Drive a Blender animation of the engine straight from the kinematic core.

Nothing here is hand-keyframed. For every frame the crank turns a little further,
and the core computes exactly where each piston, connecting rod, crank pin and
valve must be; those numbers are written as Blender keyframes. So the animation is
a *visualisation of the equations* — if the slider-crank maths were wrong the
pistons would visibly miss the rods, and they do not.

Rendered with Cycles: real metal materials (cast-iron block, steel shaft, alloy
pistons), bevelled edges, soft area lighting and a glowing combustion flash, then
assembled into an MP4 and a GIF with ffmpeg. Blender is optional — the motion is
fully defined and tested without it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from .engine import Engine

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


def sample_motion(engine: Engine, frames: int, revolutions: float):
    """Per-frame kinematics for an inline engine, ready to inline into a script."""
    n = engine.n_cylinders
    r, l = engine.crank_radius, engine.rod_length
    spacing = 1.5 * engine.bore
    x = [(i - (n - 1) / 2.0) * spacing for i in range(n)]
    head_z = (r + l) + 0.35

    crank_deg = np.linspace(0.0, 360.0 * revolutions, frames, endpoint=False)
    piston_z = np.zeros((frames, n))
    rod_mid = np.zeros((frames, n, 2))
    rod_ang = np.zeros((frames, n))
    valve_z = np.zeros((frames, n, 2))
    spark = np.zeros((frames, n))

    for f, cd in enumerate(crank_deg):
        pz = _piston_pos(engine, cd)
        along, perp = engine.crank_pins(cd)
        piston_z[f] = pz
        for i in range(n):
            az, ay = along[i], perp[i]
            by, bz = 0.0, pz[i]
            rod_mid[f, i] = ((ay + by) / 2, (az + bz) / 2)
            rod_ang[f, i] = np.arctan2(ay - by, az - bz)
        lift = engine.valve_lift(cd)
        valve_z[f, :, :] = head_z + 0.28 + (-0.16) * lift
        fire = engine.firing_cylinder(cd)
        if fire is not None:
            spark[f, fire - 1] = 1.0

    return {
        "n": n, "x": x, "r": r, "l": l, "bore": engine.bore, "head_z": head_z,
        "throw_offsets": [float(o) for o in engine.throw_offsets],
        "crank_deg": crank_deg.tolist(),
        "piston_z": piston_z.tolist(),
        "rod_mid": rod_mid.tolist(),
        "rod_ang": rod_ang.tolist(),
        "valve_z": valve_z.tolist(),
        "spark": spark.tolist(),
        "spacing": spacing,
    }


def _piston_pos(engine, crank_deg):
    from . import slider_crank as sc
    theta = np.radians(crank_deg) + np.radians(engine.throw_offsets)
    return sc.piston_position(theta, engine.crank_radius, engine.rod_length)




def build_script(data, frame_dir, fps=24, resolution=(640, 480), samples=32) -> str:
    return f"""import bpy, math, mathutils, bmesh

D = {data!r}
n = D['n']; x = D['x']; r = D['r']; l = D['l']; bore = D['bore']; head_z = D['head_z']
frames = len(D['crank_deg'])

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
coll = bpy.context.collection

# -- materials -------------------------------------------------------------
def pbr(name, base, metallic, rough, emis=None, es=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if emis is not None:
        b.inputs["Emission Color"].default_value = (*emis, 1)
        b.inputs["Emission Strength"].default_value = es
    return m

IRON  = pbr("iron",  (0.075, 0.078, 0.085), 0.7, 0.55)
STEEL = pbr("steel", (0.52, 0.54, 0.58), 1.0, 0.28)
ALLOY = pbr("alloy", (0.80, 0.81, 0.84), 1.0, 0.18)
BRASS = pbr("brass", (0.71, 0.53, 0.20), 1.0, 0.30)
VALVE = pbr("valvemat", (0.72, 0.73, 0.78), 1.0, 0.14)
EMBER = pbr("ember", (0.02, 0.02, 0.02), 0.0, 0.4, emis=(1.0, 0.42, 0.10), es=18.0)

def finish(o, bevel=0.02, smooth=False):
    mod = o.modifiers.new("bev", 'BEVEL'); mod.width = bevel; mod.segments = 2
    if smooth:
        for p in o.data.polygons: p.use_smooth = True

def box(name, size, loc, mat, bevel=0.02):
    m = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=bm.verts, vec=size); bm.to_mesh(m); bm.free()
    o = bpy.data.objects.new(name, m); coll.objects.link(o); o.location = loc
    o.data.materials.append(mat); finish(o, bevel); return o

def cyl(name, radius, depth, loc, mat, axis='Z'):
    m = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=40, radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(m); bm.free()
    o = bpy.data.objects.new(name, m); coll.objects.link(o); o.location = loc
    if axis == 'X': o.rotation_euler = (0, math.radians(90), 0)
    o.data.materials.append(mat)
    for p in o.data.polygons: p.use_smooth = True
    finish(o, 0.0, smooth=True)
    return o

# -- static frame ----------------------------------------------------------
base = box("base", (n*1.5*bore+bore, bore*2.2, 0.4), (0, 0, -r-0.3), IRON, bevel=0.05)
crank_axis = bpy.data.objects.new("crank", None); coll.objects.link(crank_axis)
journal = cyl("journal", r*0.35, n*1.5*bore+bore, (0,0,0), STEEL, axis='X'); journal.parent = crank_axis
fly = cyl("flywheel", r*1.6, 0.25, (x[0]-1.2*bore, 0, 0), STEEL, axis='X'); fly.parent = crank_axis

pistons=[]; rods=[]; valves=[]; sparks=[]
for i in range(n):
    box(f"head{{i}}", (bore*1.15, bore*1.15, 0.5), (x[i], 0, head_z+0.35), IRON, bevel=0.04)
    pistons.append(cyl(f"piston{{i}}", bore*0.45, bore*0.7, (x[i],0,0), ALLOY))
    rods.append(box(f"rod{{i}}", (bore*0.16, bore*0.16, l), (x[i],0,0), STEEL, bevel=0.015))
    off = math.radians(D['throw_offsets'][i])
    pin = cyl(f"pin{{i}}", r*0.18, bore*0.5, (x[i], r*math.sin(off), r*math.cos(off)), BRASS, axis='X')
    pin.parent = crank_axis
    valves.append(cyl(f"vin{{i}}", bore*0.12, 0.5, (x[i]-bore*0.3,0,head_z), VALVE))
    valves.append(cyl(f"vex{{i}}", bore*0.12, 0.5, (x[i]+bore*0.3,0,head_z), VALVE))
    sparks.append(cyl(f"spark{{i}}", bore*0.24, 0.05, (x[i],0,head_z-0.15), EMBER))

# -- keyframe every frame from the sampled kinematics ----------------------
for f in range(frames):
    scene.frame_set(f+1)
    crank_axis.rotation_euler = (math.radians(D['crank_deg'][f]), 0, 0)
    crank_axis.keyframe_insert("rotation_euler")
    for i in range(n):
        pistons[i].location = (x[i], 0, D['piston_z'][f][i]); pistons[i].keyframe_insert("location")
        my, mz = D['rod_mid'][f][i]
        rods[i].location = (x[i], my, mz); rods[i].rotation_euler = (D['rod_ang'][f][i], 0, 0)
        rods[i].keyframe_insert("location"); rods[i].keyframe_insert("rotation_euler")
        valves[2*i].location = (x[i]-bore*0.3, 0, D['valve_z'][f][i][0]); valves[2*i].keyframe_insert("location")
        valves[2*i+1].location = (x[i]+bore*0.3, 0, D['valve_z'][f][i][1]); valves[2*i+1].keyframe_insert("location")
        s = 1.0 + 1.1*D['spark'][f][i]
        sparks[i].scale = (s, s, 1.0 + 4.0*D['spark'][f][i]); sparks[i].keyframe_insert("scale")

for o in bpy.data.objects:
    if o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

# -- ground, camera, lights ------------------------------------------------
gm = bpy.data.meshes.new("ground"); bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=40); bm.to_mesh(gm); bm.free()
ground = bpy.data.objects.new("ground", gm); coll.objects.link(ground); ground.location = (0,0,-r-0.5)
ground.data.materials.append(pbr("floor", (0.11,0.12,0.14), 0.0, 0.35))

cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); coll.objects.link(cam)
span = n*1.5*bore + 3
cam.location = (span*0.62, -span*1.05, span*0.55)
cam.rotation_euler = (mathutils.Vector((0,0,r*0.4)) - cam.location).to_track_quat('-Z','Y').to_euler()
scene.camera = cam

def area(loc, energy, size, color=(1,1,1)):
    d = bpy.data.lights.new("a", 'AREA'); d.energy = energy; d.size = size; d.color = color
    o = bpy.data.objects.new("a", d); coll.objects.link(o); o.location = loc
    o.rotation_euler = (mathutils.Vector((0,0,0)) - mathutils.Vector(loc)).to_track_quat('-Z','Y').to_euler()
    return o
area((span, -span*0.6, span), 4000, 8, (1.0,0.97,0.9))     # key
area((-span, -span*0.3, span*0.7), 1500, 10, (0.7,0.8,1.0))  # cool fill
area((0, span, span*0.6), 1800, 8, (1.0,0.85,0.7))           # warm rim

world = bpy.data.worlds.new("w"); world.use_nodes = True; scene.world = world
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05,0.055,0.07,1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.4

# -- Cycles + denoise + glare bloom ---------------------------------------
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = {samples}
scene.render.use_persistent_data = True   # reuse BVH between frames — big anim speedup
try:
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
scene.use_nodes = True
nt = scene.node_tree; nt.nodes.clear()
rl = nt.nodes.new('CompositorNodeRLayers'); comp = nt.nodes.new('CompositorNodeComposite')
gl = nt.nodes.new('CompositorNodeGlare'); gl.glare_type = 'FOG_GLOW'; gl.quality = 'MEDIUM'; gl.threshold = 1.2
nt.links.new(rl.outputs['Image'], gl.inputs['Image']); nt.links.new(gl.outputs['Image'], comp.inputs['Image'])

scene.frame_start = 1; scene.frame_end = frames
scene.render.fps = {fps}
scene.render.resolution_x = {resolution[0]}; scene.render.resolution_y = {resolution[1]}
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = {os.path.join(frame_dir, "f_")!r}
bpy.ops.render.render(animation=True)
print("RENDER_DONE")
"""


def render(engine: Engine, out_dir: str, name: str = "engine", frames: int = 80,
           revolutions: float = 2.0, fps: int = 24, resolution=(640, 480), samples: int = 32,
           blender_bin=None, make_gif: bool = True, timeout: int = 3000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"{name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    data = sample_motion(engine, frames, revolutions)
    script = build_script(data, frame_dir, fps=fps, resolution=resolution, samples=samples)
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
        subprocess.run([ff, "-y", "-framerate", str(fps), "-i", pattern,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", mp4], capture_output=True)
        result["mp4"] = mp4
        if make_gif:
            gif = os.path.join(out_dir, f"{name}.gif")
            palette = os.path.join(out_dir, "_pal.png")
            subprocess.run([ff, "-y", "-i", pattern, "-vf", "fps=%d,scale=520:-1:flags=lanczos,palettegen" % fps, palette], capture_output=True)
            subprocess.run([ff, "-y", "-framerate", str(fps), "-i", pattern, "-i", palette,
                            "-lavfi", "fps=%d,scale=520:-1:flags=lanczos [x]; [x][1:v] paletteuse" % fps, gif],
                           capture_output=True)
            result["gif"] = gif
    return result
