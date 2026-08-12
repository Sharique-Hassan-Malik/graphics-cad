#!/usr/bin/env python3
"""Model a six-sided die from scratch with the SDF algebra, mesh it, verify it,
and render it in Blender.

    python3 examples/model_a_die.py

The die is one arithmetic expression: a rounded cube, minus twenty-one little
spheres for the pips. No mesh editing, no booleans-on-triangles — the hard part
(cutting 21 holes and staying a watertight solid) is free when you model with
distance fields and mesh the result with marching tetrahedra.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdfkit.blender_export import mesh_to_bpy, render  # noqa: E402
from sdfkit.marching import triangulate  # noqa: E402
from sdfkit.scenes import die  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
os.makedirs(out, exist_ok=True)

sdf, resolution, expected_chi = die()
print(f"Meshing a die (rounded cube minus 21 pip-spheres) at resolution {resolution}…\n")
mesh = triangulate(sdf, resolution=resolution)

print(f"  watertight ............ {mesh.is_watertight()}")
print(f"  consistently oriented . {mesh.is_consistently_oriented()}")
print(f"  edge-manifold ......... {mesh.is_edge_manifold()}")
print(f"  Euler characteristic .. {mesh.euler_characteristic()}  "
      f"(expected {expected_chi}: still one solid ball after 21 cuts)")
print(f"  volume ................ {mesh.volume():.4f}")
print(f"  faces ................. {mesh.face_count:,}")

stl = mesh.save_stl(os.path.join(out, "die.stl"))
print(f"\nWrote {stl} ({os.path.getsize(stl):,} bytes).")

script_path = os.path.join(out, "die_build.py")
with open(script_path, "w") as handle:
    handle.write(mesh_to_bpy(mesh, name="die"))
print(f"Wrote a Blender build script to {script_path}.")

result = render(mesh, out_dir=out, name="example_die")
if result["ran"]:
    print(f"Blender rendered {result['png']}.")
else:
    print("Blender not found — the STL and build script above are complete without it.")
