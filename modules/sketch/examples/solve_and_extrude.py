#!/usr/bin/env python3
"""Build a parametric sketch by hand, solve it, and extrude the result in Blender.

    python3 examples/solve_and_extrude.py

This is the whole point of the project in one file: a shape described only by
constraints (not coordinates), solved to machine precision, then handed to
Blender to become a solid.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sketchkit import Sketch  # noqa: E402
from sketchkit.blender_export import render, to_bpy_script  # noqa: E402
from sketchkit.constraints import Distance, Horizontal, Perpendicular, Vertical  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
os.makedirs(out, exist_ok=True)

# A 60 x 40 keying plate: four corners, described with no coordinates — just
# "these edges are horizontal / vertical" and "these edges are this long".
s = Sketch()
p0 = s.point(0, 0)
p1 = s.point(55, 3)      # deliberately crooked starting guesses
p2 = s.point(58, 44)
p3 = s.point(-2, 39)

s.fix(p0)                                    # anchor one corner
bottom = s.line(p0, p1)
right = s.line(p1, p2)
s.add(
    Horizontal(p0, p1),                      # bottom edge horizontal
    Vertical(p1, p2),                        # right edge vertical
    Horizontal(p2, p3),                      # top edge horizontal
    Vertical(p3, p0),                        # left edge vertical
    Perpendicular(bottom, right),            # (redundant, to show it's detected)
    Distance(p0, p1, 60.0),                  # width
    Distance(p1, p2, 40.0),                  # height
)

result = s.solve()
print("Solved a plate described only by constraints:\n")
print(f"  {result.describe()}")
print(f"  status: {result.status}"
      + ("  (the Perpendicular is implied by the H/V edges)" if result.redundant else ""))

corners = s.coords([p0, p1, p2, p3])
print("  solved corners:", ", ".join(f"({x:.3f}, {y:.3f})" for x, y in corners))

# Emit the Blender build/extrude script regardless of whether Blender is present.
script_path = os.path.join(out, "plate_build.py")
with open(script_path, "w") as handle:
    handle.write(to_bpy_script(corners, thickness=5.0, name="plate"))
print(f"\nWrote a Blender build+extrude script to {script_path}.")

r = render(corners, out_dir=out, name="example_plate", thickness=5.0)
if r["ran"]:
    print(f"Blender extruded the profile 5 mm and rendered {r['png']}.")
else:
    print("Blender not found — the solved profile and build script above are complete without it.")
