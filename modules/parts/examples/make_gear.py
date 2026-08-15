#!/usr/bin/env python3
"""Generate a bored spur gear, verify it, export an STL, and emit a Blender
build script you can run to view or render it.

    python3 examples/make_gear.py

This uses no Blender and no CAD library — everything below is computed from the
mesh's own arrays. If Blender is on PATH (or $BLENDER_BIN is set) the last step
also produces a .blend and a rendered PNG.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partkit.blender_export import mesh_to_bpy, render  # noqa: E402
from partkit.gears import involute_deviation, measure_gear, spur_gear  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
os.makedirs(out, exist_ok=True)

# A 24-tooth, module-3 gear, 8 mm thick, with a 10 mm bore.
gear = spur_gear(module=3.0, teeth=24, thickness=8.0, bore_diameter=10.0)

print("Generated a module-3, 24-tooth gear with a 10 mm bore.\n")
print(f"  watertight ............ {gear.is_watertight()}")
print(f"  consistently oriented . {gear.is_consistently_oriented()}")
print(f"  Euler characteristic .. {gear.euler_characteristic()}  (0 = has a through-hole)")
print(f"  volume ................ {gear.volume():.2f} mm^3")

measured = measure_gear(gear)
print(f"  measured back ......... {measured['teeth']} teeth, "
      f"module {measured['module']:.4f}, outer radius {measured['outer_radius']:.3f} mm")

dev = involute_deviation(gear, module=3.0, teeth=24)
print(f"  involute deviation .... {dev['max_deviation']:.2e} mm max "
      f"(over {dev['flank_points']} flank vertices)")

stl = gear.save_stl(os.path.join(out, "example_gear.stl"))
print(f"\nWrote {stl} ({os.path.getsize(stl):,} bytes).")

script_path = os.path.join(out, "example_gear_build.py")
with open(script_path, "w") as handle:
    handle.write(mesh_to_bpy(gear, name="gear"))
print(f"Wrote a Blender build script to {script_path}.")
print("  Run it with:  blender --background --python", script_path)

result = render(gear, out_dir=out, name="example_gear")
if result["ran"]:
    print(f"\nBlender rendered {result['png']}.")
else:
    print("\nBlender not found — the STL and build script above are complete without it.")
