#!/usr/bin/env python3
"""Print an inline-four's timing, then render its animation.

    python3 examples/animate_inline4.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enginekit.animate import render  # noqa: E402
from enginekit.engine import Engine  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
engine = Engine.inline4()

print("Inline-four, firing order 1-3-4-2:\n")
fa = engine.firing_angles()
for cyl in sorted(fa):
    disp = engine.piston_displacements(fa[cyl])[cyl - 1]
    print(f"  cylinder {cyl} fires at {fa[cyl]:>5.0f}° — piston displacement {disp:.1e} (0 = TDC)")

print("\nRendering the animation (96 frames, 2 crank revolutions)…")
result = render(engine, out_dir=out, name="inline4", frames=96, revolutions=2)
if result["ran"]:
    print(f"  wrote {result.get('mp4')}")
    if "gif" in result:
        print(f"  wrote {result['gif']}")
else:
    print(f"  Blender not found — the timing above is verified; script at {result['script']}")
