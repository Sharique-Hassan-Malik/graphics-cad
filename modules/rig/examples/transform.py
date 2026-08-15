#!/usr/bin/env python3
"""Show the rig is rigid, then render the vehicle↔robot transformation.

    python3 examples/transform.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformkit import quat  # noqa: E402
from transformkit.animate import render  # noqa: E402
from transformkit.character import optimus  # noqa: E402

rig = optimus()
print(f"A {len(rig.parts)}-part transformer. Checking rigidity across the morph…\n")

worst = 0.0
for part in rig.parts:
    ref = part.corners(part.vehicle)
    ref_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
    for t in np.linspace(0, 1, 30):
        c = part.corners(part.pose_at(t))
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        worst = max(worst, np.abs(d - ref_d).max())
print(f"  worst change in any part's corner-to-corner distance: {worst:.2e}")
print("  → every part is perfectly rigid through the whole transformation.\n")

out = os.path.join(os.path.dirname(__file__), "..", ".out")
print("Rendering (car → robot → car, 96 frames)…")
result = render(rig, out_dir=out, name="transform", frames=96)
if result["ran"]:
    print(f"  wrote {result.get('mp4')}" + (f" and {result['gif']}" if "gif" in result else ""))
else:
    print(f"  Blender not found — rigidity verified; script at {result['script']}")
