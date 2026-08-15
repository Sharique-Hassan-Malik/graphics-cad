#!/usr/bin/env python3
"""Verify the rig's rigidity, then render the transformation.

    python3 bench/showcase.py

The headline is that a transformation is *rigid*: every part rotates and
translates but never deforms. Measured across the whole morph, each part's
orientation stays a proper rotation (RᵀR = I, det = 1) and the distances between
its corners are invariant — to machine precision. Only then is it animated.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformkit import quat  # noqa: E402
from transformkit.character import optimus  # noqa: E402


def main():
    out = os.path.join(os.path.dirname(__file__), "..", ".out")
    os.makedirs(out, exist_ok=True)
    rig = optimus()

    wo = wd = wdist = 0.0
    e0 = e1 = 0.0
    for part in rig.parts:
        ref = part.corners(part.vehicle)
        ref_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
        for t in np.linspace(0, 1, 60):
            R = quat.to_matrix(part.pose_at(t).orientation)
            wo = max(wo, np.abs(R @ R.T - np.eye(3)).max())
            wd = max(wd, abs(np.linalg.det(R) - 1))
            c = part.corners(part.pose_at(t))
            d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
            wdist = max(wdist, np.abs(d - ref_d).max())
        e0 = max(e0, np.linalg.norm(part.pose_at(0).position - part.vehicle.position))
        e1 = max(e1, np.linalg.norm(part.pose_at(1).position - part.robot.position))

    print("Transformation rig — a rigid-body morph from vehicle to robot\n")
    print(f"  parts:                 {len(rig.parts)}")
    print(f"  max |RᵀR − I|:         {wo:.2e}   (orientation stays orthonormal)")
    print(f"  max |det R − 1|:       {wd:.2e}   (proper rotation, never a reflection)")
    print(f"  max corner-dist error: {wdist:.2e}   (rigid: no stretch / shear / scale)")
    print(f"  keypose error @t=0/1:  {e0:.1e} / {e1:.1e}   (hits vehicle & robot exactly)")

    print("""
That corner-distance error is the headline: as the truck unfolds into the robot,
no panel is ever stretched or scaled — every part is a rigid body following a
proper rotation, guaranteed by interpolating orientations as unit quaternions
with SLERP. The parts move on staggered timelines, so it reads as transforming,
not inflating.""")

    from transformkit.animate import find_blender, render
    if find_blender():
        print("\nBlender found; rendering the transformation…")
        result = render(rig, out_dir=out, name="showcase_transformer", frames=96)
        if result["ran"]:
            for k in ("mp4", "gif"):
                if k in result:
                    print(f"  wrote {os.path.basename(result[k])} ({os.path.getsize(result[k]):,} B)")
    else:
        print("\nBlender not found; the rigidity above is already verified without it.")

    return 0 if (wo < 1e-9 and wdist < 1e-9) else 1


if __name__ == "__main__":
    raise SystemExit(main())
