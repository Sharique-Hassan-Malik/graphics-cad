"""Command line: check the rig's rigidity, or render the transformation.

    python3 -m transformkit check
    python3 -m transformkit animate --frames 96 --out ./out
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import quat
from .character import optimus


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="transformkit", description="Vehicle↔robot transformer")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify every part stays a rigid body")
    a = sub.add_parser("animate", help="render the transformation")
    a.add_argument("--frames", type=int, default=96)
    a.add_argument("--fps", type=int, default=24)
    a.add_argument("--out", default="./out")
    a.add_argument("--no-gif", action="store_true")
    args = p.parse_args(argv)

    rig = optimus()
    if args.command == "check":
        wo = wd = wdist = 0.0
        for part in rig.parts:
            ref = part.corners(part.vehicle)
            ref_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
            for t in np.linspace(0, 1, 50):
                R = quat.to_matrix(part.pose_at(t).orientation)
                wo = max(wo, np.abs(R @ R.T - np.eye(3)).max())
                wd = max(wd, abs(np.linalg.det(R) - 1))
                c = part.corners(part.pose_at(t))
                d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
                wdist = max(wdist, np.abs(d - ref_d).max())
        print(f"{len(rig.parts)} rigid parts, sampled across the whole transformation:")
        print(f"  max |RᵀR − I|      = {wo:.2e}   (orientation is orthonormal)")
        print(f"  max |det R − 1|    = {wd:.2e}   (proper rotation, no reflection)")
        print(f"  max corner-dist err= {wdist:.2e}   (rigid: no stretch/shear/scale)")
        return 0

    from .animate import render
    print(f"rendering the transformation — {args.frames} frames…")
    result = render(rig, out_dir=args.out, frames=args.frames, fps=args.fps, make_gif=not args.no_gif)
    if result["ran"]:
        print(f"  wrote {result.get('mp4')}" + (f" and {result['gif']}" if "gif" in result else ""))
    else:
        print(f"  {result.get('note', 'Blender did not run')}; script at {result['script']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
