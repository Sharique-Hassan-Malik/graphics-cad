"""Command line: inspect an engine's timing, or animate it in Blender.

    python3 -m enginekit timing --engine inline4
    python3 -m enginekit animate --engine inline4 --frames 96 --out ./out
    python3 -m enginekit animate --engine v8 --frames 120 --revolutions 2 --out ./out
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .engine import Engine

_PRESETS = {"inline4": Engine.inline4, "inline6": Engine.inline6, "v8": Engine.v8}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="enginekit", description="Four-stroke engine kinematics")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("timing", help="print firing order, throws, and TDC check")
    t.add_argument("--engine", choices=list(_PRESETS), default="inline4")

    a = sub.add_parser("animate", help="render a Blender animation")
    a.add_argument("--engine", choices=list(_PRESETS), default="inline4")
    a.add_argument("--frames", type=int, default=96)
    a.add_argument("--revolutions", type=float, default=2.0)
    a.add_argument("--fps", type=int, default=24)
    a.add_argument("--out", default="./out")
    a.add_argument("--no-gif", action="store_true")

    args = p.parse_args(argv)
    engine = _PRESETS[args.engine]()

    if args.command == "timing":
        fa = engine.firing_angles()
        print(f"{args.engine}: {engine.n_cylinders} cylinders, {engine.layout}")
        print(f"  firing order:   {engine.firing_order}")
        print(f"  crank throws:   {[round(o) for o in engine.throw_offsets]}°")
        print(f"  firing every:   {engine.firing_interval:.0f}° of crank")
        print("  cyl  fires@   piston-disp-at-fire (0 = TDC)")
        for cyl in sorted(fa):
            disp = engine.piston_displacements(fa[cyl])[cyl - 1]
            print(f"   {cyl:>2}   {fa[cyl]:>5.0f}°   {disp:.2e}")
        spacing = np.diff(sorted(fa.values()) + [sorted(fa.values())[0] + 720])
        print(f"  even firing:    {np.allclose(spacing, engine.firing_interval)}")
        return 0

    from .animate import render
    print(f"animating {args.engine} — {args.frames} frames over {args.revolutions} revolutions…")
    result = render(engine, out_dir=args.out, name=args.engine, frames=args.frames,
                    revolutions=args.revolutions, fps=args.fps, make_gif=not args.no_gif)
    if result["ran"]:
        print(f"  wrote {result.get('mp4', '(frames)')}")
        if "gif" in result:
            print(f"  wrote {result['gif']}")
    else:
        print(f"  {result.get('note', 'Blender did not run')}; script at {result['script']}")
        if "stderr_tail" in result:
            print("  " + "\n  ".join(result["stderr_tail"]))
    return 0 if result["ran"] or "note" in result else 1


if __name__ == "__main__":
    sys.exit(main())
