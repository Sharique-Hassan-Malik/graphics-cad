"""Command line: solve a parametric sketch, print its degrees-of-freedom
verdict, and optionally extrude the solved profile in Blender.

    python3 -m sketchkit rectangle --width 40 --height 25
    python3 -m sketchkit polygon --sides 6 --radius 20 --render ./out --thickness 6
    python3 -m sketchkit bracket --length 40 --height 30 --wall 8 --render ./out
    python3 -m sketchkit descartes --r1 3 --r2 4 --r3 5   # nonlinear, checked vs theory
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import demos
from .blender_export import render


def _emit(sketch, points, label, args):
    result = sketch.solve()
    print(f"{label}")
    print(f"  {result.describe()}")
    coords = sketch.coords(points)
    pretty = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in coords)
    print(f"  solved profile: {pretty}")
    if args.render:
        out = render(coords, out_dir=args.render, name=label.split()[0],
                     thickness=args.thickness)
        if out["ran"]:
            print(f"  extruded {args.thickness} mm and rendered → {out['png']}")
        else:
            print(f"  {out.get('note', 'Blender did not run')}; script at {out['script']}")
    return 0 if result.status == "fully-constrained" else 0  # status is informational


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sketchkit", description="2D constraint solver")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rectangle", help="a rectangle from H/V edges + two dimensions")
    r.add_argument("--width", type=float, default=40.0)
    r.add_argument("--height", type=float, default=25.0)

    p = sub.add_parser("polygon", help="a regular n-gon from a centre, radius and angles")
    p.add_argument("--sides", type=int, default=6)
    p.add_argument("--radius", type=float, default=20.0)

    b = sub.add_parser("bracket", help="an L-shaped profile (one reflex corner)")
    b.add_argument("--length", type=float, default=40.0)
    b.add_argument("--height", type=float, default=30.0)
    b.add_argument("--wall", type=float, default=8.0)

    d = sub.add_parser("descartes", help="tangent circles, checked vs Descartes' theorem")
    d.add_argument("--r1", type=float, default=3.0)
    d.add_argument("--r2", type=float, default=4.0)
    d.add_argument("--r3", type=float, default=5.0)

    for sp in (r, p, b):
        sp.add_argument("--render", metavar="DIR", default=None, help="extrude and render in Blender")
        sp.add_argument("--thickness", type=float, default=6.0, help="extrusion depth (mm)")

    args = parser.parse_args(argv)

    if args.command == "rectangle":
        return _emit(*demos.rectangle(args.width, args.height), args)
    if args.command == "polygon":
        return _emit(*demos.regular_polygon(args.sides, args.radius), args)
    if args.command == "bracket":
        return _emit(*demos.l_bracket(args.length, args.height, args.wall), args)
    if args.command == "descartes":
        sketch, circles = demos.tangent_circles(args.r1, args.r2, args.r3)
        result = sketch.solve()
        r4 = sketch.radius(circles[3])
        predicted = demos.descartes_radius(args.r1, args.r2, args.r3)
        print("Three mutually tangent circles + a fourth in the gap")
        print(f"  {result.describe()}")
        for i, c in enumerate(circles):
            cx, cy = sketch.xy(c.center)
            print(f"  circle {i + 1}: centre ({cx:+.4f}, {cy:+.4f}), radius {sketch.radius(c):.6f}")
        print(f"  solved inner radius   {r4:.12f} mm")
        print(f"  Descartes' theorem    {predicted:.12f} mm")
        print(f"  difference            {abs(r4 - predicted):.2e} mm")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
