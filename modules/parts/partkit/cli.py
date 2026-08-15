"""Command line: generate a part, verify it, export STL, optionally render.

    python3 -m partkit gear   --module 2 --teeth 20 --thickness 6 --bore 8 --stl gear.stl --verify
    python3 -m partkit plate  --width 50 --depth 30 --thickness 4 --radius 5 --stl plate.stl
    python3 -m partkit washer --outer 20 --inner 10 --thickness 3 --stl washer.stl
    python3 -m partkit gear   --module 3 --teeth 24 --bore 10 --render ./out   # runs Blender if present
"""

from __future__ import annotations

import argparse
import sys

from . import parts
from .blender_export import render
from .gears import involute_deviation, measure_gear, spur_gear
from .mesh import Mesh


def _report(mesh: Mesh, gear_spec=None) -> None:
    print(f"  vertices {mesh.vertex_count}, faces {mesh.face_count}")
    print(f"  watertight: {mesh.is_watertight()}   oriented: {mesh.is_consistently_oriented()}"
          f"   Euler χ: {mesh.euler_characteristic()}")
    print(f"  volume: {mesh.volume():.3f} mm³   bounds: {mesh.size().round(3).tolist()} mm")
    if gear_spec:
        module, teeth, pa = gear_spec
        dev = involute_deviation(mesh, module, teeth, pa)
        m = measure_gear(mesh)
        print(f"  measured: {m['teeth']} teeth, module {m['module']:.4f}, "
              f"outer Ø {2 * m['outer_radius']:.3f} mm")
        print(f"  involute deviation: max {dev['max_deviation']:.2e} mm over {dev['flank_points']} flank points")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="partkit", description="Parametric CAD parts")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gear", help="an involute spur gear")
    g.add_argument("--module", type=float, required=True)
    g.add_argument("--teeth", type=int, required=True)
    g.add_argument("--thickness", type=float, required=True)
    g.add_argument("--bore", type=float, default=0.0, help="bore diameter (0 = solid)")
    g.add_argument("--pa", type=float, default=20.0, help="pressure angle, degrees")

    p = sub.add_parser("plate", help="a rectangular plate")
    p.add_argument("--width", type=float, required=True)
    p.add_argument("--depth", type=float, required=True)
    p.add_argument("--thickness", type=float, required=True)
    p.add_argument("--radius", type=float, default=0.0, help="corner radius")

    w = sub.add_parser("washer", help="a flat annulus")
    w.add_argument("--outer", type=float, required=True)
    w.add_argument("--inner", type=float, required=True)
    w.add_argument("--thickness", type=float, required=True)

    b = sub.add_parser("bracket", help="an L-bracket")
    b.add_argument("--length", type=float, required=True)
    b.add_argument("--height", type=float, required=True)
    b.add_argument("--thickness", type=float, required=True)
    b.add_argument("--wall", type=float, required=True)

    for p_ in (g, p, w, b):
        p_.add_argument("--stl", default=None, help="write a binary STL here")
        p_.add_argument("--obj", default=None, help="write a Wavefront OBJ here")
        p_.add_argument("--render", default=None, metavar="DIR", help="render with Blender into DIR")
        p_.add_argument("--verify", action="store_true", help="print topology + geometry checks")

    args = parser.parse_args(argv)

    gear_spec = None
    if args.command == "gear":
        mesh = spur_gear(args.module, args.teeth, args.thickness, args.pa, args.bore)
        gear_spec = (args.module, args.teeth, args.pa)
        name = f"gear_m{args.module}_z{args.teeth}"
    elif args.command == "plate":
        mesh = parts.plate(args.width, args.depth, args.thickness, args.radius)
        name = "plate"
    elif args.command == "washer":
        mesh = parts.washer(args.outer, args.inner, args.thickness)
        name = "washer"
    else:
        mesh = parts.l_bracket(args.length, args.height, args.thickness, args.wall)
        name = "bracket"

    print(f"{args.command}: generated")
    if args.verify:
        _report(mesh, gear_spec)
    if args.stl:
        mesh.save_stl(args.stl)
        print(f"  wrote {args.stl}")
    if args.obj:
        with open(args.obj, "w") as handle:
            handle.write(mesh.to_obj())
        print(f"  wrote {args.obj}")
    if args.render:
        result = render(mesh, out_dir=args.render, name=name)
        if result["ran"]:
            print(f"  rendered: {result['png']} and {result['blend']}")
        else:
            print(f"  {result.get('note', 'render did not complete')}; script at {result['script']}")

    if args.verify and not mesh.is_watertight():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
