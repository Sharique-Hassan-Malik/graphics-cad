"""Command line: mesh a built-in SDF scene, verify it, export STL, optionally render.

    python3 -m sdfkit list
    python3 -m sdfkit die --resolution 96 --stl die.stl --verify
    python3 -m sdfkit bracket --render ./out
    python3 -m sdfkit blob --resolution 80 --render ./out --stl blob.stl
"""

from __future__ import annotations

import argparse
import sys

from . import scenes
from .blender_export import render
from .marching import triangulate


def _verify(mesh, expected_chi):
    print(f"  vertices {mesh.vertex_count}, faces {mesh.face_count}")
    print(f"  watertight: {mesh.is_watertight()}   oriented: {mesh.is_consistently_oriented()}"
          f"   edge-manifold: {mesh.is_edge_manifold()}")
    chi = mesh.euler_characteristic()
    tag = "ok" if chi == expected_chi else f"EXPECTED {expected_chi}"
    print(f"  Euler χ: {chi} ({tag})   genus: {mesh.genus()}")
    print(f"  volume: {mesh.volume():.4f}   bounds: {mesh.size().round(3).tolist()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sdfkit", description="SDF modelling → watertight mesh")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the built-in scenes")

    for name in scenes.SCENES:
        sp = sub.add_parser(name, help=f"the {name} scene")
        sp.add_argument("--resolution", type=int, default=None, help="grid cells along the longest axis")
        sp.add_argument("--stl", default=None, help="write a binary STL here")
        sp.add_argument("--obj", default=None, help="write a Wavefront OBJ here")
        sp.add_argument("--render", default=None, metavar="DIR", help="render with Blender into DIR")
        sp.add_argument("--verify", action="store_true", help="print topology + geometry checks")

    args = parser.parse_args(argv)

    if args.command == "list":
        for name, fn in scenes.SCENES.items():
            _, res, chi = fn()
            print(f"  {name:9s} default resolution {res:3d}, expected Euler χ = {chi}")
        return 0

    sdf, default_res, expected_chi = scenes.SCENES[args.command]()
    res = args.resolution or default_res
    print(f"{args.command}: meshing SDF at resolution {res}…")
    mesh = triangulate(sdf, resolution=res)

    if args.verify:
        _verify(mesh, expected_chi)
    if args.stl:
        mesh.save_stl(args.stl)
        print(f"  wrote {args.stl}")
    if args.obj:
        with open(args.obj, "w") as handle:
            handle.write(mesh.to_obj())
        print(f"  wrote {args.obj}")
    if args.render:
        result = render(mesh, out_dir=args.render, name=args.command)
        if result["ran"]:
            print(f"  rendered {result['png']}")
        else:
            print(f"  {result.get('note', 'render did not complete')}; script at {result['script']}")

    return 0 if mesh.is_watertight() else 1


if __name__ == "__main__":
    sys.exit(main())
