"""Command line: generate terrain, erode it, save a 2D map and STL, optionally
render it in 3D with Blender.

    python3 -m terrainkit --size 256 --seed 2026 --map terrain.png
    python3 -m terrainkit --size 256 --seed 7 --droplets 60000 --map map.png --stl terrain.stl
    python3 -m terrainkit --size 200 --seed 3 --render ./out
    python3 -m terrainkit --size 200 --seed 3 --no-erode --map raw.png    # skip erosion
"""

from __future__ import annotations

import argparse
import sys

from .colormap import save_map
from .erosion import erode
from .mesh import terrain_mesh
from .noise import heightmap


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="terrainkit", description="Procedural terrain")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--octaves", type=int, default=7)
    p.add_argument("--droplets", type=int, default=40000)
    p.add_argument("--no-erode", action="store_true", help="skip hydraulic erosion")
    p.add_argument("--height-scale", type=float, default=3.2)
    p.add_argument("--map", default=None, help="write a 2D hypsometric PNG here")
    p.add_argument("--stl", default=None, help="write the terrain solid as STL here")
    p.add_argument("--render", metavar="DIR", default=None, help="render in 3D with Blender into DIR")
    args = p.parse_args(argv)

    print(f"generating {args.size}×{args.size} terrain, seed {args.seed}…")
    h = heightmap(size=args.size, seed=args.seed, octaves=args.octaves)

    if not args.no_erode:
        h, stats = erode(h, droplets=args.droplets, seed=args.seed)
        print(f"  eroded with {stats.droplets:,} droplets: "
              f"lifted {stats.eroded:.2f}, deposited {stats.deposited:.2f}; "
              f"mass error {stats.mass_error:.2e}")

    if args.map:
        save_map(h, args.map, scale=1 if args.size >= 400 else 2)
        print(f"  wrote 2D map {args.map}")
    if args.stl:
        mesh = terrain_mesh(h, height_scale=args.height_scale)
        mesh.save_stl(args.stl)
        print(f"  wrote {args.stl} ({mesh.face_count:,} triangles, "
              f"watertight={mesh.is_watertight()})")
    if args.render:
        from .blender_export import render
        out = render(h, out_dir=args.render, height_scale=args.height_scale)
        if out["ran"]:
            print(f"  rendered 3D terrain → {out['png']}")
        else:
            print(f"  {out.get('note', 'Blender did not run')}; script at {out['script']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
