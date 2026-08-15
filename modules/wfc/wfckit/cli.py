"""Command line: generate a tiling, verify it is legal, and save a PNG (and, for
terrain, optionally a 3D Blender render of the island).

    python3 -m wfckit pipes   --width 32 --height 32 --seed 7 --out pipes.png
    python3 -m wfckit terrain --width 40 --height 40 --seed 3 --out map.png --scale 2
    python3 -m wfckit terrain --width 24 --height 24 --render ./out   # 3D island in Blender
"""

from __future__ import annotations

import argparse
import sys

from . import verify
from .render import save
from .solver import collapse
from .tiles import TILESETS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wfckit", description="Wave Function Collapse")
    parser.add_argument("tileset", choices=list(TILESETS), help="which tileset to use")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the tiling as a PNG here")
    parser.add_argument("--scale", type=int, default=1, help="pixel scale-up for the PNG")
    parser.add_argument("--render", metavar="DIR", default=None,
                        help="(terrain only) render a 3D island in Blender into DIR")
    parser.add_argument("--attempts", type=int, default=40)
    args = parser.parse_args(argv)

    tileset = TILESETS[args.tileset]()
    border = "shut" if args.tileset == "pipes" else None
    result = collapse(tileset, args.width, args.height, seed=args.seed,
                      max_attempts=args.attempts, border=border)

    if not result.success:
        print(f"failed to converge after {result.attempts} attempts (try another seed)")
        return 1

    violations = verify.adjacency_violations(result.grid, tileset)
    edges = verify.edge_count(args.height, args.width)
    print(f"{args.tileset}: {args.width}×{args.height}, seed {args.seed}")
    print(f"  solved in {result.attempts} attempt(s), {result.collapses} collapses")
    print(f"  legality: {edges - len(violations)}/{edges} shared edges satisfied, "
          f"{len(violations)} violations")

    if args.out:
        save(result.grid, tileset, args.out, scale=args.scale)
        print(f"  wrote {args.out}")
    if args.render:
        if args.tileset != "terrain":
            print("  --render only supports the terrain tileset")
        else:
            from .blender_export import render as render3d
            out = render3d(result, out_dir=args.render)
            if out["ran"]:
                print(f"  rendered 3D island → {out['png']}")
            else:
                print(f"  {out.get('note', 'Blender did not run')}; script at {out['script']}")

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
