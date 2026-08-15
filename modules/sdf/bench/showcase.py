#!/usr/bin/env python3
"""Mesh a catalogue of SDF models, verify each against a number, export it, and
(if Blender is installed) render the flagship die.

    python3 bench/showcase.py

"Verify" is measured, never eyeballed. For every model: is the extracted surface
watertight and consistently oriented (a real solid), and is its Euler
characteristic the one the shape demands — a die is a ball (χ=2), a four-hole
bracket is genus-4 (χ=−6), a ring is a torus (χ=0). Getting χ right is proof the
booleans joined up cleanly, with no leaked holes or doubled walls.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdfkit import scenes  # noqa: E402
from sdfkit.blender_export import find_blender, render  # noqa: E402
from sdfkit.marching import triangulate  # noqa: E402
from sdfkit.sdf import Sphere  # noqa: E402


def main():
    out = os.path.join(os.path.dirname(__file__), "..", ".out")
    os.makedirs(out, exist_ok=True)

    rows = []
    all_ok = True
    meshes = {}
    for name, fn in scenes.SCENES.items():
        sdf, res, expected_chi = fn()
        mesh = triangulate(sdf, resolution=res)
        meshes[name] = mesh
        chi = mesh.euler_characteristic()
        ok = mesh.is_watertight() and mesh.is_consistently_oriented() and chi == expected_chi
        all_ok = all_ok and ok
        rows.append([
            name,
            "yes" if mesh.is_watertight() else "NO",
            "yes" if mesh.is_consistently_oriented() else "NO",
            f"{chi}",
            f"{expected_chi}",
            "ok" if chi == expected_chi else "BAD",
            f"{mesh.genus() if mesh.genus() is not None else '-'}",
            f"{mesh.face_count:,}",
        ])

    print("SDF models — every column is a measured number, not a look\n")
    header = ["model", "watertight", "oriented", "χ", "want", "χ ok", "genus", "faces"]
    print(_table(rows, header))

    # A convergence line: a meshed unit sphere → analytic volume and area.
    print("\nConvergence — a unit sphere's meshed volume/area vs the analytic 4/3·π, 4·π:")
    for res in (16, 32, 64):
        m = triangulate(Sphere(1.0), resolution=res)
        ve = abs(m.volume() - 4 / 3 * math.pi) / (4 / 3 * math.pi)
        ae = abs(m.area() - 4 * math.pi) / (4 * math.pi)
        print(f"  {res:3d}³ grid: volume err {ve * 100:5.2f}%, area err {ae * 100:5.2f}%")

    print("\nExported STL (binary):")
    for name, mesh in meshes.items():
        path = os.path.join(out, f"{name}.stl")
        mesh.save_stl(path)
        print(f"  {os.path.getsize(path):>9,} bytes  {name}.stl")

    print("""
The headline is topology. Marching tetrahedra guarantees a watertight, manifold
surface by construction, and the Euler characteristic then reads back the shape
the booleans actually produced: a die remains a single ball, a bolt-hole bracket
is exactly genus-4, a ring is a torus. Volume and area converge to the analytic
values as the grid refines. That is the checklist that separates a printable,
raymarchable, physics-ready solid from a pretty but hollow shell.""")

    blender = find_blender()
    if blender:
        print(f"\nBlender found ({blender}); rendering the die…")
        result = render(meshes["die"], out_dir=out, name="showcase_die", samples=32)
        if result["ran"]:
            print(f"  wrote {os.path.basename(result['blend'])} and "
                  f"{os.path.basename(result['png'])} ({os.path.getsize(result['png']):,} B)")
        else:
            print("  Blender ran but did not finish; see .out for the script")
    else:
        print("\nBlender not found. Every model above is verified without it; run any "
              "model's emitted _build.py in Blender to view or render it.")

    return 0 if all_ok else 1


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


if __name__ == "__main__":
    raise SystemExit(main())
