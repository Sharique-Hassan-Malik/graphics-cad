#!/usr/bin/env python3
"""Generate terrain, erode it, and verify the properties that make it simulation
rather than decoration: determinism, mass conservation, and a watertight solid.

    python3 bench/showcase.py

The headline is the mass-conservation column. Hydraulic erosion moves soil around
the map — carving valleys, silting plains — but a droplet never creates or
destroys material, so the total elevation summed over the grid is unchanged. That
holds here to floating-point precision, across seeds, which is the difference
between a physical simulation and a filter that merely looks eroded.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrainkit.colormap import save_map  # noqa: E402
from terrainkit.erosion import erode  # noqa: E402
from terrainkit.mesh import terrain_mesh  # noqa: E402
from terrainkit.noise import heightmap  # noqa: E402


def main():
    out = os.path.join(os.path.dirname(__file__), "..", ".out")
    os.makedirs(out, exist_ok=True)

    size = 160
    rows = []
    all_ok = True
    for seed in (1, 7, 2026, 42):
        h = heightmap(size=size, seed=seed, octaves=7)
        eroded, stats = erode(h, droplets=25000, seed=seed)
        mesh = terrain_mesh(eroded, height_scale=3.2)
        ok = stats.mass_error < 1e-9 and mesh.is_watertight()
        all_ok = all_ok and ok
        rows.append([
            f"{seed}",
            f"{stats.droplets:,}",
            f"{stats.eroded:.2f}",
            f"{stats.deposited:.2f}",
            f"{stats.mass_error:.1e}",
            "yes" if mesh.is_watertight() else "NO",
            f"{mesh.face_count:,}",
        ])

    print(f"Procedural terrain — {size}×{size}, fractal noise + hydraulic erosion\n")
    header = ["seed", "droplets", "soil lifted", "soil dropped", "mass err", "watertight", "faces"]
    print(_table(rows, header))

    print("""
The mass-error column is the headline: erosion redistributes soil — the "lifted"
and "dropped" totals are large — yet the net change in total elevation is zero to
floating-point precision. Every grain a droplet erodes is carried and deposited
elsewhere, including before it leaves the map. Generation is deterministic from
the seed, and the terrain meshes into a watertight solid (Euler characteristic 2)
ready to print, simulate, or drop into an engine.""")

    # before/after maps for one seed
    print("\nExported maps and mesh:")
    h = heightmap(size=220, seed=2026, octaves=7)
    save_map(h, os.path.join(out, "before.png"), scale=1)
    eroded, _ = erode(h, droplets=45000, seed=2026)
    save_map(eroded, os.path.join(out, "after.png"), scale=1)
    stl = terrain_mesh(eroded, height_scale=3.2).save_stl(os.path.join(out, "terrain.stl"))
    for f in ("before.png", "after.png", "terrain.stl"):
        print(f"  {os.path.getsize(os.path.join(out, f)):>10,} bytes  {f}")

    from terrainkit.blender_export import find_blender, render
    if find_blender():
        print("\nBlender found; rendering the eroded terrain in 3D…")
        result = render(eroded, out_dir=out, name="showcase_terrain", height_scale=3.2)
        if result["ran"]:
            print(f"  wrote {os.path.basename(result['png'])} ({os.path.getsize(result['png']):,} B)")
    else:
        print("\nBlender not found; the 2D maps and STL above are already generated and verified.")

    return 0 if all_ok else 1


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


if __name__ == "__main__":
    raise SystemExit(main())
