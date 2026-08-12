#!/usr/bin/env python3
"""Generate a terrain, erode it, and produce a 2D map, an STL, and a 3D render.

    python3 examples/make_terrain.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrainkit.blender_export import render  # noqa: E402
from terrainkit.colormap import save_map  # noqa: E402
from terrainkit.erosion import erode  # noqa: E402
from terrainkit.mesh import terrain_mesh  # noqa: E402
from terrainkit.noise import heightmap  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
os.makedirs(out, exist_ok=True)

print("Generating a 200×200 terrain (fractal noise), then eroding it…\n")
raw = heightmap(size=200, seed=2026, octaves=7)
eroded, stats = erode(raw, droplets=45000, seed=2026)

print(f"  hydraulic erosion: {stats.droplets:,} droplets")
print(f"    soil lifted   {stats.eroded:.3f}")
print(f"    soil deposited {stats.deposited:.3f}")
print(f"    mass error    {stats.mass_error:.2e}  (0 = perfectly conserved)")

save_map(raw, os.path.join(out, "map_before.png"), scale=2)
save_map(eroded, os.path.join(out, "map_after.png"), scale=2)
print("\n  wrote map_before.png and map_after.png (2D hypsometric maps)")

mesh = terrain_mesh(eroded, width=10, height_scale=3.2)
stl = mesh.save_stl(os.path.join(out, "terrain.stl"))
print(f"  wrote {os.path.basename(stl)} — {mesh.face_count:,} triangles, "
      f"watertight={mesh.is_watertight()}")

r = render(eroded, out_dir=out, name="example_terrain", height_scale=3.2)
if r["ran"]:
    print(f"\nBlender rendered the 3D terrain → {r['png']}.")
else:
    print("\nBlender not found — the maps and STL above are complete without it.")
