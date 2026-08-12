#!/usr/bin/env python3
"""Generate a game map with Wave Function Collapse, prove it is legal, and save
it — a 2D PNG, and a 3D island if Blender is present.

    python3 examples/generate_map.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wfckit import verify  # noqa: E402
from wfckit.blender_export import render  # noqa: E402
from wfckit.render import save  # noqa: E402
from wfckit.solver import collapse  # noqa: E402
from wfckit.tiles import TILESETS  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", ".out")
os.makedirs(out, exist_ok=True)

ts = TILESETS["terrain"]()
result = collapse(ts, width=36, height=36, seed=2026)

print("Generated a 36×36 terrain map with Wave Function Collapse:\n")
print(f"  converged after {result.attempts} attempt(s), {result.collapses} cells collapsed")

violations = verify.adjacency_violations(result.grid, ts)
edges = verify.edge_count(36, 36)
print(f"  legality: {edges - len(violations)}/{edges} shared edges satisfied — "
      f"{len(violations)} violations")
print(f"  tile usage: {verify.tile_histogram(result.grid, ts).tolist()}")

png = save(result.grid, ts, os.path.join(out, "map.png"), scale=2)
print(f"\nWrote 2D map {png} ({os.path.getsize(png):,} bytes).")

r = render(result, out_dir=out, name="example_island")
if r["ran"]:
    print(f"Blender rendered a 3D island → {r['png']}.")
else:
    print("Blender not found — the 2D map above is already generated and verified.")
