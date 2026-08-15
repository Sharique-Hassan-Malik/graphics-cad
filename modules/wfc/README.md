# Wave Function Collapse

> Part of [Graphics & CAD](../../README.md). Runs standalone from this folder;
> the mesh type and the Blender path come from `geokit`.

**Wave Function Collapse** (WFC) is the constraint-based procedural generator behind a wave of modern game content — tilemaps, dungeons, textures, level layouts. You give it a set of tiles and the rules for which tiles may sit next to which; it fills a grid so that **every** adjacency is legal, producing output that looks hand-authored but is generated from a seed. This is a from-scratch implementation: the tile/socket model, the collapse-and-propagate solver, an independent legality verifier, a from-scratch PNG writer, and a lift into 3D via Blender.

Despite the quantum-flavoured name, WFC is pure constraint solving. Every cell begins in *superposition* (all tiles possible); the solver repeatedly collapses the most-constrained cell to one tile and propagates the consequences to its neighbours until the grid is consistent — arc-consistency by another name.

## The headline: provably legal output

The correctness claim is a hard guarantee expressed as a number. An **independent** verifier — which never looks at the solver's state, only at the finished grid — re-checks every shared edge against the adjacency rules. Across every seed and every one of those tens of thousands of edges, the violation count is **zero**:

```
$ python3 bench/showcase.py

Wave Function Collapse — 32×32 grids, 24 seeds each

tileset  tiles  converged  avg tries  edges checked  violations  det.
-------  -----  ---------  ---------  -------------  ----------  ----
pipes    12     24/24      1.00       95,232         0           yes
terrain  16     24/24      1.00       95,232         0           yes
```

The output is not merely *plausible* — it **provably** satisfies every constraint, it is **deterministic** from the seed (same seed → identical grid, byte for byte), and it converges on the first attempt for these tilesets.

## From 2D tilemaps to a 3D island

**pipes** — a connected-conduit set (the classic "circuit"/"Knots" tiles). Every edge is *open* or *shut*, and open only ever meets open, so the network always joins up with no dangling ends. A `--border shut` constraint fences the whole board.

![a generated pipe network](docs/pipes.png)

**terrain** — Wang tiles keyed on their four *corners* (land or water). Shared edges must agree on both corner values, which forces coastlines to line up: coherent islands and lakes with sandy shores.

![a generated terrain map](docs/terrain.png)

Because WFC guarantees neighbouring terrain tiles agree on shared corners, the grid of tile-corners is a globally consistent binary heightmap — extracted for free from the solution and lifted into a **watertight 3D island** for Blender:

![the terrain lifted into a 3D island](docs/island.png)

*The corner lattice becomes a heightmap; raised land, low water, closed underneath. That the lattice is consistent at all is a second, structural proof the adjacency held — if any shared corner disagreed, the lift would fail.*

## Quick start

```bash
pip install -r requirements.txt          # numpy, pytest (PNG writing is from scratch)

python3 -m wfckit pipes   --width 40 --height 40 --seed 7 --out pipes.png
python3 -m wfckit terrain --width 40 --height 40 --seed 3 --out map.png --scale 2
python3 -m wfckit terrain --width 28 --height 28 --render ./out   # + a 3D island in Blender

python3 examples/generate_map.py         # a worked end-to-end example
python3 bench/showcase.py                # the legality table above
pytest tests/                            # 21 tests, headed by "every output is legal"
```

## As a library

```python
from wfckit import collapse, is_valid, TILESETS

tiles = TILESETS["terrain"]()
result = collapse(tiles, width=40, height=40, seed=2026)

assert result.success
assert is_valid(result.grid, tiles)      # 0 adjacency violations, guaranteed
```

## How the solver works

1. **Observe** — of the cells not yet decided, pick the one with the lowest *entropy* (fewest remaining options, weighted by tile frequency) and collapse it to a single tile, chosen at random by weight.
2. **Propagate** — that choice forbids any neighbour option no longer compatible across the shared edge; each elimination can force more, so the wave is pushed out to a fixpoint.
3. **Contradiction?** — if propagation empties a cell, this tiled formulation restarts deterministically; in practice it succeeds within a couple of attempts.

## Layout

| file | what it holds |
|---|---|
| `wfckit/tiles.py` | the `Tile`/`TileSet` model, socket-based adjacency, and the built-in tilesets |
| `wfckit/solver.py` | the collapse/propagate solver with min-entropy observation and restart |
| `wfckit/verify.py` | independent legality check — re-tests every edge against the rules |
| `wfckit/render.py` | assemble a solved grid into a seamless image |
| `wfckit/image.py` | a from-scratch PNG writer (stdlib `zlib` only, no PIL) |
| `wfckit/model3d.py` | lift a terrain tiling into a watertight island heightmap mesh |
| `wfckit/blender_export.py` | render the 3D island headless in Blender |
| `wfckit/cli.py` | `python3 -m wfckit …` |
| `tests/` | 21 tests, headed by "every output satisfies all adjacencies" |
| `bench/showcase.py` | the legality/determinism table |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the model — how sockets encode adjacency, why min-entropy observation and propagation converge, how contradictions are handled, and why the corner lattice is the right bridge to 3D.

## License

MIT — see [`LICENSE`](./LICENSE).
