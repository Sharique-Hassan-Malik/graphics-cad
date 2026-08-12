# Procedural Graphics & Games

Procedural 2D and 3D content generation, built from scratch — the algorithms behind game levels, terrain, textures, and props. Each project implements the hard core in plain Python and NumPy (no engine, no CAD or meshing library), proves the result correct with **measured numbers** rather than a screenshot, and uses **Blender** as an optional render/model backend. Every result is fully generated and verified without Blender in the loop.

A collection of 3 self-contained projects. Each lives in its own subdirectory with its own `README.md`, `ARCHITECTURE.md`, `LICENSE`, and test suite, and can be built and run independently.

## Projects

| project | 2D / 3D | what it is |
|---|---|---|
| [`SDF-Mesher`](./SDF-Mesher) | 3D | Model solids with **signed distance fields** and CSG (union/intersect/difference/blend), then extract a watertight mesh with **marching tetrahedra**. Topology is provable: a die stays a ball (χ=2) after 21 pips are carved out, a bolt-hole bracket is exactly genus-4, and volume/area converge to analytic. The tech behind metaballs and voxel/destructible terrain. |
| [`Wave-Function-Collapse`](./Wave-Function-Collapse) | 2D → 3D | The constraint-based procedural generator (**WFC**) used for tilemaps and levels. Give it tiles and adjacency rules; it fills a grid so **every** adjacency is legal — zero violations across tens of thousands of edges, verified independently, deterministic from the seed. Terrain tilings lift into a watertight 3D island. |
| [`Procedural-Terrain`](./Procedural-Terrain) | 3D | Landscapes from **fractal noise** plus **droplet-based hydraulic erosion**, rendered as a hillshaded 2D map and a watertight 3D solid. The erosion is a real simulation: it **conserves mass to floating-point precision** — every grain of soil is moved, never created or destroyed. |

## The common idea

Each project draws a hard line between content that merely *looks* right and content that provably *is* right — and the proof is always a measured number, not the render:

- **`SDF-Mesher`** — the extracted surface is watertight and manifold *by construction*, and its Euler characteristic reads back the exact topology the CSG produced.
- **`Wave-Function-Collapse`** — an independent verifier confirms every shared edge satisfies the adjacency rules; the count of violations is zero for every seed.
- **`Procedural-Terrain`** — hydraulic erosion moves large amounts of soil, yet total elevation is invariant to ~1e-16; and the terrain meshes into a genuine watertight solid.

Blender is a backend for rendering and 3D modelling throughout, deliberately kept out of the path that decides correctness.

## Repository layout

Each subdirectory is a standalone project; there is no shared build. Enter one and follow its README:

```bash
cd SDF-Mesher
cat README.md
python3 bench/showcase.py     # verify the models (and render if Blender is present)
pytest tests/
```

Rendering is optional throughout. To enable it, put a `blender` binary on `PATH` or set `BLENDER_BIN` to point at one; without it, every project still generates and fully verifies its geometry, and writes STLs / PNGs / build scripts.

## License

MIT — see the `LICENSE` file in each project.
