# Parametric Parts

> Part of [Graphics & CAD](../../README.md). Runs standalone from this folder;
> the mesh type and the Blender path come from `geokit`.

A small parametric CAD kernel that generates **manufacturable** mechanical parts — spur gears, plates, washers, brackets — as watertight triangle meshes, exports them to STL, and hands them to **Blender** for rendering. Written from scratch: no CAD library, no meshing library, no solid modeller. A part is two NumPy arrays (vertices and triangles), and every claim made about it is computed from those arrays directly.

The headline: **every gear's tooth flanks lie within about 1×10⁻¹⁴ mm of the mathematically exact involute of its base circle** — machine precision, verified against the involute equation rather than eyeballed in a render. That is the single property that makes an involute gear transmit constant angular velocity, and here it is a measured number in the test suite, not an aspiration.

## Why this is the hard version

Anyone can make a mesh that *looks* like a gear. The difference between a picture and a part you can 3D-print is almost entirely topology and geometry that never shows up on screen:

- **Watertight** — every edge shared by exactly two triangles. One edge used once is a hole; used three times is a non-manifold seam. Either makes "inside" undefined and a slicer refuses the file.
- **Consistently oriented** — every triangle winds outward, so the solid has a well-defined interior. Caught by directed-edge uniqueness, which a flipped triangle fails even when the mesh is still watertight.
- **The right topology** — a solid disk has Euler characteristic χ = 2; a bored gear (a through-hole, i.e. a torus) has χ = 0. A value that is neither is a broken mesh, detected without looking at geometry at all.
- **Correct volume** — the signed volume via the divergence theorem matches the analytic answer to well under a percent, and is positive only when orientation is correct.
- **Exact flanks** — for gears, the involute deviation above.

Every one of these is asserted against a number in `tests/`, so a regression in the geometry cannot hide behind a plausible render.

## The measured result

```
$ python3 bench/showcase.py

part                watertight  χ  χ ok  volume   vs analytic  faces  note
------------------  ----------  -  ----  -------  -----------  -----  -----------------------
gear m2 z20 solid   yes         2  ok    7352.2   0.00%        1680   involute dev 1.8e-14 mm
gear m3 z24 bored   yes         0  ok    31313.9  0.00%        3840   involute dev 5.0e-14 mm
gear m1.5 z40 pa25  yes         2  ok    14050.3  0.29%        3040   involute dev 2.4e-14 mm
plate 50x30 r5      yes         2  ok    5911.5   —            124
L-bracket 40x30     yes         2  ok    1920.0   0.00%        20     reflex profile
washer 20/10        yes         0  ok    705.7    0.16%        512
```

Every column is a computed quantity. `vs analytic` is the mesh volume against a closed-form value; `involute dev` is the maximum distance of any tooth-flank vertex from the exact involute.

## Blender integration

Blender is the *renderer and viewer*, and it is optional. The kernel emits a self-contained Python build script that reconstructs the mesh with `bpy.data.meshes.new(...).from_pydata(...)`; running it inside Blender rebuilds the exact part. The `render()` helper drives Blender headless (`--background --factory-startup`), assigns a brushed-steel Principled-BSDF material, auto-frames a camera to the part's bounding sphere, and renders a PNG with Cycles on CPU (no GPU or display required).

If Blender is not installed, nothing above breaks — every part is fully verified without it, and you still get the STL and the build script. Set `$BLENDER_BIN`, or put `blender` on `PATH`, to enable rendering.

![the flagship gear, rendered in Blender](docs/gear.png)

*A module-3, 24-tooth spur gear with a 10 mm bore — the mesh from `spur_gear(3, 24, 8, bore_diameter=10)`, rebuilt in Blender from the emitted `bpy` script and rendered with Cycles.*

## Quick start

```bash
pip install -r requirements.txt          # numpy, pytest

# Generate a part, verify it, export an STL — no Blender needed:
python3 -m partkit gear --module 2 --teeth 20 --thickness 6 --bore 8 --stl gear.stl --verify

# Render it (uses Blender if present, otherwise writes the build script):
python3 -m partkit gear --module 3 --teeth 24 --bore 10 --render ./out

python3 -m partkit plate  --width 50 --depth 30 --thickness 4 --radius 5 --stl plate.stl
python3 -m partkit washer --outer 20 --inner 10 --thickness 3 --stl washer.stl
python3 -m partkit bracket --length 40 --height 30 --thickness 5 --wall 6 --stl bracket.stl
```

The `gear ... --verify` output measures the part *back* from its own vertices — tooth count from angular clustering, module from the outer radius — and reports the involute deviation, so the numbers are evidence rather than an echo of the inputs.

```bash
python3 examples/make_gear.py    # a worked end-to-end example
python3 bench/showcase.py        # the catalogue + verification table above
pytest tests/                    # 20 tests, all measured assertions
```

## As a library

```python
from partkit.gears import spur_gear, involute_deviation

gear = spur_gear(module=2.0, teeth=24, thickness=6.0, bore_diameter=8.0)
assert gear.is_watertight() and gear.euler_characteristic() == 0
gear.save_stl("gear.stl")

dev = involute_deviation(gear, module=2.0, teeth=24)
print(dev["max_deviation"])       # ~1e-14 mm
```

## Layout

| file | what it holds |
|---|---|
| `partkit/mesh.py` | the `Mesh` type and every topology/geometry check; STL and OBJ I/O, all written by hand |
| `partkit/profiles.py` | 2D profiles — circles, rounded rectangles, and the involute gear outline |
| `partkit/solids.py` | turning a 2D profile into a watertight solid: ear-clipping, star-fan and ring extruders |
| `partkit/gears.py` | `spur_gear`, `measure_gear` (the inverse), and `involute_deviation` (the correctness proof) |
| `partkit/parts.py` | plates, washers, L-brackets |
| `partkit/blender_export.py` | emit a `bpy` build script; drive Blender headless to render a PNG |
| `partkit/cli.py` | `python3 -m partkit …` |
| `tests/` | 20 tests, each asserting a measured property |
| `bench/showcase.py` | the catalogue and verification table |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the geometry — how the involute profile is derived, how each extruder caps its ends, and how a finished gear is measured back to its parameters.

## License

MIT — see [`LICENSE`](./LICENSE).
