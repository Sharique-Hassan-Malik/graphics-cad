# Graphics & CAD

Seven geometry generators — procedural terrain, an SDF mesher, wave function
collapse, parametric CAD parts, a sketch constraint solver, an engine
animation and a character rig — producing **one mesh type**, validated by
**one set of topology checks**, and reaching Blender through **one path**.

```
geo modules                       # what is here, and how to run each alone
geo make parts --part gear        # run a generator (its own flags, unchanged)
geo check model.stl               # is this actually manufacturable?
geo render model.stl              # put any mesh in front of Blender
```

```
$ geo check bracket.stl

  vertices                 12
  vertices_in_file         60
  faces                    20
  watertight               True  ✓
  edge_manifold            True  ✓
  consistently_oriented    True  ✓
  euler_characteristic     2
  genus                    0
  volume                   1056.0
  boundary_edges           0

  Closed, orientable and manifold — a slicer will accept this.
```

That command works on the output of *any* of the seven. It could not before:
each generator carried its own `Mesh` class with its own subset of the checks,
so a terrain mesh could not be validated with the CAD tools and vice versa.

## What was written seven times

`find_blender()` — the function that looks for a Blender binary in `$BLENDER_BIN`,
then `PATH`, then the usual install locations — existed **once per generator**,
along with the `from_pydata` boilerplate and the headless-invocation plumbing.
There is now one copy in [`geokit/blender.py`](geokit/blender.py), and a test
asserts there is still only one.

`Mesh` existed three times. All three agreed on the representation — vertices
N×3, triangles M×3 — and disagreed about which checks existed: only the CAD one
had manifold and orientation checks, only the SDF one had `genus` and per-face
normals. [`geokit/mesh.py`](geokit/mesh.py) is the **union**, so the merge added
capability to every generator rather than taking the largest and discarding the
rest.

**What is not shared: the scene.** A terrain slab lit by a low sun, a CAD part
on a neutral backdrop and an animated crank are genuinely different scenes.
Collapsing them into one `render()` with fifteen flags would have been a worse
abstraction than the duplication it replaced, so `geokit.blender` knows how to
find Blender, turn a mesh into `from_pydata` calls, frame a camera and run a
script — and each generator still writes its own scene.

## Blender is a backend, never a dependency

Every generator produces and validates geometry with plain numpy. The manifold
checks are counting edges, not asking a kernel. With no Blender installed,
`geo render` still writes a runnable build script and says so:

```
  ran              False
  note             no Blender binary found; run the emitted script yourself
```

That is deliberate. The mesh was already correct; losing it to a missing
optional renderer would be absurd.

## The seven generators

| Module | Produces | What it is |
|---|---|---|
| [`terrain`](modules/terrain) | mesh | Layered noise, hydraulic erosion, a biome colour ramp, meshed into a slab with skirt walls. |
| [`sdf`](modules/sdf) | mesh | Signed distance fields with boolean and smooth-blend operators, surfaced by marching cubes. |
| [`wfc`](modules/wfc) | layout | Constraint-propagating tile solver with backtracking, in 2D and its 3D realisation. |
| [`parts`](modules/parts) | mesh | CAD primitives, profiles, revolves and involute gears — checked for manufacturability, not looks. |
| [`sketch`](modules/sketch) | mesh | 2D geometric constraints solved by Newton iteration, then extruded. |
| [`engine`](modules/engine) | animation | Slider-crank kinematics for an inline-four, sampled over a cycle. |
| [`rig`](modules/rig) | animation | A jointed character rig with quaternion interpolation between poses. |

## One thing worth knowing about STL

`geo check` welds vertices before checking topology, because binary STL stores
three vertices per triangle with **no sharing**. A closed mesh written to STL
and read back has no shared edges at all, so every topology check reports it as
open — a fact about the file format, not the geometry. The `vertices_in_file`
line above shows the difference: 60 stored, 12 real.

`--no-weld` checks the file exactly as stored, which is occasionally what you
want and never the default.

## Using one generator on its own

```bash
cd modules/terrain && python -m terrainkit --size 256 --erode 40
cd modules/sdf     && python -m sdfkit --scene die --resolution 96
cd modules/wfc     && python -m wfckit --width 24 --height 24
cd modules/parts   && python -m partkit --part gear --teeth 18
cd modules/sketch  && python -m sketchkit --demo bracket
cd modules/engine  && python -m enginekit --cylinders 4 --frames 120
cd modules/rig     && python -m transformkit --frames 90
```

`geo make <generator> …` delegates to exactly those CLIs, flags and `--help`
included.

## Install

```bash
pip install -e .
```

numpy is the only dependency. Blender is found if present and worked around if
not.

## Getting Blender

Every generator works without it — meshes, solvers and exporters are pure
Python. Blender is needed only for the rendering path, and the seven tests that
exercise it skip themselves by name when it is absent.

It does not need installing or root. The official tarball is self-contained:

```bash
mkdir -p ~/.local/opt && cd ~/.local/opt
curl -O https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz
```

`find_blender()` looks there, so nothing else is required. Set `$BLENDER_BIN`
to override. Verified against 4.0.2.

## Tests

```bash
pytest                    # everything, 149 tests
pytest modules/parts      # one generator
```

## Licence

MIT — see [LICENSE](LICENSE).
