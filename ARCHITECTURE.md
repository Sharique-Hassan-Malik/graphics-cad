# Architecture

Seven generators, one core. Each generator's own design is in [`docs/`](docs);
this is about what they share.

```
                      geokit/
              ┌──────────┴──────────┐
          mesh.py               blender.py
      the triangle mesh      find, script, run
      + topology checks      (never the scene)
              │                      │
   ┌─────┬────┴──┬───────┬──────┬────┴──┬──────┐
terrain  sdf    wfc    parts  sketch  engine  rig
```

## One mesh, by union not by selection

Three `Mesh` classes existed. Picking the largest and deleting the others would
have been a regression for whichever generator depended on a method the winner
lacked — `genus`, `face_normals`, `face_centroids` and `oriented` were only in
the SDF mesher's copy, while the manifold and orientation checks were only in
the CAD one.

`geokit.mesh` is the union of all three, and a test names those four methods
specifically so a future tidy-up cannot quietly drop them again.

Each generator's `mesh.py` is a re-export, so `from partkit.mesh import Mesh`
resolves to the shared type. That keeps each module's own suite meaningful: the
122 tests across the seven exercise the shared `Mesh` through each generator's
own import path, rather than a private copy that happens to agree.

## One Blender path, and a deliberate line

Shared: finding a binary, `from_pydata` generation, camera framing, Cycles
setup, and headless invocation with a done-marker check. Blender exits zero
after failing halfway, so "did it run" is decided by a marker the generated
script prints at the end, not by the return code.

Not shared: the scene. Seven generators light and stage seven different things,
and the version of this file that had one `render()` taking `sun_angle`,
`biome_ramp`, `frame_count`, `material_preset`… would have been harder to read
than the duplication.

The API is arranged so a generator composes a scene out of pieces
(`mesh_to_pydata` + `frame_camera` + `cycles_setup` + `finish`) rather than
configuring one.

## Failure is a state, not an exception

`find_blender()` returns `None`. `run_script()` writes the script and returns a
`RenderResult` with `ran=False` and a note. Nothing raises.

This is the right shape because the geometry does not depend on Blender at all.
The interesting properties — watertight, manifold, oriented, genus, volume —
are computed by counting edges and summing signed tetrahedra, and none of it
involves a modelling kernel. Turning a missing optional renderer into an
exception would throw away a valid result.

## STL and welding

`geo check` welds before it checks, and this is the subtlety most worth writing
down. Binary STL is a flat list of triangles, each carrying its own three
vertices with no index sharing. Round-tripping a closed mesh through it
produces a mesh where *no edge is used twice* — so `is_watertight()` correctly
reports `False` about a mesh that is, in fact, closed.

The check is on the geometry, so the load path welds within a tolerance and
reports both counts. `--no-weld` exists for looking at the file as stored.

## Layout

Each generator is its own source root under `modules/`, so
`cd modules/parts && python -m partkit` works with nothing installed. `geokit`
is reached with a three-line `sys.path` bootstrap from each module — the same
resolution in both contexts, so standalone and integrated cannot drift apart.

`pytest` from the root needs `--import-mode=importlib` and a root `conftest.py`
adding each module folder, because several generators ship their own `tests`
package and the default import mode resolves them all to one top-level name.
