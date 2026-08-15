# Architecture

The pipeline is short and one-directional:

```
parameters ──▶ 2D profile ──▶ extrude to a solid ──▶ Mesh ──▶ STL / Blender
 (module,        (profiles.py)     (solids.py)        (mesh.py)   (blender_export.py)
  teeth, …)                                              │
                                                         └──▶ verify: topology, volume,
                                                              involute deviation
```

Everything is two NumPy arrays — an `N×3` array of vertices and an `M×3` array of triangle indices. There is no scene graph, no B-rep, no half-edge structure kept around; each stage produces arrays and the next stage consumes them. That is what makes the verification honest: a check reads the finished arrays, so it cannot be fooled by state the generator left behind.

## 1. The mesh and its checks (`mesh.py`)

`Mesh` is a dataclass of `vertices` and `faces`. The load-bearing routine is `_edge_use_counts()`: it walks every triangle's three edges, keys each by its unordered vertex pair, and counts. From that one histogram:

- **watertight** ⟺ every count is exactly 2.
- **edge-manifold** ⟺ no count exceeds 2.
- **boundary edges** = the edges with count 1 (the holes).
- **Euler characteristic** χ = V − E + F, where E is the number of distinct edges. χ = 2 for a closed genus-0 solid, χ = 0 for one through-hole (a torus).

Orientation is a separate, *directed* check: collect every directed edge `(u,v)` in face-winding order; if any directed edge appears twice, two faces wind the same way across it and one of them faces inward. A flipped triangle passes watertightness but fails this.

`volume()` is the signed volume by the divergence theorem — sum over triangles of `((v1−v0) × (v2−v0)) · v0 / 6`. It is the part's volume *and* an orientation witness: it comes out positive exactly when the surface is closed and outward-facing. Comparing it to a closed-form value (πr²h, area×thickness) is the cheapest full-mesh correctness test there is.

`welded()` merges coincident vertices — necessary because the extruders build the wall and the caps as separate vertex sets, and two triangles only count as "adjacent" if they reference the *same* index. It quantises coordinates to a tolerance, uses `np.unique` to find the representative of each cluster, remaps the faces through the inverse index, and drops any triangle whose corners collapsed together.

STL (binary) and OBJ are written by hand. Binary STL is an 80-byte header, a `uint32` triangle count, then 50 bytes per triangle: a float32 normal, three float32 vertices, and a 2-byte attribute word. `from_stl_binary()` parses it back for round-trip tests.

## 2. Profiles (`profiles.py`)

A profile is a closed 2D loop as an ordered `K×2` array, counter-clockwise. The simple ones (circle, rectangle, rounded rectangle, regular polygon) are direct. The interesting one is the gear.

### The involute tooth

An involute gear's flank is the *involute of a base circle* of radius `rb`: the curve traced by the end of a string unwound from that circle. Its constant-velocity property — the reason involute gears are universal — is a direct consequence of this definition. In parametric form, unwinding by angle `t`:

```
x(t) = rb·(cos t + t·sin t)
y(t) = rb·(sin t − t·cos t)
```

The standard proportions, from the module `m`, tooth count `z`, and pressure angle `α`:

| radius | value | meaning |
|---|---|---|
| pitch  | `r = m·z / 2`      | where meshing gears roll without sliding |
| base   | `rb = r·cos α`     | the circle the involute unwinds from |
| outer / addendum | `ra = r + m`   | tooth tip |
| root / dedendum  | `rf = r − 1.25 m` | tooth valley |

For a point at radius ρ on the flank, the angle from the tooth's centreline is

```
flank(ρ) = θ0 − inv(arccos(rb/ρ)),   where  inv(a) = tan a − a  (the involute function)
           θ0 = π/(2z) + inv(α)
```

`θ0` places the flank so the tooth has the correct thickness at the pitch circle (a tooth and a gap each span half the pitch there). `involute_gear_profile()` walks one tooth as a strictly counter-clockwise (increasing-angle) sequence — a root point, the left flank sampled base→tip, a short tip arc, the right flank tip→base, another root point, then the root arc to the next tooth — and rotates that block `z` times. The **CCW ordering matters**: a clockwise traversal makes the polygon's signed area negative, and the tooth ends up *subtracted* from the disk instead of added.

## 3. Extrusion (`solids.py`)

An extruder turns a closed profile into a watertight prism: a wall of quads (two triangles each) connecting the top and bottom copies of the loop, plus a cap over each end. The whole game is capping the ends into a valid triangulation, and the right strategy depends on the loop's shape — so there are three:

- **`triangulate_simple` — ear clipping.** The general method for any simple (non-self-intersecting) polygon, including ones with reflex corners like the L-bracket. Repeatedly find an "ear" (a convex corner whose triangle contains no other vertex) and clip it. O(n²), but robust and dependency-free. `extrude()` uses it.
- **`extrude_star` — centre fan.** A gear outline is *star-shaped* from its centre (a ray from the origin crosses the boundary once), so it caps trivially by fanning triangles from the centroid to each edge. This sidesteps a real failure of ear-clipping on gears: ~400 near-collinear flank vertices make the reflex test numerically fragile, and it wrongly rejects valid ears. Solid gears use this.
- **`extrude_ring` — matched quad cap.** For a part with a hole (a bored gear, a washer), the cap is the *annulus* between an outer loop and an inner loop. If the two loops have the same length and are radially aligned — inner vertex `i` at the same polar angle as outer vertex `i` — the cap is just one quad per index, no triangulation search at all. `bore_matching()` builds exactly such an inner loop: it samples a circle at the *outer* outline's vertex angles, guaranteeing the alignment `extrude_ring` needs. Bored gears use this.

The earlier annulus code compared *angular step sizes* while sweeping the two loops, which is invalid when the outer loop's angles aren't monotone (a gear's are not — they back up over each tooth). Sampling the bore at the outline's own angles removes the sweep entirely.

## 4. Gears, forward and backward (`gears.py`)

`spur_gear()` is the forward direction: build the profile, pick the extruder by whether there's a bore, centre the solid on z, weld. Ten lines.

The backward direction is what makes the result trustworthy:

- **`measure_gear()`** recovers geometry from *the vertices alone*, without reading the build parameters. Outer and root radii are the extreme distances from the axis; tooth count comes from angular clustering of the tip vertices (`_count_clusters` finds the gaps between tips); module follows from `ra = m·(z/2 + 1)`. The tests compare these to the analytic values — real evidence the part *is* the gear it claims to be, not a tautology.

- **`involute_deviation()`** is the correctness proof for the tooth shape, and it is direct. For every flank vertex (radius between the flank start and the tip), it computes where the exact involute of the base circle places a point at that radius, and measures the arc distance. Two subtleties:

  - **The flank band.** The involute part of a flank runs from the base circle outward — *unless* the root circle sits outside the base circle (`rf > rb`, which happens on high-tooth-count gears that don't undercut), in which case the flank begins at the root and the little arc below it is not involute. Using `flank_start = max(rb, rf)` excludes those root-arc vertices, which would otherwise read as flank points sitting off the involute.
  - **Folding by symmetry.** Rather than track which tooth a vertex belongs to, reduce its polar angle modulo the angular pitch `2π/z` into the window `(−pitch/2, +pitch/2]` — which must span a *whole* tooth, whose flanks reach ±θ0 (wider than half the tooth angle near the base). The absolute folded angle then compares directly to the single canonical flank angle, covering all `z` teeth and both flanks at once.

  Across gears from 12 to 60 teeth, modules 1 to 4, and pressure angles 20°–25°, the maximum deviation is ~10⁻¹⁴ mm — the floating-point noise floor. The flanks *are* the involute.

## 5. Blender (`blender_export.py`)

Blender is the renderer, held at arm's length and optional. `mesh_to_bpy()` emits a self-contained Python fragment that rebuilds the mesh with `bpy.data.meshes.new(...).from_pydata(verts, [], faces)` — the arrays inlined, no import of this package needed on the Blender side. `render_script()` wraps that with a Cycles-CPU setup: a brushed-steel Principled BSDF, a three-point-ish light rig, and a camera auto-framed to the part's bounding sphere so any size of part fills the frame. `render()` locates Blender (`$BLENDER_BIN`, then `PATH`, then common install paths), runs it headless with `--background --factory-startup`, and returns a dict recording what was written.

The degradation is deliberate and total: with no Blender, `render()` still writes the `.blend`-building script and reports `ran: False`, and *nothing else in the project depends on Blender at all* — every part is fully generated and fully verified without it. Blender turns a verified mesh into a picture; it is never in the path that decides whether the mesh is correct.
