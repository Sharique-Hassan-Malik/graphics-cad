# Architecture

The pipeline is a straight line from an arithmetic expression to a verified solid:

```
SDF expression tree ──sample──▶ scalar field on a grid ──marching tetrahedra──▶ triangle soup
   (sdf.py)                        (marching.py)                                   │
                                                                          weld + orient
                                                                                   │  (mesh.py)
                                                                                   ▼
                                                          watertight manifold ──▶ topology / volume checks
                                                                                   └──▶ STL / Blender
```

## 1. Signed distance fields (`sdf.py`)

An SDF is a function `f(p)` giving the signed distance from `p` to a surface: negative inside, zero on it, positive outside. Every field here is a small object with a vectorised `distance(points)` — call it on an `(N,3)` array, get back `N` distances — and Python operators build an expression tree:

- **Primitives** are closed-form distances. A sphere is `‖p−c‖ − r`. A box is the classic `‖max(|p−c|−h, 0)‖ + min(max(qₓ,q_y,q_z), 0)` (exact outside *and* inside). A torus reduces the 3D distance to a 2D one on `(√(x²+y²) − R, z)`. A cylinder is a 2D box in `(radial, axial)`.
- **Combinators** are the reason to use SDFs at all: `Union = min`, `Intersection = max`, `Difference = max(A, −B)`. `SmoothUnion` replaces the hard `min` with a soft blend of width `k`, producing the rounded fillet where two shapes meet — the metaball look. `Round(f, r) = f − r` inflates the field, rounding every convex edge to radius `r`. `Shell(f, t) = |f| − t/2` turns a solid into a wall.

Two services every node provides, used by the mesher:

- **`gradient(p)`** — central differences of the field. For a distance field the gradient points *outward* (toward increasing distance) everywhere the field is smooth, which is what lets the mesher decide which way a face should point.
- **`bounds()`** — a conservative axis-aligned box that contains the surface, combined up the tree (a union's box is the union of boxes, a difference fits inside its first operand). The mesher meshes inside this box, padded, so it never has to guess a domain.

## 2. Marching tetrahedra (`marching.py`)

**The decomposition.** The domain is a regular grid of samples. Each grid *cube* is split into six *tetrahedra* using the Freudenthal (Kuhn) decomposition: the six tetrahedra correspond to the six orderings of the axes, each a path from corner `(0,0,0)` to `(1,1,1)` taking one unit axis-step at a time. All six therefore share the cube's main diagonal `0↔7`. The point of this particular decomposition is that it tiles space *consistently*: apply it identically to every cube and the diagonal chosen on any face shared by two cubes is the same from both sides. Shared faces triangulate identically, so the extracted surface has no cracks — it is watertight before any welding.

**The cases.** A tetrahedron has four corners, each inside (`f<0`) or outside. Up to symmetry there are only three outcomes:

- **0|4** — all same sign: the surface misses this tetrahedron, no triangles.
- **1|3** — one corner is the minority: the surface cuts the three edges leaving that corner, giving **one triangle**.
- **2|2** — two against two: the surface cuts the four edges between the groups, giving a **quad → two triangles**.

That is the entire algorithm; there is no 256-entry table and no ambiguous case. The code handles it generically: for a 1|3 split it interpolates the three edges from the lone corner; for a 2|2 split with inside corners `{a,b}` and outside `{c,d}` it walks the quad `(a,c)→(a,d)→(b,d)→(b,c)` and emits two triangles across it. Each crossing point is a linear interpolation `p_a + t·(p_b − p_a)` with `t = f_a/(f_a − f_b)` — and because that value depends only on the two shared grid samples, the *same* point is produced from every tetrahedron that shares the edge, which is why welding closes the surface exactly.

**Vectorisation.** The naïve version loops over hundreds of thousands of tetrahedra in Python. Instead the field is sampled in one call, the eight corner blocks of every cube are gathered by array slicing, and each of the six tetrahedron types is processed across *all* cells at once — the 1|3 and 2|2 cases become boolean masks over the whole grid. A 64³ field meshes in well under a second.

Winding is deliberately *not* tracked here; the marching step emits triangles in whatever order is convenient and leaves orientation to the mesh.

## 3. From soup to solid (`mesh.py`)

The marching step emits independent triangles — a "soup" in which the two triangles meeting at a shared edge hold *coincident but distinct* vertices. Two operations turn it into a solid:

**Welding** quantises vertex coordinates to a tolerance, fuses duplicates with `np.unique`, remaps the faces through the inverse index, and drops triangles that collapsed to a line. After this, "adjacent" means "shares a vertex index," and the counting checks below become meaningful.

**Orientation** is done in two stages, and the second stage is subtler than it looks:

1. *Consistency* — a flood fill across the manifold. Two faces sharing an edge are consistently wound exactly when that edge runs *opposite* ways through them; any neighbour that agrees is flipped. This is purely topological (no normals), which makes it robust across the sharp creases of CSG results, where a finite-difference gradient is noisy and a per-face normal test would flip adjacent faces inconsistently.
2. *Global sign, per connected component* — decided by a **majority vote of the SDF gradient** over each component's faces, not by forcing positive volume. That distinction matters for **internal cavities**: cut a small sphere entirely inside a box and the result is a solid box with a spherical void. The void's surface is a second connected component whose outward-facing normals point *into* the void, so it must contribute *negative* volume. Forcing every component to positive volume would wrongly inflate the total (and indeed did, until the gradient vote replaced it); the gradient — which points out of the *material* everywhere — gets both the outer shell and the cavity right. Such a solid correctly reports χ=4 (two closed shells) and a volume of box minus sphere.

## 4. Why the Euler characteristic is the right check

For a closed orientable surface, χ = V − E + F = 2 − 2g, where g is the genus (the number of through-holes). So χ is a cheap integer fingerprint of topology that a broken boolean cannot fake:

- a **ball** (die, blob, cross) has χ = 2, g = 0 — one closed shell, no holes;
- a **torus** (ring) has χ = 0, g = 1 — one through-hole;
- an **n-bolt-hole plate** has χ = 2 − 2n — the bracket's four holes give χ = −6, g = 4;
- a **solid with a cavity** has χ = 4 — two shells.

If a union leaked a hole, or a difference doubled a wall, or the mesh were merely a hollow shell, χ would come out wrong. Getting it right — together with a volume that matches the analytic answer and converges under refinement — is what certifies the extracted surface as a genuine, printable, physics-ready solid rather than a convincing picture.
