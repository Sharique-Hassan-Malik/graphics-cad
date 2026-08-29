# Architecture

The whole system is one idea carried through cleanly: a sketch is a vector of
numbers **q** and a set of equations **r(q) = 0**, and everything — solving,
extruding, the "fully defined" verdict — is an operation on those two objects.

```
Sketch (points, lines, circles, constraints)
        │  assemble
        ▼
   r(q)  and  J(q) = ∂r/∂q          (constraints.py)
        │  root-find
        ▼
   solved q  +  rank(J) analysis     (solver.py)
        │  read profile back
        ▼
   closed 2D loop ──▶ Blender bmesh extrude ──▶ solid   (blender_export.py)
```

## 1. The parameter vector and entities (`sketch.py`)

There is exactly one array of unknowns: `Sketch._params`, a flat list of floats. Entities are thin handles into it:

- a **Point** owns two slots — the indices of its x and y in the vector;
- a **Circle** is a centre Point plus one radius slot;
- a **Line** owns *no* slots at all — it is a pair of references to two Points.

That last choice is the reason a constraint solver feels connected. A line and the points it is drawn through are the *same* points, so a constraint on the line writes derivatives against those points' slots, and moving a point moves every line built on it. There is no separate "geometry" and "constraint" copy of a coordinate to keep in sync.

A parallel boolean array, `_free`, marks which slots the solver may change. `fix(point)` clears both of a point's flags: the coordinate becomes a constant, not an unknown. Anchoring one point (and one direction) is how a sketch loses its free translation and rotation so it can be fully defined.

## 2. Constraints: residual + analytic Jacobian (`constraints.py`)

Each constraint is one or more scalar equations that should equal zero, plus the partial derivatives of those equations. A constraint exposes two methods:

- `residuals(q) → array` — the current violation(s);
- `jacobian(q) → list of {index: ∂r/∂param}` — one sparse dict per residual, holding only the handful of parameters that equation touches.

The residuals are written in **smooth, root-findable** form so Newton sees a well-behaved system:

| constraint | residual | note |
|---|---|---|
| Coincident(p,q) | `pₓ−qₓ`, `p_y−q_y` | linear, two equations |
| Distance(p,q,d) | `‖p−q‖² − d²` | squared, so it is smooth through zero |
| Horizontal / Vertical | `p_y−q_y` / `pₓ−qₓ` | linear |
| Parallel(l₁,l₂) | `u×v` | cross product of directions |
| Perpendicular(l₁,l₂) | `u·v` | dot product |
| EqualLength(l₁,l₂) | `‖u‖² − ‖v‖²` | squared lengths |
| PointOnLine(p,l) | `(b−a)×(p−a)` | signed area = 0 |
| PointOnCircle / Radius | `‖p−c‖² − R²` / `R − R₀` | |
| Tangent(c₁,c₂) | `‖c₁−c₂‖² − (R₁±R₂)²` | + external, − internal |
| Angle(l₁,l₂,θ) | `cosθ·(u×v) − sinθ·(u·v)` | see below |

The **Angle** residual is the one worth dwelling on. Writing the angle directly needs an `arctan2` and a division by lengths — non-smooth and awkward to differentiate. Instead, since `u×v = ‖u‖‖v‖sinφ` and `u·v = ‖u‖‖v‖cosφ` for the angle φ between the directions,

```
cosθ·(u×v) − sinθ·(u·v) = ‖u‖‖v‖·(sinφ cosθ − cosφ sinθ) = ‖u‖‖v‖·sin(φ − θ),
```

which is zero exactly when φ = θ, is polynomial in the coordinates, and has a Jacobian that is just a linear blend of the cross-product rows (as in Parallel) and the dot-product rows (as in Perpendicular). No trigonometry of the unknowns, no division.

**Why analytic derivatives, and how they are trusted.** An exact Jacobian is what gives Newton its quadratic convergence — the reason the solver needs five or six iterations, not fifty. But a hand-derived derivative is exactly the kind of thing that is subtly wrong and still *almost* works. So the test suite takes every constraint at a random configuration and compares its analytic Jacobian, entry by entry, against a central finite-difference of its own `residuals()`. A wrong sign or a missing term shows up as a mismatch, not as a mysteriously slow solve.

## 3. The solver (`solver.py`)

`Sketch.solve()` assembles the global residual vector (concatenate every constraint's residuals) and the global Jacobian (scatter every sparse row into an `m × n` matrix), restricts the Jacobian to the free columns, and hands them to `solve()`.

The step is **Gauss–Newton first, Levenberg–Marquardt as a safety net**:

1. Try the full least-squares step `δ = argmin ‖J δ + r‖`, via `lstsq`. Near the solution this converges quadratically, and `lstsq` also does the right thing when `J` is rank-deficient (an under-constrained sketch) by returning the minimum-norm step instead of blowing up.
2. If that step does not *decrease the objective* `‖r‖²`, it overshot — fall back to the damped normal equations `(JᵀJ + λI) δ = −Jᵀr`, growing λ until a step reduces the objective. λ shrinks again on success, so the method drifts back toward pure Newton as it homes in.

Acceptance is tested on the least-squares objective `‖r‖²`, not the max-norm — using the max-norm as the gate rejects good steps that trade a large drop in most residuals for a tiny rise in one, and turns a six-iteration solve into a hundred-iteration crawl.

## 4. Degrees of freedom = rank of the Jacobian

This is the part that makes the tool a *sketcher* and not just a solver. At the solution, with `n` free parameters and `m` constraint equations, take the numerical rank `r` of the Jacobian (via SVD, with a scale-aware singular-value threshold):

- **remaining DOF = n − r** — how many independent ways the geometry can still move without breaking any constraint. Zero means fully defined.
- **redundant = m − r** — how many constraints are linearly dependent on the others. Greater than zero means over-defined (some constraint is implied, like a perpendicularity that two horizontal/vertical edges already force).

So `status` reads straight off two subtractions: `dof > 0` → under-constrained, else `redundant > 0` → over-constrained, else fully-constrained. The showcase's rectangle-minus-a-dimension reports 1 DOF; adding an implied perpendicular reports 1 redundant; a well-posed sketch reports 0 and 0.

**Conditioning is a design choice.** Two constraint sets can define the *same* regular hexagon and behave completely differently. "Every vertex on a circle, every edge equal length" is ill-conditioned — equal chords leave a soft breathing mode, condition number ~5×10⁴, and the solve crawls. "A fixed centre, every vertex at radius R, equal angles between the radial lines" — which is how a CAD user actually draws a polygon — is condition ~7 and solves in six steps to the same answer. `demos.py` uses the second, deliberately.

## 5. Extrusion in Blender (`blender_export.py`)

The contrast with a mesh-first kernel is intentional. Here the solver produces a *solved, constrained outline* and Blender does the modelling. The emitted script builds the polygon with `bmesh`, then calls `bmesh.ops.extrude_face_region` and translates the new face along +Z — Blender's own extrude feature, applied to a profile whose vertices satisfy the constraints exactly. `recalc_face_normals` orients it into a valid solid, and the same Cycles-CPU render setup as elsewhere (auto-framed camera, brass material) turns it into a picture.

As everywhere, Blender is held at arm's length: `to_bpy_script()` returns a program you can run with `blender --background --python`, and if no Blender binary is found `render()` writes the script and reports `ran: False`. The solve — the hard part, the part with the correctness proof — never touches Blender.
