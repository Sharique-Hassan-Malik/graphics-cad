# Sketch Solver

> Part of [Graphics & CAD](../../README.md). Runs standalone from this folder;
> the mesh type and the Blender path come from `geokit`.

A 2D **geometric constraint solver** — the mathematical engine underneath every parametric CAD sketcher (SolidWorks, Fusion 360, FreeCAD, Onshape). You describe a shape not by its coordinates but by its *relationships* — "these edges are equal," "this line is horizontal," "these two circles are tangent," "this distance is 40 mm" — and the solver finds the coordinates that satisfy all of them at once, then reports whether your sketch is **fully defined, under-defined, or over-defined**. Solved profiles are extruded into solids by **Blender**.

Written from scratch: the constraint equations, their analytic derivatives, the Gauss–Newton / Levenberg–Marquardt solver, and the degrees-of-freedom analysis are all here in plain NumPy — no CAD library, no `scipy.optimize`.

## Two headline results

**1. It solves — to machine precision, in a handful of Newton steps — and it tells you the truth about your sketch.** The rank of the constraint Jacobian at the solution gives the exact degrees of freedom, which is the "fully defined" verdict a CAD sketcher shows you, computed rather than guessed.

**2. On a nonlinear problem, the answer is provably correct.** Give it three mutually tangent circles and a fourth nestled in the gap, constrained only by tangency, and the radius it finds matches **Descartes' Circle Theorem** to ~2×10⁻¹⁶ mm. The solver never knows the theorem; it finds the geometry, and the geometry obeys the theorem.

```
$ python3 bench/showcase.py

sketch                 verdict            dof  redund  residual  iters  geometric check
---------------------  -----------------  ---  ------  --------  -----  ---------------------
rectangle 40x25        fully-constrained  0    0       1.0e-27   6      corner ⟂ err: 2.5e-26
regular 6-gon r20      fully-constrained  0    0       5.7e-14   6      side spread: 1.1e-14
L-bracket 40x30        fully-constrained  0    0       5.8e-25   6      corner err: 5.8e-25
tangent circles 3/4/5  fully-constrained  0    0       1.4e-14   5      vs Descartes: 2.2e-16
```

Every column is a measured number. `verdict`/`dof`/`redund` come from the Jacobian rank; `residual` is the max constraint violation at the solution; `geometric check` is a shape-specific truth test (are the corners actually square, are the sides actually equal, does the radius actually match theory).

## What "solving" means

A sketch is a vector of parameters **q** (every point's x and y, every circle's radius) and a set of scalar equations **r(q) = 0**, one per constraint. The equations are nonlinear — a distance is quadratic, a tangency quadratic, an angle trigonometric — so solving is a root-find, done with Gauss–Newton and Levenberg–Marquardt damping. Two properties make it trustworthy rather than merely convergent:

- **The Jacobians are analytic and exact.** Every constraint supplies its own partial derivatives, and `tests/` cross-checks each one against a finite-difference of its residual. A wrong derivative can't hide behind a solve that still limps to an answer.
- **Degrees of freedom come from linear algebra, not heuristics.** With *n* free parameters and constraint Jacobian of rank *r*: **n − r** degrees of freedom remain (0 ⟺ fully defined) and **m − r** constraints are redundant (>0 ⟺ over-defined). That is exactly the verdict a CAD sketcher computes to colour your sketch "fully constrained."

## Constraint vocabulary

| constraint | meaning |
|---|---|
| `Coincident(p, q)` | two points are the same point |
| `Distance(p, q, d)` | fixed distance between two points |
| `Horizontal(p, q)` / `Vertical(p, q)` | the segment is axis-aligned |
| `Parallel(l1, l2)` / `Perpendicular(l1, l2)` | angle between two lines is 0 / 90° |
| `Angle(l1, l2, θ)` | fixed directed angle between two lines |
| `EqualLength(l1, l2)` | two segments are the same length |
| `PointOnLine(p, line)` | a point lies on a line |
| `PointOnCircle(p, circle)` | a point lies on a circle |
| `Radius(circle, r)` | fixed radius |
| `Tangent(c1, c2, internal=…)` | two circles are tangent (externally or internally) |

Points can be **pinned** (`sketch.fix(p)`), removing them from the unknowns — the usual way to anchor a sketch so it has no free rigid-body motion.

## Blender integration

Blender is the modelling backend, and it is optional. Once a sketch is solved, its ordered points are a closed profile; `blender_export` emits a self-contained script that builds that polygon with `bmesh` and pulls it into a solid with Blender's own `extrude_face_region` operator — "solve the sketch, then extrude the feature," the way a CAD package works. `render()` runs it headless (Cycles on CPU, no GPU or display needed) to produce a `.blend` and a PNG.

If Blender is absent, nothing breaks: every sketch is fully solved and verified without it, and you still get the solved coordinates and the build script. Set `$BLENDER_BIN` or put `blender` on `PATH` to enable rendering.

![a solved hexagon, extruded in Blender](docs/sketch.png)

*The `regular 6-gon` from the table above — six points solved to lie on a circle at equal angular spacing, then extruded 8 mm by Blender's bmesh into a hexagonal prism.*

## Quick start

```bash
pip install -r requirements.txt          # numpy, pytest

# Solve a parametric sketch and read its DOF verdict (no Blender needed):
python3 -m sketchkit rectangle --width 40 --height 25
python3 -m sketchkit polygon --sides 6 --radius 20
python3 -m sketchkit bracket --length 40 --height 30 --wall 8

# The nonlinear one, checked against Descartes' theorem:
python3 -m sketchkit descartes --r1 3 --r2 4 --r3 5

# Solve, extrude and render (uses Blender if present):
python3 -m sketchkit polygon --sides 6 --radius 20 --render ./out --thickness 8
```

```bash
python3 examples/solve_and_extrude.py    # build a sketch by hand, solve, extrude
python3 bench/showcase.py                # the verification table above
pytest tests/                            # 26 tests, all measured assertions
```

## As a library

```python
from sketchkit import Sketch
from sketchkit.constraints import Horizontal, Vertical, Distance

s = Sketch()
p0, p1, p2, p3 = s.point(0, 0), s.point(38, 2), s.point(41, 27), s.point(-1, 24)
s.fix(p0)
s.add(Horizontal(p0, p1), Vertical(p1, p2), Horizontal(p2, p3), Vertical(p3, p0),
      Distance(p0, p1, 40), Distance(p1, p2, 25))

result = s.solve()
print(result.status)          # "fully-constrained"
print(result.dof)             # 0
print(s.coords([p0, p1, p2, p3]))   # an exact 40 x 25 rectangle
```

## Layout

| file | what it holds |
|---|---|
| `sketchkit/solver.py` | the numerical core: Gauss–Newton + Levenberg–Marquardt, and the DOF/rank analysis |
| `sketchkit/constraints.py` | every constraint as a residual plus its analytic Jacobian rows |
| `sketchkit/sketch.py` | the `Sketch` builder: points, lines, circles, pinning, `solve()`, profile extraction |
| `sketchkit/demos.py` | parametric example sketches (rectangle, polygon, bracket, tangent circles) |
| `sketchkit/blender_export.py` | emit a `bmesh` build+extrude script; drive Blender headless to render |
| `sketchkit/cli.py` | `python3 -m sketchkit …` |
| `tests/` | 26 tests, each a measured assertion |
| `bench/showcase.py` | the verification table |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the mathematics — how each constraint's residual and derivative are formed, how the solver's step is chosen, and how the degrees-of-freedom verdict falls out of the singular values.

## License

MIT — see [`LICENSE`](./LICENSE).
