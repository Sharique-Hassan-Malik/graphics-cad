# Architecture

```
slider_crank.py  ──▶  engine.py  ──▶  animate.py  ──▶  Blender (Workbench) ──▶ ffmpeg ──▶ mp4 / gif
 (one piston's         (N cylinders,     (per-frame
  exact kinematics)     firing order,     keyframes)
                        cam timing)
```

The design rule is that **Blender never computes motion** — it only draws numbers the tested core produced. That is what lets the animation double as a proof: a wrong equation is visible as a piston tearing away from its rod.

## 1. The slider-crank (`slider_crank.py`)

One cylinder is a crank of radius `r` turning through θ, a rod of length `l`, and a piston constrained to the bore axis. Projually, the piston's distance from the crank axis is `x(θ) = r·cosθ + √(l² − r²·sin²θ)`: the crank throw projected onto the bore, plus the rod foreshortened by its own swing. From this one expression:

- **Top / bottom dead centre and stroke.** `x(0) = r + l`, `x(π) = l − r`, so the stroke is exactly `2r` — the tests assert these against the closed form.
- **Rod angle.** The rod leans by `asin(r·sinθ / l)`, bounded by `±asin(r/l)`.
- **Velocity.** `dx/dt = (dx/dθ)·ω` in closed form; the tests check it against a numeric derivative of the position.
- **The crank-pin invariant.** The crank pin is at `(r·cosθ, r·sinθ)` in (bore-axis, perpendicular) coordinates; the piston pin is at `(x(θ), 0)`. Their distance is `√((x − r·cosθ)² + (r·sinθ)²) = √((√(l²−r²sin²θ))² + r²sin²θ) = l`, *identically*. The animation rig leans on exactly this: it can place a rigid rod of fixed length `l` and only rotate it, because the geometry guarantees the endpoints are always `l` apart.

## 2. The engine (`engine.py`)

An N-cylinder four-stroke is N slider-cranks sharing a shaft, each with its crank throw offset, plus the timing that makes them cooperate.

**Throws derived from the firing order.** A cylinder must be at top dead centre at the instant it fires. The firing order lays the power strokes out evenly — the k-th cylinder to fire does so at `k·(720/N)` degrees — so the crank throw that puts a cylinder at TDC at its firing angle is simply the negative of that angle (mod 360). Deriving the throws this way, rather than hard-coding them, makes "fires at TDC" true *by construction* and reproduces the classic cranks: inline-four → `0-180-180-0`, inline-six → `0-240-120-120-240-0`, and the V8 → a cross-plane `0/90/180/270` crank with each throw shared by two cylinders. The tests confirm the piston displacement at each firing angle is zero to machine precision and that the firing events are evenly spaced.

**Valve timing.** Each cylinder's four strokes — power, exhaust, intake, compression — occupy successive 180° windows measured from its own firing angle. A raised-cosine cam lobe opens the exhaust valve through the exhaust up-stroke and the intake valve through the intake down-stroke, with only a brief overlap near TDC (checked in the tests: the two valves are never both fully open for long). The lobe gives the animation smoothly bobbing valves.

**State query.** For any crank angle the engine returns every piston displacement, rod angle, crank-pin position, the `(intake, exhaust)` valve lift per cylinder, and which cylinder (if any) is firing — everything the rig needs, and nothing about Blender.

## 3. Driving the animation (`animate.py`)

`sample_motion` walks the crank through a whole number of revolutions in `frames` steps and, at each step, records where every part must be: piston heights on the bore axis, rod midpoints and lean angles (from the crank-pin and piston-pin positions), valve heights from the cam lift, and a spark flag on the firing cylinder. These arrays are inlined into a generated Blender script.

The script builds the rig once from primitives — a base, a crankshaft empty with a journal, flywheel and crank pins parented to it, and per cylinder a head, a piston, a connecting rod, two valves and a spark marker — then loops over the frames inserting keyframes straight from the sampled arrays, with linear interpolation so the playback is exactly the sampled motion (no Bézier easing smuggled in between frames). The crankshaft is a single empty rotated by θ, so its parented pins and flywheel spin rigidly with it.

**Rendering.** The renderer is **Cycles** (CPU, path-traced, headless — no GPU needed), with `use_persistent_data` so the scene's acceleration structure is reused between frames. Parts are built at their true size (so a single bevel modifier chamfers every edge uniformly) and given physically-based materials — cast-iron block, steel shaft and rods, alloy pistons, brass crank pins. The combustion flash is an emissive material whose emitter grows on the firing cylinder, lit for real into the head above it and bloomed by a compositor glare node; the OpenImageDenoise denoiser keeps the sample count (and so the render time) low. `ffmpeg` assembles the PNG frames into an MP4 and a palette-optimised GIF. None of this is load-bearing for correctness: the kinematics are fully defined and tested with no Blender in the loop, and the module degrades to just emitting the build script.
