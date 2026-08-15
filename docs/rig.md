# Architecture

```
quat.py  ──▶  rig.py (Part: size + two poses + window)  ──▶  character.py (the 11-part rig)
                     │                                              │
                     └── pose_at(t) = lerp(pos) + slerp(quat) ──────┘
                                          │
                                          ▼
                              animate.py ──▶ Blender (Workbench) ──▶ ffmpeg ──▶ mp4 / gif
```

The invariant the whole project defends is **rigidity**: a transforming part may be reororiented and repositioned but must never deform. Everything below exists to make that true by construction and to measure that it stayed true.

## 1. Quaternions and SLERP (`quat.py`)

A rotation is stored as a unit quaternion `(w, x, y, z)`. The module implements the algebra from scratch — product, axis-angle and Euler constructors, the quaternion→matrix map, and the angle between two orientations — but the load-bearing routine is **SLERP**:

```
slerp(a, b, t) = a·cos(θt) + q⊥·sin(θt),   θ = angle between a and b,  q⊥ = normalize(b − a·cosθ)
```

Two properties make it the right tool. First, if `a` and `b` are unit quaternions then `slerp(a, b, t)` is a unit quaternion for every `t` — it walks the great-circle arc on the unit sphere — so the rotation it yields is always *proper* (orthonormal, `det = +1`). That is the algebraic reason parts stay rigid, and the tests confirm `‖slerp‖ = 1` throughout. Second, it moves at **constant angular speed** (the angle from `a` grows linearly in `t`), so a part rotates evenly rather than lurching — also tested. Two guards matter in practice: if the quaternions point into opposite hemispheres (`a·b < 0`) one is negated so the interpolation takes the *short* way (a quaternion and its negation are the same rotation), and if they are nearly parallel it falls back to a normalized lerp to avoid dividing by `sin θ ≈ 0`.

## 2. Parts, poses, and the staggered timeline (`rig.py`)

A `Part` is a box with a **fixed size** — the size is never interpolated, which is the first half of rigidity — plus a vehicle `Pose` and a robot `Pose`, each a position and a unit quaternion, and a time window `[t_start, t_end]`.

`pose_at(t)` maps the global morph time to that part's local progress with a smoothstep over its window, then lerps the position and SLERPs the orientation. Because each part has its own window, the parts do **not** move in lockstep: at the middle of the morph they are at a spread of progress values (the tests assert a spread > 0.3), which is exactly what makes the motion read as *transforming*. `corners(pose)` returns the eight world-space box corners under a pose — the probe the rigidity tests use: for a rigid body every pairwise corner distance is invariant, and it is, to `~1e-15`.

## 3. The character (`character.py`)

An original stylised rig of eleven boxes: head, torso, pelvis, two upper arms, two forearms, two thighs, two shins, in a red/blue/silver palette. Each part is given two poses — one that assembles a compact vehicle (limbs folded flat, four wheel-blocks at the corners, a cab on top) and one that assembles a standing humanoid — and a window on the shared timeline. The windows are ordered to stage the classic beats: torso and pelvis rise first (`0.0–0.55`), the legs deploy (`0.10–0.72`), the arms unfold (`0.30–0.85`), and the head emerges last (`0.55–0.85`). It is a homage in boxes, not a screen-accurate model; the fidelity lives in the *motion*, not the mesh.

## 4. Driving the animation (`animate.py`)

`sample` walks a raised-cosine morph time `t = ½ − ½cos(2πf/F)` so the clip runs vehicle → robot → vehicle and loops seamlessly, and records each part's `(position, quaternion)` per frame. The generated Blender script builds one cube per part (scaled to the part's size, tinted its colour), then keyframes `location` and — importantly — `rotation_quaternion`, so Blender receives the orientations in exactly the representation the maths produced, with no Euler-order translation to get wrong. A parented empty gives the camera a gentle orbit.

Rendering is **Cycles** (CPU, path-traced, headless — no GPU), with `use_persistent_data` to reuse the scene between frames. Each part is built at true size and bevelled once (uniform chamfers), and given a material chosen from its colour — near-grey brights become chrome, everything else automotive paint with a clear coat — over a softly reflective floor the robot mirrors in, lit three-point. The OpenImageDenoise denoiser keeps samples low, and `ffmpeg` assembles the frames into an MP4 and a GIF. As with every project in this group, none of the correctness lives in Blender — the rigidity is defined and verified in NumPy, and the module degrades to emitting the build script when Blender is absent.
