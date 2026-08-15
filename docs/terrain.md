# Architecture

```
fractal noise ──▶ heightmap ──▶ hydraulic erosion ──▶ eroded heightmap
  (noise.py)                        (erosion.py)              │
                                                              ├──▶ hypsometric map (2D)   (colormap.py, image.py)
                                                              └──▶ watertight solid ──▶ Blender (3D)   (mesh.py, blender_export.py)
```

## 1. Fractal noise (`noise.py`)

**Value noise, one octave.** Fill a coarse `cells × cells` lattice with seeded random values and interpolate it up to the target resolution. The interpolation weight is `smoothstep(t) = t²(3 − 2t)` rather than a straight line, so the field has continuous first derivatives — no creased artefacts along cell boundaries (a property the tests check by bounding the per-pixel step). One octave is gentle, single-scale hills.

**Fractal Brownian motion.** Real terrain has structure at every scale, so `fbm` sums several octaves, each at `lacunarity`× the frequency and `persistence`× the amplitude of the last (2× and ½× by default). Broad landmasses come from the low octaves, ridges from the middle, and fine roughness from the high ones. The sum is normalised by the total amplitude so the result stays in `[0, 1]`. Because every octave is seeded deterministically (`seed + 1013·octave`), the whole heightmap is a pure function of the seed. A final mild exponent (`relief`) sharpens peaks and flattens valleys, which reads as more mountainous.

## 2. Hydraulic erosion (`erosion.py`)

This is the interesting half, and the model is the standard droplet simulation. A droplet starts at a random point carrying water and no sediment, and for a fixed lifetime:

1. **Flow downhill.** Compute the bilinear height and gradient under the droplet's floating-point position, and steer it in the negative-gradient direction, blended with its previous direction by an `inertia` term (so it doesn't turn on a dime). Step one cell along that direction.
2. **Erode or deposit.** Compare the new height to the old. If the droplet moved *uphill* (into a pit), it deposits enough sediment to fill the step. If it moved downhill, its **carrying capacity** is proportional to the slope, its speed, and its water; carrying more than capacity, it deposits the excess, and carrying less, it erodes a fraction of the deficit from the terrain (never more than the height step, to avoid digging spikes).
3. **Update.** Speed grows with the drop (`v ← √(v² + Δh·g)`), water evaporates a little, and the droplet advances.

Both erosion and deposition are spread **bilinearly** over the four cells under the droplet — the same weights used to read the height — so the terrain stays smooth and no single vertex gets a spike.

**Why it conserves mass — exactly.** Every gram of soil a droplet lifts is added to the droplet's `sediment`, and every gram it drops is taken from `sediment`. The only way mass could leak is a droplet ending its life still loaded, or running off the edge of the map with a load. Both are handled: a droplet deposits *all* remaining sediment when its lifetime ends, when it stalls, and — crucially — right before it crosses the border. So over the whole run the soil lifted equals the soil deposited, and the total elevation summed over the grid is invariant. The code tracks `eroded` and `deposited` totals and the height-sum before and after; the tests assert the relative change is below `1e-9` (in practice it is at the `1e-16` floor). That invariant is what distinguishes a physical simulation from a texture filter that merely looks eroded.

The simulation is deterministic: one seeded generator draws every droplet's start position, so the same seed reproduces the landscape bit for bit.

## 3. The hypsometric map (`colormap.py`, `image.py`)

The 2D map is elevation → colour through the familiar cartographic ramp (deep water, shallows, sand, grass, forest, rock, scree, snow), interpolated between stops. On its own that is a flat paint-by-numbers; **hillshading** gives it depth. The surface normal is reconstructed from the heightmap gradient, and each pixel is scaled by the Lambert term `n·l` against a low "sun," so slopes facing the light brighten and those facing away fall into shadow — which is exactly what makes the eroded drainage network visible. The flat water surface is left unshaded so it reads as water rather than lit rock. The image is written by a small hand-rolled PNG encoder (signature, `IHDR`, `zlib`-compressed `IDAT`, `IEND`), so the only dependency is NumPy.

## 4. From heightmap to solid (`mesh.py`)

A heightmap is a *surface*, and a surface is not a printable or simulable object — it has an open boundary all along its rim. `terrain_mesh` closes it into a solid slab: the top is the heightmap, triangulated two triangles per cell; a flat underside sits below the lowest point; and four side walls connect the two along the border. Every edge then belongs to exactly two triangles, which the tests verify (and the Euler characteristic comes out 2, a genus-0 solid). The mesh writes to binary STL by hand — an 80-byte header, a triangle count, then 50 bytes per triangle.

## 5. Rendering (`blender_export.py`)

For 3D, the solid is rebuilt in Blender with `from_pydata` and each face is assigned a biome material by its mean height — water, sand, grass, rock, snow — then lit by a single low sun and rendered with Cycles on the CPU (headless, no GPU or display). As with every project in this group, Blender is optional: the heightmap, the erosion invariant, the 2D map, and the watertight STL are all produced and verified without it; Blender only turns the verified terrain into a 3D picture.
