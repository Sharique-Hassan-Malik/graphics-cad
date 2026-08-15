#!/usr/bin/env python3
"""Verify the engine timing for every preset, then render the inline-four.

    python3 bench/showcase.py

The numbers are the point: for each engine, every cylinder reaches top dead
centre exactly when the firing order says it fires (piston displacement 0 at the
firing angle), and the power strokes are spaced evenly around the cycle. The
slider-crank position matches the rod-length geometry to machine precision. Only
then is it worth rendering.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enginekit import slider_crank as sc  # noqa: E402
from enginekit.engine import Engine  # noqa: E402


def main():
    out = os.path.join(os.path.dirname(__file__), "..", ".out")
    os.makedirs(out, exist_ok=True)

    # slider-crank accuracy
    r, l = 0.5, 1.6
    theta = np.linspace(0, 2 * np.pi, 4000)
    x = sc.piston_position(theta, r, l)
    b = -2 * r * np.cos(theta)
    x_check = (-b + np.sqrt(b * b - 4 * (r * r - l * l))) / 2
    print(f"slider-crank vs rod-length geometry: max deviation {np.max(np.abs(x - x_check)):.1e}\n")

    rows = []
    all_ok = True
    for name, engine in (("inline4", Engine.inline4()), ("inline6", Engine.inline6()), ("v8", Engine.v8())):
        fa = engine.firing_angles()
        tdc_err = max(abs(engine.piston_displacements(a)[c - 1]) for c, a in fa.items())
        gaps = np.diff(sorted(fa.values()) + [sorted(fa.values())[0] + 720])
        even = np.allclose(gaps, engine.firing_interval)
        ok = tdc_err < 1e-9 and even
        all_ok = all_ok and ok
        rows.append([name, str(engine.n_cylinders), str(engine.firing_order),
                     f"{engine.firing_interval:.0f}°", f"{tdc_err:.1e}", "yes" if even else "NO"])

    print("Engine timing — every cylinder must fire at top dead centre\n")
    print(_table(rows, ["engine", "cyl", "firing order", "interval", "TDC err @fire", "even"]))

    print("""
The TDC-error column is the headline: at the crank angle where the firing order
says a cylinder fires, that cylinder's piston is at top dead centre to machine
precision — the crank throws, the firing order, and the slider-crank geometry all
agree. The power strokes are evenly spaced around the cycle, which is what makes
an engine run without a stumble. The animation is then just these numbers, drawn.""")

    from enginekit.animate import find_blender, render
    if find_blender():
        print("\nBlender found; rendering the inline-four…")
        result = render(Engine.inline4(), out_dir=out, name="showcase_engine",
                        frames=96, revolutions=2)
        if result["ran"]:
            for k in ("mp4", "gif"):
                if k in result:
                    print(f"  wrote {os.path.basename(result[k])} ({os.path.getsize(result[k]):,} B)")
    else:
        print("\nBlender not found; the timing above is already verified without it.")

    return 0 if all_ok else 1


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


if __name__ == "__main__":
    raise SystemExit(main())
