"""Tests for the engine kinematics.

The animation is only as honest as the maths behind it, so the maths is what is
tested: the slider-crank position against the geometry it is derived from, the
piston staying inside its stroke, the analytic velocity against a numeric
derivative, and — the property that makes an engine *run* — every cylinder
reaching top dead centre exactly when the firing order says it fires, with the
power strokes evenly spaced.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enginekit import slider_crank as sc  # noqa: E402
from enginekit.engine import Engine  # noqa: E402


# ---------------------------------------------------------------------------
# slider-crank geometry
# ---------------------------------------------------------------------------


def test_piston_matches_the_law_of_cosines():
    r, l = 0.5, 1.6
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = sc.piston_position(theta, r, l)
    # independent solution of the rod-length constraint x² − 2r·cosθ·x + (r²−l²) = 0
    b = -2 * r * np.cos(theta)
    c = r * r - l * l
    x_check = (-b + np.sqrt(b * b - 4 * c)) / 2
    assert np.max(np.abs(x - x_check)) < 1e-12


def test_tdc_bdc_and_stroke():
    r, l = 0.5, 1.6
    assert sc.piston_position(0.0, r, l) == pytest.approx(r + l)       # top dead centre
    assert sc.piston_position(math.pi, r, l) == pytest.approx(l - r)   # bottom dead centre
    assert sc.stroke(r) == pytest.approx(2 * r)


def test_piston_stays_within_the_stroke():
    r, l = 0.5, 1.6
    theta = np.linspace(0, 2 * np.pi, 2000)
    disp = sc.piston_displacement(theta, r, l)
    assert disp.min() >= -1e-12
    assert disp.max() <= sc.stroke(r) + 1e-12


def test_rod_angle_is_bounded():
    r, l = 0.5, 1.6
    theta = np.linspace(0, 2 * np.pi, 2000)
    ang = sc.rod_angle(theta, r, l)
    assert np.max(np.abs(ang)) <= math.asin(r / l) + 1e-12


def test_analytic_velocity_matches_numeric():
    r, l = 0.5, 1.6
    theta = np.linspace(0.01, 2 * np.pi - 0.01, 2000)
    v = sc.piston_velocity(theta, r, l, omega=1.0)
    x = sc.piston_position(theta, r, l)
    v_num = np.gradient(x, theta)
    assert np.max(np.abs(v - v_num)[5:-5]) < 1e-4


def test_crank_pin_is_exactly_one_rod_length_from_the_piston():
    # the whole slider-crank premise: pin-to-piston distance is constant = l
    r, l = 0.5, 1.6
    theta = np.linspace(0, 2 * np.pi, 500)
    along, perp = sc.crank_pin(theta, r)     # crank pin: along-bore, perpendicular
    piston = sc.piston_position(theta, r, l)  # piston pin: on the bore axis (perp = 0)
    dist = np.hypot(along - piston, perp)
    assert np.allclose(dist, l, atol=1e-12)


# ---------------------------------------------------------------------------
# engine timing — the headline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [Engine.inline4, Engine.inline6, Engine.v8])
def test_every_cylinder_fires_at_top_dead_centre(factory):
    engine = factory()
    for cyl, angle in engine.firing_angles().items():
        disp = engine.piston_displacements(angle)[cyl - 1]
        assert abs(disp) < 1e-9, (factory.__name__, cyl, disp)


@pytest.mark.parametrize("factory", [Engine.inline4, Engine.inline6, Engine.v8])
def test_power_strokes_are_evenly_spaced(factory):
    engine = factory()
    angles = sorted(engine.firing_angles().values())
    gaps = np.diff(angles + [angles[0] + 720])
    assert np.allclose(gaps, engine.firing_interval)


def test_inline4_reproduces_the_classic_crank_and_order():
    engine = Engine.inline4()
    assert engine.firing_order == [1, 3, 4, 2]
    assert [round(o) for o in engine.throw_offsets] == [0, 180, 180, 0]


def test_v8_is_cross_plane():
    engine = Engine.v8()
    throws = sorted(set(round(o) % 360 for o in engine.throw_offsets))
    assert throws == [0, 90, 180, 270]        # cross-plane V8 crank


def test_pistons_move_in_antiphase_on_an_inline_four():
    engine = Engine.inline4()
    disp = engine.piston_displacements(0.0)
    # cylinders 1 & 4 at TDC, 2 & 3 at BDC
    assert disp[0] == pytest.approx(0.0, abs=1e-9)
    assert disp[3] == pytest.approx(0.0, abs=1e-9)
    assert disp[1] == pytest.approx(engine.stroke, abs=1e-9)
    assert disp[2] == pytest.approx(engine.stroke, abs=1e-9)


# ---------------------------------------------------------------------------
# valve timing
# ---------------------------------------------------------------------------


def test_valves_open_and_close_over_the_cycle():
    engine = Engine.inline4()
    lift = np.array([engine.valve_lift(a) for a in range(0, 720, 5)])
    # both valves reach near-full lift somewhere and are shut somewhere
    assert lift[:, 0, 0].max() > 0.9 and lift[:, 0, 0].min() < 0.05   # intake
    assert lift[:, 0, 1].max() > 0.9 and lift[:, 0, 1].min() < 0.05   # exhaust


def test_intake_and_exhaust_do_not_peak_together():
    engine = Engine.inline4()
    # a four-stroke opens intake and exhaust on different strokes
    both_high = 0
    for a in range(0, 720, 3):
        intake, exhaust = engine.valve_lift(a)[0]
        if intake > 0.5 and exhaust > 0.5:
            both_high += 1
    assert both_high < 6   # only a brief valve overlap near TDC


# ---------------------------------------------------------------------------
# animation sampling (no Blender needed)
# ---------------------------------------------------------------------------


def test_motion_sampling_has_finite_consistent_shapes():
    from enginekit.animate import sample_motion

    data = sample_motion(Engine.inline4(), frames=40, revolutions=2)
    assert len(data["piston_z"]) == 40
    assert len(data["piston_z"][0]) == 4
    assert np.all(np.isfinite(np.array(data["rod_mid"])))
    # rod midpoints and piston heights are bounded by the mechanism size
    assert np.max(np.abs(np.array(data["rod_mid"]))) < data["r"] + data["l"]


def _blender():
    from enginekit.animate import find_blender

    return find_blender()


@pytest.mark.skipif(_blender() is None, reason="Blender not installed")
def test_blender_renders_animation(tmp_path):
    from enginekit.animate import render

    result = render(Engine.inline4(), out_dir=str(tmp_path), name="e",
                    frames=6, revolutions=1, resolution=(160, 120), make_gif=False)
    assert result["ran"], result
    frames = [f for f in os.listdir(result["frame_dir"]) if f.endswith(".png")]
    assert len(frames) == 6
