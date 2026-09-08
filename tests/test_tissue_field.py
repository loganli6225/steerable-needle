"""Tests for the spatially varying tissue field.

The important one is test_thin_capsule_reaches_its_value: a thin
high-contrast layer is exactly what a naive blend washes out, and the capsule
is the most interesting feature of the profile.

Run:  pytest tests/test_tissue_field.py -v
"""

import math

import pytest

from needlesim.models.tissue_field import PROSTATE_PATH, TissueField, TissueLayer


def test_layer_interiors_hold_their_values():
    """Away from boundaries, kappa should equal the layer's own value. If the
    interiors are blended, the field is not representing distinct media."""
    f = TissueField()
    assert f.kappa_at(0.0, 30.0) == pytest.approx(1 / 34, rel=1e-3)  # fat
    assert f.kappa_at(0.0, 55.0) == pytest.approx(1 / 22, rel=1e-3)  # muscle
    assert f.kappa_at(0.0, 100.0) == pytest.approx(1 / 28, rel=1e-3)  # gland


def test_thin_capsule_reaches_its_value():
    """THE test for the blend construction. The capsule is 5mm thick against
    2mm transitions, so a naive chained-sigmoid blend lets the two adjacent
    transitions overlap and the layer never attains its own kappa -- measured
    at R=19.5mm instead of the intended 15mm, washing out the most
    interesting feature of the profile.

    The membership-weighted construction keeps it: R=15.1mm at the midpoint.
    """
    f = TissueField()
    r_mid = 1.0 / f.kappa_at(0.0, 67.5)
    assert r_mid == pytest.approx(15.0, abs=0.5), (
        f"capsule reached only R={r_mid:.1f}mm at its midpoint, not ~15mm — "
        f"the thin layer is being washed out by its own boundary smoothing, "
        f"which is what the membership-weighted blend exists to prevent"
    )


def test_transitions_are_smooth_and_monotone_between_layers():
    """Crossing an interface, kappa should move steadily from one layer's
    value to the next rather than jumping. A discontinuity would make
    d(kappa)/dy a delta function, which the EKF's linearisation cannot
    represent and the planner's Dubins geometry would see as an instantaneous
    change in turning radius mid-arc."""
    f = TissueField()
    ys = [40.0, 42.0, 44.0, 45.0, 46.0, 48.0, 50.0]  # across fat/muscle
    ks = [f.kappa_at(0.0, y) for y in ys]
    assert all(
        b >= a - 1e-12 for a, b in zip(ks, ks[1:])
    ), "kappa is not monotone across the fat/muscle interface"
    # no single step larger than the whole difference between the two layers
    span = abs(1 / 22 - 1 / 34)
    assert (
        max(abs(b - a) for a, b in zip(ks, ks[1:])) < span
    ), "kappa jumps discontinuously at the interface"


def test_gradient_is_zero_in_interiors_and_nonzero_at_boundaries():
    """The gradient is what the Jacobians will need once a field is in use:
    theta_dot = v*kappa(x,y)*b depends on position, so the theta row of
    NeedleEKF's Jacobian stops being [0, 0, 1]. In a layer interior the field
    is flat and the new terms vanish; at a boundary they do not."""
    f = TissueField()
    _, dk_dy_interior = f.kappa_gradient_at(0.0, 30.0)
    _, dk_dy_boundary = f.kappa_gradient_at(0.0, 45.0)
    assert abs(dk_dy_interior) < 1e-6, "field should be flat inside a layer"
    assert abs(dk_dy_boundary) > 1e-5, "field should vary at an interface"


def test_x_gradient_is_zero_for_a_depth_only_field():
    """PROSTATE_PATH varies with depth only, so d(kappa)/dx is identically
    zero. This test should FAIL if a laterally varying field is later added,
    which is the intended signal that call sites assuming depth-only need
    revisiting."""
    f = TissueField()
    for y in (30.0, 45.0, 67.5, 100.0):
        dk_dx, _ = f.kappa_gradient_at(0.0, y)
        assert abs(dk_dx) < 1e-12


def test_kappa_stays_physical_everywhere():
    """Over the whole workspace and beyond it, kappa must stay positive and
    within the layer stack's range -- the blend must not overshoot. A
    non-positive kappa would divide by zero in dubins_full (R = 1/kappa)."""
    f = TissueField()
    lo = min(l.kappa for l in PROSTATE_PATH)
    hi = max(l.kappa for l in PROSTATE_PATH)
    for y in range(-20, 200):
        k = f.kappa_at(0.0, float(y))
        assert k > 0.0, f"kappa non-positive at y={y}"
        assert lo - 1e-9 <= k <= hi + 1e-9, (
            f"kappa {k} at y={y} is outside the layer range [{lo}, {hi}] — "
            f"the blend is overshooting"
        )


def test_layer_at_names_the_right_medium():
    f = TissueField()
    assert f.layer_at(0.0, 30.0) == "fat"
    assert f.layer_at(0.0, 55.0) == "muscle"
    assert f.layer_at(0.0, 67.5) == "capsule"
    assert f.layer_at(0.0, 100.0) == "gland"


def test_single_layer_field_is_constant():
    """Degenerate case: one layer means constant kappa, which must reproduce
    the pre-4b behaviour exactly. This is the bridge that lets the existing
    constant-kappa tests keep passing."""
    f = TissueField(
        layers=(TissueLayer(kappa=1 / 50, upper_edge_mm=None, name="uniform"),)
    )
    for y in (0.0, 45.0, 67.5, 150.0):
        assert f.kappa_at(0.0, y) == pytest.approx(1 / 50)
        assert f.kappa_gradient_at(0.0, y) == pytest.approx((0.0, 0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
