"""Phase 4b: the tissue field wired into the SIMULATOR (via NeedleParams).

These tests check the ONE integration 4b adds -- that a `kappa_field` on
NeedleParams reaches `_time_deriv` and bends the trajectory -- plus the guards
that keep a field out of the constant-kappa filters (that Jacobian work is 4c).

The field's own construction (blend, gradient, layer naming) is tested
separately in test_tissue_field.py; this file is about the model coupling.

Run:  pytest tests/test_field_integration.py -v
"""

import math

import pytest

from needlesim.estimation.ekf import EKFConfig, NeedleEKF
from needlesim.models.tissue_field import PROSTATE_PATH, TissueField, TissueLayer
from needlesim.models.unicycle_needle import (
    Control,
    NeedleParams,
    State,
    rollout,
    step,
)


def test_field_changes_the_trajectory():
    """A field-carrying sim must produce a DIFFERENT path from a scalar-kappa
    one under otherwise identical controls. If it does not, the field is not
    reaching `_time_deriv` at all -- the whole phase is a no-op."""
    scalar = NeedleParams(kappa=1.0 / 50.0)
    field = NeedleParams(kappa=1.0 / 50.0, kappa_field=TissueField())

    # Insert upward from the fat layer, curving, so the path sweeps through
    # depths where the field (R = 34 -> 22 -> 15 -> 28mm) differs sharply from
    # the scalar R = 50mm.
    start = State(75.0, 25.0, math.pi / 2)
    controls = [Control(v=5.0, b=+1) for _ in range(300)]

    end_scalar = rollout(start, controls, 0.05, scalar)[-1]
    end_field = rollout(start, controls, 0.05, field)[-1]

    sep = math.hypot(end_field.x - end_scalar.x, end_field.y - end_scalar.y)
    assert sep > 5.0, (
        f"field and scalar trajectories only diverged {sep:.3f}mm -- the "
        f"kappa_field is not reaching _time_deriv"
    )


@pytest.mark.parametrize("depth", [30.0, 45.0, 55.0, 67.5, 80.0, 100.0])
def test_local_turn_rate_tracks_the_field(depth):
    """Rolling through the layers, the needle's instantaneous curvature must
    match kappa_at at that depth. Measured as dtheta/ds over a step so short
    the field is effectively constant across it (ds ~ 5e-3mm, vs 2mm
    transitions), so this holds even at the boundary depths 45 and 67.5."""
    field = NeedleParams(kappa=1.0 / 50.0, kappa_field=TissueField())
    # Heading +x so the needle stays essentially at this depth over the step.
    s0 = State(0.0, depth, 0.0)
    v, dt = 5.0, 1e-3
    s1 = step(s0, Control(v=v, b=+1), dt, field)

    ds = math.hypot(s1.x - s0.x, s1.y - s0.y)
    measured_kappa = (s1.theta - s0.theta) / ds
    expected = field.kappa_field.kappa_at(s0.x, s0.y)
    assert measured_kappa == pytest.approx(expected, rel=1e-3), (
        f"at depth {depth}: local kappa {measured_kappa:.5f} != field "
        f"{expected:.5f}"
    )


def test_single_layer_field_is_bit_identical_to_scalar():
    """The bridge that lets every constant-kappa test keep passing: a
    one-layer field returns numer/denom = (1.0*k)/1.0 == k exactly, so the
    field path is not merely close to the scalar path but bit-identical.

    Asserted with `==`, not approx, on purpose -- any drift here means the
    blend is doing arithmetic it should not for the degenerate case."""
    k = 1.0 / 37.0
    scalar = NeedleParams(kappa=k)
    field = NeedleParams(
        kappa=k,
        kappa_field=TissueField(
            layers=(TissueLayer(kappa=k, upper_edge_mm=None, name="uniform"),)
        ),
    )
    start = State(10.0, 20.0, 0.3)
    controls = [Control(v=5.0, b=(1 if i % 4 else -1)) for i in range(200)]

    trace_scalar = rollout(start, controls, 0.05, scalar)
    trace_field = rollout(start, controls, 0.05, field)

    for a, b in zip(trace_scalar, trace_field):
        assert (a.x, a.y, a.theta) == (b.x, b.y, b.theta)


def test_needle_ekf_rejects_field_carrying_params():
    """The filter's Jacobian is constant-kappa; a field would propagate the
    mean correctly through `step` while F stays [0,0,1] and is silently wrong.
    Construction must refuse it rather than degrade silently. (Field-aware
    filtering is 4c.)"""
    field_params = NeedleParams(kappa=1.0 / 50.0, kappa_field=TissueField())
    with pytest.raises(AssertionError):
        NeedleEKF(field_params, EKFConfig(), State(0.0, 20.0, math.pi / 2))


def test_needle_ekf_accepts_scalar_params():
    """The guard must not fire on the ordinary scalar case -- otherwise it
    would break every existing filter construction."""
    NeedleEKF(
        NeedleParams(kappa=1.0 / 50.0), EKFConfig(), State(0.0, 20.0, math.pi / 2)
    )


def test_field_params_are_still_hashable():
    """NeedleParams is frozen so it stays hashable; a field must not break
    that. This is why TissueField is frozen too -- a frozen dataclass holding
    an unhashable field raises only when hash() is actually called."""
    p = NeedleParams(kappa=1.0 / 50.0, kappa_field=TissueField(layers=PROSTATE_PATH))
    hash(p)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
