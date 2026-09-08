"""Bevel-tip needle kinematic model (2D unicycle / nonholonomic model).

THIS IS YOUR TASK 1 FILE. The interface is defined so the rest of the
codebase can be built against it, but the kinematics are left for you to
implement. Fill in `step` (and adjust `State`/`Control` if you want).

Model (Webster/Cowan nonholonomic bevel-tip model), 2D:

    state  = (x, y, theta)
    control = (v, b)   with v = insertion speed, b in {-1, +1} bevel direction

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = v * kappa * b

There is deliberately no "straight" control: the needle always curves at
kappa; you only choose the sign. Effective curvature in [0, kappa] emerges
from duty-cycling b, you should NOT hardcode it. For a fraction p of steps
with b=+1 (rest b=-1), kappa_eff = kappa * (2p - 1): p=0.5 is a straight
line, p=0.75 gives kappa/2, p=1.0 gives full curvature.

Note b in {-1, +1} is a 2D caricature: physically, duty-cycling means
axially spinning the needle base so the bevel direction rotates through 3D
tissue. The p-fraction relationship above is the 2D analogue of that, not
the mechanism itself -- worth being explicit before comparing against 3D
literature.

ACCEPTANCE TESTS (see tests/test_needle_model.py):
    1. Constant v, b=+1  -> circular arc of radius 1/kappa (fit and check;
       smoke test only -- see IMPLEMENTATION NOTES on why this tolerance
       doesn't distinguish Euler from RK4).
    2. Flip b at the midpoint -> symmetric S-curve (point reflection in x,y
       about the midpoint, not just a theta check -- theta integrates
       exactly under any integrator here, see below).
    3. kappa_eff = kappa * (2p - 1) emerges from a p-fraction duty cycle on
       b, for p in {0.5, 0.625, 0.75, 0.875, 1.0}.

IMPLEMENTATION NOTES:
    - Use RK4, not forward Euler. At this model's operating point (constant
      kappa, small dt) the two are numerically indistinguishable on the
      circle test -- theta_dot = v*kappa*b doesn't depend on x, y, or theta,
      so heading integrates exactly under any integrator, and Euler's
      position error here is a fixed sub-1e-5 bias, not a growing spiral.
      RK4 is kept anyway for when kappa becomes state-dependent (inhomogeneous
      tissue, learned deflection models, later phases): once theta_dot
      depends on position, the decoupling above breaks and integrator choice
      starts to matter. See test_rk4_matches_analytic_quarter_circle for a
      tolerance that does discriminate the two.
    - Keep `step` PURE: (state, control, dt) -> new_state, no mutation, no
      plotting, no global state. The planner will call it thousands of times
      per second on hypothetical states.
    - Keep `true` vs `model` parameters separate via NeedleParams (see below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from needlesim.models.tissue_field import TissueField


@dataclass(frozen=True)
class NeedleParams:
    """Physical parameters of the needle+tissue.

    Two instances of this exist in every experiment: the ground-truth params
    the simulator uses, and the (possibly wrong) params the planner/filter
    believe. Keep them as separate objects from day one.
    """

    kappa: float = 1.0 / 50.0  # natural curvature [1/mm], ~1/50 is typical

    # Phase 4b: an optional spatially varying curvature field. When None
    # (the default), `_time_deriv` uses the scalar `kappa` exactly as before,
    # so every existing call site is unchanged. When set, `_time_deriv` reads
    # kappa_field.kappa_at(x, y) at each RK4 sub-evaluation instead.
    #
    # WHY THE FIELD LIVES ON params (option (a), not a separate step_field or
    # an always-callable kappa): every field-relevant consumer already routes
    # through `step` -- the simulator, the collision checker
    # (GridEnvironment.is_arc_free), the KinodynamicRRT rollout, and the
    # filters' mean-propagation all call step(state, control, dt, params). So
    # attaching the field to `params` propagates it to all of them for free,
    # through code that already exists. A separate step_field would need a
    # branch at every one of those call sites and two code paths in
    # is_arc_free; making `kappa` always-callable would break every
    # `R = 1/params.kappa` in the Dubins tree while only MOVING the
    # scalar-sampling problem the geometry has, not removing it.
    #
    # The scalar `kappa` is KEPT as the fallback and as "the" curvature a
    # field-carrying params still reports when a scalar is asked for -- it is
    # what the planner and filters legitimately consume (see the true/model
    # split: in 4b the world gets a field, the belief stays scalar).
    kappa_field: TissueField | None = None
    # Room to grow: process_noise_std, tissue_inhomogeneity, etc.


@dataclass(frozen=True)
class State:
    """Needle tip pose in 2D."""

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class Control:
    """Control input over one step."""

    v: float  # insertion speed [mm/s] (or per-step distance if you prefer)
    b: int  # bevel direction, -1 or +1


def _time_deriv(
    current_state: np.ndarray, control: Control, params: NeedleParams
) -> np.ndarray:
    """Time derivative of (x, y, theta)."""
    x, y, theta = current_state
    x_dot = control.v * np.cos(theta)
    y_dot = control.v * np.sin(theta)
    # Phase 4b: with a field, curvature depends on position, so it is read at
    # THIS sub-evaluation's (x, y) -- which is why RK4 (k1..k4 at different
    # points) matters once kappa varies, exactly as the module docstring
    # anticipated. With no field this is bit-identical to `params.kappa`.
    kappa = (
        params.kappa
        if params.kappa_field is None
        else params.kappa_field.kappa_at(x, y)
    )
    theta_dot = control.v * kappa * control.b
    return np.array([x_dot, y_dot, theta_dot])


def step(state: State, control: Control, dt: float, params: NeedleParams) -> State:
    """Advance the needle one timestep via RK4 integration of the kinematics
    in the module docstring.

    Pure function: no mutation of inputs, no side effects.

    Args:
        state: current tip pose.
        control: (v, b) applied over this step.
        dt: timestep [s].
        params: needle/tissue parameters (use params.kappa).

    Returns:
        The new State after time dt.
    """
    # First-order reference, kept for comparison; see
    # test_rk4_matches_analytic_quarter_circle for why RK4 below is used
    # instead.
    # x = state.x + (control.v * np.cos(state.theta)) * dt
    # y = state.y + (control.v * np.sin(state.theta)) * dt
    # theta = state.theta + (control.v * params.kappa * control.b) * dt
    # return State(x, y, theta)

    current_state = np.array([state.x, state.y, state.theta])
    k1 = _time_deriv(current_state, control, params)
    k2 = _time_deriv(current_state + ((dt / 2) * k1), control, params)
    k3 = _time_deriv(current_state + ((dt / 2) * k2), control, params)
    k4 = _time_deriv(current_state + ((dt) * k3), control, params)
    return State(*(current_state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)))


def rollout(state: State, controls, dt: float, params: NeedleParams):
    """Convenience: apply a sequence of controls, returning the full trace.

    Works as soon as `step` is implemented. Useful for the acceptance-test
    plots. Returns a list of States including the initial one.
    """
    trace = [state]
    for control in controls:
        trace.append(step(trace[-1], control, dt, params))
    return trace


def rollout_variable(state: State, control_dt_pairs, params: NeedleParams):
    trace = [state]
    for control, dt in control_dt_pairs:
        trace.append(step(trace[-1], control, dt, params))
    return trace
