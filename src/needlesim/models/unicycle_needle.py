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
from duty-cycling b, you should NOT hardcode it.

ACCEPTANCE TESTS (see tests/test_needle_model.py):
    1. Constant v, b=+1  -> circular arc of radius 1/kappa (fit and check).
    2. Flip b at the midpoint -> symmetric S-curve.
    3. ~50% duty cycle on b -> effective curvature ~ kappa/2.

IMPLEMENTATION NOTES:
    - Use RK4, not forward Euler. Euler visibly spirals on the circle test.
    - Keep `step` PURE: (state, control, dt) -> new_state, no mutation, no
      plotting, no global state. The planner will call it thousands of times
      per second on hypothetical states.
    - Keep `true` vs `model` parameters separate via NeedleParams (see below).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeedleParams:
    """Physical parameters of the needle+tissue.

    Two instances of this exist in every experiment: the ground-truth params
    the simulator uses, and the (possibly wrong) params the planner/filter
    believe. Keep them as separate objects from day one.
    """

    kappa: float = 1.0 / 50.0  # natural curvature [1/mm], ~1/50 is typical
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


def step(state: State, control: Control, dt: float, params: NeedleParams) -> State:
    """Advance the needle one timestep. IMPLEMENT ME (Task 1).

    Must be a pure function: no mutation of inputs, no side effects.
    Recommended: RK4 integration of the kinematics in the module docstring.

    Args:
        state: current tip pose.
        control: (v, b) applied over this step.
        dt: timestep [s].
        params: needle/tissue parameters (use params.kappa).

    Returns:
        The new State after time dt.
    """
    raise NotImplementedError("Task 1: implement the bevel-tip kinematics here.")


def rollout(state: State, controls, dt: float, params: NeedleParams):
    """Convenience: apply a sequence of controls, returning the full trace.

    Works as soon as `step` is implemented. Useful for the acceptance-test
    plots. Returns a list of States including the initial one.
    """
    trace = [state]
    for control in controls:
        trace.append(step(trace[-1], control, dt, params))
    return trace
