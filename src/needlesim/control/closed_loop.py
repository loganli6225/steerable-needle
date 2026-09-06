"""Closed-loop needle steering: plan, execute, estimate, replan.

THIS IS YOUR PHASE 3.5 FILE. Interface and structure laid out; the control
loop and the drift trigger are yours. Per the working agreement, the loop
logic is load-bearing; the experiment harness and plots are delegable.

WHAT THIS CLOSES
----------------
Everything so far runs open-loop: the planner produces a control sequence and
the needle executes all of it, blind. The EKF (Phase 3) estimates where the
needle actually is, but nothing consumes that estimate. This module wires the
two together:

    plan from the estimate -> execute a prefix -> measure -> update the
    estimate -> if it has drifted from the plan, replan from where you now
    believe you are

That is what a real needle-steering system does, and it is the only mechanism
by which model error gets CORRECTED rather than ACCUMULATED. Phase 3 showed
that under a 2x kappa mismatch the filter tracks (bounded error) and its
innovation goes biased -- it knows something is wrong. Closed-loop is what
lets the system act on that.

THE EXPERIMENT THIS ENABLES
---------------------------
Sweep model kappa from matched to badly wrong. Run each level open-loop and
closed-loop, same seeds, same scenario. Expect open-loop final error to grow
with the mismatch while closed-loop stays bounded. That contrast is the
Phase 3.5 result, and it is the direct setup for Phase 4: a learned kappa is
worth having because of what it does to the CLOSED loop, not because it fits
data well in isolation.

DESIGN CHOICES (settled)
------------------------
- REPLAN ON A DRIFT TRIGGER, not every measurement. Replanning is expensive
  (kinodynamic takes ~1.6s), and re-solving when you are still on track buys
  nothing. Drift is measured as cross-track distance from the planned path.
- CHECK DRIFT ONLY ON MEASUREMENT STEPS. Between measurements the estimate
  moves by PREDICTION alone, so any drift it perceives is unconfirmed model
  error -- replanning from it means re-aiming from a pose the filter may be
  wrong about. It also cuts replan attempts ~20x, which matters because a
  failing replan burns the planner's whole iteration budget (measured: this
  change took a sweep from ~20 min/seed to ~2 min/seed).
- PLAN FROM THE ESTIMATE MEAN, ignoring the covariance. Using the uncertainty
  to bias the plan (prefer routes that are robust to where you might actually
  be) is uncertainty-aware planning -- Phase 4 territory. Note the hook, do
  not build it here.
- HALT WHEN REPLANNING GIVES UP. Measured: continuing to execute a stale plan
  after the replan cap made things WORSE, stretching runs to 2-3x the normal
  step count and ending 80-105mm out, because a stale plan steers away from
  the pose the needle has already left. Halting reports where the controller
  gave up, which is bounded and interpretable.
- METRICS: final tip error at the goal, collisions along the EXECUTED
  trajectory, and the TERMINATION REASON -- the error number alone conflates
  "stopped at 60mm because it gave up" with "wandered to 104mm because it
  would not".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from needlesim.benchmark.vanilla_tracker import crosstrack_distance
from needlesim.estimation.ekf import EKFConfig, NeedleEKF
from needlesim.models.unicycle_needle import NeedleParams, State, step


@dataclass
class ClosedLoopConfig:
    """Tunables for the control loop itself (the planner and filter carry
    their own configs)."""

    # Cross-track distance from the planned path that triggers a replan [mm].
    # Should sit above the measurement noise (1mm) so the trigger is not
    # firing on sensor jitter, and below the planning margin (2mm) plus a
    # sensible allowance -- if you have drifted further than the margin, the
    # plan you are following is already unsafe. Sweep it rather than trusting
    # this default.
    drift_threshold_mm: float = 4.0

    # Cap on replans, so a pathological run cannot spin forever.
    max_replans: int = 50

    # Consecutive replan FAILURES tolerated before the controller gives up and
    # halts. A single failure is not decisive -- the planner is randomised and
    # fails from good poses too -- but a run of them means the drifted pose is
    # one the planner cannot recover from. Set to 0 to abort on the first
    # failure. (This replaces an earlier continue_on_replan_failure flag,
    # whose False branch aborted on the first failure and stopped runs at
    # 141-301 steps, well short of the goal.)
    max_replan_failures: int = 10

    seed: int = 0


@dataclass
class ClosedLoopResult:
    """Everything needed to score and plot one run."""

    executed_states: list[State]        # where the needle ACTUALLY went
    estimated_states: list[State]       # what the filter believed
    plans: list[list[State]]            # every path planned, in order
    replan_steps: list[int]             # step indices where a replan happened
    n_replan_failures: int
    reached_goal: bool
    final_error_mm: float               # true tip to goal at the end
    collided: bool
    left_bounds: bool
    n_steps: int
    # WHY the run ended. Distinguishes outcomes the error number alone
    # conflates: stopping at 60mm because the controller gave up is a
    # different result from wandering to 104mm because it would not.
    #   "goal"            -- reached the goal tolerance
    #   "plan_exhausted"  -- executed a whole plan without drifting or arriving
    #   "replan_cap"      -- gave up replanning after max_replan_failures
    #   "max_steps"       -- ran out of budget
    #   "no_initial_plan" -- the first plan() call failed
    termination_reason: str = "unknown"


# ---------------------------------------------------------------------------
# The drift trigger.
# ---------------------------------------------------------------------------


def cross_track_distance(point: State, path: list[State]) -> float:
    """Shortest distance from `point` to the planned polyline.

    Distance to the POLYLINE, not to the nearest node: a point can sit exactly
    on a segment while being far from both its endpoints, and node distance
    would trigger replans on straight stretches purely because the nodes are
    sparse there.

    Delegates to the vanilla tracker's implementation rather than
    reimplementing -- two copies of the same projection geometry would drift
    apart, and the cross-track number would then mean subtly different things
    in the benchmark and in the controller.
    """
    return crosstrack_distance(point, path)


# ---------------------------------------------------------------------------
# The control loop.
# ---------------------------------------------------------------------------


def run_closed_loop(
    planner,
    env,
    true_params: NeedleParams,
    model_params: NeedleParams,
    start: State,
    goal: State,
    ekf_config: EKFConfig,
    loop_config: ClosedLoopConfig,
    max_steps: int = 2000,
) -> ClosedLoopResult:
    """Plan, execute, estimate, replan until the goal is reached, the
    controller gives up, or the step budget runs out.

    FOUR INVARIANTS
    ---------------

    1. THE TRUE/MODEL SPLIT. The simulator steps with TRUE params; the filter
       predicts with MODEL params; the PLANNER is built with MODEL params by
       the caller. If any of the three sees the wrong object the experiment
       measures nothing. Third place in the codebase this matters.

    2. REPLAN FROM THE ESTIMATE, NOT FROM TRUTH. `ekf.state`, never
       `true_state`. Planning from truth is the cheat this phase exists to
       avoid -- it would make closed-loop look perfect for the wrong reason.

    3. DISCARD THE STALE PLAN. On a successful replan the remaining controls
       of the old plan were computed from a pose the needle is no longer at,
       so break out of that sequence entirely.

    4. DRIFT IS MEASURED ON THE ESTIMATE, not on truth. The controller only
       has the estimate; using truth to decide when to replan is the same
       cheat as (2), just subtler.

    COLLISION is checked on the EXECUTED trajectory, not the planned one. The
    plan is collision-free by construction; the question is whether the needle
    stayed clear while following it under model error.
    """
    executed_states = []
    estimated_states = []
    plans = []
    replan_steps = []
    n_replan_failures = 0

    rrt_result = planner.plan(start, goal)
    if not rrt_result.success:
        return ClosedLoopResult(
            executed_states=[], estimated_states=[], plans=[], replan_steps=[],
            n_replan_failures=0, reached_goal=False,
            final_error_mm=float("inf"), collided=False, left_bounds=False, n_steps=0,
            termination_reason="no_initial_plan",
        )
    plans.append(rrt_result.path)

    ekf = NeedleEKF(model_params, ekf_config, start)
    true_state = start
    dt = planner.config.step_dt
    n_per_edge = planner.config.n_steps_per_extend
    # Measurement noise belongs to the WORLD, not the filter -- its own
    # generator, as in run_estimation.
    world_rng = np.random.default_rng(loop_config.seed + 1000)
    k = 0                       # GLOBAL step counter. Never reset on replan:
                                # imaging fires on its own schedule, not the
                                # controller's.
    done = False
    reached = False
    reason = "plan_exhausted"   # default: fell off the end of a plan

    while not done and k < max_steps and len(replan_steps) <= loop_config.max_replans:
        # controls are one Control per EDGE; each edge is n_per_edge steps.
        steps = [c for c in rrt_result.controls for _ in range(n_per_edge)]
        replanned = False

        for control in steps:
            true_state = step(true_state, control, dt, true_params)   # TRUE
            ekf.predict(control, dt)                                  # MODEL

            if k % ekf_config.measurement_interval == 0:
                noise = world_rng.normal(0.0, ekf_config.measurement_noise_std)
                ekf.update((true_state.x + noise[0], true_state.y + noise[1]))

            executed_states.append(true_state)
            estimated_states.append(ekf.state)
            k += 1

            if (
                math.hypot(true_state.x - goal.x, true_state.y - goal.y)
                < planner.config.goal_tolerance
            ):
                reached = True
                done = True
                reason = "goal"
                break

            if k >= max_steps:
                done = True
                reason = "max_steps"
                break

            if not (0 <= true_state.x <= env.width and 0 <= true_state.y <= env.height):
                done = True
                reason = "left_bounds"
                break

            # Give up: the controller cannot replan from where it now is, so
            # halt rather than keep executing a plan computed for a pose the
            # needle left long ago. Measured: continuing stretched runs to
            # 2-3x the normal step count and ended 80-105mm out, because a
            # stale plan steers AWAY from the drifted pose.
            if n_replan_failures >= loop_config.max_replan_failures:
                done = True
                reason = "replan_cap"
                break

            # TERMINAL-APPROACH GUARD. At radius R = 1/kappa, correcting a
            # heading error of `off` radians costs R*off of arc. Near the goal
            # the distance remaining shrinks faster than the heading error
            # does, so that correction can cost MORE travel than remains -- at
            # which point the planner can only answer with a loop.
            # Measured (open, seed 1, model 1/25): a replan 15.6mm from goal
            # with 0.89 rad of heading offset needed ~22mm of travel to
            # straighten and returned a 399mm plan, 25x direct. Executing it
            # drove the needle off the workspace. Both curvatures loop there;
            # the true-kappa planner failed outright at 20,000 iterations.
            # So: when the correction is not geometrically possible, do not
            # ask for one -- carry on with the current plan.
            d_goal = math.hypot(ekf.state.x - goal.x, ekf.state.y - goal.y)
            bearing = math.atan2(goal.y - ekf.state.y, goal.x - ekf.state.x)
            head_off = abs(
                (ekf.state.theta - bearing + math.pi) % (2 * math.pi) - math.pi
            )
            correctable = (1.0 / model_params.kappa) * head_off <= d_goal

            measurement_step = (k - 1) % ekf_config.measurement_interval == 0
            if (
                measurement_step
                and correctable
                and cross_track_distance(ekf.state, rrt_result.path)
                > loop_config.drift_threshold_mm
            ):
                new_result = planner.plan(ekf.state, goal)   # from the ESTIMATE
                if new_result.success:
                    rrt_result = new_result
                    plans.append(new_result.path)
                    replan_steps.append(k)
                    replanned = True
                    break            # discard the stale plan's remaining controls
                else:
                    n_replan_failures += 1

        if not replanned:
            done = True              # plan exhausted without triggering a replan

    final_error_mm = math.hypot(true_state.x - goal.x, true_state.y - goal.y)
    in_bounds = lambda s: (0 <= s.x <= env.width and 0 <= s.y <= env.height)
    collided = any(
        in_bounds(s) and not env.is_free(s, planner.config.margin)
        for s in executed_states
    )
    left_bounds = any(not in_bounds(s) for s in executed_states)

    return ClosedLoopResult(
        executed_states=executed_states,
        estimated_states=estimated_states,
        plans=plans,
        replan_steps=replan_steps,
        n_replan_failures=n_replan_failures,
        reached_goal=reached,
        final_error_mm=final_error_mm,
        collided=collided,
        left_bounds=left_bounds,
        n_steps=k,
        termination_reason=reason,
    )


def run_open_loop(
    planner,
    env,
    true_params: NeedleParams,
    model_params: NeedleParams,
    start: State,
    goal: State,
    max_steps: int = 2000,
) -> ClosedLoopResult:
    """The baseline: plan once, execute the whole sequence blind.

    No filter, no replanning -- plan from `start` with MODEL params and roll
    the entire control sequence through the TRUE model. Fills the same result
    type so the two are directly comparable; `estimated_states` and
    `replan_steps` stay empty.

    This is what every experiment before Phase 3.5 did implicitly. Making it
    an explicit baseline is what lets the closed-loop result mean something.
    """
    executed_states = []

    rrt_result = planner.plan(start, goal)
    if not rrt_result.success:
        return ClosedLoopResult(
            executed_states=[], estimated_states=[], plans=[], replan_steps=[],
            n_replan_failures=0, reached_goal=False,
            final_error_mm=float("inf"), collided=False, left_bounds=False, n_steps=0,
            termination_reason="no_initial_plan",
        )

    true_state = start
    dt = planner.config.step_dt
    n_per_edge = planner.config.n_steps_per_extend
    k = 0
    reached = False
    reason = "plan_exhausted"

    steps = [c for c in rrt_result.controls for _ in range(n_per_edge)]

    for control in steps:
        true_state = step(true_state, control, dt, true_params)
        executed_states.append(true_state)
        k += 1

        if (
            math.hypot(true_state.x - goal.x, true_state.y - goal.y)
            < planner.config.goal_tolerance
        ):
            reached = True
            reason = "goal"
            break

        if k >= max_steps:
            reason = "max_steps"
            break

        if not (0 <= true_state.x <= env.width and 0 <= true_state.y <= env.height):
            done = True
            reason = "left_bounds"
            break

    final_error_mm = math.hypot(true_state.x - goal.x, true_state.y - goal.y)
    in_bounds = lambda s: (0 <= s.x <= env.width and 0 <= s.y <= env.height)
    collided = any(
        in_bounds(s) and not env.is_free(s, planner.config.margin)
        for s in executed_states
    )
    left_bounds = any(not in_bounds(s) for s in executed_states)

    return ClosedLoopResult(
        executed_states=executed_states,
        estimated_states=[],
        plans=[rrt_result.path],
        replan_steps=[],
        n_replan_failures=0,
        reached_goal=reached,
        final_error_mm=final_error_mm,
        collided=collided,
        left_bounds=left_bounds,
        n_steps=k,
        termination_reason=reason,
    )