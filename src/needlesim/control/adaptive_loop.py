"""Adaptive closed-loop steering: replan from a LEARNED kappa.

THIS IS THE SECOND HALF OF PHASE 4a. The first half showed the augmented EKF
recovers kappa from a 2x wrong prior to within 2-5% using only noisy position
measurements. But nothing consumed that estimate -- the planner still held its
original wrong belief. This module wires it in.

Kept SEPARATE from control/closed_loop.py rather than added as a flag, so
Phase 3.5's verified loop stays untouched. The duplication is deliberate.

THE FIVE-WAY COMPARISON THIS COMPLETES
---------------------------------------
    1. open-loop                              run_open_loop        (Phase 3.5)
    2. closed-loop, drift trigger, FIXED kappa run_closed_loop     (Phase 3.5)
    3. closed-loop, drift trigger, LEARNED kappa   <- here, DRIFT
    4. closed-loop, KAPPA trigger, LEARNED kappa   <- here, KAPPA_CHANGE
    5. closed-loop, EITHER trigger, LEARNED kappa  <- here, UNION

1 vs 2 isolates what replanning buys (Phase 3.5's answer: helps under
constraint, hurts in open space). 2 vs 3 isolates what fixing the MODEL buys
on top of that -- replanning re-aims from a better pose, but until now every
new plan carried the same wrong kappa. 3 vs 4 asks whether drift is even the
right thing to trigger on. 5 asks whether catching BOTH signals beats catching
either -- and the measured answer, that it does not, is itself the result (see
the UNION enum comment).

WHY 4 IS WORTH RUNNING. Phase 3.5 measured that drift-triggered replanning
HURTS where open-loop already lands within tolerance: it is an intervention
when nothing needs correcting. A kappa-change trigger replans when there is a
REASON -- the current plan was computed under a belief no longer held -- rather
than when the needle has wandered. Expect it to fire a few times early (kappa
converges fast: most of the movement is in the first ~200 steps) and then stop.

CONTROLLED COMPARISON. Conditions 2 through 5 all start from the SAME wrong
kappa guess, so they differ only in whether the belief updates and what
triggers a replan. Any difference is attributable to that, not to a different
starting point. (A separate and also interesting question -- how wrong can the
prior be before learning stops rescuing it -- is a sweep over the initial
guess, not a sixth condition.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from needlesim.control.closed_loop import ClosedLoopResult, cross_track_distance
from needlesim.estimation.ekf_augmented import AugmentedEKFConfig, AugmentedNeedleEKF
from needlesim.models.unicycle_needle import NeedleParams, State, step


class ReplanTrigger(Enum):
    """What causes a replan.

    DRIFT       -- cross-track distance from the plan exceeds a threshold.
                   Phase 3.5's trigger, kept so conditions 2 and 3 differ only
                   in whether kappa is learned.
    KAPPA_CHANGE -- the curvature belief has moved materially since the
                   current plan was made, so that plan was computed under a
                   belief no longer held.
    """

    DRIFT = "drift"
    KAPPA_CHANGE = "kappa_change"
    UNION = "union"
    # Replan on EITHER signal. Originally motivated by a plausible but WRONG
    # inference: run alone, the two triggers fire at almost disjoint times --
    # kappa-change at steps ~21-81 (while the estimate moves fast), drift at
    # ~101-281 (once error accumulates) -- so a union "should" catch two
    # distinct signals. That inference has a hole, and the combined run exposes
    # it: each trigger changes the trajectory the OTHER one sees. Under a union
    # kappa-change fires early and often, every replan refreshes the path, and
    # cross-track drift therefore never accumulates past threshold. Drift is
    # pre-empted.
    #
    # MEASURED: UNION == KAPPA_CHANGE in 9 of 10 runs (5 seeds x 2 scenarios).
    # The sole exception, open seed 1, added one drift replan at step 381
    # ([21,41,61,81] -> [21,41,61,81,381]) and it helped, 3.9mm -> 3.0mm. So
    # the union tracks condition 4, not the sum of 3 and 4. The methodological
    # point worth keeping: disjointness observed in INDEPENDENT single-trigger
    # runs does not survive combination.
    #
    # The condition is retained because that near-identity IS the finding:
    # drift is largely inert once model-triggered replanning is active, i.e.
    # cross-track drift is mostly a SYMPTOM of model error rather than an
    # independent signal. See docs/roadmap.md, Phase 4a.


@dataclass
class AdaptiveLoopConfig:
    """Tunables for the adaptive loop. Mirrors ClosedLoopConfig, plus the
    kappa trigger."""

    trigger: ReplanTrigger = ReplanTrigger.DRIFT

    # DRIFT trigger: same threshold and reasoning as Phase 3.5 -- above the
    # 1mm measurement noise so it does not fire on sensor jitter.
    drift_threshold_mm: float = 4.0

    # KAPPA_CHANGE trigger: replan when |log(kappa_now) - log(kappa_at_plan)|
    # exceeds this. LOG units, so it is a RELATIVE change: 0.095 is ~10%,
    # 0.2 is ~22%, 0.69 is a factor of two. Matching the filter's own
    # parameterisation keeps "materially different" meaning the same thing in
    # both places.
    kappa_change_threshold: float = 0.095

    max_replans: int = 50
    max_replan_failures: int = 10

    seed: int = 0


@dataclass
class AdaptiveLoopResult(ClosedLoopResult):
    """Everything ClosedLoopResult carries, plus the kappa history -- without
    which you cannot tell whether a poor run was bad control or a bad
    estimate."""

    kappa_estimates: list[float] = None      # kappa_hat at every step
    kappa_at_replans: list[float] = None     # the belief each plan was made under
    final_kappa_error: float = 0.0           # |kappa_hat_final - kappa_true|


def run_closed_loop_adaptive(
    planner,
    env,
    true_params: NeedleParams,
    initial_kappa_guess: float,
    start: State,
    goal: State,
    ekf_config: AugmentedEKFConfig,
    loop_config: AdaptiveLoopConfig,
    max_steps: int = 2000,
) -> AdaptiveLoopResult:
    """Plan, execute, estimate (pose AND kappa), replan from the LEARNED kappa.

    IMPLEMENT ME. Structurally this is Phase 3.5's run_closed_loop with three
    changes. Start from that function and adapt it rather than writing fresh.

    CHANGE 1 -- the filter is AugmentedNeedleEKF, and there is no
    `model_params` argument. Kappa is a state now, so the only model input is
    `initial_kappa_guess`: the filter's starting belief, expected to be wrong.

    CHANGE 2 -- REBUILD THE PLANNER at each replan, from the current estimate:

        new_planner = type(planner)(env, ekf.params, planner.config)
        new_result = new_planner.plan(ekf.state, goal)

    This is the whole point of the phase. Note it must be a FRESH planner:
    NeedleParams is frozen and the planner reads its kappa at construction, so
    mutating is not an option. Constructing one is cheap next to planning.

    Two things to be careful of:
      - `type(planner)(...)` preserves whichever planner the caller passed.
        Do not hardcode KinodynamicRRT.
      - The planner carries RNG state seeded from its config. A fresh planner
        restarts that sequence, so replans are not independent draws the way
        they were in Phase 3.5. Note it; it is a difference between the
        conditions that is worth being conscious of rather than surprised by.

    CHANGE 3 -- THE TERMINAL-APPROACH GUARD MUST USE THE LEARNED KAPPA.

    Phase 3.5's guard suppresses replanning when R*heading_offset exceeds the
    distance remaining, because the correction then costs more travel than is
    left and the planner can only answer with a loop. R = 1/kappa there came
    from the fixed model belief. Here it must come from `ekf.kappa` -- the
    current estimate -- or the guard is reasoning about a needle that does not
    exist. As kappa_hat improves, the guard's geometry gets more accurate too.

    THE TRIGGER, per loop_config.trigger:

        DRIFT: exactly Phase 3.5 -- cross_track_distance(ekf.state,
        current_path) > drift_threshold_mm, checked on measurement steps only.

        KAPPA_CHANGE: replan when
            abs(log(ekf.kappa) - log(kappa_when_current_plan_was_made))
                > loop_config.kappa_change_threshold
        Also checked on measurement steps only (kappa only moves at updates,
        so checking between them is wasted work). Record kappa at each plan so
        the comparison has a reference.

    EVERYTHING ELSE IS UNCHANGED from Phase 3.5, and should stay that way so
    the conditions are comparable: the same measurement interval and world RNG
    convention, the same replan-failure cap and halting behaviour, the same
    termination reasons, collision checked on the EXECUTED trajectory, and the
    true/model split -- `step` with true_params, the filter with its own
    belief.

    RECORD kappa_estimates every step and kappa_at_replans at each plan. A run
    that ends badly is ambiguous without them: bad control and a bad estimate
    look the same in the final error alone.
    """
    executed_states = []
    estimated_states = []
    kappa_estimates = []
    kappa_at_replans = []
    plans = []
    replan_steps = []
    n_replan_failures = 0

    rrt_result = planner.plan(start, goal)
    if not rrt_result.success:
        return AdaptiveLoopResult(
            executed_states=[], estimated_states=[], plans=[], replan_steps=[],
            n_replan_failures=0, reached_goal=False,
            final_error_mm=float("inf"), collided=False, left_bounds=False, n_steps=0,
            termination_reason="no_initial_plan",
            kappa_estimates=kappa_estimates, kappa_at_replans=kappa_at_replans, final_kappa_error=0.0
        )
    plans.append(rrt_result.path)
    kappa_at_replans.append(initial_kappa_guess)

    ekf = AugmentedNeedleEKF(ekf_config, start, initial_kappa_guess)
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
            kappa_estimates.append(ekf.kappa)
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
            correctable = (1.0 / ekf.kappa) * head_off <= d_goal

            measurement_step = (k - 1) % ekf_config.measurement_interval == 0
            if measurement_step:
                # WHICH trigger, per loop_config.trigger. All arms are gated
                # by measurement_step and the terminal-approach guard above:
                # kappa only moves on updates, and a near-goal correction is a
                # geometry problem no matter what asked for the replan.
                # DRIFT and KAPPA_CHANGE each isolate a single trigger (so
                # conditions 3 and 4 are clean comparisons); UNION fires on
                # either, and measures the fact that the drift arm is nearly
                # inert once kappa-replanning is active -- see the five-way
                # comparison and the UNION enum comment.
                drift_fired = (
                    cross_track_distance(ekf.state, rrt_result.path)
                    > loop_config.drift_threshold_mm
                )
                kappa_fired = (
                    abs(math.log(ekf.kappa) - math.log(kappa_at_replans[-1]))
                    > loop_config.kappa_change_threshold
                )
                if loop_config.trigger == ReplanTrigger.DRIFT:
                    should_replan = correctable and drift_fired
                elif loop_config.trigger == ReplanTrigger.KAPPA_CHANGE:
                    # kappa_at_replans[-1] is the belief the CURRENT plan was
                    # made under. Compare in LOG space to match the filter's
                    # own parameterisation, so the threshold means the same
                    # relative change here as it does there.
                    should_replan = correctable and kappa_fired
                else:  # UNION
                    should_replan = correctable and (drift_fired or kappa_fired)

                if should_replan:
                    new_planner = type(planner)(env, ekf.params, planner.config)
                    new_result = new_planner.plan(ekf.state, goal)
                    if new_result.success:
                        rrt_result = new_result
                        plans.append(new_result.path)
                        replan_steps.append(k)
                        kappa_at_replans.append(ekf.kappa)
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

    return AdaptiveLoopResult(
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
        kappa_estimates=kappa_estimates,
        kappa_at_replans=kappa_at_replans,
        final_kappa_error=abs(ekf.kappa - true_params.kappa)
    )
