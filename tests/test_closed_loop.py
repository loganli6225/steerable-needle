"""Tests for closed-loop needle steering.

The central claim this phase makes is CONDITIONAL: under model mismatch,
replanning from the filter's estimate keeps the needle on target where
precision is required -- and is an unnecessary intervention where open-loop
already suffices. These tests pin the mechanism, and the last two pin both
halves of that claim.

Run:  pytest tests/test_closed_loop.py -v
"""

import math

import pytest

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.control.closed_loop import (
    ClosedLoopConfig,
    cross_track_distance,
    run_closed_loop,
    run_open_loop,
)
from needlesim.environments.grid_environment import GridEnvironment
from needlesim.estimation.ekf import EKFConfig
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

TRUE_KAPPA = 1.0 / 50.0
START = State(30.0, 20.0, math.pi / 2)
GOAL = State(30.0, 130.0, math.pi / 2)


def make_env():
    env = GridEnvironment(width=150.0, height=150.0, resolution=0.5)
    env.add_circle(75.0, 75.0, 20.0)
    env.bake()
    return env


def make_planner(env, kappa, seed=1):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, NeedleParams(kappa=kappa), cfg)


# =====================================================================
# 1. THE DRIFT TRIGGER -- geometry, testable in isolation.
# =====================================================================


def test_cross_track_distance_on_segment_interior():
    """The case that node-distance gets wrong: a point nearest to the MIDDLE
    of a long segment. Node distance would report the distance to an
    endpoint, which is much larger, and would trigger spurious replans on
    straight stretches where nodes are sparse."""
    path = [State(0.0, 0.0, 0.0), State(100.0, 0.0, 0.0)]
    point = State(50.0, 7.0, 0.0)
    assert cross_track_distance(point, path) == pytest.approx(7.0, abs=1e-9)


def test_cross_track_distance_clamps_beyond_ends():
    """Projection must be CLAMPED to the segment. An unclamped projection
    would report the perpendicular distance to the infinite line, which for a
    point past the end is smaller than the true distance to the path."""
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0)]
    point = State(20.0, 0.0, 0.0)   # 10mm beyond the segment's end
    assert cross_track_distance(point, path) == pytest.approx(10.0, abs=1e-9)


def test_cross_track_distance_takes_minimum_over_segments():
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0), State(10.0, 10.0, 0.0)]
    point = State(12.0, 5.0, 0.0)   # nearest the SECOND segment
    assert cross_track_distance(point, path) == pytest.approx(2.0, abs=1e-9)


def test_cross_track_distance_handles_duplicate_points():
    """Degenerate segment: the projection divides by squared segment length.
    Must not raise."""
    path = [State(5.0, 5.0, 0.0), State(5.0, 5.0, 0.0)]
    assert cross_track_distance(State(5.0, 8.0, 0.0), path) == pytest.approx(3.0)


# =====================================================================
# 2. THE LOOP -- mechanics.
# =====================================================================


def test_matched_params_barely_replans():
    """With a correct model there is nothing to correct: the needle follows
    its plan, drift stays under the threshold, and replans should be rare or
    absent. If this run replans constantly, the trigger is firing on noise
    rather than on genuine drift -- check the threshold against the
    measurement noise."""
    env = make_env()
    result = run_closed_loop(
        planner=make_planner(env, TRUE_KAPPA),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=TRUE_KAPPA),
        start=START,
        goal=GOAL,
        ekf_config=EKFConfig(),
        loop_config=ClosedLoopConfig(),
    )
    assert result.reached_goal
    assert len(result.replan_steps) <= 3, (
        f"replanned {len(result.replan_steps)} times with a correct model — "
        f"the drift trigger is probably firing on measurement noise"
    )


def test_mismatch_triggers_replanning():
    """The complement: with a wrong model the needle DOES drift, so the
    trigger must fire. If it never fires under a 2x kappa error, the trigger
    is not working and closed-loop reduces to open-loop."""
    env = make_env()
    result = run_closed_loop(
        planner=make_planner(env, 1.0 / 25.0),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=1.0 / 25.0),
        start=START,
        goal=GOAL,
        ekf_config=EKFConfig(),
        loop_config=ClosedLoopConfig(),
    )
    assert len(result.replan_steps) >= 1, (
        "no replan under a 2x kappa mismatch — the drift trigger is not firing"
    )


def test_open_loop_baseline_runs():
    """The baseline must produce a comparable result object, so the two can
    be scored side by side."""
    env = make_env()
    result = run_open_loop(
        planner=make_planner(env, TRUE_KAPPA),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=TRUE_KAPPA),
        start=START,
        goal=GOAL,
    )
    assert result.n_steps > 0
    assert result.replan_steps == []
    assert result.estimated_states == []


# =====================================================================
# 3. THE CLAIM -- and it is CONDITIONAL, so it takes two tests.
#
# Replanning helps where precision is required and is an unnecessary
# intervention where open-loop already lands within tolerance. Asserting only
# the flattering half would misrepresent the finding, so both directions are
# pinned.
# =====================================================================


def test_closed_loop_beats_open_loop_under_constraint():
    """Phase 3.5's headline, on the scenario where precision matters.

    Measured over 5 seeds at true kappa 1/50, model 1/25, scenario
    `constrained_passage`: open-loop median 18.6mm, closed-loop 4.4mm, and
    closed-loop won 5/5 seeds. Seed 3 specifically gave open-loop 17.2mm and
    closed-loop 2.8mm. The bar below is set from that pair with headroom for
    seed variation, not tuned until green.

    WHY THIS SCENARIO AND NOT `open`. The result is CONDITIONAL, and asserting
    it unconditionally would be false -- see the companion test below.

    SEED CHOICE. `constrained_passage` is 19/30 for KinodynamicRRT even
    unperturbed, so some seeds fail to plan at all and produce inf. Seed 3
    reached the goal at every mismatch level in the sweep, so it exercises the
    comparison rather than the scenario's own difficulty.
    """
    model_kappa = 1.0 / 25.0
    seed = 3
    start, goal = CONSTRAINED_PASSAGE.start, CONSTRAINED_PASSAGE.goal

    env = build_env(CONSTRAINED_PASSAGE)
    ol = run_open_loop(
        planner=make_planner(env, model_kappa, seed),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=model_kappa),
        start=start,
        goal=goal,
    )

    env2 = build_env(CONSTRAINED_PASSAGE)
    cl = run_closed_loop(
        planner=make_planner(env2, model_kappa, seed),
        env=env2,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=model_kappa),
        start=start,
        goal=goal,
        ekf_config=EKFConfig(seed=seed),
        loop_config=ClosedLoopConfig(seed=seed),
    )

    # Guard against the scenario's own failure rate masking the comparison: if
    # either run could not plan at all, this is not measuring what it claims.
    assert math.isfinite(ol.final_error_mm), "open-loop failed to plan; pick another seed"
    assert math.isfinite(cl.final_error_mm), "closed-loop failed to plan; pick another seed"

    # Measured 2.8 vs 17.2mm at this seed. Half of open-loop's error is a wide
    # margin over that, and still fails loudly if replanning stops helping.
    assert cl.final_error_mm < 0.5 * ol.final_error_mm, (
        f"closed-loop {cl.final_error_mm:.1f}mm did not meaningfully beat "
        f"open-loop {ol.final_error_mm:.1f}mm under 2x kappa mismatch on "
        f"constrained_passage (closed-loop ended: {cl.termination_reason})"
    )


def test_closed_loop_does_not_help_in_open_space():
    """The other half of the conditional result, and the one more likely to be
    quietly dropped later because it is the unflattering direction.

    On `open`, open-loop is essentially immune to curvature model error: it
    lands 2.9mm from the goal at every mismatch level from 1/50 to 1/25,
    because KinodynamicRRT's controls encode turn DIRECTIONS (b = +/-1) rather
    than turn magnitudes, and through an open corridor the executed path
    barely changes with kappa. There is nothing for replanning to correct.

    Closed-loop there is an intervention without a problem, and measurably so:
    median 6.0mm at 2x mismatch against open-loop's 2.9mm, winning only 2 of 5
    seeds. This test pins that closed-loop does NOT beat open-loop here -- if
    it ever starts to, either the scenario or the loop has changed and the
    conditional claim in docs/roadmap.md needs revisiting.

    Deliberately a weak inequality with slack: the point is that closed-loop
    fails to WIN, not that it fails by any particular amount.
    """
    model_kappa = 1.0 / 25.0
    seed = 1
    start, goal = OPEN.start, OPEN.goal

    env = build_env(OPEN)
    ol = run_open_loop(
        planner=make_planner(env, model_kappa, seed),
        env=env,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=model_kappa),
        start=start,
        goal=goal,
    )

    env2 = build_env(OPEN)
    cl = run_closed_loop(
        planner=make_planner(env2, model_kappa, seed),
        env=env2,
        true_params=NeedleParams(kappa=TRUE_KAPPA),
        model_params=NeedleParams(kappa=model_kappa),
        start=start,
        goal=goal,
        ekf_config=EKFConfig(seed=seed),
        loop_config=ClosedLoopConfig(seed=seed),
    )

    assert math.isfinite(ol.final_error_mm)
    assert math.isfinite(cl.final_error_mm)

    # Open-loop should be at or near the goal tolerance here -- that is the
    # premise of the whole test. If it is not, the scenario has changed.
    assert ol.final_error_mm < 5.0, (
        f"open-loop was expected to land near tolerance on `open` regardless "
        f"of kappa error, but got {ol.final_error_mm:.1f}mm — the premise of "
        f"this test no longer holds"
    )

    # The actual claim: closed-loop does not meaningfully beat it. Measured
    # 9.8mm (closed) vs 2.8mm (open) at this seed.
    assert cl.final_error_mm >= 0.8 * ol.final_error_mm, (
        f"closed-loop {cl.final_error_mm:.1f}mm now beats open-loop "
        f"{ol.final_error_mm:.1f}mm on `open` — the conditional finding in "
        f"docs/roadmap.md (replanning helps under constraint, not in open "
        f"space) may no longer hold and should be re-examined"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
