"""Tests for closed-loop needle steering.

The central claim this phase makes: under model mismatch, replanning from the
filter's estimate keeps the needle on target where open-loop execution drifts.
These tests pin the mechanism, and the last one pins the claim.

Run:  pytest tests/test_closed_loop.py -v
"""

import math

import pytest

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
# 3. THE CLAIM -- closed-loop beats open-loop under mismatch.
# =====================================================================


def test_closed_loop_beats_open_loop_under_mismatch():
    """The Phase 3.5 result. Same scenario, same seeds, same 2x kappa error:
    open-loop executes blind and drifts; closed-loop replans from the
    estimate and stays on target.

    TODO: fill this in once the loop runs. Assert the closed-loop final error
    is meaningfully smaller than open-loop's -- but derive the bar from a
    measured pair rather than tuning until green, and note what you measured
    in a comment. If closed-loop does NOT win, that is a finding worth
    understanding before adjusting anything: it would mean the drift trigger,
    the replanning, or the estimate is not doing what this phase assumes.

    Worth also asserting on collisions: open-loop under mismatch should
    sometimes hit the obstacle, closed-loop should not.
    """
    pytest.skip("fill in with measured values once the loop runs")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
