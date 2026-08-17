"""Tests for the VanillaRRT open-loop segment-following tracker.

Three things must hold, each pinned against a hand-computable case:
  1. The b sign rule: a left (CCW) turn in the reference polyline -> b=+1, a
     right (CW) turn -> b=-1. This guards the "b=+1 turns left" convention; a
     silent flip would mirror every trajectory.
  2. Cross-track distance is to the POLYLINE (nearest segment), including the
     case where the nearest point is a segment's interior, not an endpoint.
  3. A trajectory that passes through an obstacle reports collides=True.
"""

from __future__ import annotations

import math

from needlesim.benchmark.vanilla_tracker import (
    _point_segment_distance,
    crosstrack_distance,
    track_path,
    tracking_controls,
)
from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State, rollout

KAPPA = 1.0 / 50.0
PARAMS = NeedleParams(kappa=KAPPA)
EDGE_V = 5.0
STEP_DT = 0.05


# --- 1. the b sign rule -----------------------------------------------------


def test_b_sign_left_turn_is_positive():
    """Polyline that goes east then turns NORTH (a left/CCW turn) must be
    tracked with b=+1 on the second segment. Start heading = east, so the
    first (east) segment has turn 0 -> b=-1 by the >0 rule, and the north
    segment turns +90deg -> b=+1."""
    start = State(0.0, 0.0, 0.0)  # pointing east
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0), State(10.0, 10.0, 0.0)]
    controls = tracking_controls(path, start, EDGE_V, STEP_DT)
    # First segment: bearing 0, turn wrap(0 - 0) = 0, not > 0 -> b = -1.
    assert controls[0].b == -1
    # Last control belongs to the north segment: bearing +pi/2, turn +pi/2 -> +1.
    assert controls[-1].b == 1


def test_b_sign_right_turn_is_negative():
    """Mirror of the above: east then turn SOUTH (a right/CW turn) -> b=-1."""
    start = State(0.0, 0.0, 0.0)
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0), State(10.0, -10.0, 0.0)]
    controls = tracking_controls(path, start, EDGE_V, STEP_DT)
    # South segment: bearing -pi/2, turn wrap(-pi/2 - 0) = -pi/2, < 0 -> b=-1.
    assert controls[-1].b == -1


def test_b_sign_convention_matches_model():
    """The convention rests on the model: b=+1 must actually INCREASE theta.
    If someone flips theta_dot's sign in the model, this fails loudly rather
    than the tracker silently mirroring every trajectory."""
    from needlesim.models.unicycle_needle import Control

    after = rollout(State(0.0, 0.0, 0.0), [Control(EDGE_V, 1)], STEP_DT, PARAMS)[-1]
    assert after.theta > 0.0  # b=+1 turned left (theta increased)


def test_first_segment_turn_measured_from_start_heading():
    """A single segment whose bearing is LEFT of the start heading -> b=+1;
    right of it -> b=-1. Confirms bearing_{-1} is the start pose's theta."""
    # Segment points east (bearing 0); start points south-east-ish (theta<0),
    # so the turn to east is positive (left) -> b=+1.
    start_left = State(0.0, 0.0, -0.5)
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0)]
    assert tracking_controls(path, start_left, EDGE_V, STEP_DT)[0].b == 1
    # Start points north-east-ish (theta>0); turn to east is negative -> b=-1.
    start_right = State(0.0, 0.0, 0.5)
    assert tracking_controls(path, start_right, EDGE_V, STEP_DT)[0].b == -1


# --- 2. cross-track distance ------------------------------------------------


def test_point_segment_distance_interior_projection():
    """A point above the middle of a horizontal segment is nearest to the
    segment INTERIOR, not either endpoint. Distance must be the perpendicular
    offset (3), not the distance to an endpoint (sqrt(5^2+3^2))."""
    d = _point_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(d - 3.0) < 1e-12


def test_point_segment_distance_beyond_endpoint():
    """A point past the segment's end clamps to the endpoint."""
    d = _point_segment_distance(15.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(d - 5.0) < 1e-12


def test_crosstrack_uses_nearest_segment():
    """An L-shaped polyline: a query point near the elbow's interior of the
    SECOND segment must measure to that segment, catching the case where it is
    far from the first segment's nodes."""
    path = [State(0.0, 0.0, 0.0), State(10.0, 0.0, 0.0), State(10.0, 10.0, 0.0)]
    # Point at (13, 5): nearest to the vertical segment interior, offset 3.
    d = crosstrack_distance(State(13.0, 5.0, 0.0), path)
    assert abs(d - 3.0) < 1e-12


# --- 3. collision detection -------------------------------------------------


def _straight_up_env():
    """150x150 world with a circle centred at (0, 15) so a needle driving
    north out of the origin passes through it."""
    env = GridEnvironment(width=150.0, height=150.0, resolution=0.5)
    env.add_circle(0.0, 15.0, 5.0)
    env.bake()
    return env


def test_track_reports_collision_through_obstacle():
    """A reference path that runs straight up through an obstacle: the executed
    trajectory hits it, so collides=True and first_collision_index is set."""
    env = _straight_up_env()
    start = State(0.0, 0.0, math.pi / 2)  # pointing north
    path = [State(0.0, 0.0, math.pi / 2), State(0.0, 40.0, math.pi / 2)]
    tracked = track_path(
        path, start, path[-1], EDGE_V, STEP_DT, env, PARAMS, margin=0.0
    )
    assert tracked.collides is True
    assert tracked.first_collision_index is not None


def test_track_no_collision_in_open_space():
    """Same tracker on an obstacle-free world does not report a collision."""
    env = GridEnvironment(width=150.0, height=150.0, resolution=0.5)
    env.bake()
    start = State(20.0, 20.0, math.pi / 2)
    path = [State(20.0, 20.0, math.pi / 2), State(20.0, 60.0, math.pi / 2)]
    tracked = track_path(
        path, start, path[-1], EDGE_V, STEP_DT, env, PARAMS, margin=2.0
    )
    assert tracked.collides is False
    assert tracked.first_collision_index is None
