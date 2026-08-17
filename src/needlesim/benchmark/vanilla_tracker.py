"""Open-loop segment-following tracker for VanillaRRT paths (benchmark-only).

WHY THIS EXISTS. VanillaRRT's `extend` stores a single Control(v, b=+1) per
edge, because all of vanilla's steering information lives in the per-node
headings it synthesises (pointing at each sample) and then discards. Executing
that stored control sequence just drives the needle in a circle of radius
1/kappa, so the old `endpoint_error_mm` measured where that circle happened to
end -- a STORAGE CONVENTION, not path quality, and no real system would deploy
raw controls. This module instead reads vanilla's PATH (`result.path`) as a
reference polyline and derives feasible curved controls that ATTEMPT to follow
it -- the most charitable feasible reading of what vanilla output.

The tracker is VanillaRRT-ONLY, on purpose. KinodynamicRRT and RRTStar generate
their controls WITH the model, so their planned and executed paths are the same
curve and `endpoint_error_mm` is already exact (~0 heading disc, mm-scale
error). Re-deriving their controls from node bearings would manufacture a small
error that says nothing about the planner (an arc's chord bearing differs from
its tangent) and degrade a metric that is currently exact. Do not apply this to
them.

OPEN-LOOP BY DELIBERATE CHOICE. The whole b-sequence is decided up front from
the polyline's geometry; the needle's actual position never feeds back. This
grants vanilla no capability it lacked -- no sensing, no correction -- so it
tests the path as an intent, not a tracking controller's skill. A closed-loop
version (pure pursuit) is explicitly out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]. Same convention as harness._wrap; duplicated here so
    the tracker stays importable BY the harness without an import cycle."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def tracking_controls(
    path: list[State], start: State, edge_velocity: float, step_dt: float
) -> list[Control]:
    """Derive an open-loop control sequence that follows `path` from `start`.

    For each segment k of the path, from path[k] to path[k+1]:
      1. bearing_k = atan2(path[k+1].y - path[k].y, path[k+1].x - path[k].x).
      2. turn = wrap(bearing_k - bearing_{k-1}), giving a value in (-pi, pi].
         bearing_{-1} is the START POSE's heading -- the first segment has no
         preceding segment, so its turn is measured from where the needle
         actually points.
      3. b = +1 if turn > 0 (CCW / left), else b = -1 (CW / right).
         b=+1 turns LEFT: theta_dot = v * kappa * b with kappa > 0, so a
         positive b increases theta. This convention was confirmed empirically
         against the model. A future sign flip would silently MIRROR every
         trajectory, so tests/test_vanilla_tracker.py pins it against a
         known-geometry path -- if you flip the rule, that test must fail.
      4. Hold that b for ceil(seg_len / (edge_velocity * step_dt)) steps at
         step_dt.

    Degenerate zero-length segments carry no bearing and are skipped.
    """
    controls: list[Control] = []
    prev_bearing = start.theta  # bearing_{-1}: measured from the start heading
    for k in range(len(path) - 1):
        p0, p1 = path[k], path[k + 1]
        dx, dy = p1.x - p0.x, p1.y - p0.y
        seg_len = math.hypot(dx, dy)
        if seg_len == 0.0:
            continue
        bearing = math.atan2(dy, dx)
        turn = _wrap(bearing - prev_bearing)
        b = 1 if turn > 0 else -1
        n_steps = math.ceil(seg_len / (edge_velocity * step_dt))
        controls.extend(Control(edge_velocity, b) for _ in range(n_steps))
        prev_bearing = bearing
    return controls


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Euclidean distance from point (px, py) to the SEGMENT a-b.

    Distance to the segment, not the infinite line and not just the nearest
    endpoint: a point can sit nearest to a segment's INTERIOR while being far
    from both its endpoints. The projection parameter t is clamped to [0, 1].
    """
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:  # degenerate segment -> distance to the point
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def crosstrack_distance(state: State, path: list[State]) -> float:
    """Minimum distance from `state` to the reference POLYLINE.

    The min over all segments of point-to-segment distance -- to the polyline,
    NOT to the nearest node. Requires at least two path points.
    """
    return min(
        _point_segment_distance(
            state.x, state.y, path[k].x, path[k].y, path[k + 1].x, path[k + 1].y
        )
        for k in range(len(path) - 1)
    )


@dataclass
class TrackedRun:
    """Result of executing vanilla's path as open-loop tracked controls."""

    trace: list[State]  # the executed trajectory, including the start pose
    endpoint_error_mm: float  # |final - goal|
    max_crosstrack_mm: float  # max over the trace of distance to the polyline
    collides: bool  # any executed state violates is_free(state, margin)
    first_collision_index: int | None  # index into trace, or None -- for figures


def track_path(
    path: list[State],
    start: State,
    goal: State,
    edge_velocity: float,
    step_dt: float,
    env,
    params: NeedleParams,
    margin: float,
) -> TrackedRun:
    """Execute vanilla's `path` as open-loop tracked controls and measure it.

    Rolls the derived controls (see `tracking_controls`) through the REAL model
    from `start`, then computes the three vanilla metrics:
      - endpoint_error_mm: distance from the executed final state to the goal.
      - max_crosstrack_mm: max distance from any executed state to the polyline.
      - collides: does any executed state violate env.is_free(state, margin)?
        The harness's edge_velocity*step_dt <= 0.5*resolution invariant makes
        the rollout dense enough not to step over a thin obstacle.
    """
    controls = tracking_controls(path, start, edge_velocity, step_dt)
    trace = rollout(start, controls, step_dt, params)

    final = trace[-1]
    endpoint_error = math.hypot(final.x - goal.x, final.y - goal.y)
    max_ct = max(crosstrack_distance(s, path) for s in trace)

    first_collision: int | None = None
    for i, s in enumerate(trace):
        if not env.is_free(s, margin):
            first_collision = i
            break

    return TrackedRun(
        trace=trace,
        endpoint_error_mm=endpoint_error,
        max_crosstrack_mm=max_ct,
        collides=first_collision is not None,
        first_collision_index=first_collision,
    )
