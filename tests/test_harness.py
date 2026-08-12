"""Tests for the benchmark harness metrics and its resume logic.

Fast by construction: the metric functions read only `result.controls` and
`result.path`, so they are exercised on tiny hand-built results (a
SimpleNamespace stands in for RRTResult) rather than by running planners. The
one thing that MUST hold -- and the benchmark's claim rests on it -- is that a
kinematically-consistent path has ~0 heading discontinuity while a
heading-teleporting one does not.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

from needlesim.benchmark.harness import (
    CSV_COLUMNS,
    _existing_keys,
    endpoint_error_mm,
    heading_discontinuity,
    make_config,
    path_cost_mm,
)
from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout

KAPPA = 1.0 / 50.0
PARAMS = NeedleParams(kappa=KAPPA)


def _consistent_path(cfg, n_edges=3):
    """Build a path by ACTUALLY rolling the model, so each stored node state is
    the rollout endpoint of its edge -- the kinodynamic invariant. Returns a
    result-like object with matching path + one control per edge."""
    control = Control(v=cfg.edge_velocity, b=1)
    state = State(10.0, 10.0, 0.3)
    path = [state]
    controls = []
    for _ in range(n_edges):
        state = rollout(
            path[-1], [control] * cfg.n_steps_per_extend, cfg.step_dt, PARAMS
        )[-1]
        path.append(state)
        controls.append(control)
    return SimpleNamespace(path=path, controls=controls)


def test_path_cost_matches_closed_form():
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=4)
    expected = 4 * cfg.n_steps_per_extend * cfg.edge_velocity * cfg.step_dt
    assert path_cost_mm(result, cfg) == expected


def test_heading_discontinuity_zero_for_consistent_path():
    # Kinodynamic case: arrival heading (re-rolled) == stored departure heading.
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=3)
    max_d, mean_d, count = heading_discontinuity(result, cfg, PARAMS)
    assert max_d < 1e-9
    assert mean_d < 1e-9
    assert count == 0


def test_heading_discontinuity_positive_for_teleported_heading():
    # Vanilla case: stored departure heading is teleported away from the
    # heading the previous edge actually produces.
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=3)
    # Teleport the interior node's stored heading by ~1 rad.
    k = 1
    p = result.path[k]
    result.path[k] = State(p.x, p.y, p.theta + 1.0)
    max_d, mean_d, count = heading_discontinuity(result, cfg, PARAMS)
    assert count >= 1
    assert max_d > 0.5


def test_heading_discontinuity_empty_for_single_edge():
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=1)  # only start + one node: no interior
    assert heading_discontinuity(result, cfg, PARAMS) == (0.0, 0.0, 0)


def test_endpoint_error_zero_when_goal_is_the_rolled_endpoint():
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=3)
    start = result.path[0]
    goal = result.path[-1]  # exactly where the controls land
    err = endpoint_error_mm(result, start, goal, cfg, PARAMS)
    assert err < 1e-9


def test_endpoint_error_nonzero_for_offset_goal():
    cfg = make_config(seed=0)
    result = _consistent_path(cfg, n_edges=3)
    start = result.path[0]
    landed = result.path[-1]
    goal = State(landed.x + 5.0, landed.y - 3.0, 0.0)
    err = endpoint_error_mm(result, start, goal, cfg, PARAMS)
    assert abs(err - (5.0**2 + 3.0**2) ** 0.5) < 1e-9


def test_existing_keys_reads_back_written_rows(tmp_path):
    # The resume mechanism: keys already in the CSV are skipped on rerun.
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for name, planner, seed in [
            ("open", "VanillaRRT", 0),
            ("open", "KinodynamicRRT", 3),
        ]:
            row = {c: "" for c in CSV_COLUMNS}
            row.update(scenario_name=name, planner=planner, seed=seed)
            w.writerow(row)
    keys = _existing_keys(path)
    assert keys == {("open", "VanillaRRT", "0"), ("open", "KinodynamicRRT", "3")}


def test_existing_keys_empty_when_no_file(tmp_path):
    assert _existing_keys(tmp_path / "missing.csv") == set()
