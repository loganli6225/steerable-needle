"""Smoke tests for the hand-designed benchmark scenarios.

Cheap invariants only: every scenario builds, and its start/goal poses are
collision-free at the benchmark margin. A scenario whose start or goal sits
inside an obstacle (or its margin) is degenerate -- no planner can begin or
finish -- and that is exactly the kind of geometry bug these guard against.

Difficulty (success-rate band) is NOT tested here; that is measured by
scripts/verify_scenarios.py and is a design decision, not a pass/fail.
"""

from __future__ import annotations

import pytest

from needlesim.benchmark.scenarios import (
    COMMON_CONDITIONS,
    HAND_DESIGNED,
    SCENARIOS,
    build_env,
)

MARGIN = COMMON_CONDITIONS["margin"]


@pytest.mark.parametrize("scenario", HAND_DESIGNED, ids=lambda s: s.name)
def test_scenario_builds(scenario):
    env = build_env(scenario)
    assert env.sdf is not None  # bake() ran
    assert env.width == scenario.width
    assert env.height == scenario.height


@pytest.mark.parametrize("scenario", HAND_DESIGNED, ids=lambda s: s.name)
def test_start_and_goal_free_with_margin(scenario):
    env = build_env(scenario)
    assert env.is_free(scenario.start, margin=MARGIN), "start inside obstacle/margin"
    assert env.is_free(scenario.goal, margin=MARGIN), "goal inside obstacle/margin"


@pytest.mark.parametrize("scenario", HAND_DESIGNED, ids=lambda s: s.name)
def test_start_and_goal_in_bounds(scenario):
    for pose in (scenario.start, scenario.goal):
        assert 0.0 <= pose.x <= scenario.width
        assert 0.0 <= pose.y <= scenario.height


def test_registry_matches_and_names_unique():
    names = [s.name for s in HAND_DESIGNED]
    assert len(names) == len(set(names)), "duplicate scenario names"
    assert set(SCENARIOS) == set(names)
    assert len(HAND_DESIGNED) == 4


def test_scenario_is_frozen_and_hashable():
    # Data-first: a Scenario must be immutable and hashable so it can be a dict
    # key / serialised uniformly with the PART 2 random scenarios.
    s = HAND_DESIGNED[0]
    assert hash(s) is not None
    with pytest.raises(Exception):
        s.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PART 2: the random scenario generator. Same cheap invariants (builds,
# start/goal free) plus the distribution ranges and reproducibility, since the
# generator -- unlike the hand-designed four -- is where a range bug would hide.
# ---------------------------------------------------------------------------

from needlesim.benchmark.random_scenarios import (  # noqa: E402
    CIRCLE_R_MAX,
    CIRCLE_R_MIN,
    MAX_OBSTACLES,
    MIN_OBSTACLES,
    RECT_SIDE_MAX,
    RECT_SIDE_MIN,
    Circle,
    Rect,
    generate_scenarios,
    load_scenarios,
    save_scenarios,
    scenario_from_dict,
    scenario_to_dict,
)

_RANDOM_SET = generate_scenarios(seed=123, n=20)


@pytest.mark.parametrize("scenario", _RANDOM_SET, ids=lambda s: s.name)
def test_random_scenario_builds_and_endpoints_free(scenario):
    env = build_env(scenario)
    assert env.sdf is not None
    # The generator's per-obstacle reject test must agree with the baked SDF.
    assert env.is_free(scenario.start, margin=MARGIN), "start inside obstacle/margin"
    assert env.is_free(scenario.goal, margin=MARGIN), "goal inside obstacle/margin"


def test_random_obstacle_counts_in_range():
    for s in _RANDOM_SET:
        assert MIN_OBSTACLES <= len(s.obstacles) <= MAX_OBSTACLES


def test_random_obstacle_sizes_in_range():
    for s in _RANDOM_SET:
        for o in s.obstacles:
            if isinstance(o, Circle):
                assert CIRCLE_R_MIN <= o.r <= CIRCLE_R_MAX
            else:
                assert isinstance(o, Rect)
                w, h = abs(o.x1 - o.x0), abs(o.y1 - o.y0)
                assert RECT_SIDE_MIN <= w <= RECT_SIDE_MAX
                assert RECT_SIDE_MIN <= h <= RECT_SIDE_MAX


def test_random_generator_count_and_fixed_endpoints():
    scens = generate_scenarios(seed=7, n=30)
    assert len(scens) == 30
    # start/goal are FIXED across all scenarios -- only obstacles vary.
    starts = {s.start for s in scens}
    goals = {s.goal for s in scens}
    assert len(starts) == 1 and len(goals) == 1


def test_random_generator_is_reproducible():
    # Same seed -> identical set (Scenario is a frozen, __eq__-able dataclass).
    a = generate_scenarios(seed=42, n=15)
    b = generate_scenarios(seed=42, n=15)
    assert a == b
    # Different seed -> different set (guards against an ignored seed).
    c = generate_scenarios(seed=43, n=15)
    assert a != c


def test_random_scenario_serialisation_round_trip():
    scen = _RANDOM_SET[0]
    assert scenario_from_dict(scenario_to_dict(scen)) == scen


def test_random_scenario_file_round_trip(tmp_path):
    scens = generate_scenarios(seed=99, n=10)
    path = tmp_path / "scenarios.json"
    save_scenarios(path, scens, seed=99)
    loaded_seed, loaded = load_scenarios(path)
    assert loaded_seed == 99
    assert loaded == scens
