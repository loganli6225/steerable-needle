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
