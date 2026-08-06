"""Shared scaffolding for all needlesim planners.

Why a base class exists: the project's deliverable is a head-to-head benchmark
of VanillaRRT vs KinodynamicRRT vs RRTStar (see docs/research_question.md).
That comparison is only valid if all three planners run on provably identical
scaffolding -- same sampling, same nearest-neighbour, same distance metric,
same goal check. Those primitives therefore live HERE, once, and the planners
inherit them:

    PlannerBase
    |-- RRT (the shared RRT loop; extend() is abstract)
    |     |-- VanillaRRT       (extend = straight line, ignores curvature)
    |     |-- KinodynamicRRT   (extend = curvature-constrained arcs)
    |-- RRTStar (its own loop: choose-parent + rewire + full-budget
                 best-tracking)

Vanilla and kinodynamic RRT are the same algorithm differing only in `extend`,
so they share the RRT middle class -- the whole plan loop lives there once.
RRT* has a structurally different loop, so it is a sibling of RRT, sharing
only the base primitives. This makes "the only difference between vanilla and
kinodynamic is extend" and "all three share identical sampling/distance/
nearest" structurally true -- which is what makes the benchmark comparison
valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from needlesim.models.unicycle_needle import NeedleParams, State


@dataclass
class BasePlannerConfig:
    """Tunables shared by every planner. Config-driven + seeded (see CLAUDE.md
    conventions)."""

    max_iterations: int = 5000
    goal_tolerance: float = 3.0  # [mm] how close to goal counts as reached
    goal_sample_rate: float = 0.05  # fraction of samples drawn AT the goal (bias)
    step_dt: float = 0.05  # per-step dt handed to is_arc_free/step
    edge_velocity: float = 5.0  # v used when extending [mm/s]
    # [mm] clearance margin passed to collision checks. NOTE: RRTConfig and
    # RRTStarConfig currently override this with DIFFERENT defaults (2.0 vs 0.0)
    # to preserve historical behavior. The benchmark harness MUST set margin
    # explicitly and identically for all planners -- comparing planners at
    # different margins is invalid. See docs/roadmap.md.
    margin: float = 0.0
    seed: int = 0


class PlannerBase:
    """Shared planner primitives: sampling, spatial queries, distance metric,
    goal check.

    `plan` is deliberately NOT here -- each planner family owns its loop.
    Subclasses set `self.goal` in plan() before sampling; sample() reads it
    for goal biasing.
    """

    def __init__(self, env, params: NeedleParams, config: BasePlannerConfig) -> None:
        """
        Args:
            env: the Task 2 GridEnvironment (queried, never inspected).
            params: the MODEL needle params (NOT ground truth -- a planner
                that can see true params invalidates every model-mismatch
                result later; see docs/roadmap.md).
            config: the planner's config (a BasePlannerConfig subclass).
        """
        self.env = env
        self.params = params  # MODEL params, never ground truth
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Validate the is_arc_free spacing invariant up front so its assert
        # can never fire mid-run.
        assert config.edge_velocity * config.step_dt <= 0.5 * env.resolution

    def sample(self) -> State:
        """Draw a random target pose from the environment bounds.

        With probability config.goal_sample_rate, returns the goal pose
        instead of a random one (goal biasing). Uses self.rng, not np.random,
        so runs are reproducible from the seed.
        """
        if self.rng.random() < self.config.goal_sample_rate:
            return self.goal
        x_min, y_min, x_max, y_max = self.env.bounds
        x = self.rng.uniform(x_min, x_max)
        y = self.rng.uniform(y_min, y_max)
        theta = self.rng.uniform(0, 2 * np.pi)
        return State(x, y, theta)

    def nearest(self, nodes: list, target: State) -> int:
        """Return the INDEX of the tree node closest to `target`.

        "Closest" is defined by self.distance. Linear scan, O(n) per call --
        deliberately so for now (see docs/roadmap.md; a KD-tree is a later,
        separate step).
        """
        closest_node_idx = -1
        closest_distance = np.inf
        for idx, node in enumerate(nodes):
            node_distance = self.distance(target, node.state)
            if closest_distance > node_distance:
                closest_distance = node_distance
                closest_node_idx = idx
        return closest_node_idx

    def near_indices(self, nodes: list, target: State, radius: float) -> list[int]:
        """Indices of all nodes within `radius` of target.

        Unlike nearest(), returns the whole neighbourhood (RRT* needs it for
        choose-parent and rewire). Lives in the base alongside nearest() on
        purpose: both spatial queries are co-located so a later step can back
        them with one shared structure. Linear scan, O(n) per call.
        """
        indices_within_radius = []
        for idx, node in enumerate(nodes):
            if self.distance(target, node.state) <= radius:
                indices_within_radius.append(idx)
        return indices_within_radius

    def distance(self, a: State, b: State) -> float:
        """Distance between two poses -- position-only Euclidean."""
        # Position-only by deliberate choice: all three planners use it. A
        # heading-weighted metric sqrt(dx^2 + dy^2 + w*wrap(dtheta)^2) with
        # w ~ (1/kappa)^2 was tried for kinodynamic RRT and STARVED THE TREE at
        # w=(1/kappa)^2=400 (heading error dominated position). If a future
        # scenario demands tight heading-matching to thread a gap, reintroduce
        # the weighted form as a KinodynamicRRT.distance override, wrap dtheta
        # to [-pi, pi] via (d + pi) % (2*pi) - pi, and start with SMALL w.
        return math.hypot(a.x - b.x, a.y - b.y)

    def reached_goal(self, state: State) -> bool:
        """True if `state` is within config.goal_tolerance of the goal."""
        return self.distance(state, self.goal) < self.config.goal_tolerance
