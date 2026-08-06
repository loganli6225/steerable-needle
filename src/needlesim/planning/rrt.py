"""RRT planners for the steerable needle: one loop, two extend strategies.

Hierarchy (see planning/base.py for the full rationale):

    PlannerBase -> RRT -> {VanillaRRT, KinodynamicRRT}

Vanilla and kinodynamic RRT are the SAME algorithm differing only in how the
tree grows toward a sample, so the loop (`plan`, `reconstruct`) lives once in
the RRT middle class and each concrete subclass supplies only `extend`:

- VanillaRRT: straight-line extend. Treats the needle as a point robot that
  can snap its heading toward each sample, so it happily produces paths the
  real needle cannot follow. That is deliberate: vanilla is a baseline the
  benchmark needs, and its jagged infeasible paths versus kinodynamic's
  feasible arcs is the motivating comparison figure (docs/roadmap.md).
- KinodynamicRRT: curvature-constrained extend. Picks the bevel direction
  b in {+1, -1} that bends the needle most toward the sample and rolls the
  Task 1 model forward.

Sampling, nearest-neighbour, the distance metric, and the goal check are
inherited from PlannerBase, so all planners (RRT* included) are compared on
identical scaffolding.

What this planner needs from earlier tasks
------------------------------------------
- Task 1 model: `step` (and NeedleParams/State/Control) to roll arcs forward.
- Task 2 environment: `is_free`, `is_arc_free`, `clearance`, `bounds`.
  The planner ONLY calls these; it never reaches into the SDF or the grid.
- CRITICAL: the planner consumes the MODEL needle params, never the
  simulator's ground truth. See the model_needle vs true_needle rule in
  docs/roadmap.md. A planner that can see true params invalidates every
  model-mismatch result later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout
from needlesim.planning.base import BasePlannerConfig, PlannerBase

# ---------------------------------------------------------------------------
# Plain containers -- scaffolding, safe to delegate. These just hold data.
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A vertex in the RRT tree.

    Stores the pose AND how we got here, so the control sequence can be
    reconstructed by walking parent pointers back to the root. `control` is the
    control that produced THIS node from its parent (None for the root).
    """

    state: State
    parent: int | None = None  # index into the tree's node list
    control: Control | None = None  # control that produced this node


@dataclass
class RRTResult:
    """Everything a caller (or a plot) needs after a run.

    Full tree is returned, not just the path, so exploration can be visualised
    and later benchmarked (nodes expanded, coverage, etc.).
    """

    nodes: list[Node]  # the whole tree
    path: list[State] | None  # goal->start reversed to start->goal, or None
    controls: list[Control] | None  # controls along the path, or None
    success: bool
    n_iterations: int


# ---------------------------------------------------------------------------
# Config -- scaffolding.
# ---------------------------------------------------------------------------


@dataclass
class RRTConfig(BasePlannerConfig):
    """RRT-family tunables. Shared fields come from BasePlannerConfig."""

    n_steps_per_extend: int = 20  # steps taken per extend (arc length = v*dt*n)
    # [mm] clearance required on top of the point-robot arc, so the vanilla
    # and curvature-constrained planners are compared under the same
    # physical-width assumption. Bevel-tip needles used in the literature run
    # ~1.2-2mm OD (radius ~0.6-1.0mm); add ~1mm for tracking/model-mismatch
    # safety -> ~2mm total. Revisit if a specific needle gauge is adopted.
    margin: float = 2.0


# ---------------------------------------------------------------------------
# The planners.
# ---------------------------------------------------------------------------


class RRT(PlannerBase):
    """The shared RRT loop. Concrete subclasses implement `extend`."""

    def extend(self, from_state: State, toward: State) -> tuple[State, Control] | None:
        """Grow from `from_state` toward `toward` by one edge.

        This is the HEART of the planner and the one method that changes
        between the vanilla and curvature-constrained versions. Either way,
        the edge must be collision-checked with is_arc_free BEFORE it is
        accepted. Returning None means "couldn't grow this way."
        """
        raise NotImplementedError(
            "subclasses implement the steering: "
            "VanillaRRT (straight) or KinodynamicRRT (arcs)"
        )

    # --- goal + assembly --------------------------------------------------

    def reconstruct(
        self, nodes: list[Node], goal_idx: int
    ) -> tuple[list[State], list[Control]]:
        """Walk parent pointers from goal_idx back to the root.

        Collects states and controls, then reverses so the path runs
        start -> goal. The root's control is None and does not appear in the
        control list.
        """
        states_flipped = []
        controls_flipped = []
        idx = goal_idx
        while idx is not None:
            node = nodes[idx]
            states_flipped.append(node.state)
            if node.control is not None:
                controls_flipped.append(node.control)
            idx = node.parent
        states_corrected = states_flipped[::-1]
        controls_corrected = controls_flipped[::-1]
        return (states_corrected, controls_corrected)

    # --- the main loop ----------------------------------------------------

    def plan(self, start: State, goal: State) -> RRTResult:
        """Run RRT from `start` to `goal`.

        Stores the goal (self.goal = goal) before the loop so sample() can
        bias toward it. Returns an RRTResult either way; on failure the tree
        so far is still returned (useful to plot -- shows what it explored).
        """
        self.goal = goal
        tree = [Node(start, parent=None, control=None)]
        for num_iterations in range(1, self.config.max_iterations + 1):
            target = self.sample()
            i = self.nearest(tree, target)
            result = self.extend(tree[i].state, target)
            if result is None:
                continue
            new_state, control = result
            tree.append(Node(new_state, parent=i, control=control))
            if self.reached_goal(new_state):
                goal_idx = len(tree) - 1
                success_path, success_controls = self.reconstruct(tree, goal_idx)
                return RRTResult(
                    nodes=tree,
                    path=success_path,
                    controls=success_controls,
                    success=True,
                    n_iterations=num_iterations,
                )
        return RRTResult(
            nodes=tree,
            path=None,
            controls=None,
            success=False,
            n_iterations=self.config.max_iterations,
        )


class VanillaRRT(RRT):
    """Straight-line RRT: the point-robot baseline the benchmark needs."""

    def __init__(self, env, params: NeedleParams, config: RRTConfig) -> None:
        super().__init__(env, params, config)
        # kappa=0 turns the Task 1 model into a straight-line integrator --
        # the point-robot cheat. These two attributes are the single swap
        # point for the vanilla edge.
        self._vanilla_params = NeedleParams(kappa=0.0)
        self._vanilla_control = Control(v=config.edge_velocity, b=1)

    def extend(self, from_state: State, toward: State) -> tuple[State, Control] | None:
        """Head straight at `toward`: compute the bearing from `from_state`,
        synthesise a State whose theta points at the target, and roll
        straight. Ignores that the real needle cannot snap its heading."""
        if self.distance(from_state, toward) < 1e-9:
            return None
        theta = np.arctan2(toward.y - from_state.y, toward.x - from_state.x)
        from_state_copy = State(from_state.x, from_state.y, theta)
        if self.env.is_arc_free(
            from_state_copy,
            self._vanilla_control,
            self.config.step_dt,
            self.config.n_steps_per_extend,
            self._vanilla_params,
            self.config.margin,
        ):
            # rollout of n separate steps. With kappa=0 a single big step would be
            # exact (constant derivative), but stepping n times is what the
            # curvature-constrained version needs, so keep this form.
            controls = [self._vanilla_control] * self.config.n_steps_per_extend
            trace = rollout(
                from_state_copy, controls, self.config.step_dt, self._vanilla_params
            )
            new_state = trace[-1]
            return (new_state, self._vanilla_control)
        return None


class KinodynamicRRT(RRT):
    """Curvature-constrained RRT: extend respects the needle's kinematics."""

    def extend(self, from_state: State, toward: State) -> tuple[State, Control] | None:
        """Pick the control that bends the needle MOST TOWARD `toward`: try
        b=+1 and b=-1, roll each forward n_steps_per_extend with the Task 1
        `step`, and keep whichever lands nearer `toward` by self.distance --
        provided is_arc_free passes."""
        if self.distance(from_state, toward) < 1e-9:
            return None

        best_scenario = None
        for b in [1, -1]:
            control = Control(v=self.config.edge_velocity, b=b)
            if not self.env.is_arc_free(
                from_state,
                control,
                self.config.step_dt,
                self.config.n_steps_per_extend,
                self.params,
                self.config.margin,
            ):
                continue
            controls = [control] * self.config.n_steps_per_extend
            trace = rollout(from_state, controls, self.config.step_dt, self.params)
            end_state = trace[-1]
            d = self.distance(end_state, toward)
            if best_scenario is None or d < best_scenario[0]:
                best_scenario = (d, end_state, control)

        if best_scenario is None:
            return None
        _, new_state, control = best_scenario
        return new_state, control
