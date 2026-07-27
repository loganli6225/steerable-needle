"""RRT planner for the steerable needle.

THIS IS YOUR TASK 3 FILE. The interface and the RRT skeleton are laid out so
the pieces are clear, but the load-bearing methods are left for you to
implement (marked IMPLEMENT ME). Per the working agreement, the steering
function, the distance metric, and the main loop are YOURS; the plotting and
the plain data containers are scaffolding you may delegate.

Plan of attack (see docs/roadmap.md for why this order)
-------------------------------------------------------
VANILLA FIRST. Start with a straight-line `extend` that ignores the needle's
curvature constraint -- treat it as a point robot that can move directly
toward a sample. This will happily produce paths the real needle cannot
follow. That is deliberate: it exercises the tree, nearest-neighbour search,
sampling, goal-checking, and the is_arc_free plumbing with a steering function
simple enough that any bug is obviously NOT in the steering.

THEN curvature-constrained. Swap ONLY `extend` (pick the b in {-1,+1} that
bends the needle most toward the sample and roll forward with the Task 1
model) and the distance metric (start caring about heading). The tree/loop/
goal code does not change. Plot both: the straight-line paths cutting through
gaps the needle cannot actually take, versus the feasible arcs. That before/
after is a figure for the writeup.

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
class RRTConfig:
    """Tunables for a run. Config-driven + seeded (see CLAUDE.md conventions)."""

    max_iterations: int = 5000
    goal_tolerance: float = 3.0  # [mm] how close to goal counts as reached
    goal_sample_rate: float = 0.05  # fraction of samples drawn AT the goal (bias)
    step_dt: float = 0.05  # per-step dt handed to is_arc_free/step
    n_steps_per_extend: int = 20  # steps taken per extend (arc length = v*dt*n)
    edge_velocity: float = 5.0  # v used when extending [mm/s]
    seed: int = 0
    # [mm] clearance required on top of the point-robot arc, so the vanilla
    # and curvature-constrained planners are compared under the same
    # physical-width assumption. Bevel-tip needles used in the literature run
    # ~1.2-2mm OD (radius ~0.6-1.0mm); add ~1mm for tracking/model-mismatch
    # safety -> ~2mm total. Revisit if a specific needle gauge is adopted.
    margin: float = 2.0
    # NOTE: step_dt must satisfy the is_arc_free spacing assert:
    #   edge_velocity * step_dt <= 0.5 * env.resolution
    # Validate this once when the planner is constructed, not per call.


# ---------------------------------------------------------------------------
# The planner. The methods below marked IMPLEMENT ME are YOURS.
# ---------------------------------------------------------------------------


class RRT:
    def __init__(self, env, params: NeedleParams, config: RRTConfig) -> None:
        """
        Args:
            env: the Task 2 GridEnvironment (queried, never inspected).
            params: the MODEL needle params (NOT ground truth -- see docstring).
            config: RRTConfig.
        """
        self.env = env
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._theta_weight = 0.0  # (1/params.kappa)**2

        ### VANILLA
        # self._vanilla_params = NeedleParams(kappa=0.0)
        # self._vanilla_control = Control(v=config.edge_velocity, b=1)

        # Recommended: validate the spacing invariant up front so the
        # is_arc_free assert can never fire mid-run. IMPLEMENT (small).
        assert config.edge_velocity * config.step_dt <= 0.5 * env.resolution

    # --- the four pieces of the RRT loop ---------------------------------

    def sample(self) -> State:
        """Draw a random target pose from the environment bounds.

        IMPLEMENT ME. With probability config.goal_sample_rate, return the goal
        pose instead of a random one (goal biasing) -- this is what makes RRT
        actually reach the goal in reasonable time. Use self.rng, not
        np.random directly, so runs are reproducible from the seed.

        For vanilla RRT you can sample theta arbitrarily (it's ignored by the
        straight-line extend). For the curvature-constrained version you'll
        care about it. self.goal is set in plan(); you may store it on self.
        """
        if self.rng.random() < self.config.goal_sample_rate:
            return self.goal
        x_min, y_min, x_max, y_max = self.env.bounds
        x = self.rng.uniform(x_min, x_max)
        y = self.rng.uniform(y_min, y_max)
        theta = self.rng.uniform(0, 2 * np.pi)
        return State(x, y, theta)

    def nearest(self, nodes: list[Node], target: State) -> int:
        """Return the INDEX of the tree node closest to `target`.

        IMPLEMENT ME. "Closest" is defined by self.distance. Linear scan over
        nodes is fine to start (this is O(n) per iteration; a KD-tree is a
        later optimisation, do not bother now).
        """
        closest_node_idx = -1
        closest_distance = np.inf
        for idx, node in enumerate(nodes):
            node_distance = self.distance(target, node.state)
            if closest_distance > node_distance:
                closest_distance = node_distance
                closest_node_idx = idx
        return closest_node_idx

    def distance(self, a: State, b: State) -> float:
        """Distance between two poses.

        IMPLEMENT ME.
        VANILLA: Euclidean distance in (x, y) only -- ignore theta.
        LATER (curvature-constrained): you will need to account for heading,
        because two poses can be near in position yet hard to connect given the
        needle can only turn at curvature kappa. Leave a note to yourself here.

        """
        # TODO (curvature-constrained): position-only distance is wrong for a
        # nonholonomic needle -- two poses can be 1mm apart but require a long
        # detour to connect if their headings differ (you can't pivot in place).
        # Will need a weighted metric: sqrt(dx^2 + dy^2 + w * angle_diff^2), with
        # w tuned relative to the turning radius 1/kappa. Angle diff must wrap
        # to [-pi, pi].
        ### FOR VANILLA
        # return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

        # Heuristic metric: weighted Euclidean, not true Dubins cost-to-connect.
        # Good enough for kinodynamic RRT; RRT* will need the real Dubins distance.
        d_theta = (a.theta - b.theta + np.pi) % (2 * np.pi) - np.pi
        return np.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + self._theta_weight * (d_theta) ** 2
        )

    def extend(self, from_state: State, toward: State) -> tuple[State, Control] | None:
        """Grow from `from_state` toward `toward` by one edge.

        This is the HEART of the planner and the one method that changes
        between the vanilla and curvature-constrained versions.

        IMPLEMENT ME.

        VANILLA (do this first):
            Treat the needle as a point robot. Head straight at `toward`:
            compute the bearing from from_state to toward, MOVE a fixed arc
            length in that direction (you can synthesise a State whose theta
            points at the target and roll straight), and return the new state
            plus the control used. Ignore that the real needle cannot snap its
            heading. Check the swept edge with self.env.is_arc_free; return None
            if the edge collides.

        CURVATURE-CONSTRAINED (swap in later):
            You cannot point at the target for free. Instead pick the control
            that bends the needle MOST TOWARD `toward`: try b=+1 and b=-1
            (optionally a straight/duty-cycled option), roll each forward
            n_steps_per_extend with the Task 1 `step`, and keep whichever lands
            nearer `toward` by self.distance -- provided is_arc_free passes.
            Return (new_state, control) or None if both collide.

        Either way: the edge must be collision-checked with is_arc_free BEFORE
        you accept it. Returning None means "couldn't grow this way."
        """
        ### FOR VANILLA
        # if self.distance(from_state, toward) < 1e-9:
        #     return None
        # theta = np.arctan2(toward.y - from_state.y, toward.x - from_state.x)
        # from_state_copy = State(from_state.x, from_state.y, theta)
        # if self.env.is_arc_free(from_state_copy, self._vanilla_control, self.config.step_dt, self.config.n_steps_per_extend, self._vanilla_params, self.config.margin):
        #     # rollout of n separate steps. With kappa=0 a single big step would be
        #     # exact (constant derivative), but stepping n times is what the
        #     # curvature-constrained version needs, so keep this form.
        #     controls = [self._vanilla_control] * self.config.n_steps_per_extend
        #     trace = rollout(from_state_copy, controls, self.config.step_dt, self._vanilla_params)
        #     new_state = trace[-1]
        #     return (new_state, self._vanilla_control)
        # return None

        ### KINODYNAMIC
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

    # --- goal + assembly --------------------------------------------------

    def reached_goal(self, state: State) -> bool:
        """True if `state` is within config.goal_tolerance of the goal.

        IMPLEMENT ME (small). Use self.distance or plain (x,y) distance; decide
        whether heading matters for "reached" (for vanilla, position is fine).
        """
        return self.distance(state, self.goal) < self.config.goal_tolerance

    def reconstruct(
        self, nodes: list[Node], goal_idx: int
    ) -> tuple[list[State], list[Control]]:
        """Walk parent pointers from goal_idx back to the root.

        IMPLEMENT ME (small, pure bookkeeping -- delegable). Collect states and
        controls, then reverse so the path runs start -> goal. The root's
        control is None and should not appear in the control list.
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

        IMPLEMENT ME. The skeleton, in words:

            1. tree = [Node(start, parent=None, control=None)]
            2. repeat up to config.max_iterations:
                 a. target = self.sample()
                 b. i = self.nearest(tree, target)
                 c. result = self.extend(tree[i].state, target)
                 d. if result is None: continue        # edge blocked
                 e. new_state, control = result
                 f. append Node(new_state, parent=i, control=control)
                 g. if self.reached_goal(new_state):
                        build path via self.reconstruct and RETURN success
            3. if the loop ends without reaching goal: return failure with the
               tree so far (still useful to plot -- shows what it explored).

        Store the goal (self.goal = goal) before the loop so sample() can bias
        toward it. Return an RRTResult either way.
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
