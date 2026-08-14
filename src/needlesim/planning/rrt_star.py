"""
RRT* planner for the steerable needle, built on CCC Dubins steering.

Hierarchy note (see planning/base.py for the full rationale): RRTStar extends
PlannerBase directly, as a SIBLING of the RRT class -- not a subclass of it.
Its loop is structurally different from RRT's (choose-parent + rewire +
full-budget best-tracking instead of return-on-first-contact), so only the
base primitives (sample / nearest / near_indices / distance / reached_goal)
are shared. That sharing is what makes the three-planner benchmark
comparison valid.

RRT* = RRT + two additions that plain RRT lacks:
  1. CHOOSE-BEST-PARENT: a new node attaches not to the nearest node, but to
     whichever nearby node yields the lowest cost-from-start (via a collision-
     free Dubins connection).
  2. REWIRE: after adding a node, check whether nearby nodes could reach the
     start more cheaply THROUGH the new node; re-parent them if so. This is
     what drives convergence toward the optimal path.

Both require EXACT pose-to-pose steering with a length -- that is dubins_ccc
(Task 3.5). dubins_ccc returns None when two poses cannot be connected; RRT*
handles that by simply not forming that edge.

TWO DESIGN CHOICES, DELIBERATELY ISOLATED (see the two functions near the top):
  - COST: start length-only (standard RRT*, simplest correct baseline), then
    swap to length + clearance penalty once the machinery is trusted. The
    penalty makes cost non-metric, so strict RRT*-optimality no longer holds --
    it becomes cost-aware RRT*, biased toward safer paths. Build length-only
    FIRST; debugging rewiring is far easier when "shorter == better" holds.
  - REWIRE RADIUS: shrinking r(n) = gamma * (log n / n)^(1/d). d is the config-
    space dimension. For a Dubins (x,y,theta) system d=3 is the defensible
    default (NOT 2 -- that's the holonomic-point value). gamma is tuned. Both
    are documented constants, not magic numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from needlesim.models.unicycle_needle import NeedleParams, State, rollout_variable
from needlesim.planning.base import BasePlannerConfig, PlannerBase
from needlesim.planning.dubins import DubinsPath, dubins_ccc, dubins_full

# ---------------------------------------------------------------------------
# THE TWO ISOLATED CHOICES. Change behaviour here, not scattered through code.
# ---------------------------------------------------------------------------


def edge_cost(
    path: DubinsPath, env, params: NeedleParams, clearance_weight: float
) -> float:
    """Cost of traversing a Dubins edge.

    PHASE A (current): return path.length. Pure length-only RRT*.

    PHASE B (swap in later): add a clearance penalty, e.g.
        length + clearance_weight * (penalty integrated along the edge),
    where the penalty is larger where clearance is SMALL (hugging obstacles is
    expensive). Roll the edge out (rollout_variable), sample clearance at each
    pose via env.clearance, and accumulate. Keep clearance_weight=0 equivalent
    to Phase A so you can A/B them.

    NOTE: once the penalty is nonzero, cost is no longer a metric and strict
    RRT*-optimality no longer holds. That's an accepted trade for safer paths;
    state it in the writeup rather than claiming the guarantee.
    """
    return path.length


def rewire_radius(n_nodes: int, gamma: float, dim: int, max_radius: float) -> float:
    """Shrinking RRT* neighbourhood radius.

        r(n) = min(max_radius, gamma * (log(n) / n) ** (1/dim))

    dim = 3 for a Dubins (x, y, theta) system (config-space dimension), NOT 2.
    gamma is tuned; too small -> tree never improves, too large -> every
    iteration scans most of the tree. Cap at max_radius so early (small-n)
    iterations don't use an enormous radius. Guard n_nodes < 2 (log blows up).
    """
    if n_nodes < 2:
        return max_radius
    return min(max_radius, gamma * (math.log(n_nodes) / n_nodes) ** (1 / dim))


# ---------------------------------------------------------------------------
# Containers -- delegable.
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """Tree vertex. cost_from_start is maintained incrementally as the tree
    grows and rewires -- keep it in sync whenever parent changes.

    Intentionally DIFFERENT from the RRT Node (carries cost bookkeeping and a
    Dubins edge instead of a single Control); do not unify them.
    """

    state: State
    parent: int | None = None
    cost_from_start: float = 0.0
    # The Dubins edge from parent -> this node, kept so the final path can be
    # reconstructed as an executable (control, dt) sequence.
    edge_from_parent: DubinsPath | None = None


@dataclass
class RRTStarResult:
    nodes: list[Node]
    path: list[State] | None
    control_dt_pairs: list | None  # concatenated (Control, dt) along the path
    success: bool
    n_iterations: int
    best_cost: float  # cost_from_start of the goal node, or inf


@dataclass
class RRTStarConfig(BasePlannerConfig):
    """RRT*-specific tunables. Shared fields come from BasePlannerConfig."""

    # rewire radius
    gamma: float = 40.0  # TUNE. Scales the shrinking radius.
    dim: int = 3  # config-space dim for Dubins (x,y,theta)
    max_radius: float = 40.0  # cap for small-n iterations [mm]
    # cost
    clearance_weight: float = 0.0  # 0.0 = length-only (Phase A). >0 = Phase B.
    margin: float = 0.0  # is_arc_free margin [mm]
    use_full_dubins: bool = True  # False = CCC-only (arc-only steering).
    # CCC-only collapses at realistic curvature (~98.5% sample rejection at
    # R=50mm in a 150mm world) -- keep the switch so that finding stays
    # reproducible.
    # STEERING HORIZON: reject any Dubins edge longer than this [mm]. Purpose is
    # to test whether forbidding the long heading-reconciliation loops (~314mm at
    # R=50mm) forces the tree onto short edges and lowers path cost, or simply
    # makes arbitrary-heading node pairs unconnectable and starves the tree. The
    # default inf imposes NO constraint, preserving existing behaviour exactly.
    # Checked immediately after the Dubins geometry returns and BEFORE the
    # collision rollout (cheap-first) in choose_parent/rewire/steer.
    max_edge_length: float = float("inf")


# ---------------------------------------------------------------------------
# The planner.
# ---------------------------------------------------------------------------


class RRTStar(PlannerBase):
    # __init__ is inherited from PlannerBase, which now does everything the
    # old RRTStar.__init__ did, plus the spacing-invariant assert.

    # --- steering: the Dubins bridge ---------------------------------------

    def steer(self, from_state: State, to_state: State) -> DubinsPath | None:
        """Exact Dubins connection from -> to, or None if unconnectable OR the
        edge collides. Composed convenience: geometry + collision in one call. choose_parent and rewire use the two halves separately in cheap-first order — see their bodies.
        """
        steer_fn = dubins_full if self.config.use_full_dubins else dubins_ccc
        path = steer_fn(
            from_state,
            to_state,
            self.params,
            self.config.edge_velocity,
            self.config.step_dt,
        )
        if path is None:
            return None
        if path.length > self.config.max_edge_length:
            return None

        if not self._edge_collision_free(from_state, path):
            return None
        return path

    def _edge_collision_free(self, from_state: State, path: DubinsPath) -> bool:
        states = rollout_variable(from_state, path.controls, self.params)
        for state in states:
            if not self.env.is_free(state, self.config.margin):
                return False
        return True

    # --- the two RRT* steps: the heart -------------------------------------

    def choose_parent(
        self, nodes: list[Node], new_state: State, neighbourhood: list[int]
    ):
        """Among neighbourhood nodes that can steer to new_state collision-free,
        pick the one giving the lowest cost_from_start for new_state.

        Returns (best_parent_idx, best_edge, best_cost) or None if no
        neighbour can connect.
        """
        steer_fn = dubins_full if self.config.use_full_dubins else dubins_ccc
        all_connected_paths = []
        for idx in neighbourhood:
            path = steer_fn(
                nodes[idx].state,
                new_state,
                self.params,
                self.config.edge_velocity,
                self.config.step_dt,
            )
            if path is None:
                continue
            # Steering horizon: reject over-long edges before the rollout (cheap:
            # length is already on the returned path). inf default = no-op.
            if path.length > self.config.max_edge_length:
                continue
            cost = nodes[idx].cost_from_start + edge_cost(
                path, self.env, self.params, self.config.clearance_weight
            )
            all_connected_paths.append((path, idx, cost))
        # Sort ascending by cost, then rollout in that order and take the FIRST
        # collision-free candidate: exact, because every later candidate is
        # costlier. Python's sort is stable, so exact cost ties keep
        # neighbourhood order (matching the old strict-< first-wins behaviour).
        #
        # TODO (Phase B): with a clearance penalty, this order is by LOWER
        # BOUND (path.length), not true cost. Stopping at the first feasible
        # candidate is then NOT exact -- you may only stop once the surviving
        # candidate's TRUE cost beats the next candidate's lower bound.
        # Otherwise a costlier-by-length edge with better clearance could win.
        paths_sorted_by_cost = sorted(all_connected_paths, key=lambda item: item[2])
        for path, idx, cost in paths_sorted_by_cost:
            if self._edge_collision_free(nodes[idx].state, path):
                return (idx, path, cost)
        return None

    def rewire(self, nodes: list[Node], new_idx: int, neighbourhood: list[int]) -> None:
        """For each neighbour, if reaching it THROUGH the new node is cheaper,
        re-parent it to the new node and update its cost."""
        steer_fn = dubins_full if self.config.use_full_dubins else dubins_ccc
        for idx in neighbourhood:
            if idx == new_idx or idx == nodes[new_idx].parent:
                continue
            path = steer_fn(
                nodes[new_idx].state,
                nodes[idx].state,
                self.params,
                self.config.edge_velocity,
                self.config.step_dt,
            )
            if path is None:
                continue
            # Steering horizon (see choose_parent): reject over-long edges before
            # the cost/rollout work. inf default = no-op.
            if path.length > self.config.max_edge_length:
                continue
            new_cost = nodes[new_idx].cost_from_start + edge_cost(
                path, self.env, self.params, self.config.clearance_weight
            )
            # LOWER-BOUND PRUNE: skip the expensive collision rollout when this
            # edge cannot improve on the neighbour's current cost. Exact today
            # because edge_cost is pure geometry (path.length). Under Phase B's
            # clearance penalty this REMAINS valid as a lower-bound prune: the
            # penalty is additive and non-negative, so path.length <= true cost.
            # Phase B must then compute the true (penalised) cost AFTER the
            # rollout for candidates that survive this bail.
            if new_cost >= nodes[idx].cost_from_start:
                continue
            if not self._edge_collision_free(nodes[new_idx].state, path):
                continue
            nodes[idx].parent = new_idx
            nodes[idx].cost_from_start = new_cost
            nodes[idx].edge_from_parent = path
        # NOTE: re-parenting i lowers its cost; i's descendants keep their (now
        # slightly stale) cost_from_start. Accepted simplification: paths stay
        # valid, only optimality is mildly affected. Not propagating the delta
        # through the subtree (Node has no child pointers; would need a scan).

    # --- assembly + main loop ----------------------------------------------

    def reconstruct(self, nodes: list[Node], goal_idx: int):
        """Walk parents to root; return (states, concatenated (control, dt)
        pairs, total path length). Uses edge_from_parent for the control
        sequence; the root has None."""
        states = []
        controls_lists = []
        total_length = 0.0
        current_node_idx = goal_idx
        while current_node_idx is not None:
            state = nodes[current_node_idx].state
            states.append(state)
            if nodes[current_node_idx].edge_from_parent is not None:
                controls_lists.append(nodes[current_node_idx].edge_from_parent.controls)
                total_length += nodes[current_node_idx].edge_from_parent.length
            current_node_idx = nodes[current_node_idx].parent
        states.reverse()
        controls_lists.reverse()
        controls_flat = []
        for one_list_controls in controls_lists:
            for pair in one_list_controls:
                controls_flat.append(pair)
        return states, controls_flat, total_length

    def plan(self, start: State, goal: State) -> RRTStarResult:
        """RRT* main loop.

        Unlike RRT, does NOT return on first goal contact -- RRT* keeps
        improving. Tracks the best goal node seen and its cost; returns the
        best at the end. The tree is always returned (for plotting), success
        or not.
        """
        self.goal = goal
        tree = [Node(start, parent=None, cost_from_start=0.0)]
        best_goal_idx, best_cost = None, np.inf
        for i in range(self.config.max_iterations):
            sampled = self.sample()  # State; theta is discarded below
            idx_near_target = self.nearest(tree, State(sampled.x, sampled.y, 0.0))
            near_state = tree[idx_near_target].state
            # Target heading is derived from nearest-node-toward-sample because
            # raw sampled headings are mostly CCC-unreachable (measured: ~87%
            # sample rejection before this derivation was introduced).
            theta = math.atan2(sampled.y - near_state.y, sampled.x - near_state.x)
            target = State(sampled.x, sampled.y, theta)

            r = rewire_radius(
                len(tree), self.config.gamma, self.config.dim, self.config.max_radius
            )
            neighbourhood_indices = self.near_indices(tree, target, r)
            parent = self.choose_parent(tree, target, neighbourhood_indices)
            if parent is None:
                continue
            parent_idx, edge, new_cost = parent
            tree.append(Node(target, parent_idx, new_cost, edge))
            new_idx = len(tree) - 1
            self.rewire(tree, new_idx, neighbourhood_indices)
            if self.reached_goal(target) and new_cost < best_cost:
                best_goal_idx, best_cost = new_idx, new_cost

        if best_goal_idx is None:
            return RRTStarResult(
                nodes=tree,
                path=None,
                control_dt_pairs=None,
                success=False,
                n_iterations=self.config.max_iterations,
                best_cost=np.inf,
            )
        path, pairs, true_cost = self.reconstruct(tree, best_goal_idx)
        return RRTStarResult(
            nodes=tree,
            path=path,
            control_dt_pairs=pairs,
            success=True,
            n_iterations=self.config.max_iterations,
            best_cost=true_cost,
        )
