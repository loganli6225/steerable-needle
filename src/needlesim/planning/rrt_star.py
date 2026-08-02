"""
RRT* planner for the steerable needle, built on CCC Dubins steering.

THIS IS YOUR TASK 4 FILE. Interface + skeleton laid out; the load-bearing
methods (choose_parent, rewire, the main loop) are yours. Per the working
agreement, cost/steering/rewiring logic is yours; plain containers and plotting
are delegable.

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
from needlesim.planning.dubins import DubinsPath, dubins_ccc

# ---------------------------------------------------------------------------
# THE TWO ISOLATED CHOICES. Change behaviour here, not scattered through code.
# ---------------------------------------------------------------------------


def edge_cost(
    path: DubinsPath, env, params: NeedleParams, clearance_weight: float
) -> float:
    """Cost of traversing a Dubins edge.

    IMPLEMENT ME.

    PHASE A (do first): return path.length. Pure length-only RRT*.

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

    IMPLEMENT ME (small).

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
    grows and rewires -- keep it in sync whenever parent changes."""

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
class RRTStarConfig:
    max_iterations: int = 5000
    goal_tolerance: float = 3.0
    goal_sample_rate: float = 0.05
    step_dt: float = 0.05
    edge_velocity: float = 5.0
    # rewire radius
    gamma: float = 40.0  # TUNE. Scales the shrinking radius.
    dim: int = 3  # config-space dim for Dubins (x,y,theta)
    max_radius: float = 40.0  # cap for small-n iterations [mm]
    # cost
    clearance_weight: float = 0.0  # 0.0 = length-only (Phase A). >0 = Phase B.
    margin: float = 0.0  # is_arc_free margin [mm]
    seed: int = 0


# ---------------------------------------------------------------------------
# The planner. Methods marked IMPLEMENT ME are yours.
# ---------------------------------------------------------------------------


class RRTStar:
    def __init__(self, env, params: NeedleParams, config: RRTStarConfig) -> None:
        self.env = env
        self.params = params  # MODEL params, never ground truth
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # --- steering: the Dubins bridge (delegable wrapper, but check it) -----

    def steer(self, from_state: State, to_state: State) -> DubinsPath | None:
        """Exact Dubins connection from -> to, or None if unconnectable OR the
        edge collides. IMPLEMENT ME (small wrapper):
            1. path = dubins_ccc(from_state, to_state, self.params,
                                 config.edge_velocity, config.step_dt)
            2. if path is None: return None
            3. collision-check the edge: roll it out and require is_free at
               every pose (or use a variable-dt arc check). Return None if it
               hits. Otherwise return path.
        This is the ONE connection primitive choose_parent and rewire both use.
        """
        path = dubins_ccc(
            from_state,
            to_state,
            self.params,
            self.config.edge_velocity,
            self.config.step_dt,
        )
        if path is None:
            return None

        states = rollout_variable(from_state, path.controls, self.params)
        for state in states:
            if not self.env.is_free(state, self.config.margin):
                return None
        return path

    # --- sampling / nearest (same as RRT; can reuse) -----------------------

    def sample(self) -> tuple[float, float]:
        """Random pose in bounds, goal-biased. IMPLEMENT (same as your RRT)."""
        if self.rng.random() < self.config.goal_sample_rate:
            return self.goal.x, self.goal.y
        x_min, y_min, x_max, y_max = self.env.bounds
        x = self.rng.uniform(x_min, x_max)
        y = self.rng.uniform(y_min, y_max)
        return x, y

    def near_indices(
        self, nodes: list[Node], target: State, radius: float
    ) -> list[int]:
        """Indices of all nodes within `radius` of target (position distance is
        fine here). IMPLEMENT ME. Unlike RRT's single nearest(), RRT* needs the
        whole neighbourhood for choose-parent and rewire."""
        indices_within_radius = []
        for idx, node in enumerate(nodes):
            if self.distance(target, node.state) <= radius:
                indices_within_radius.append(idx)
        return indices_within_radius

    # --- the two RRT* steps: the heart -------------------------------------

    def choose_parent(
        self, nodes: list[Node], new_state: State, neighbourhood: list[int]
    ):
        """Among neighbourhood nodes that can steer to new_state collision-free,
        pick the one giving the lowest cost_from_start for new_state.

        IMPLEMENT ME. Returns (best_parent_idx, best_edge, best_cost) or None if
        no neighbour can connect. For each candidate i:
            edge = self.steer(nodes[i].state, new_state)
            if edge is None: skip
            c = nodes[i].cost_from_start + edge_cost(edge, ...)
            keep the minimum.
        """
        best_parent_idx = None
        best_edge = None
        best_cost = np.inf
        for idx in neighbourhood:
            edge = self.steer(nodes[idx].state, new_state)
            if edge is not None:
                cost = nodes[idx].cost_from_start + edge_cost(
                    edge, self.env, self.params, self.config.clearance_weight
                )
                if cost < best_cost:
                    best_parent_idx = idx
                    best_edge = edge
                    best_cost = cost
        if best_edge is None:
            return None
        return (best_parent_idx, best_edge, best_cost)

    def rewire(self, nodes: list[Node], new_idx: int, neighbourhood: list[int]) -> None:
        """For each neighbour, if reaching it THROUGH new node is cheaper,
        re-parent it to new node and update its cost.

        IMPLEMENT ME. For each i in neighbourhood (i != new node's parent):
            edge = self.steer(nodes[new_idx].state, nodes[i].state)
            if edge is None: skip
            new_cost = nodes[new_idx].cost_from_start + edge_cost(edge, ...)
            if new_cost < nodes[i].cost_from_start:
                nodes[i].parent = new_idx
                nodes[i].cost_from_start = new_cost
                nodes[i].edge_from_parent = edge
                # NOTE: descendants of i now have stale cost_from_start. For a
                # first correct version, propagate the delta to the subtree, OR
                # accept slightly stale costs (common simplification). Document
                # whichever you choose -- it affects optimality, not validity.
        """
        for idx in neighbourhood:
            if idx == new_idx or idx == nodes[new_idx].parent:
                continue
            edge = self.steer(nodes[new_idx].state, nodes[idx].state)
            if edge is None:
                continue
            new_cost = nodes[new_idx].cost_from_start + edge_cost(
                edge, self.env, self.params, self.config.clearance_weight
            )
            if new_cost < nodes[idx].cost_from_start:
                nodes[idx].parent = new_idx
                nodes[idx].cost_from_start = new_cost
                nodes[idx].edge_from_parent = edge
        # NOTE: re-parenting i lowers its cost; i's descendants keep their (now
        # slightly stale) cost_from_start. Accepted simplification: paths stay
        # valid, only optimality is mildly affected. Not propagating the delta
        # through the subtree (Node has no child pointers; would need a scan).

    # --- goal + assembly + main loop ---------------------------------------

    def distance(self, a: State, b: State) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def reached_goal(self, state: State) -> bool:
        """IMPLEMENT (small). Position within config.goal_tolerance of goal."""
        return self.distance(state, self.goal) < self.config.goal_tolerance

    def reconstruct(self, nodes: list[Node], goal_idx: int):
        """Walk parents to root; return (states, concatenated (control,dt) pairs).
        IMPLEMENT (bookkeeping, delegable). Use edge_from_parent for the
        control sequence; root has None."""
        states = []
        controls_lists = []
        current_node_idx = goal_idx
        while current_node_idx is not None:
            state = nodes[current_node_idx].state
            states.append(state)
            if nodes[current_node_idx].edge_from_parent is not None:
                controls_lists.append(nodes[current_node_idx].edge_from_parent.controls)
            current_node_idx = nodes[current_node_idx].parent
        states.reverse()
        controls_lists.reverse()
        controls_flat = []
        for one_list_controls in controls_lists:
            for pair in one_list_controls:
                controls_flat.append(pair)
        return states, controls_flat

    def nearest(self, nodes: list[Node], target_x: float, target_y: float) -> int:
        """Return the INDEX of the tree node closest to `target`.

        IMPLEMENT ME. "Closest" is defined by self.distance. Linear scan over
        nodes is fine to start (this is O(n) per iteration; a KD-tree is a
        later optimisation, do not bother now).
        """
        closest_node_idx = -1
        closest_distance = np.inf
        for idx, node in enumerate(nodes):
            node_distance = math.hypot(node.state.x - target_x, node.state.y - target_y)
            if closest_distance > node_distance:
                closest_distance = node_distance
                closest_node_idx = idx
        return closest_node_idx

    def plan(self, start: State, goal: State) -> RRTStarResult:
        """RRT* main loop. IMPLEMENT ME.

        Unlike RRT, do NOT return on first goal contact -- RRT* keeps improving.
        Track the best goal node seen and its cost; return the best at the end.

        Skeleton:
            self.goal = goal
            nodes = [Node(start, parent=None, cost_from_start=0.0)]
            best_goal_idx, best_cost = None, inf
            for it in 1..max_iterations:
                target = sample()
                # nearest single node to steer TOWARD (like RRT), then the
                # steered new_state; OR sample a state and use it directly.
                r = rewire_radius(len(nodes), gamma, dim, max_radius)
                neigh = near_indices(nodes, new_state, r)
                cp = choose_parent(nodes, new_state, neigh)
                if cp is None: continue
                parent_idx, edge, cost = cp
                append Node(new_state, parent_idx, cost, edge); new_idx = last
                rewire(nodes, new_idx, neigh)
                if reached_goal(new_state) and cost < best_cost:
                    best_goal_idx, best_cost = new_idx, cost
            build result from best_goal_idx (or failure with the tree).

        Return RRTStarResult either way (tree always returned for plotting).
        """
        self.goal = goal
        tree = [Node(start, parent=None, cost_from_start=0.0)]
        best_goal_idx, best_cost = None, np.inf
        for i in range(self.config.max_iterations):
            target_x, target_y = self.sample()
            idx_near_target = self.nearest(tree, target_x, target_y)
            near_state = tree[idx_near_target].state
            theta = math.atan2(target_y - near_state.y, target_x - near_state.x)
            target = State(target_x, target_y, theta)

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
        path, pairs = self.reconstruct(tree, best_goal_idx)
        return RRTStarResult(
            nodes=tree,
            path=path,
            control_dt_pairs=pairs,
            success=True,
            n_iterations=self.config.max_iterations,
            best_cost=best_cost,
        )
