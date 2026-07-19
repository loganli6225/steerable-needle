"""2D grid environment with a signed distance field (SDF).

THIS IS YOUR TASK 2 FILE. The interface is defined so Task 3's planner can be
built against it, but the geometry is left for you to implement. Fill in the
methods marked IMPLEMENT ME.

Representation
--------------
The world is a rectangle in millimetres. Obstacles are painted onto a boolean
occupancy mask, then a SIGNED DISTANCE FIELD is baked ONCE at construction and
frozen. Every query afterwards is a pure lookup against that fixed array.

    sdf[i, j] = signed distance from cell (i, j) to the nearest obstacle SURFACE
                > 0  outside all obstacles  (value = clearance)
                = 0  exactly on a surface
                < 0  inside an obstacle

So there is ONE number per cell. Collision is its sign; clearance is its
magnitude. Nothing is stored per query and nothing changes after construction;
the only thing that varies between queries is which cell you look up.

Build the SDF with two Euclidean distance transforms
(scipy.ndimage.distance_transform_edt): one on the free space, one on the
obstacle space, combined as  outside_distance - inside_distance.

Coordinate frames
-----------------
Poses live in continuous world coordinates (mm). The grid is integer indices.
ALL conversion goes through world_to_grid / grid_to_world so any half-cell
offset lives in exactly one place. This transform is the single most likely
home for an off-by-one bug; test it directly (see the resolution-convergence
test in tests/test_grid_environment.py).
"""

from __future__ import annotations

import numpy as np

from needlesim.models.unicycle_needle import Control, NeedleParams, State, step


class GridEnvironment:
    """A rectangular 2D world with obstacles and a signed distance field.

    Typical use:
        env = GridEnvironment(width=100.0, height=100.0, resolution=0.5)
        env.add_circle(cx=50.0, cy=50.0, r=15.0)
        env.bake()                      # freezes the SDF; call once, after obstacles
        env.is_free(State(10, 10, 0))   # queries are pure lookups thereafter
    """

    def __init__(self, width: float, height: float, resolution: float) -> None:
        """Create an empty world (no obstacles yet).

        Args:
            width:  world width  [mm] (x extent, from 0 to width).
            height: world height [mm] (y extent, from 0 to height).
            resolution: grid spacing [mm] per cell. Smaller = finer = slower.

        Sets up an empty boolean occupancy mask. The SDF is NOT built here --
        obstacles are added first, then bake() computes the field.
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # Grid shape. Decide and document your convention: which axis is rows,
        # which is columns, and how many cells span the world. Keep it
        # consistent with world_to_grid / grid_to_world below.
        self.n_rows = int(np.ceil(height / resolution))
        self.n_cols = int(np.ceil(width / resolution))

        # Occupancy: True where an obstacle is. Painted by add_* helpers.
        self.occupancy = np.zeros((self.n_rows, self.n_cols), dtype=bool)

        # Signed distance field, filled by bake(). None until then.
        self.sdf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Building the world (obstacles + bake). You may let these be written
    # for you -- they are scaffolding, not load-bearing geometry -- but the
    # world<->grid transform below is yours.
    # ------------------------------------------------------------------

    def add_circle(self, cx: float, cy: float, r: float) -> None:
        """Paint a filled circular obstacle onto the occupancy mask.

        IMPLEMENT ME (or delegate -- this is a helper, not core geometry).
        Must be called before bake().
        """
        i_min, j_min = self.world_to_grid(cx - r, cy - r)
        i_max, j_max = self.world_to_grid(cx + r, cy + r)
        i_min = max(i_min, 0)
        j_min = max(j_min, 0)
        i_max = min(i_max, self.n_rows - 1)
        j_max = min(j_max, self.n_cols - 1)

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                x, y = self.grid_to_world(i, j)
                if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                    self.occupancy[i, j] = True

    def add_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Paint a filled axis-aligned rectangular obstacle.

        IMPLEMENT ME (or delegate). Must be called before bake().
        """
        i_1, j_1 = self.world_to_grid(x0, y0)
        i_2, j_2 = self.world_to_grid(x1, y1)
        i_min, i_max, j_min, j_max = (
            min(i_1, i_2),
            max(i_1, i_2),
            min(j_1, j_2),
            max(j_1, j_2),
        )
        i_min = max(i_min, 0)
        j_min = max(j_min, 0)
        i_max = min(i_max, self.n_rows - 1)
        j_max = min(j_max, self.n_cols - 1)

        self.occupancy[i_min : i_max + 1, j_min : j_max + 1] = True

    def bake(self) -> None:
        from scipy.ndimage import distance_transform_edt as edt

        """Compute the signed distance field from the occupancy mask, ONCE.

        IMPLEMENT ME. Two distance transforms, combined so that the result is
        negative inside obstacles and positive outside, in millimetres (mind
        the resolution: the transform returns distances in CELLS, multiply by
        resolution to get mm). After this, self.sdf is frozen and every query
        reads from it.
        """
        outside = edt(~self.occupancy)
        inside = edt(self.occupancy)

        if self.occupancy.all():
            raise ValueError("bake(): entire map is obstacle")

        self.sdf = (outside - inside) * self.resolution

    # ------------------------------------------------------------------
    # Coordinate transform -- YOURS to write. Single source of the off-by-one.
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Map continuous world coordinates (mm) to integer grid indices.

        IMPLEMENT ME. Define and document your rounding/offset convention (do
        cell centres sit at integer multiples of resolution, or at half-cell
        offsets?). Whatever you choose, grid_to_world must invert it.
        """
        i = int(y // self.resolution)
        j = int(x // self.resolution)
        return (i, j)

    def grid_to_world(self, i: int, j: int) -> tuple[float, float]:
        """Map integer grid indices back to world coordinates (mm).

        IMPLEMENT ME. Must be the inverse of world_to_grid (to within
        half-cell quantisation).
        """
        x = (j + 0.5) * self.resolution
        y = (i + 0.5) * self.resolution
        return (x, y)

    # ------------------------------------------------------------------
    # The four queries the planner needs. YOURS to write.
    # ------------------------------------------------------------------

    def is_free(self, state: State, margin: float = 0.0) -> bool:
        """True if the pose has at least `margin` mm of clearance.

        IMPLEMENT ME. Look up the SDF at the pose's cell; return
        (sdf_value >= margin). A margin > 0 both accounts for the needle's
        physical radius and keeps poses off the numerically dangerous exact
        boundary. A pose outside the world bounds should count as NOT free.

        Note only (x, y) matter here; theta does not affect point collision.
        """
        return self.clearance(state) >= margin

    def clearance(self, state: State) -> float:
        """Signed distance from the pose to the nearest obstacle surface [mm].

        IMPLEMENT ME. This is just the SDF value at the pose's cell: positive
        outside (distance to nearest obstacle), negative inside. Decide what to
        return for a pose outside the world bounds and document it.
        """
        if self.sdf is None:
            raise RuntimeError("call bake() before querying")
        if not (0 <= state.x <= self.width and 0 <= state.y <= self.height):
            return (
                -1.0
            )  # TODO Out-of-bounds should eventually return a graded signed distance for RRT* and Phase 4
        i, j = self.world_to_grid(state.x, state.y)
        return self.sdf[i, j]

    def is_arc_free(
        self,
        state: State,
        control: Control,
        dt: float,
        n_samples: int,
        params: NeedleParams,
        margin: float = 0.0,
    ) -> bool:
        """True if the swept arc from `state` under `control` is collision-free.

        IMPLEMENT ME. Roll the needle forward using the Task 1 model: sample
        n_samples poses along the arc (reuse `step`, do NOT reimplement the
        kinematics here) and require is_free(margin) at every one. This is the
        query RRT calls most; it will dominate runtime, so keep it lean.

        Design notes you should settle:
        - Include both endpoints in the samples, or just interior points?
        - n_samples should be dense enough that no arc step skips over a thin
          obstacle. What relates the right n_samples to arc length and
          resolution? (You do not have to solve this perfectly now, but write
          down your assumption.)
        """
        step_spacing = abs(control.v) * dt
        assert step_spacing <= 0.5 * self.resolution, (
            f"arc sample spacing {step_spacing:.3f} mm exceeds half-cell "
            f"{0.5*self.resolution:.3f} mm; caller must use finer dt"
        )

        if not self.is_free(state, margin):
            return False
        new_state = state
        for _ in range(n_samples):
            new_state = step(new_state, control, dt, params)
            if not self.is_free(new_state, margin):
                return False
        return True

    # ------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Sampling domain as (x_min, y_min, x_max, y_max) in mm.

        IMPLEMENT ME (trivial): the planner samples poses inside this.
        """
        return (0, 0, self.width, self.height)
